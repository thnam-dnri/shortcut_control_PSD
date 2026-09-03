#!/usr/bin/env python3
"""Run isotope-independent quality control on a session of ROOT waveform files.

The routine checks DAQ/electronics health only.  It deliberately does not use
source-dependent energy spectra, peak positions, FWHM, absolute amplitudes, or
trigger-rate limits as universal quality criteria.

The NKFADC500 files contain an unsplit composite ``event`` branch with many
small baskets. Native ROOT is used for bounded sampling because uproot's
structured-branch array reader is impractically slow for this layout.

ROOT sample numbers are one-based in the report.  Each waveform is first
baseline-subtracted using the mean of samples 1..200.  Baseline RMS is then
measured on the corrected samples 1..200.  Raw samples 1..1000 remain the
pedestal/drift inspection window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ROOT
except Exception:  # pragma: no cover - depends on the external ROOT setup
    ROOT = None


EXPECTED_ENTRIES = 100_000
EXPECTED_SAMPLES = 4_500
EXPECTED_SAMPLE_EVENTS = 1_000
BASELINE_SUBTRACTION_SLICE = slice(0, 200)
BASELINE_SUBTRACTION_SAMPLES_INCLUSIVE = [1, 200]
BASELINE_SLICE = slice(0, 1_000)
BASELINE_SAMPLES_INCLUSIVE = [1, 1_000]
BASELINE_NOISE_SLICE = BASELINE_SUBTRACTION_SLICE
BASELINE_NOISE_SAMPLES_INCLUSIVE = BASELINE_SUBTRACTION_SAMPLES_INCLUSIVE
BASELINE_BLOCKS = 10
TRIGGER_TIMER_PERIOD_S = (2**24) / 1_000_000.0
DEFAULT_NOISE_WARN_FACTOR = 1.25
DEFAULT_NOISE_FAIL_FACTOR = 1.50

EXPECTED_METADATA = {
    "sample_rate_msps": "250",
    "sample_period_ns": "4",
    "stored_samples": "4500",
    "pretrigger_us": "6",
    "posttrigger_us": "12",
    "trigger_threshold_adc": "10",
}


class QualityError(RuntimeError):
    """Raised for an unrecoverable input or configuration problem."""


def quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float | None]:
    """Return JSON-safe quantiles for finite values."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    names = {0.01: "p01", 0.50: "p50", 0.95: "p95", 0.99: "p99"}
    if finite.size == 0:
        return {names[p]: None for p in probabilities}
    return {names[p]: float(np.quantile(finite, p)) for p in probabilities}


def finite_float(value: float | int | np.number | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def read_named_title(root_file: Any, name: str) -> str | None:
    """Read the title of a TNamed metadata object."""

    try:
        metadata = root_file.Get(name)
        return str(metadata.GetTitle()) if metadata else None
    except Exception:
        return None


def unwrap_trigger_timestamps(timestamps: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Unwrap the digitizer's 24-bit microsecond timer when needed.

    The current DAQ writer already performs this correction.  Keeping the
    correction here makes the QC robust to older files that stored the raw
    timer value.  Small backward steps remain errors.
    """

    values = np.asarray(timestamps, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return values.copy(), 0, 0

    unwrapped = values.copy()
    offset = 0.0
    rollover_count = 0
    nonrollover_backward_count = 0
    for index in range(1, len(values)):
        previous = values[index - 1] + offset
        current = values[index] + offset
        if current < previous:
            backward = previous - current
            if backward > 0.5 * TRIGGER_TIMER_PERIOD_S:
                offset += TRIGGER_TIMER_PERIOD_S
                rollover_count += 1
            else:
                nonrollover_backward_count += 1
        unwrapped[index] = values[index] + offset
    return unwrapped, rollover_count, nonrollover_backward_count


def read_sample(
    tree: Any,
    entries: int,
    sample_events: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read either the first N events or N deterministic blocks with ROOT."""

    if entries <= 0:
        raise QualityError("tree contains no entries")
    if sample_events <= 0:
        raise QualityError("sample_events must be positive")
    if ROOT is None:
        raise QualityError(
            "PyROOT is unavailable; source /mnt/Data/FADC500/root/bin/thisroot.sh "
            "before running QC"
        )

    count = min(sample_events, entries)
    if mode == "first" or count < BASELINE_BLOCKS:
        starts = [0]
        block_lengths = [count]
    else:
        block_count = min(BASELINE_BLOCKS, count)
        block_length = count // block_count
        starts = np.linspace(0, entries - block_length, block_count, dtype=np.int64).tolist()
        block_lengths = [block_length] * block_count

    event_dtype = np.dtype(
        [
            ("event_id", np.uint32),
            ("waveform", np.float32, (EXPECTED_SAMPLES,)),
            ("trigger_time_s", np.float32),
        ]
    )
    arrays: list[np.ndarray] = []
    entry_indices: list[np.ndarray] = []
    for start, block_length in zip(starts, block_lengths):
        stop = min(int(start) + int(block_length), entries)
        count = stop - int(start)
        if count <= 0:
            continue

        def draw_values(expression: str, estimate: int) -> np.ndarray:
            tree.SetEstimate(int(estimate))
            selected = int(tree.Draw(expression, "", "goff", count, int(start)))
            if selected != estimate:
                raise QualityError(
                    f"ROOT sample {expression!r} returned {selected} values; "
                    f"expected {estimate}"
                )
            # TTree::Draw exposes Double_t buffers. Copy before the next Draw
            # call replaces the buffer.
            return np.frombuffer(
                tree.GetV1(), dtype=np.float64, count=estimate
            ).copy()

        event_ids = draw_values("event_id", count).astype(np.uint32, copy=False)
        waveform_values = draw_values(
            "waveform", count * EXPECTED_SAMPLES
        ).astype(np.float32, copy=False)
        timestamps = draw_values("trigger_time_s", count).astype(
            np.float32, copy=False
        )

        block = np.empty(count, dtype=event_dtype)
        block["event_id"] = event_ids
        block["waveform"] = waveform_values.reshape(count, EXPECTED_SAMPLES)
        block["trigger_time_s"] = timestamps
        arrays.append(block)
        entry_indices.append(np.arange(int(start), stop, dtype=np.int64))

    if not arrays:
        raise QualityError("ROOT sample returned no events")
    return np.concatenate(arrays), np.concatenate(entry_indices)


def analyze_file(path: Path, sample_events: int, sample_mode: str) -> dict[str, Any]:
    """Extract the five source-independent QC metric groups from one ROOT file."""

    report: dict[str, Any] = {
        "file": str(path),
        "file_name": path.name,
        "status": "FAIL",
        "failures": [],
        "warnings": [],
    }

    try:
        if ROOT is None:
            raise QualityError(
                "PyROOT is unavailable; source /mnt/Data/FADC500/root/bin/thisroot.sh "
                "before running QC"
            )
        root_file = ROOT.TFile.Open(str(path), "READ")
        if not root_file or root_file.IsZombie():
            raise QualityError("cannot open ROOT file or file is a zombie")
        try:
            tree = root_file.Get("HPGE")
            if not tree:
                raise QualityError("missing HPGE tree")
            entries = int(tree.GetEntries())
            event, entry_indices = read_sample(tree, entries, sample_events, sample_mode)

            metadata = {
                name: read_named_title(root_file, name)
                for name in EXPECTED_METADATA
            }
        finally:
            root_file.Close()

    except Exception as error:
        report["failures"].append(f"cannot read ROOT input: {error}")
        return report

    report["integrity"] = {
        "entries": entries,
        "expected_entries": EXPECTED_ENTRIES,
        "sampled_entries": int(len(event)),
        "expected_sampled_entries": int(min(sample_events, EXPECTED_ENTRIES)),
        "sample_mode": sample_mode,
        "waveform_shape": list(event["waveform"].shape),
        "expected_waveform_samples": EXPECTED_SAMPLES,
        "metadata": metadata,
        "expected_metadata": EXPECTED_METADATA,
    }

    try:
        waveform = np.asarray(event["waveform"], dtype=np.float64)
        event_ids = np.asarray(event["event_id"])
        timestamps = np.asarray(event["trigger_time_s"], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as error:
        report["failures"].append(f"missing or invalid event fields: {error}")
        return report

    if entries != EXPECTED_ENTRIES:
        report["failures"].append(
            f"entry count {entries} != expected {EXPECTED_ENTRIES}"
        )
    if waveform.ndim != 2 or waveform.shape[1] != EXPECTED_SAMPLES:
        report["failures"].append(
            f"waveform shape {list(waveform.shape)} != expected (*, {EXPECTED_SAMPLES})"
        )
    if len(event) != min(sample_events, entries):
        report["failures"].append("requested waveform sample could not be read completely")

    metadata_mismatches = [
        f"{name}={metadata[name]!r} (expected {expected!r})"
        for name, expected in EXPECTED_METADATA.items()
        if metadata[name] != expected
    ]
    if metadata_mismatches:
        report["failures"].append("metadata mismatch: " + ", ".join(metadata_mismatches))

    finite_samples = np.isfinite(waveform)
    finite_event = np.all(finite_samples, axis=1) if len(event) else np.array([], dtype=bool)
    nonfinite_sample_fraction = 1.0 - float(finite_samples.mean()) if finite_samples.size else 1.0
    nonfinite_event_fraction = 1.0 - float(finite_event.mean()) if finite_event.size else 1.0
    if nonfinite_sample_fraction > 0.0:
        report["failures"].append("non-finite waveform sample detected")

    expected_event_ids = entry_indices + 1
    event_id_sequence = bool(np.array_equal(event_ids.astype(np.int64), expected_event_ids))
    if not event_id_sequence:
        report["failures"].append("event IDs are not sequential for the sampled entries")

    timestamp_finite = bool(np.isfinite(timestamps).all()) if timestamps.size else False
    unwrapped_timestamps, rollover_count, nonrollover_backward_count = unwrap_trigger_timestamps(timestamps)
    raw_deltas = (
        np.diff(unwrapped_timestamps)
        if unwrapped_timestamps.size > 1 else np.array([], dtype=np.float64)
    )
    contiguous_sample = (
        np.diff(entry_indices) == 1
        if entry_indices.size > 1 else np.array([], dtype=bool)
    )
    contiguous_deltas = raw_deltas[contiguous_sample]
    negative_delta_count = int(np.count_nonzero(raw_deltas < 0.0))
    nonpositive_delta_fraction = (
        float(np.mean(contiguous_deltas <= 0.0)) if contiguous_deltas.size else 0.0
    )
    if not timestamp_finite:
        report["failures"].append("non-finite trigger timestamp detected")
    if nonrollover_backward_count or negative_delta_count:
        report["failures"].append("trigger timestamps decrease outside a recognized timer rollover")

    # Estimate and subtract each event's baseline using samples 1..200 before
    # measuring baseline RMS.  Keep the raw offset for pedestal/drift checks.
    baseline_reference = waveform[:, BASELINE_SUBTRACTION_SLICE]
    valid_baseline_reference = np.isfinite(baseline_reference).all(axis=1)
    baseline_offset = np.full(len(waveform), np.nan, dtype=np.float64)
    if np.any(valid_baseline_reference):
        baseline_offset[valid_baseline_reference] = baseline_reference[
            valid_baseline_reference
        ].mean(axis=1)
    baseline_corrected = waveform - baseline_offset[:, None]
    corrected_noise_window = baseline_corrected[:, BASELINE_NOISE_SLICE]
    valid_baseline_noise = np.isfinite(corrected_noise_window).all(axis=1)
    baseline_noise = (
        np.sqrt(np.mean(np.square(corrected_noise_window[valid_baseline_noise]), axis=1))
        if np.any(valid_baseline_noise) else np.array([])
    )

    baseline_block_medians: list[float] = []
    if baseline_offset.size:
        for block in np.array_split(baseline_offset, min(BASELINE_BLOCKS, len(baseline_offset))):
            finite_block = block[np.isfinite(block)]
            if len(finite_block):
                baseline_block_medians.append(float(np.median(finite_block)))
    drift_range = (
        float(max(baseline_block_medians) - min(baseline_block_medians))
        if baseline_block_medians else None
    )
    first_last_drift = (
        float(baseline_block_medians[-1] - baseline_block_medians[0])
        if len(baseline_block_medians) >= 2 else None
    )

    report["baseline"] = {
        "window_samples_inclusive": BASELINE_SAMPLES_INCLUSIVE,
        "window_numpy_slice": "0:1000",
        "window_duration_us": 4.0,
        "subtraction_samples_inclusive": BASELINE_SUBTRACTION_SAMPLES_INCLUSIVE,
        "subtraction_numpy_slice": "0:200",
        "subtraction_window_duration_us": 0.8,
        "noise_samples_inclusive": BASELINE_NOISE_SAMPLES_INCLUSIVE,
        "noise_numpy_slice": "0:200",
        "noise_window_duration_us": 0.8,
        "valid_baseline_event_fraction": float(valid_baseline_noise.mean()) if len(valid_baseline_noise) else 0.0,
        "baseline_offset_adc": quantiles(baseline_offset, (0.01, 0.50, 0.99)),
        "baseline_noise_rms_adc": quantiles(baseline_noise, (0.50, 0.95, 0.99)),
        "block_median_adc": baseline_block_medians,
        "block_drift_range_adc": drift_range,
        "first_to_last_block_drift_adc": first_last_drift,
    }

    lower_rail_events = np.any(waveform <= 0.0, axis=1) if len(event) else np.array([], dtype=bool)
    upper_rail_events = np.any(waveform >= 4095.0, axis=1) if len(event) else np.array([], dtype=bool)
    flatline_events = np.ptp(waveform, axis=1) == 0.0 if len(event) else np.array([], dtype=bool)
    duplicate_events = (
        np.all(waveform[1:] == waveform[:-1], axis=1) & contiguous_sample
        if len(event) > 1 else np.array([], dtype=bool)
    )
    report["waveform_integrity"] = {
        "finite_sample_fraction": float(finite_samples.mean()) if finite_samples.size else 0.0,
        "nonfinite_sample_fraction": nonfinite_sample_fraction,
        "nonfinite_event_fraction": nonfinite_event_fraction,
        "lower_rail_event_fraction": float(lower_rail_events.mean()) if len(event) else 1.0,
        "upper_rail_event_fraction": float(upper_rail_events.mean()) if len(event) else 1.0,
        "flatline_event_fraction": float(flatline_events.mean()) if len(event) else 1.0,
        "duplicate_consecutive_event_fraction": float(duplicate_events.mean()) if len(duplicate_events) else 0.0,
    }
    if np.any(lower_rail_events) or np.any(upper_rail_events):
        report["warnings"].append("ADC rail/clipping events detected; review against source and gain")
    if np.any(flatline_events):
        report["warnings"].append("flatline waveform detected")
    if np.any(duplicate_events):
        report["warnings"].append("consecutive duplicate waveform detected")

    positive_deltas = contiguous_deltas[contiguous_deltas > 0.0]
    report["timing"] = {
        "timestamp_finite": timestamp_finite,
        "timestamp_monotonic": negative_delta_count == 0 and nonrollover_backward_count == 0,
        "negative_delta_count_after_unwrap": negative_delta_count,
        "nonrollover_backward_count": nonrollover_backward_count,
        "recognized_timer_rollover_count": rollover_count,
        "nonpositive_delta_fraction": nonpositive_delta_fraction,
        "contiguous_timestamp_delta_s": quantiles(positive_deltas, (0.50, 0.95, 0.99)),
        "observed_rate_hz": (
            float((entry_indices[-1] - entry_indices[0]) / (unwrapped_timestamps[-1] - unwrapped_timestamps[0]))
            if len(unwrapped_timestamps) > 1 and unwrapped_timestamps[-1] > unwrapped_timestamps[0] else None
        ),
        "rate_used_as_pass_fail": False,
    }

    report["diagnostics"] = {
        "sampled_entry_start": int(entry_indices[0]) if len(entry_indices) else None,
        "sampled_entry_end": int(entry_indices[-1]) if len(entry_indices) else None,
    }
    if report["failures"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"
    return report


def compare_to_reference(
    report: dict[str, Any],
    reference: dict[str, Any],
    noise_warn_factor: float,
    noise_fail_factor: float,
    pedestal_warn_adc: float,
    pedestal_fail_adc: float,
    drift_warn_adc: float,
    drift_fail_adc: float,
) -> None:
    """Apply source-independent reference comparisons and update status."""

    if report["status"] == "FAIL":
        return

    ref_baseline = reference.get("baseline", {})
    test_baseline = report.get("baseline", {})
    ref_noise = ref_baseline.get("baseline_noise_rms_adc", {})
    test_noise = test_baseline.get("baseline_noise_rms_adc", {})

    for quantile_name in ("p95", "p99"):
        ref_value = ref_noise.get(quantile_name)
        test_value = test_noise.get(quantile_name)
        if ref_value is None or test_value is None:
            report["failures"].append(f"missing baseline noise {quantile_name}")
            continue
        if test_value > ref_value * noise_fail_factor:
            report["failures"].append(
                f"baseline noise {quantile_name}={test_value:.3f} ADC exceeds "
                f"reference-based failure limit {ref_value * noise_fail_factor:.3f} ADC"
            )
        elif test_value > ref_value * noise_warn_factor:
            report["warnings"].append(
                f"baseline noise {quantile_name}={test_value:.3f} ADC is elevated "
                f"relative to reference {ref_value:.3f} ADC"
            )

    ref_offset = ref_baseline.get("baseline_offset_adc", {}).get("p50")
    test_offset = test_baseline.get("baseline_offset_adc", {}).get("p50")
    if ref_offset is not None and test_offset is not None:
        pedestal_delta = abs(test_offset - ref_offset)
        if pedestal_delta > pedestal_fail_adc:
            report["failures"].append(
                f"baseline offset shift={pedestal_delta:.3f} ADC exceeds "
                f"failure limit {pedestal_fail_adc:.3f} ADC"
            )
        elif pedestal_delta > pedestal_warn_adc:
            report["warnings"].append(
                f"baseline offset shift={pedestal_delta:.3f} ADC exceeds "
                f"warning limit {pedestal_warn_adc:.3f} ADC"
            )

    drift = test_baseline.get("block_drift_range_adc")
    if drift is not None:
        if drift > drift_fail_adc:
            report["failures"].append(
                f"baseline block drift range={drift:.3f} ADC exceeds failure limit {drift_fail_adc:.3f} ADC"
            )
        elif drift > drift_warn_adc:
            report["warnings"].append(
                f"baseline block drift range={drift:.3f} ADC exceeds warning limit {drift_warn_adc:.3f} ADC"
            )

    integrity = report.get("waveform_integrity", {})
    if integrity.get("duplicate_consecutive_event_fraction", 0.0) > 0.01:
        report["failures"].append("more than 1% consecutive duplicate waveforms")
    if integrity.get("flatline_event_fraction", 0.0) > 0.01:
        report["failures"].append("more than 1% flatline waveforms")

    if report["failures"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"


def format_cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def flatten_row(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report.get("baseline", {})
    noise = baseline.get("baseline_noise_rms_adc", {})
    offset = baseline.get("baseline_offset_adc", {})
    integrity = report.get("waveform_integrity", {})
    timing = report.get("timing", {})
    return {
        "status": report.get("status"),
        "file": report.get("file_name"),
        "entries": report.get("integrity", {}).get("entries"),
        "sampled_entries": report.get("integrity", {}).get("sampled_entries"),
        "baseline_offset_median_adc": offset.get("p50"),
        "baseline_noise_p50_adc": noise.get("p50"),
        "baseline_noise_p95_adc": noise.get("p95"),
        "baseline_noise_p99_adc": noise.get("p99"),
        "baseline_drift_range_adc": baseline.get("block_drift_range_adc"),
        "lower_rail_fraction": integrity.get("lower_rail_event_fraction"),
        "upper_rail_fraction": integrity.get("upper_rail_event_fraction"),
        "duplicate_fraction": integrity.get("duplicate_consecutive_event_fraction"),
        "timestamp_monotonic": timing.get("timestamp_monotonic"),
        "observed_rate_hz": timing.get("observed_rate_hz"),
        "warning_count": len(report.get("warnings", [])),
        "failure_count": len(report.get("failures", [])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-files", nargs="+", type=Path, required=True,
        help="The newly collected ROOT files; pass exactly one session's files.",
    )
    parser.add_argument(
        "--reference-root", type=Path, required=True,
        help="Confirmed good ROOT file used as the source-independent reference.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/data_quality"),
        help="Directory under which a session report directory is created.",
    )
    parser.add_argument(
        "--session-id", default=None,
        help="Session label for the output directory; defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--expected-files", type=int, default=10,
        help="Expected ROOT files in one DAQ session (default: 10).",
    )
    parser.add_argument(
        "--sample-events", type=int, default=EXPECTED_SAMPLE_EVENTS,
        help="Events sampled from each ROOT file (default: 1000).",
    )
    parser.add_argument(
        "--sample-mode", choices=("first", "distributed"), default="first",
        help="Read the first N events, or deterministic blocks across each file.",
    )
    parser.add_argument(
        "--noise-warn-factor", type=float, default=DEFAULT_NOISE_WARN_FACTOR,
        help="Warn when corrected baseline RMS exceeds this reference factor.",
    )
    parser.add_argument(
        "--noise-fail-factor", type=float, default=DEFAULT_NOISE_FAIL_FACTOR,
        help="Fail when corrected baseline RMS exceeds this reference factor.",
    )
    parser.add_argument("--pedestal-warn-adc", type=float, default=5.0)
    parser.add_argument("--pedestal-fail-adc", type=float, default=10.0)
    parser.add_argument("--drift-warn-adc", type=float, default=5.0)
    parser.add_argument("--drift-fail-adc", type=float, default=10.0)
    parser.add_argument(
        "--fail-on-warn", action="store_true",
        help="Return failure exit code when any file is WARN.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_files <= 0 or args.sample_events <= 0:
        print("expected-files and sample-events must be positive", file=sys.stderr)
        return 2
    if args.noise_fail_factor <= args.noise_warn_factor or args.noise_warn_factor <= 1.0:
        print("noise failure factor must be greater than warning factor, both > 1", file=sys.stderr)
        return 2
    if args.pedestal_fail_adc <= args.pedestal_warn_adc or args.drift_fail_adc <= args.drift_warn_adc:
        print("failure ADC limits must be greater than warning limits", file=sys.stderr)
        return 2

    root_files = sorted({path.resolve() for path in args.root_files})
    if not root_files:
        print("no ROOT files supplied", file=sys.stderr)
        return 2
    missing = [str(path) for path in root_files if not path.is_file()]
    if not args.reference_root.is_file():
        print(f"reference ROOT file does not exist: {args.reference_root}", file=sys.stderr)
        return 2
    if missing:
        print("missing ROOT files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    session_id = args.session_id or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S_utc")
    session_dir = args.output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    reference = analyze_file(args.reference_root.resolve(), args.sample_events, args.sample_mode)
    if reference["status"] == "FAIL":
        print("reference file failed basic analysis:", file=sys.stderr)
        for failure in reference["failures"]:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    reports = []
    for path in root_files:
        report = analyze_file(path, args.sample_events, args.sample_mode)
        compare_to_reference(
            report,
            reference,
            args.noise_warn_factor,
            args.noise_fail_factor,
            args.pedestal_warn_adc,
            args.pedestal_fail_adc,
            args.drift_warn_adc,
            args.drift_fail_adc,
        )
        reports.append(report)

    count_status = "PASS" if len(root_files) == args.expected_files else "FAIL"
    session_failures = []
    session_warnings = []
    if count_status == "FAIL":
        session_failures.append(
            f"received {len(root_files)} ROOT files; expected {args.expected_files}"
        )
    for report in reports:
        session_failures.extend(
            f"{report['file_name']}: {message}" for message in report.get("failures", [])
        )
        session_warnings.extend(
            f"{report['file_name']}: {message}" for message in report.get("warnings", [])
        )

    if session_failures:
        session_status = "FAIL"
    elif session_warnings:
        session_status = "WARN"
    else:
        session_status = "PASS"

    session_report: dict[str, Any] = {
        "session_id": session_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": session_status,
        "root_file_count": len(root_files),
        "expected_root_file_count": args.expected_files,
        "sample_events_per_file": args.sample_events,
        "sample_mode": args.sample_mode,
        "reference_root": str(args.reference_root.resolve()),
        "baseline_window_samples_inclusive": BASELINE_SAMPLES_INCLUSIVE,
        "baseline_window_numpy_slice": "0:1000",
        "baseline_subtraction_samples_inclusive": BASELINE_SUBTRACTION_SAMPLES_INCLUSIVE,
        "baseline_subtraction_numpy_slice": "0:200",
        "baseline_noise_samples_inclusive": BASELINE_NOISE_SAMPLES_INCLUSIVE,
        "baseline_noise_numpy_slice": "0:200",
        "source_independent_metrics": [
            "file_integrity",
            "baseline_noise",
            "pedestal_stability",
            "waveform_integrity",
            "timestamp_continuity",
        ],
        "thresholds": {
            "noise_warn_factor": args.noise_warn_factor,
            "noise_fail_factor": args.noise_fail_factor,
            "pedestal_warn_adc": args.pedestal_warn_adc,
            "pedestal_fail_adc": args.pedestal_fail_adc,
            "drift_warn_adc": args.drift_warn_adc,
            "drift_fail_adc": args.drift_fail_adc,
        },
        "reference_metrics": reference,
        "files": reports,
        "session_failures": session_failures,
        "session_warnings": session_warnings,
    }
    json_path = session_dir / "session_qc_report.json"
    json_path.write_text(json.dumps(session_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    csv_path = session_dir / "session_qc_summary.csv"
    rows = [flatten_row(report) for report in reports]
    fieldnames = list(rows[0]) if rows else ["status", "file"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Session: {session_id}")
    print(f"Status:  {session_status}")
    print("STATUS  FILE  ENTRIES  BASELINE_NOISE_P95  BASELINE_OFFSET  DRIFT_RANGE  RAIL  DUP  RATE_HZ")
    for row in rows:
        rail = max(float(row["lower_rail_fraction"] or 0.0), float(row["upper_rail_fraction"] or 0.0))
        print(
            f"{str(row['status']):<6}  {str(row['file']):<55} "
            f"{format_cell(row['entries'], 0):>7}  "
            f"{format_cell(row['baseline_noise_p95_adc'], 4):>18}  "
            f"{format_cell(row['baseline_offset_median_adc'], 3):>15}  "
            f"{format_cell(row['baseline_drift_range_adc'], 3):>11}  "
            f"{rail:>4.3f}  {float(row['duplicate_fraction'] or 0.0):>4.3f}  "
            f"{format_cell(row['observed_rate_hz'], 2):>7}"
        )
    print(f"JSON report: {json_path}")
    print(f"CSV summary: {csv_path}")

    if session_status == "FAIL":
        return 2
    if session_status == "WARN" and args.fail_on_warn:
        return 2
    if session_status == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
