#!/usr/bin/env python3
"""Preprocess source-labeled 250-MSPS ROOT files into one-to-one HDF5 products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import uproot


EXPECTED_SAMPLES = 4_500
SAMPLE_RATE_MSPS = 250
SAMPLE_PERIOD_US = 0.004
POLE_ZERO_TAU_US = 100.0
RISE_SAMPLES = 1_125
FLAT_SAMPLES = 200
CALIBRATION_KEV_PER_UNIT = 13.620393994376382
CALIBRATION_INTERCEPT_KEV = 0.8706343947031012
CALIBRATION_METHOD = (
    "provisional multi-source affine calibration from resolved Ba-133, Na-22, "
    "Cs-137, and Co-60 photopeaks"
)
BASELINE_SLICES = tuple(slice(start, start + 200) for start in range(0, 1_000, 200))
PULSE_START = 1_250
PULSE_STOP = 2_000
ROOT_METADATA_NAMES = (
    "source",
    "sample_rate_msps",
    "sample_period_ns",
    "stored_samples",
    "pretrigger_us",
    "posttrigger_us",
    "trigger_threshold_adc",
)
QC_REJECTION_BIT_DEFINITIONS = {
    "bit_0": "nonfinite_reconstructed_energy",
    "bit_1": "reconstructed_energy_below_minimum",
    "bit_2": "reconstructed_energy_above_maximum",
    "bit_3": "baseline_section_noise_above_limit",
    "bit_4": "pulse_extremum_outside_window",
}


def infer_source(input_root: Path) -> str:
    name = input_root.name.lower()
    for source in ("ba133", "co60", "cs137", "na22", "th232"):
        if name.startswith(source):
            return source
    return "unknown"


def pole_zero_correct(signal: np.ndarray, tau_us: float = POLE_ZERO_TAU_US) -> np.ndarray:
    centered = signal - signal[:, :1_000].mean(axis=1, keepdims=True)
    alpha = float(np.exp(-SAMPLE_PERIOD_US / tau_us))
    difference = np.empty_like(centered)
    difference[:, 0] = centered[:, 0]
    difference[:, 1:] = centered[:, 1:] - alpha * centered[:, :-1]
    corrected = np.cumsum(difference, axis=1, dtype=np.float32)
    corrected -= corrected[:, :1_000].mean(axis=1, keepdims=True)
    return corrected


def trapezoid_max(signal: np.ndarray) -> np.ndarray:
    stop = signal.shape[1]
    span = 2 * RISE_SAMPLES + FLAT_SAMPLES
    positions = stop - span + 1
    prefix = np.empty((signal.shape[0], stop + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(signal, axis=1, dtype=np.float64, out=prefix[:, 1:])
    first = prefix[:, RISE_SAMPLES : RISE_SAMPLES + positions] - prefix[:, :positions]
    delayed = RISE_SAMPLES + FLAT_SAMPLES
    second = (
        prefix[:, delayed + RISE_SAMPLES : delayed + RISE_SAMPLES + positions]
        - prefix[:, delayed : delayed + positions]
    )
    values = (second - first) / float(RISE_SAMPLES)
    return values.max(axis=1).astype(np.float32)


def reconstruct_energy(waveforms: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray]:
    signal = -waveforms if polarity == "negative" else waveforms
    shaped = trapezoid_max(pole_zero_correct(signal))
    energy = CALIBRATION_KEV_PER_UNIT * shaped + CALIBRATION_INTERCEPT_KEV
    return energy.astype(np.float32), shaped


def section_noise_rms(waveforms: np.ndarray) -> np.ndarray:
    return np.stack([waveforms[:, section].std(axis=1) for section in BASELINE_SLICES], axis=1)


def pulse_extremum(waveforms: np.ndarray, polarity: str) -> tuple[np.ndarray, np.ndarray]:
    search = waveforms[:, PULSE_START : PULSE_STOP + 1]
    if polarity == "negative":
        relative = np.argmin(search, axis=1)
    else:
        relative = np.argmax(search, axis=1)
    values = search[np.arange(len(search)), relative]
    return (relative + PULSE_START).astype(np.int32), values.astype(np.float32)


def root_metadata(root_file: uproot.ReadOnlyDirectory) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for name in ROOT_METADATA_NAMES:
        if name not in root_file:
            continue
        obj = root_file[name]
        title = getattr(obj, "title", None)
        metadata[name] = str(title if title is not None else obj)
    return metadata


def create_datasets(output: h5py.File) -> None:
    output.create_dataset(
        "waveform",
        shape=(0, EXPECTED_SAMPLES),
        maxshape=(None, EXPECTED_SAMPLES),
        chunks=(256, EXPECTED_SAMPLES),
        dtype="f4",
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    output.create_dataset("event_id", shape=(0,), maxshape=(None,), chunks=True, dtype="u4")
    output.create_dataset("root_entry_index", shape=(0,), maxshape=(None,), chunks=True, dtype="u4")
    output.create_dataset("trigger_time_s", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset("reconstructed_energy_kev", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset("shaped_energy_unit", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset(
        "noise_rms_adc",
        shape=(0, len(BASELINE_SLICES)),
        maxshape=(None, len(BASELINE_SLICES)),
        chunks=(256, len(BASELINE_SLICES)),
        dtype="f4",
    )
    output.create_dataset("pulse_extremum_index", shape=(0,), maxshape=(None,), chunks=True, dtype="i4")
    output.create_dataset("pulse_extremum_adc", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset("qc_rejection_bits", shape=(0,), maxshape=(None,), chunks=True, dtype="u2")


def append_datasets(output: h5py.File, values: dict[str, np.ndarray], current: int) -> int:
    count = len(values["event_id"])
    if count == 0:
        return current
    end = current + count
    for name, value in values.items():
        dataset = output[name]
        dataset.resize((end,) + dataset.shape[1:])
        dataset[current:end] = value
    return end


def iter_chunks(tree: uproot.TTree, step_size: int) -> Iterator[tuple[Any, np.ndarray]]:
    for arrays in tree.iterate(["event"], step_size=step_size, library="np"):
        event = arrays["event"]
        waveforms = np.asarray(event["waveform"], dtype=np.float32)
        if waveforms.ndim != 2 or waveforms.shape[1] != EXPECTED_SAMPLES:
            raise ValueError(f"expected waveform shape (*, {EXPECTED_SAMPLES}); received {waveforms.shape}")
        yield event, waveforms


def set_attributes(
    output: h5py.File,
    input_root: Path,
    source: str,
    metadata: dict[str, str],
    total: int,
    retained: int,
    analysis_accepted: int,
    args: argparse.Namespace,
    status: str = "OK",
    error: str | None = None,
) -> None:
    output.attrs.update(
        {
            "preprocessing_version": "2",
            "processing_status": status,
            "input_root": str(input_root.resolve()),
            "input_tree": "HPGE/event",
            "source_label": source,
            "input_entries": total,
            "accepted_entries": retained,
            "retained_entries": retained,
            "analysis_accepted_entries": analysis_accepted,
            "retention_mode": "all_events_with_qc_flags" if args.retain_all_events else "accepted_only",
            "qc_rejection_bit_definitions": json.dumps(QC_REJECTION_BIT_DEFINITIONS, sort_keys=True),
            "polarity": args.polarity,
            "energy_reconstruction": "FADC500 first-order pole-zero correction followed by maximum difference of two moving averages",
            "pole_zero_tau_us": POLE_ZERO_TAU_US,
            "rise_samples": RISE_SAMPLES,
            "flat_samples": FLAT_SAMPLES,
            "energy_calibration_kev_per_shaped_unit": CALIBRATION_KEV_PER_UNIT,
            "energy_calibration_intercept_kev": CALIBRATION_INTERCEPT_KEV,
            "energy_calibration_method": CALIBRATION_METHOD,
            "energy_range_kev_inclusive": json.dumps([args.min_energy_kev, args.max_energy_kev]),
            "noise_definition": "standard deviation within each local 200-sample section",
            "noise_sections_zero_based_inclusive": json.dumps(
                [[section.start, section.stop - 1] for section in BASELINE_SLICES]
            ),
            "max_section_noise_adc": args.max_section_noise_adc,
            "pulse_extremum_window_zero_based_inclusive": json.dumps([args.pulse_start, args.pulse_stop]),
            "raw_waveform": True,
            **{f"root_{key}": value for key, value in metadata.items()},
        }
    )
    if error is not None:
        output.attrs["processing_error"] = error


def process_one(input_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output_dir / f"{input_root.stem}.h5"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = args.source_label or infer_source(input_root)
    total = retained = analysis_accepted = 0
    rejection_counts = {
        "nonfinite_energy": 0,
        "energy_below_minimum_kev": 0,
        "energy_above_maximum_kev": 0,
        "noise_over_limit_adc": 0,
        "pulse_extremum_outside_window": 0,
    }

    with h5py.File(output_path, "w") as output:
        create_datasets(output)
        try:
            with uproot.open(input_root) as root_file:
                metadata = root_metadata(root_file)
                if "HPGE" not in root_file:
                    set_attributes(
                        output,
                        input_root,
                        source,
                        metadata,
                        0,
                        0,
                        0,
                        args,
                        "INPUT_INVALID",
                        "HPGE tree is missing",
                    )
                    return {
                        "input_root": str(input_root), "output_hdf5": str(output_path), "source": source,
                        "input_entries": 0, "accepted_entries": 0, "rejected_entries": 0,
                        "status": "INPUT_INVALID", "error": "HPGE tree is missing",
                    }
                tree = root_file["HPGE"]
                current = 0
                root_entry_offset = 0
                for event, waveforms in iter_chunks(tree, args.step_size):
                    energies, shaped = reconstruct_energy(waveforms, args.polarity)
                    noise = section_noise_rms(waveforms)
                    pulse_indices, pulse_values = pulse_extremum(waveforms, args.polarity)
                    finite_energy = np.isfinite(energies)
                    below = finite_energy & (energies < args.min_energy_kev)
                    above = finite_energy & (energies > args.max_energy_kev)
                    noisy = np.any(noise > args.max_section_noise_adc, axis=1)
                    misplaced = (pulse_indices < args.pulse_start) | (pulse_indices > args.pulse_stop)
                    analysis_keep = finite_energy & ~below & ~above & ~noisy & ~misplaced
                    keep = np.ones(len(waveforms), dtype=bool) if args.retain_all_events else analysis_keep

                    rejection_bits = np.zeros(len(waveforms), dtype=np.uint16)
                    rejection_bits |= (~finite_energy).astype(np.uint16) << 0
                    rejection_bits |= below.astype(np.uint16) << 1
                    rejection_bits |= above.astype(np.uint16) << 2
                    rejection_bits |= noisy.astype(np.uint16) << 3
                    rejection_bits |= misplaced.astype(np.uint16) << 4

                    rejection_counts["nonfinite_energy"] += int(np.count_nonzero(~finite_energy))
                    rejection_counts["energy_below_minimum_kev"] += int(np.count_nonzero(below))
                    rejection_counts["energy_above_maximum_kev"] += int(np.count_nonzero(above))
                    rejection_counts["noise_over_limit_adc"] += int(np.count_nonzero(noisy))
                    rejection_counts["pulse_extremum_outside_window"] += int(np.count_nonzero(misplaced))
                    entry_indices = np.arange(
                        root_entry_offset,
                        root_entry_offset + len(waveforms),
                        dtype=np.uint32,
                    )
                    total += len(waveforms)
                    root_entry_offset += len(waveforms)
                    analysis_accepted += int(np.count_nonzero(analysis_keep))
                    selected = {
                        "waveform": waveforms[keep],
                        "event_id": event["event_id"][keep],
                        "root_entry_index": entry_indices[keep],
                        "trigger_time_s": event["trigger_time_s"][keep],
                        "reconstructed_energy_kev": energies[keep],
                        "shaped_energy_unit": shaped[keep],
                        "noise_rms_adc": noise[keep].astype(np.float32),
                        "pulse_extremum_index": pulse_indices[keep],
                        "pulse_extremum_adc": pulse_values[keep],
                        "qc_rejection_bits": rejection_bits[keep],
                    }
                    current = append_datasets(output, selected, current)
                retained = current
                set_attributes(output, input_root, source, metadata, total, retained, analysis_accepted, args)
        except Exception as error:
            set_attributes(output, input_root, source, {}, total, retained, analysis_accepted, args, "PROCESSING_ERROR", str(error))
            raise

    return {
        "input_root": str(input_root),
        "output_hdf5": str(output_path),
        "source": source,
        "status": "OK",
        "input_entries": total,
        "accepted_entries": retained,
        "retained_entries": retained,
        "analysis_accepted_entries": analysis_accepted,
        "rejected_entries": total - analysis_accepted,
        "acceptance_fraction": analysis_accepted / total if total else 0.0,
        "retention_mode": "all_events_with_qc_flags" if args.retain_all_events else "accepted_only",
        "rejection_counts_nonexclusive": rejection_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data/waveform_hdf5"))
    parser.add_argument("--polarity", choices=("positive", "negative"), default="negative")
    parser.add_argument("--source-label", type=str, default=None)
    parser.add_argument("--retain-all-events", action="store_true")
    parser.add_argument("--min-energy-kev", type=float, default=80.0)
    parser.add_argument("--max-energy-kev", type=float, default=2700.0)
    parser.add_argument("--max-section-noise-adc", type=float, default=1.4)
    parser.add_argument("--pulse-start", type=int, default=PULSE_START)
    parser.add_argument("--pulse-stop", type=int, default=PULSE_STOP)
    parser.add_argument("--step-size", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/waveform_hdf5/preprocessing_manifest.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.min_energy_kev >= args.max_energy_kev:
        raise ValueError("minimum energy must be below maximum energy")
    if args.pulse_start < 0 or args.pulse_stop < args.pulse_start or args.pulse_stop >= EXPECTED_SAMPLES:
        raise ValueError("pulse window must be inside the stored waveform")
    if args.step_size <= 0:
        raise ValueError("step size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for input_root in sorted(args.input_roots):
        result = process_one(input_root, args)
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preprocessing_version": "2",
        "input_file_count": len(results),
        "output_dir": str(args.output_dir),
        "configuration": {
            "polarity": args.polarity,
            "source_label": args.source_label,
            "retention_mode": "all_events_with_qc_flags" if args.retain_all_events else "accepted_only",
            "qc_rejection_bit_definitions": QC_REJECTION_BIT_DEFINITIONS,
            "energy_range_kev_inclusive": [args.min_energy_kev, args.max_energy_kev],
            "max_section_noise_adc": args.max_section_noise_adc,
            "noise_sections_zero_based_inclusive": [[s.start, s.stop - 1] for s in BASELINE_SLICES],
            "pulse_extremum_window_zero_based_inclusive": [args.pulse_start, args.pulse_stop],
            "energy_calibration_kev_per_shaped_unit": CALIBRATION_KEV_PER_UNIT,
            "energy_calibration_intercept_kev": CALIBRATION_INTERCEPT_KEV,
        },
        "files": results,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "processed_files": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
