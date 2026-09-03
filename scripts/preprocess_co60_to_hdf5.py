#!/usr/bin/env python3
"""Filter one direct-preamp Co-60 ROOT file and write accepted events to HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import uproot
from scipy.optimize import curve_fit


SAMPLE_RATE_MSPS = 250
SAMPLE_PERIOD_NS = 4.0
STORED_SAMPLES = 4500
CO60_ENERGIES_KEV = np.array([1173.228, 1332.492], dtype=np.float64)
BASELINE_SLICE = slice(250, 1000)
PLATEAU_SLICE = slice(1350, 1600)
NOISE_SECTION_SLICES = tuple(slice(start, start + 200) for start in range(0, 1000, 200))
PULSE_SEARCH_START = 1250
PULSE_SEARCH_STOP = 2000


def gaussian_linear(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_photopeak(amplitudes: np.ndarray, low: float, high: float) -> dict[str, float]:
    edges = np.arange(low, high + 0.1, 0.1)
    counts, _ = np.histogram(amplitudes, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() <= 0:
        raise RuntimeError(f"No amplitude entries in photopeak fit window {low:g}–{high:g} ADC")
    initial_mean = float(centers[np.argmax(counts)])
    fitted, covariance = curve_fit(
        gaussian_linear,
        centers,
        counts,
        p0=[float(counts.max()), initial_mean, 0.5, float(np.median(counts)), 0.0],
        bounds=([0.0, low, 0.05, 0.0, -np.inf], [np.inf, high, 3.0, np.inf, np.inf]),
        maxfev=100000,
    )
    errors = np.sqrt(np.diag(covariance))
    return {
        "mean_adc": float(fitted[1]),
        "mean_error_adc": float(errors[1]),
        "sigma_adc": float(fitted[2]),
        "fwhm_adc": float(2.354820045 * fitted[2]),
    }


def reconstruct_amplitude(waveforms: np.ndarray, polarity: str) -> np.ndarray:
    baseline = waveforms[:, BASELINE_SLICE].mean(axis=1)
    plateau = waveforms[:, PLATEAU_SLICE].mean(axis=1)
    if polarity == "negative":
        return baseline - plateau
    return plateau - baseline


def section_noise_rms(waveforms: np.ndarray) -> np.ndarray:
    """Return local-mean-subtracted RMS for six consecutive 200-sample sections."""
    return np.stack(
        [waveforms[:, section].std(axis=1) for section in NOISE_SECTION_SLICES],
        axis=1,
    )


def pulse_extremum_index(
    waveforms: np.ndarray,
    polarity: str,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    search = waveforms[:, start : stop + 1]
    if polarity == "negative":
        relative = np.argmin(search, axis=1)
        values = search[np.arange(len(search)), relative]
    else:
        relative = np.argmax(search, axis=1)
        values = search[np.arange(len(search)), relative]
    return relative + start, values


def root_metadata(root_file: uproot.ReadOnlyDirectory) -> dict[str, str]:
    names = (
        "source",
        "sample_rate_msps",
        "sample_period_ns",
        "stored_samples",
        "pretrigger_us",
        "posttrigger_us",
        "trigger_threshold_adc",
    )
    metadata: dict[str, str] = {}
    for name in names:
        if name not in root_file:
            continue
        obj = root_file[name]
        title = getattr(obj, "title", None)
        metadata[name] = str(title if title is not None else obj)
    return metadata


def iter_waveform_chunks(tree: uproot.TTree, step_size: int):
    for arrays in tree.iterate(["event"], step_size=step_size, library="np"):
        event = arrays["event"]
        waveforms = event["waveform"].astype(np.float32, copy=False)
        if waveforms.ndim != 2 or waveforms.shape[1] != STORED_SAMPLES:
            raise ValueError(
                f"Expected waveform shape (N, {STORED_SAMPLES}); received {waveforms.shape}"
            )
        yield event, waveforms


def fit_energy_calibration(
    input_root: Path,
    polarity: str,
    step_size: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    amplitudes_parts: list[np.ndarray] = []
    with uproot.open(input_root) as root_file:
        tree = root_file["HPGE"]
        for _, waveforms in iter_waveform_chunks(tree, step_size):
            amplitudes_parts.append(reconstruct_amplitude(waveforms, polarity).astype(np.float64))
    amplitudes = np.concatenate(amplitudes_parts)
    finite = amplitudes[np.isfinite(amplitudes)]
    peaks = [fit_photopeak(finite, 82.0, 89.0), fit_photopeak(finite, 94.0, 101.0)]
    calibration = np.polyfit([peak["mean_adc"] for peak in peaks], CO60_ENERGIES_KEV, 1)
    if not np.all(np.isfinite(calibration)) or calibration[0] <= 0.0:
        raise RuntimeError(f"Invalid energy calibration: {calibration}")
    return calibration, peaks


def create_hdf5_datasets(output: h5py.File, samples: int) -> None:
    output.create_dataset(
        "waveform",
        shape=(0, samples),
        maxshape=(None, samples),
        chunks=(256, samples),
        dtype="f4",
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    output.create_dataset("event_id", shape=(0,), maxshape=(None,), chunks=True, dtype="u4")
    output.create_dataset("trigger_time_s", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset(
        "reconstructed_energy_kev", shape=(0,), maxshape=(None,), chunks=True, dtype="f4"
    )
    output.create_dataset(
        "noise_rms_adc",
        shape=(0, len(NOISE_SECTION_SLICES)),
        maxshape=(None, len(NOISE_SECTION_SLICES)),
        chunks=(256, len(NOISE_SECTION_SLICES)),
        dtype="f4",
    )
    output.create_dataset("pulse_extremum_index", shape=(0,), maxshape=(None,), chunks=True, dtype="i4")
    output.create_dataset("pulse_extremum_adc", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")
    output.create_dataset("charge_amplitude_adc", shape=(0,), maxshape=(None,), chunks=True, dtype="f4")


def append_hdf5(output: h5py.File, values: dict[str, np.ndarray], current: int) -> int:
    count = len(values["event_id"])
    if count == 0:
        return current
    end = current + count
    for name, value in values.items():
        dataset = output[name]
        dataset.resize((end,) + dataset.shape[1:])
        dataset[current:end] = value
    return end


def process(args: argparse.Namespace) -> dict[str, object]:
    calibration, peaks = fit_energy_calibration(args.input_root, args.polarity, args.step_size)
    slope, intercept = calibration
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.input_root.stem}.h5"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite to replace it: {output_path}")

    total = accepted = 0
    rejection_counts = {
        "nonfinite_energy": 0,
        "energy_below_80_kev": 0,
        "energy_above_2700_kev": 0,
        "noise_over_1_4_adc": 0,
        "pulse_extremum_outside_1250_2000": 0,
    }
    with uproot.open(args.input_root) as root_file:
        tree = root_file["HPGE"]
        metadata = root_metadata(root_file)
        with h5py.File(output_path, "w") as output:
            create_hdf5_datasets(output, STORED_SAMPLES)
            current = 0
            for event, waveforms in iter_waveform_chunks(tree, args.step_size):
                amplitudes = reconstruct_amplitude(waveforms, args.polarity).astype(np.float64)
                energies = slope * amplitudes + intercept
                noise = section_noise_rms(waveforms).astype(np.float64)
                pulse_indices, pulse_values = pulse_extremum_index(
                    waveforms, args.polarity, args.pulse_start, args.pulse_stop
                )

                finite_energy = np.isfinite(energies)
                below = finite_energy & (energies < args.min_energy_kev)
                above = finite_energy & (energies > args.max_energy_kev)
                noisy = np.any(noise > args.max_section_noise_adc, axis=1)
                misplaced = (pulse_indices < args.pulse_start) | (pulse_indices > args.pulse_stop)
                keep = finite_energy & ~below & ~above & ~noisy & ~misplaced

                rejection_counts["nonfinite_energy"] += int(np.count_nonzero(~finite_energy))
                rejection_counts["energy_below_80_kev"] += int(np.count_nonzero(below))
                rejection_counts["energy_above_2700_kev"] += int(np.count_nonzero(above))
                rejection_counts["noise_over_1_4_adc"] += int(np.count_nonzero(noisy))
                rejection_counts["pulse_extremum_outside_1250_2000"] += int(np.count_nonzero(misplaced))
                total += len(waveforms)

                selected = {
                    "waveform": waveforms[keep],
                    "event_id": event["event_id"][keep],
                    "trigger_time_s": event["trigger_time_s"][keep],
                    "reconstructed_energy_kev": energies[keep].astype(np.float32),
                    "noise_rms_adc": noise[keep].astype(np.float32),
                    "pulse_extremum_index": pulse_indices[keep].astype(np.int32),
                    "pulse_extremum_adc": pulse_values[keep].astype(np.float32),
                    "charge_amplitude_adc": amplitudes[keep].astype(np.float32),
                }
                current = append_hdf5(output, selected, current)
                accepted = current

            output.attrs.update(
                {
                    "preprocessing_version": "1",
                    "input_root": str(args.input_root.resolve()),
                    "input_tree": "HPGE/event",
                    "input_entries": total,
                    "accepted_entries": accepted,
                    "polarity": args.polarity,
                    "energy_calibration_kev_per_adc": float(slope),
                    "energy_calibration_intercept_kev": float(intercept),
                    "energy_range_kev_inclusive": json.dumps(
                        [args.min_energy_kev, args.max_energy_kev]
                    ),
                    "noise_definition": "per-section standard deviation after subtracting each section mean",
                    "noise_section_samples_zero_based": json.dumps(
                        [[section.start, section.stop - 1] for section in NOISE_SECTION_SLICES]
                    ),
                    "max_section_noise_adc": args.max_section_noise_adc,
                    "pulse_extremum_window_samples_zero_based_inclusive": json.dumps(
                        [args.pulse_start, args.pulse_stop]
                    ),
                    "raw_waveform": True,
                    **{f"root_{key}": value for key, value in metadata.items()},
                }
            )

    return {
        "input_root": str(args.input_root),
        "output_hdf5": str(output_path),
        "input_entries": total,
        "accepted_entries": accepted,
        "rejected_entries": total - accepted,
        "acceptance_fraction": accepted / total if total else 0.0,
        "rejection_counts_nonexclusive": rejection_counts,
        "cuts": {
            "energy_kev_inclusive": [args.min_energy_kev, args.max_energy_kev],
            "max_section_noise_adc": args.max_section_noise_adc,
            "noise_sections_zero_based_inclusive": [
                [section.start, section.stop - 1] for section in NOISE_SECTION_SLICES
            ],
            "pulse_extremum_window_zero_based_inclusive": [args.pulse_start, args.pulse_stop],
            "polarity": args.polarity,
        },
        "energy_calibration": {
            "kev_per_adc": float(slope),
            "intercept_kev": float(intercept),
            "photopeak_fits": peaks,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed_data/co60_hdf5"),
        help="Directory for the one-to-one HDF5 output (default: processed_data/co60_hdf5).",
    )
    parser.add_argument("--polarity", choices=("positive", "negative"), default="negative")
    parser.add_argument("--min-energy-kev", type=float, default=80.0)
    parser.add_argument("--max-energy-kev", type=float, default=2700.0)
    parser.add_argument("--max-section-noise-adc", type=float, default=1.4)
    parser.add_argument("--pulse-start", type=int, default=PULSE_SEARCH_START)
    parser.add_argument("--pulse-stop", type=int, default=PULSE_SEARCH_STOP)
    parser.add_argument("--step-size", type=int, default=1500)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.min_energy_kev >= args.max_energy_kev:
        raise ValueError("--min-energy-kev must be less than --max-energy-kev")
    if args.pulse_start < 0 or args.pulse_stop < args.pulse_start or args.pulse_stop >= STORED_SAMPLES:
        raise ValueError("pulse window must be within the stored waveform")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    summary = process(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
