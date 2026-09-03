#!/usr/bin/env python3
"""Apply a Co-60-fitted O2-3P energy threshold to corrected Th-232 data.

The stretched-exponential threshold fit is selected and frozen from the Co-60
validation curve before Th-232 scores are inspected. Peak windows are fitted
once on the uncut corrected-energy spectrum and reused for fixed and fitted cuts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_o2_late_fusion import O2LateFusion, extract_o2_features  # noqa: E402


MODEL_NAME = "O2-3P Late Fusion"
REFERENCE_PEAKS_KEV = (238.632, 338.320, 583.187, 911.204, 968.971, 1588.190, 2614.511)
ENERGY_MIN_KEV = 0.0
ENERGY_MAX_KEV = 3200.0
BIN_WIDTH_KEV = 1.0
ENERGY_EDGES = np.arange(ENERGY_MIN_KEV, ENERGY_MAX_KEV + BIN_WIDTH_KEV, BIN_WIDTH_KEV)
ENERGY_CENTERS = 0.5 * (ENERGY_EDGES[:-1] + ENERGY_EDGES[1:])
FIXED_BASELINE_THRESHOLD = 0.2793106436729431
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True)
class PeakWindow:
    reference_kev: float
    centroid_kev: float
    sigma_kev: float
    roi_low_kev: float
    roi_high_kev: float
    left_low_kev: float
    left_high_kev: float
    right_low_kev: float
    right_high_kev: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def stretched_exponential_threshold(
    energy_kev: np.ndarray | float,
    asymptote: float,
    amplitude: float,
    scale_kev: float,
    exponent: float,
) -> np.ndarray:
    energy = np.asarray(energy_kev, dtype=np.float64)
    offset = np.maximum(energy - 100.0, 0.0)
    return asymptote + amplitude * np.exp(-np.power(offset / scale_kev, exponent))


def load_threshold_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 5:
        raise ValueError("Too few empirical threshold points")
    energy = np.asarray([float(row["energy_center_kev"]) for row in rows])
    threshold = np.asarray([float(row["threshold"]) for row in rows])
    if not np.all(np.diff(energy) > 0.0):
        raise ValueError("Threshold energies must be strictly increasing")
    return energy, threshold, rows


def fit_threshold_curve(path: Path) -> dict[str, Any]:
    energy, threshold, rows = load_threshold_points(path)
    lower = (0.0, 0.0, 10.0, 0.1)
    upper = (1.0, 2.0, 2000.0, 4.0)
    parameters, _ = curve_fit(
        stretched_exponential_threshold,
        energy,
        threshold,
        p0=(0.23, 0.41, 240.0, 1.4),
        bounds=(lower, upper),
        maxfev=100000,
    )
    prediction = stretched_exponential_threshold(energy, *parameters)
    residual = threshold - prediction
    loo_residuals = []
    for index in range(energy.size):
        retained = np.arange(energy.size) != index
        fitted, _ = curve_fit(
            stretched_exponential_threshold,
            energy[retained],
            threshold[retained],
            p0=parameters,
            bounds=(lower, upper),
            maxfev=100000,
        )
        held_out = stretched_exponential_threshold(energy[index], *fitted)
        loo_residuals.append(float(threshold[index] - held_out))
    return {
        "model": "c + a * exp(-((max(E - 100, 0) / tau) ** p))",
        "parameters": {
            "c_asymptote": float(parameters[0]),
            "a_amplitude": float(parameters[1]),
            "tau_kev": float(parameters[2]),
            "p_exponent": float(parameters[3]),
        },
        "parameter_vector": [float(value) for value in parameters],
        "point_count": int(energy.size),
        "energy_domain_kev": [float(energy.min() - 25.0), float(energy.max() + 25.0)],
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "leave_one_bin_out_rmse": float(
            np.sqrt(np.mean(np.square(loo_residuals)))
        ),
        "threshold_at_100kev": float(
            stretched_exponential_threshold(100.0, *parameters)
        ),
        "threshold_at_1000kev": float(
            stretched_exponential_threshold(1000.0, *parameters)
        ),
        "source_rows": rows,
    }


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[O2LateFusion, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "late_fusion":
        raise ValueError("Checkpoint is not a late-fusion model")
    if checkpoint.get("architecture") != "O2_style_charge_current_late_fusion":
        raise ValueError("Unexpected model architecture")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Checkpoint test boundary is not frozen")
    model = O2LateFusion().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def th232_admission_mask(
    corrected_energy_kev: np.ndarray,
    shaped_energy_unit: np.ndarray,
    qc_rejection_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_valid = (
        np.isfinite(corrected_energy_kev)
        & (corrected_energy_kev >= ENERGY_MIN_KEV)
        & (corrected_energy_kev < ENERGY_MAX_KEV)
        & ((qc_rejection_bits.astype(np.uint16) & np.uint16(0b111)) == 0)
    )
    shaped_valid = np.isfinite(shaped_energy_unit) & (shaped_energy_unit > 0.0)
    return energy_valid & shaped_valid, energy_valid, shaped_valid


def resize_cache(handle: h5py.File, new_size: int) -> None:
    for dataset in handle.values():
        if isinstance(dataset, h5py.Group):
            continue
        dataset.resize((new_size,))


def create_score_cache(handle: h5py.File, file_count: int, chunk_size: int) -> h5py.Group:
    for name, dtype in (
        ("corrected_energy_kev", np.float32),
        ("score", np.float32),
        ("source_file_index", np.uint16),
        ("source_row", np.int64),
    ):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=dtype,
        )
    files = handle.create_group("source_files")
    files.create_dataset("path", shape=(file_count,), dtype=STRING_DTYPE)
    files.create_dataset("sha256", shape=(file_count,), dtype=STRING_DTYPE)
    files.create_dataset("input_event_count", shape=(file_count,), dtype=np.int64)
    files.create_dataset("admitted_event_count", shape=(file_count,), dtype=np.int64)
    return files


def score_th232(
    files: list[Path],
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    model, checkpoint = load_model(checkpoint_path, device)
    statistics = checkpoint["feature_statistics"]
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    counts = {
        "input_events": 0,
        "admitted_events": 0,
        "rejected_energy_or_qc_bits_0_to_2": 0,
        "rejected_nonpositive_shaped_energy": 0,
        "t10_fallback_count": 0,
    }
    file_records = []
    next_index = 0
    try:
        with h5py.File(partial, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "model_name": MODEL_NAME,
                    "checkpoint": relative(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "energy_dataset": "corrected_energy_kev",
                    "score_definition": "sigmoid of O2-3P late-fusion logit",
                    "created_utc": utc_now(),
                    "test_partition_used": False,
                }
            )
            source_table = create_score_cache(output, len(files), batch_size)
            for file_index, path in enumerate(files):
                digest = sha256_file(path)
                admitted_for_file = 0
                with h5py.File(path, "r") as source:
                    if str(source.attrs.get("processing_status")) != "OK":
                        raise ValueError(f"Non-OK processing status: {path}")
                    if str(source.attrs.get("source_label", "")).lower() != "th232":
                        raise ValueError(f"Unexpected source label: {path}")
                    event_count = int(source["waveform"].shape[0])
                    if source["waveform"].shape != (event_count, 4500):
                        raise ValueError(f"Unexpected waveform shape: {path}")
                    counts["input_events"] += event_count
                    for start in range(0, event_count, batch_size):
                        stop = min(start + batch_size, event_count)
                        energy = np.asarray(
                            source["corrected_energy_kev"][start:stop], dtype=np.float32
                        )
                        shaped = np.asarray(
                            source["shaped_energy_unit"][start:stop], dtype=np.float32
                        )
                        bits = np.asarray(
                            source["qc_rejection_bits"][start:stop], dtype=np.uint16
                        )
                        admitted, energy_valid, shaped_valid = th232_admission_mask(
                            energy, shaped, bits
                        )
                        counts["rejected_energy_or_qc_bits_0_to_2"] += int(
                            np.count_nonzero(~energy_valid)
                        )
                        counts["rejected_nonpositive_shaped_energy"] += int(
                            np.count_nonzero(energy_valid & ~shaped_valid)
                        )
                        if not np.any(admitted):
                            continue
                        waveforms = np.asarray(
                            source["waveform"][start:stop][admitted], dtype=np.float32
                        )
                        selected_shaped = shaped[admitted]
                        charge, current, fallback = extract_o2_features(
                            waveforms, selected_shaped
                        )
                        counts["t10_fallback_count"] += fallback
                        charge -= np.float32(statistics["charge_mean"])
                        charge /= np.float32(statistics["charge_std"])
                        current -= np.float32(statistics["current_mean"])
                        current /= np.float32(statistics["current_std"])
                        with torch.inference_mode():
                            score = torch.sigmoid(
                                model(
                                    torch.from_numpy(charge).to(device, non_blocking=True),
                                    torch.from_numpy(current).to(device, non_blocking=True),
                                )
                            ).cpu().numpy()
                        selected_rows = start + np.flatnonzero(admitted)
                        count = int(selected_rows.size)
                        end_index = next_index + count
                        resize_cache(output, end_index)
                        output["corrected_energy_kev"][next_index:end_index] = energy[
                            admitted
                        ]
                        output["score"][next_index:end_index] = score
                        output["source_file_index"][next_index:end_index] = file_index
                        output["source_row"][next_index:end_index] = selected_rows
                        next_index = end_index
                        admitted_for_file += count
                        counts["admitted_events"] += count
                source_table["path"][file_index] = relative(path)
                source_table["sha256"][file_index] = digest
                source_table["input_event_count"][file_index] = event_count
                source_table["admitted_event_count"][file_index] = admitted_for_file
                file_records.append(
                    {
                        "path": relative(path),
                        "sha256": digest,
                        "input_events": event_count,
                        "admitted_events": admitted_for_file,
                    }
                )
                print(
                    f"Th232 file {file_index + 1}/{len(files)} "
                    f"admitted={admitted_for_file} cumulative={next_index}",
                    flush=True,
                )
            output.attrs["event_count"] = next_index
            output.attrs["t10_fallback_count"] = counts["t10_fallback_count"]
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "score_cache": relative(output_path),
        "score_cache_sha256": sha256_file(output_path),
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "counts": counts,
        "files": file_records,
    }


def gaussian_linear(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * np.square((x - mean) / sigma)) + offset + slope * (
        x - mean
    )


def fit_peak_windows(no_cut_histogram: np.ndarray) -> list[PeakWindow]:
    smoothed = gaussian_filter1d(no_cut_histogram.astype(np.float64), 1.5)
    windows = []
    for reference in REFERENCE_PEAKS_KEV:
        search_half_width = 12.0 if reference < 1200.0 else 25.0
        search = (ENERGY_CENTERS >= reference - search_half_width) & (
            ENERGY_CENTERS <= reference + search_half_width
        )
        initial_mean = float(ENERGY_CENTERS[search][np.argmax(smoothed[search])])
        fit_half_width = 13.0 if reference < 1200.0 else 20.0
        selected = (ENERGY_CENTERS >= initial_mean - fit_half_width) & (
            ENERGY_CENTERS <= initial_mean + fit_half_width
        )
        x = ENERGY_CENTERS[selected]
        y = no_cut_histogram[selected].astype(np.float64)
        edge_count = max(2, x.size // 5)
        background = float(np.median(np.concatenate((y[:edge_count], y[-edge_count:]))))
        initial = (
            max(float(np.max(y) - background), 1.0),
            initial_mean,
            2.0,
            max(background, 0.0),
            0.0,
        )
        parameters, _ = curve_fit(
            gaussian_linear,
            x,
            y,
            p0=initial,
            bounds=(
                (0.0, initial_mean - 4.0, 0.6, 0.0, -np.inf),
                (np.inf, initial_mean + 4.0, 8.0, np.inf, np.inf),
            ),
            maxfev=20000,
        )
        mean = float(parameters[1])
        sigma = float(parameters[2])
        windows.append(
            PeakWindow(
                reference,
                mean,
                sigma,
                mean - 2.0 * sigma,
                mean + 2.0 * sigma,
                mean - 5.0 * sigma,
                mean - 3.0 * sigma,
                mean + 3.0 * sigma,
                mean + 5.0 * sigma,
            )
        )
    return windows


def interval_counts(histogram: np.ndarray, low: float, high: float) -> tuple[float, float]:
    overlap = np.maximum(
        0.0,
        np.minimum(ENERGY_EDGES[1:], high) - np.maximum(ENERGY_EDGES[:-1], low),
    ) / BIN_WIDTH_KEV
    counts = float(np.sum(histogram * overlap))
    weighted_energy = float(np.sum(histogram * overlap * ENERGY_CENTERS))
    return counts, weighted_energy


def peak_background_metrics(
    histogram: np.ndarray, window: PeakWindow
) -> dict[str, float]:
    roi_counts, _ = interval_counts(histogram, window.roi_low_kev, window.roi_high_kev)
    left_counts, left_energy = interval_counts(
        histogram, window.left_low_kev, window.left_high_kev
    )
    right_counts, right_energy = interval_counts(
        histogram, window.right_low_kev, window.right_high_kev
    )
    left_center = (
        left_energy / left_counts
        if left_counts > 0.0
        else 0.5 * (window.left_low_kev + window.left_high_kev)
    )
    right_center = (
        right_energy / right_counts
        if right_counts > 0.0
        else 0.5 * (window.right_low_kev + window.right_high_kev)
    )
    left_density = left_counts / (window.left_high_kev - window.left_low_kev)
    right_density = right_counts / (window.right_high_kev - window.right_low_kev)
    fraction = (window.centroid_kev - left_center) / (right_center - left_center)
    background_density = left_density + fraction * (right_density - left_density)
    background_counts = background_density * (window.roi_high_kev - window.roi_low_kev)
    net_peak_counts = roi_counts - background_counts
    return {
        "roi_counts": roi_counts,
        "estimated_background_counts": background_counts,
        "net_peak_counts": net_peak_counts,
        "peak_to_background": net_peak_counts / background_counts,
    }


def make_pb_rows(
    histograms: dict[str, np.ndarray],
    windows: list[PeakWindow],
    threshold_parameters: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for window in windows:
        baseline = peak_background_metrics(histograms["no_cut"], window)
        for condition in ("no_cut", "fixed_global", "fitted_energy_dependent"):
            metrics = peak_background_metrics(histograms[condition], window)
            rows.append(
                {
                    "reference_energy_kev": window.reference_kev,
                    "observed_centroid_kev": window.centroid_kev,
                    "fwhm_kev": 2.354820045 * window.sigma_kev,
                    "threshold_fit_domain": 100.0 <= window.reference_kev <= 1000.0,
                    "fitted_threshold_at_centroid": float(
                        stretched_exponential_threshold(
                            window.centroid_kev, *threshold_parameters
                        )
                    ),
                    "condition": condition,
                    **metrics,
                    "pb_improvement_factor_vs_no_cut": metrics["peak_to_background"]
                    / baseline["peak_to_background"],
                    "net_peak_retention_vs_no_cut": metrics["net_peak_counts"]
                    / baseline["net_peak_counts"],
                }
            )
    return rows


def save_pb_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_spectrum_csv(path: Path, histograms: dict[str, np.ndarray]) -> None:
    values = np.column_stack(
        (
            ENERGY_CENTERS,
            histograms["no_cut"],
            histograms["fixed_global"],
            histograms["fitted_energy_dependent"],
        )
    )
    np.savetxt(
        path,
        values,
        delimiter=",",
        header="energy_kev_bin_center,no_cut,fixed_global,fitted_energy_dependent",
        comments="",
        fmt=("%.1f", "%d", "%d", "%d"),
    )


def save_plots(
    output_dir: Path,
    threshold_points: tuple[np.ndarray, np.ndarray],
    threshold_parameters: list[float],
    histograms: dict[str, np.ndarray],
    windows: list[PeakWindow],
    pb_rows: list[dict[str, Any]],
) -> None:
    point_energy, point_threshold = threshold_points
    dense = np.linspace(100.0, 3200.0, 1200)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    in_domain = dense <= 1000.0
    axes[0].scatter(point_energy, point_threshold, label="Co-60 empirical bins")
    axes[0].plot(
        dense[in_domain],
        stretched_exponential_threshold(dense[in_domain], *threshold_parameters),
        label="stretched-exponential fit",
    )
    axes[0].set_xlim(100, 1000)
    axes[0].set_xlabel("Corrected energy (keV)")
    axes[0].set_ylabel("O2-3P threshold")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        dense,
        stretched_exponential_threshold(dense, *threshold_parameters),
        color="#d95f02",
    )
    axes[1].axvspan(100, 1000, color="#1b9e77", alpha=0.12, label="fit domain")
    axes[1].axvspan(1000, 3200, color="#d95f02", alpha=0.08, label="extrapolation")
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[1].set_ylabel("O2-3P threshold")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_fit_and_extrapolation.png", dpi=180)
    plt.close(figure)

    colors = {
        "no_cut": "black",
        "fixed_global": "#377eb8",
        "fitted_energy_dependent": "#e41a1c",
    }
    labels = {
        "no_cut": "No score cut",
        "fixed_global": "Fixed threshold 0.27931",
        "fitted_energy_dependent": "Fitted energy threshold",
    }
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for condition, histogram in histograms.items():
        for axis in axes:
            axis.step(
                ENERGY_CENTERS,
                histogram,
                where="mid",
                linewidth=0.9,
                color=colors[condition],
                label=labels[condition],
            )
    axes[0].set_ylabel("Counts / 1 keV")
    axes[0].legend()
    axes[0].set_title("Corrected Th-232 spectrum after frozen O2-3P cuts")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set_ylabel("Counts / 1 keV")
    axes[1].set_xlabel("Corrected energy (keV)")
    figure.tight_layout()
    figure.savefig(output_dir / "th232_corrected_energy_spectra.png", dpi=180)
    plt.close(figure)

    peak_labels = [f"{window.reference_kev:g}" for window in windows]
    fixed = [
        row["pb_improvement_factor_vs_no_cut"]
        for row in pb_rows
        if row["condition"] == "fixed_global"
    ]
    fitted = [
        row["pb_improvement_factor_vs_no_cut"]
        for row in pb_rows
        if row["condition"] == "fitted_energy_dependent"
    ]
    x = np.arange(len(windows))
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - 0.18, fixed, width=0.36, label="Fixed threshold")
    axis.bar(x + 0.18, fitted, width=0.36, label="Fitted energy threshold")
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xticks(x, peak_labels)
    axis.set_xlabel("Th-232 reference peak (keV)")
    axis.set_ylabel("P/B improvement factor vs no cut")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "th232_peak_background_improvement.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/co60_continuum_o2_3p_threshold_curve_20260819/threshold_curve_50kev.csv",
    )
    parser.add_argument(
        "--co60-validation-scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/co60_continuum_o2_3p_threshold_curve_20260819/validation_scores.h5",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/three_peak_weight_scan_20260819/late_fusion_best.pt",
    )
    parser.add_argument(
        "--th232-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_csv = args.threshold_csv.resolve()
    fit = fit_threshold_curve(threshold_csv)
    parameters = fit["parameter_vector"]
    point_energy, point_threshold, source_rows = load_threshold_points(threshold_csv)

    co60_bin_checks = []
    with h5py.File(args.co60_validation_scores.resolve(), "r") as source:
        co60_energy = np.asarray(source["corrected_energy_kev"], dtype=np.float32)
        co60_scores = np.asarray(source["score"], dtype=np.float32)
    for index, row in enumerate(source_rows):
        low = float(row["energy_low_kev"])
        high = float(row["energy_high_kev"])
        selected = (co60_energy >= low) & (
            (co60_energy <= high) if index == len(source_rows) - 1 else (co60_energy < high)
        )
        fitted_threshold = stretched_exponential_threshold(
            co60_energy[selected], *parameters
        )
        co60_bin_checks.append(
            {
                "energy_low_kev": low,
                "energy_high_kev": high,
                "event_count": int(np.count_nonzero(selected)),
                "fitted_curve_passing_fraction": float(
                    np.mean(co60_scores[selected] >= fitted_threshold)
                ),
                "target_passing_fraction": float(row["target_passing_fraction"]),
            }
        )

    files = sorted(args.th232_dir.resolve().glob("*.h5"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 corrected Th-232 files, found {len(files)}")
    cache_path = output_dir / "th232_o2_3p_scores.h5"
    scoring = score_th232(
        files, args.checkpoint.resolve(), cache_path, args.batch_size, device
    )
    with h5py.File(cache_path, "r") as cache:
        energy = np.asarray(cache["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(cache["score"], dtype=np.float32)
    fitted_thresholds = stretched_exponential_threshold(energy, *parameters)
    masks = {
        "no_cut": np.ones(energy.size, dtype=bool),
        "fixed_global": scores >= FIXED_BASELINE_THRESHOLD,
        "fitted_energy_dependent": scores >= fitted_thresholds,
    }
    histograms = {
        condition: np.histogram(energy[mask], ENERGY_EDGES)[0]
        for condition, mask in masks.items()
    }
    windows = fit_peak_windows(histograms["no_cut"])
    pb_rows = make_pb_rows(histograms, windows, parameters)

    pb_csv = output_dir / "th232_peak_to_background.csv"
    spectrum_csv = output_dir / "th232_corrected_spectra_1kev.csv"
    save_pb_csv(pb_csv, pb_rows)
    save_spectrum_csv(spectrum_csv, histograms)
    save_plots(
        output_dir,
        (point_energy, point_threshold),
        parameters,
        histograms,
        windows,
        pb_rows,
    )

    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "model_name": MODEL_NAME,
        "threshold_fit": {
            **fit,
            "source_csv": relative(threshold_csv),
            "source_csv_sha256": sha256_file(threshold_csv),
            "selection_rule": "fit family selected from Co-60 only before Th-232 scoring",
            "below_100kev_policy": "clamp naturally to c+a through max(E-100,0)",
            "above_1000kev_policy": "extrapolate to fitted asymptote; mark peaks above fit domain",
        },
        "co60_fitted_curve_check": {
            "score_file": relative(args.co60_validation_scores.resolve()),
            "score_file_sha256": sha256_file(args.co60_validation_scores.resolve()),
            "bins": co60_bin_checks,
            "minimum_passing_fraction": min(
                row["fitted_curve_passing_fraction"] for row in co60_bin_checks
            ),
            "maximum_passing_fraction": max(
                row["fitted_curve_passing_fraction"] for row in co60_bin_checks
            ),
        },
        "th232_scoring": scoring,
        "admission": {
            "energy_dataset": "corrected_energy_kev",
            "energy_range_kev": [ENERGY_MIN_KEV, ENERGY_MAX_KEV],
            "qc_rule": "reject qc_rejection_bits 0-2; retain noise bit 3 and pulse bit 4",
            "shaped_energy_rule": "finite and positive",
        },
        "conditions": {
            "no_cut_events": int(np.count_nonzero(masks["no_cut"])),
            "fixed_global_threshold": FIXED_BASELINE_THRESHOLD,
            "fixed_global_events": int(np.count_nonzero(masks["fixed_global"])),
            "fixed_global_fraction": float(np.mean(masks["fixed_global"])),
            "fitted_energy_dependent_events": int(
                np.count_nonzero(masks["fitted_energy_dependent"])
            ),
            "fitted_energy_dependent_fraction": float(
                np.mean(masks["fitted_energy_dependent"])
            ),
            "fitted_within_100_1000_fraction": float(
                np.mean(
                    masks["fitted_energy_dependent"]
                    & (energy >= 100.0)
                    & (energy <= 1000.0)
                )
                / np.mean((energy >= 100.0) & (energy <= 1000.0))
            ),
        },
        "peak_windows": [asdict(window) for window in windows],
        "peak_background_rows": pb_rows,
        "artifacts": {},
        "test_partition_used": False,
        "scientific_boundary": (
            "Historical corrected Th-232 cache; not a newly untouched campaign. "
            "Thresholds above 1000 keV are extrapolated beyond the Co-60 fit domain."
        ),
    }
    for name in (
        "threshold_fit_and_extrapolation.png",
        "th232_corrected_energy_spectra.png",
        "th232_peak_background_improvement.png",
        pb_csv.name,
        spectrum_csv.name,
    ):
        path = output_dir / name
        report["artifacts"][name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "th232_o2_3p_energy_threshold_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
