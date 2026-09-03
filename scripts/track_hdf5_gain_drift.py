#!/usr/bin/env python3
"""Fit isotope photopeaks in each processed HDF5 file and track gain drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class PeakDefinition:
    name: str
    reference_kev: float
    fit_half_width_kev: float
    primary: bool = False


# Reference gamma energies are in vacuum-energy keV.  Th-232 entries are the
# prominent daughter lines normally used to calibrate a Th-232 chain spectrum.
SOURCE_PEAKS: dict[str, tuple[PeakDefinition, ...]] = {
    "ba133": (
        PeakDefinition("ba133_276", 276.3989, 7.0),
        PeakDefinition("ba133_303", 302.8508, 7.0),
        PeakDefinition("ba133_356", 356.0129, 9.0, True),
        PeakDefinition("ba133_384", 383.8485, 9.0),
    ),
    "co60": (
        PeakDefinition("co60_1173", 1173.228, 18.0),
        PeakDefinition("co60_1332", 1332.492, 20.0, True),
    ),
    "cs137": (PeakDefinition("cs137_662", 661.657, 13.0, True),),
    "na22": (
        PeakDefinition("annihilation_511", 510.99895, 13.0, True),
        PeakDefinition("na22_1275", 1274.537, 20.0),
    ),
    "th232": (
        PeakDefinition("pb212_239", 238.632, 8.0),
        PeakDefinition("ac228_338", 338.320, 10.0),
        PeakDefinition("tl208_583", 583.191, 14.0, True),
        PeakDefinition("ac228_911", 911.204, 18.0),
        PeakDefinition("ac228_969", 968.971, 18.0),
        PeakDefinition("tl208_2615", 2614.511, 38.0),
    ),
}

SOURCE_DISPLAY = {
    "ba133": "Ba-133",
    "co60": "Co-60",
    "cs137": "Cs-137",
    "na22": "Na-22",
    "th232": "Th-232 chain",
}

SOURCE_PLOT_MAX_KEV = {
    "ba133": 500.0,
    "co60": 1500.0,
    "cs137": 850.0,
    "na22": 1450.0,
    "th232": 3000.0,
}

# For full-retention evaluation HDF5 files, preserve events rejected only by
# preprocessing energy bounds.  Gain drift can move a valid peak outside those
# provisional bounds.  Reject non-finite energy, baseline-noise, and pulse-time
# failures (bits 0, 3, and 4).
GAIN_DRIFT_QC_REJECT_MASK = (1 << 0) | (1 << 3) | (1 << 4)
TIMESTAMP_PATTERN = re.compile(r"_(20\d{6})_(\d{6})(?:_|\.)")


def normalize_source(value: Any, path: Path) -> str:
    text = str(value.decode() if isinstance(value, bytes) else value).strip().lower()
    text = text.replace("-", "").replace("_", "")
    aliases = {
        "ba133": "ba133",
        "co60": "co60",
        "cs137": "cs137",
        "na22": "na22",
        "th232": "th232",
    }
    if text in aliases:
        return aliases[text]
    lower_name = path.name.lower()
    for source in aliases:
        if lower_name.startswith(source):
            return source
    return "unknown"


def parse_acquisition_time(path: Path) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def gaussian_linear(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2) + offset + slope * (x - centroid)


def load_energy(path: Path, dataset_name: str = "reconstructed_energy_kev") -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"{dataset_name} dataset is missing")
        source = normalize_source(handle.attrs.get("source_label", ""), path)
        status = str(handle.attrs.get("processing_status", ""))
        energy = np.asarray(handle[dataset_name][:], dtype=np.float64)
        input_count = int(energy.size)
        keep = np.isfinite(energy)
        qc_rejected = 0
        if "qc_rejection_bits" in handle:
            bits = np.asarray(handle["qc_rejection_bits"][:], dtype=np.uint16)
            if bits.shape != energy.shape:
                raise ValueError(f"qc_rejection_bits shape does not match {dataset_name}")
            qc_keep = (bits & GAIN_DRIFT_QC_REJECT_MASK) == 0
            qc_rejected = int(np.count_nonzero(~qc_keep))
            keep &= qc_keep
        metadata = {
            "source": source,
            "processing_status": status,
            "stored_entries": input_count,
            "analysis_entries": int(np.count_nonzero(keep)),
            "nonfinite_entries": int(np.count_nonzero(~np.isfinite(energy))),
            "gain_drift_qc_rejected_entries": qc_rejected,
            "retention_mode": str(handle.attrs.get("retention_mode", "accepted_only")),
            "calibration_kev_per_shaped_unit": float(
                handle.attrs.get("energy_calibration_kev_per_shaped_unit", math.nan)
            ),
            "calibration_intercept_kev": float(
                handle.attrs.get("energy_calibration_intercept_kev", math.nan)
            ),
        }
    return energy[keep], metadata


def estimate_scale(
    energy: np.ndarray,
    peaks: tuple[PeakDefinition, ...],
) -> tuple[float, float, int, int]:
    """Estimate one common scale using all available lines to preserve identity."""

    primary = next(peak for peak in peaks if peak.primary)
    bin_width = 0.5
    maximum = max(peak.reference_kev for peak in peaks) * 1.13
    edges = np.arange(0.0, maximum + bin_width, bin_width)
    counts, _ = np.histogram(energy, bins=edges)
    scale_low, scale_high = (0.97, 1.03) if len(peaks) == 1 else (0.88, 1.12)
    primary_low = scale_low * primary.reference_kev
    primary_high = scale_high * primary.reference_kev
    primary_search_events = int(np.count_nonzero((energy >= primary_low) & (energy <= primary_high)))
    if primary_search_events < 50:
        raise RuntimeError("fewer than 50 events in primary-peak search range")

    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=2.0)
    background = median_filter(smoothed, size=81, mode="nearest")
    significance = np.maximum(0.0, smoothed - background) / np.sqrt(np.maximum(background, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    scale_grid = np.arange(scale_low, scale_high + 0.00001, 0.0001)
    line_scores = np.stack(
        [
            np.interp(peak.reference_kev * scale_grid, centers, significance, left=0.0, right=0.0)
            for peak in peaks
        ],
        axis=0,
    )
    # log1p limits domination by one intense line.  A wrong Co-60 seed that
    # maps 1332 keV onto the real 1173-keV line, for example, then loses to the
    # scale supported simultaneously by both Co-60 lines.
    total_score = np.log1p(line_scores).sum(axis=0)
    best_index = int(np.argmax(total_score))
    scale_seed = float(scale_grid[best_index])
    supported_lines = int(np.count_nonzero(line_scores[:, best_index] >= 5.0))
    required_support = 1 if len(peaks) == 1 else 2
    if supported_lines < required_support:
        raise RuntimeError(f"common scale seed is not supported by at least {required_support} reference line(s)")
    seed_kev = primary.reference_kev * scale_seed
    return scale_seed, seed_kev, primary_search_events, supported_lines


def fit_peak(
    energy: np.ndarray,
    peak: PeakDefinition,
    scale_seed: float,
    bin_width_kev: float,
) -> dict[str, Any]:
    predicted = peak.reference_kev * scale_seed
    half_width = peak.fit_half_width_kev * max(1.0, scale_seed)
    low = predicted - half_width
    high = predicted + half_width
    edges = np.arange(low, high + bin_width_kev, bin_width_kev)
    counts, _ = np.histogram(energy, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_events = int(counts.sum())
    base: dict[str, Any] = {
        "peak_name": peak.name,
        "primary_peak": peak.primary,
        "reference_energy_kev": peak.reference_kev,
        "predicted_energy_kev": predicted,
        "fit_window_low_kev": low,
        "fit_window_high_kev": high,
        "fit_bin_width_kev": bin_width_kev,
        "fit_window_events": fit_events,
    }
    if fit_events < 100 or counts.max(initial=0) < 5:
        return {**base, "fit_status": "INSUFFICIENT_COUNTS", "fit_error": "too few peak-window counts"}

    edge_count = max(3, min(10, len(counts) // 5))
    background0 = float(np.median(np.r_[counts[:edge_count], counts[-edge_count:]]))
    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    mode_index = int(np.argmax(smooth[2:-2]) + 2) if len(smooth) > 5 else int(np.argmax(smooth))
    centroid0 = float(centers[mode_index])
    amplitude0 = max(1.0, float(counts[mode_index]) - background0)
    sigma0 = min(max(1.0, peak.reference_kev * 0.0025), half_width / 4.0)
    sigma_upper = max(2.0, half_width / 2.0)
    try:
        fitted, covariance = curve_fit(
            gaussian_linear,
            centers,
            counts,
            p0=[amplitude0, centroid0, sigma0, background0, 0.0],
            bounds=(
                [0.0, low, 0.15, 0.0, -np.inf],
                [np.inf, high, sigma_upper, np.inf, np.inf],
            ),
            maxfev=100_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return {**base, "fit_status": "FIT_FAILED", "fit_error": str(error)}

    errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    amplitude, centroid, sigma, offset, slope = (float(value) for value in fitted)
    centroid_error = float(errors[1])
    model = gaussian_linear(centers, *fitted)
    residual = counts - model
    dof = max(1, len(counts) - len(fitted))
    reduced_chi_square = float(np.sum(residual**2 / np.maximum(model, 1.0)) / dof)
    gain_ratio = centroid / peak.reference_kev
    correction_factor = peak.reference_kev / centroid
    fwhm = 2.354820045 * sigma
    net_area = amplitude * sigma * math.sqrt(2.0 * math.pi) / bin_width_kev
    status = "OK"
    warnings: list[str] = []
    if sigma >= 0.95 * sigma_upper:
        warnings.append("sigma_at_upper_bound")
    boundary_margin = max(2.0 * bin_width_kev, 0.10 * half_width)
    if min(centroid - low, high - centroid) <= boundary_margin:
        warnings.append("centroid_near_fit_window_boundary")
    if centroid_error > max(1.0, 0.25 * fwhm):
        warnings.append("large_centroid_uncertainty")
    if reduced_chi_square > 20.0:
        warnings.append("poor_reduced_chi_square")
    if amplitude < 3.0 * math.sqrt(max(offset, 1.0)):
        warnings.append("low_peak_significance")
    if warnings:
        status = "WARN"
    return {
        **base,
        "fit_status": status,
        "fit_warnings": warnings,
        "fitted_centroid_kev": centroid,
        "centroid_error_kev": centroid_error,
        "sigma_kev": sigma,
        "fwhm_kev": fwhm,
        "amplitude_counts_per_bin": amplitude,
        "background_counts_per_bin": offset,
        "net_gaussian_events": net_area,
        "reduced_chi_square": reduced_chi_square,
        "centroid_to_reference_ratio": gain_ratio,
        "centroid_scale_correction": correction_factor,
        "centroid_residual_kev": centroid - peak.reference_kev,
    }


def plot_file_spectrum(
    path: Path,
    energy: np.ndarray,
    source: str,
    fits: list[dict[str, Any]],
    output_path: Path,
) -> None:
    maximum = SOURCE_PLOT_MAX_KEV[source]
    edges = np.arange(0.0, maximum + 1.0, 1.0)
    counts, _ = np.histogram(energy, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, logarithmic in zip(axes, (False, True)):
        axis.step(centers, counts, where="mid", linewidth=0.8, color="tab:blue")
        if logarithmic:
            axis.set_yscale("log")
            axis.set_ylim(bottom=0.8)
        for fit in fits:
            reference = float(fit["reference_energy_kev"])
            axis.axvline(reference, color="black", linestyle=":", linewidth=0.7, alpha=0.7)
            if fit.get("fit_status") in {"OK", "WARN"}:
                centroid = float(fit["fitted_centroid_kev"])
                axis.axvline(centroid, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.85)
        axis.set_ylabel("Counts / 1 keV")
        axis.grid(alpha=0.2)
    primary = next((fit for fit in fits if fit.get("primary_peak")), None)
    drift_text = "primary fit unavailable"
    if primary and primary.get("fit_status") in {"OK", "WARN"}:
        drift_text = (
            f"primary {primary['fitted_centroid_kev']:.3f} keV vs "
            f"{primary['reference_energy_kev']:.3f} keV; "
            f"centroid shift {primary['centroid_residual_kev']:+.3f} keV"
        )
    axes[0].set_title(f"{SOURCE_DISPLAY[source]} — {path.name}\n{drift_text}")
    axes[-1].set_xlabel("Stored reconstructed energy (keV)")
    axes[-1].set_xlim(0.0, maximum)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_source_drift(source: str, file_rows: list[dict[str, Any]], output_path: Path) -> None:
    usable = [row for row in file_rows if row.get("primary_fit_status") in {"OK", "WARN"}]
    if not usable:
        return
    usable.sort(key=lambda row: (row.get("acquisition_time") or "", row["hdf5_file"]))
    x = np.arange(len(usable))
    drift = np.asarray([row["primary_centroid_residual_kev"] for row in usable], dtype=float)
    relative = np.asarray([row["primary_relative_to_baseline_kev"] for row in usable], dtype=float)
    uncertainty = np.asarray([row["primary_centroid_error_kev"] for row in usable], dtype=float)
    labels = [Path(row["hdf5_file"]).stem for row in usable]
    fig, axis = plt.subplots(figsize=(max(11.0, 0.32 * len(usable)), 5.8))
    axis.errorbar(
        x, drift, yerr=uncertainty, fmt="o-", markersize=3.5, linewidth=0.8,
        capsize=2, label="centroid shift vs reference",
    )
    axis.plot(x, relative, "s--", markersize=3.0, linewidth=0.8, label="shift vs first fitted file")
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Primary-peak centroid shift (keV)")
    axis.set_xlabel("HDF5 file in acquisition-time order")
    axis.set_title(f"{SOURCE_DISPLAY[source]} per-file centroid-derived gain proxy")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def discover_hdf5(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*.h5") if recursive else input_dir.glob("*.h5")
    return sorted(path for path in iterator if path.is_file())


def apply_peak_baselines(peak_rows: list[dict[str, Any]]) -> None:
    """Add same-source, same-line shifts relative to the earliest usable fit."""

    peak_baselines: dict[tuple[str, str], float] = {}
    for row in sorted(
        peak_rows,
        key=lambda item: (item.get("acquisition_time") or "", item["hdf5_file"], item["peak_name"]),
    ):
        if row.get("fit_status") in {"OK", "WARN"}:
            peak_baselines.setdefault(
                (row["source"], row["peak_name"]), float(row["fitted_centroid_kev"])
            )
    for row in peak_rows:
        centroid = float(row.get("fitted_centroid_kev", math.nan))
        baseline = float(peak_baselines.get((row["source"], row["peak_name"]), math.nan))
        row["peak_baseline_centroid_kev"] = baseline
        row["relative_to_peak_baseline_kev"] = (
            centroid - baseline if np.isfinite(centroid) and np.isfinite(baseline) else math.nan
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("processed_data/waveform_hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gain_drift"))
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-spectra", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bin-width-kev", type=float, default=0.5)
    parser.add_argument("--energy-dataset", default="reconstructed_energy_kev")
    args = parser.parse_args()
    if args.bin_width_kev <= 0.0:
        parser.error("--bin-width-kev must be positive")

    input_paths = discover_hdf5(args.input_dir, args.recursive)
    if not input_paths:
        raise FileNotFoundError(f"no HDF5 files found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only artifacts generated by this script so a rerun cannot leave
    # stale spectra or source plots from files that are no longer present.
    if args.write_spectra:
        shutil.rmtree(args.output_dir / "spectra", ignore_errors=True)
    for source in SOURCE_PEAKS:
        (args.output_dir / f"{source}_gain_drift.png").unlink(missing_ok=True)

    peak_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    per_source_files: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCE_PEAKS}

    for index, path in enumerate(input_paths, start=1):
        try:
            energy, metadata = load_energy(path, args.energy_dataset)
        except (OSError, KeyError, ValueError) as error:
            skipped.append({"hdf5_file": str(path), "reason": str(error)})
            print(f"[{index}/{len(input_paths)}] SKIP {path}: {error}")
            continue
        source = metadata["source"]
        if source not in SOURCE_PEAKS:
            skipped.append({"hdf5_file": str(path), "reason": f"unsupported source: {source}"})
            print(f"[{index}/{len(input_paths)}] SKIP {path}: unsupported source {source}")
            continue
        if metadata["processing_status"] not in {"", "OK"}:
            skipped.append(
                {"hdf5_file": str(path), "reason": f"processing_status={metadata['processing_status']}"}
            )
            print(f"[{index}/{len(input_paths)}] SKIP {path}: status {metadata['processing_status']}")
            continue
        if energy.size == 0:
            skipped.append({"hdf5_file": str(path), "reason": "no usable reconstructed energies"})
            print(f"[{index}/{len(input_paths)}] SKIP {path}: no usable energies")
            continue

        peaks = SOURCE_PEAKS[source]
        primary = next(peak for peak in peaks if peak.primary)
        acquisition_time = parse_acquisition_time(path)
        try:
            scale_seed, seed_centroid, search_events, seed_support_peaks = estimate_scale(energy, peaks)
            fits = [fit_peak(energy, peak, scale_seed, args.bin_width_kev) for peak in peaks]
        except RuntimeError as error:
            skipped.append({"hdf5_file": str(path), "reason": str(error)})
            print(f"[{index}/{len(input_paths)}] SKIP {path}: {error}")
            continue

        common = {
            "hdf5_file": str(path),
            "source": source,
            "source_display": SOURCE_DISPLAY[source],
            "acquisition_time": acquisition_time.isoformat() if acquisition_time else "",
            **metadata,
            "scale_seed": scale_seed,
            "scale_seed_centroid_kev": seed_centroid,
            "primary_search_events": search_events,
            "scale_seed_supporting_peaks": seed_support_peaks,
        }
        for fit in fits:
            peak_rows.append({**common, **fit})
        primary_fit = next(fit for fit in fits if fit["primary_peak"])
        file_row = {
            **common,
            "primary_peak_name": primary_fit["peak_name"],
            "primary_fit_status": primary_fit["fit_status"],
            "primary_reference_energy_kev": primary_fit["reference_energy_kev"],
            "primary_fitted_centroid_kev": primary_fit.get("fitted_centroid_kev", math.nan),
            "primary_centroid_error_kev": primary_fit.get("centroid_error_kev", math.nan),
            "primary_centroid_residual_kev": primary_fit.get("centroid_residual_kev", math.nan),
            "primary_centroid_to_reference_ratio": primary_fit.get("centroid_to_reference_ratio", math.nan),
            "primary_centroid_scale_correction": primary_fit.get("centroid_scale_correction", math.nan),
            "successful_peak_fits": sum(fit["fit_status"] in {"OK", "WARN"} for fit in fits),
            "requested_peak_fits": len(fits),
        }
        file_rows.append(file_row)
        per_source_files[source].append(file_row)
        if args.write_spectra:
            plot_file_spectrum(
                path,
                energy,
                source,
                fits,
                args.output_dir / "spectra" / source / f"{path.stem}_spectrum.png",
            )
        print(
            f"[{index}/{len(input_paths)}] {source} {path.name}: "
            f"{primary_fit['fit_status']} "
            f"centroid_shift={primary_fit.get('centroid_residual_kev', math.nan):+.4f} keV"
        )

    for source, rows in per_source_files.items():
        usable = sorted(
            (row for row in rows if row["primary_fit_status"] in {"OK", "WARN"}),
            key=lambda row: (row.get("acquisition_time") or "", row["hdf5_file"]),
        )
        if usable:
            baseline_centroid = float(usable[0]["primary_fitted_centroid_kev"])
            for row in rows:
                centroid = float(row["primary_fitted_centroid_kev"])
                row["primary_baseline_centroid_kev"] = baseline_centroid
                row["primary_relative_to_baseline_kev"] = (
                    centroid - baseline_centroid if np.isfinite(centroid) else math.nan
                )
        plot_source_drift(source, rows, args.output_dir / f"{source}_gain_drift.png")

    # Baseline-relative peak shifts must use the same line, not the source's
    # primary-line centroid.  This also makes cross-line audits meaningful.
    apply_peak_baselines(peak_rows)

    peak_fields = [
        "hdf5_file", "source", "source_display", "acquisition_time", "processing_status",
        "retention_mode", "stored_entries", "analysis_entries", "nonfinite_entries",
        "gain_drift_qc_rejected_entries", "peak_name", "primary_peak", "fit_status",
        "reference_energy_kev", "predicted_energy_kev", "fitted_centroid_kev",
        "centroid_error_kev", "centroid_residual_kev", "centroid_to_reference_ratio",
        "centroid_scale_correction", "sigma_kev", "fwhm_kev",
        "net_gaussian_events",
        "reduced_chi_square", "fit_window_low_kev", "fit_window_high_kev",
        "fit_window_events", "fit_bin_width_kev", "scale_seed", "scale_seed_centroid_kev",
        "primary_search_events", "scale_seed_supporting_peaks", "peak_baseline_centroid_kev",
        "relative_to_peak_baseline_kev", "fit_warnings", "fit_error",
    ]
    file_fields = [
        "hdf5_file", "source", "source_display", "acquisition_time", "processing_status",
        "retention_mode", "stored_entries", "analysis_entries", "nonfinite_entries",
        "gain_drift_qc_rejected_entries", "primary_peak_name", "primary_fit_status",
        "primary_reference_energy_kev", "primary_fitted_centroid_kev",
        "primary_centroid_error_kev", "primary_centroid_residual_kev",
        "primary_centroid_to_reference_ratio", "primary_centroid_scale_correction",
        "successful_peak_fits",
        "requested_peak_fits", "scale_seed", "scale_seed_centroid_kev", "primary_search_events",
        "scale_seed_supporting_peaks", "primary_baseline_centroid_kev",
        "primary_relative_to_baseline_kev",
    ]
    write_csv(args.output_dir / "peak_positions.csv", peak_rows, peak_fields)
    write_csv(args.output_dir / "file_gain_summary.csv", file_rows, file_fields)

    source_summary: dict[str, Any] = {}
    for source, rows in per_source_files.items():
        values = np.asarray(
            [row["primary_centroid_residual_kev"] for row in rows if row["primary_fit_status"] in {"OK", "WARN"}],
            dtype=float,
        )
        ok_values = np.asarray(
            [row["primary_centroid_residual_kev"] for row in rows if row["primary_fit_status"] == "OK"],
            dtype=float,
        )
        if values.size == 0:
            continue
        status_counts = {
            status: sum(row["primary_fit_status"] == status for row in rows)
            for status in sorted({row["primary_fit_status"] for row in rows})
        }
        source_summary[source] = {
            "source_display": SOURCE_DISPLAY[source],
            "file_count": len(rows),
            "fitted_file_count": int(values.size),
            "primary_fit_status_counts": status_counts,
            "primary_peak_name": rows[0]["primary_peak_name"],
            "primary_reference_energy_kev": rows[0]["primary_reference_energy_kev"],
            "all_usable_mean_centroid_shift_kev": float(np.mean(values)),
            "all_usable_standard_deviation_centroid_shift_kev": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "all_usable_minimum_centroid_shift_kev": float(np.min(values)),
            "all_usable_maximum_centroid_shift_kev": float(np.max(values)),
            "all_usable_peak_to_peak_centroid_shift_kev": float(np.ptp(values)),
            "ok_only_file_count": int(ok_values.size),
            "ok_only_mean_centroid_shift_kev": float(np.mean(ok_values)) if ok_values.size else None,
            "ok_only_standard_deviation_centroid_shift_kev": (
                float(np.std(ok_values, ddof=1)) if ok_values.size > 1 else (0.0 if ok_values.size else None)
            ),
        }

    results = {
        "status": "OK" if file_rows else "NO_VALID_FILES",
        "input_directory": str(args.input_dir),
        "discovered_hdf5_files": len(input_paths),
        "analyzed_hdf5_files": len(file_rows),
        "skipped_hdf5_files": skipped,
        "energy_source": f"stored {args.energy_dataset} dataset",
        "qc_policy": {
            "legacy_accepted_only_files": f"all finite stored values from {args.energy_dataset}",
            "full_retention_files": "reject qc bits 0, 3, and 4; retain energy-bound-only bits 1 and 2",
            "full_retention_reject_mask": GAIN_DRIFT_QC_REJECT_MASK,
        },
        "centroid_shift_definition": "fitted_centroid_kev - reference_energy_kev",
        "metric_interpretation": (
            "centroid-derived gain-plus-offset proxy in the stored affine-calibrated energy; "
            "it is not a pure multiplicative electronics-gain estimate when the intercept also drifts"
        ),
        "relative_drift_definition": "fitted_centroid_kev - first_fitted_file_centroid_kev",
        "centroid_scale_correction_definition": "reference_energy_kev / fitted_centroid_kev",
        "reference_provenance": {
            "source": "NNDC Nuclear Wallet Cards for Radioactive Nuclides, October 2023 (ENSDF-evaluated data)",
            "publication_version": "2023-10",
            "accessed": "2026-08-15",
            "url": "https://www.nndc.bnl.gov/walletcards/doc/wallet-cards-radioactive-2023-10.pdf",
            "th232_note": "Th-232-chain entries are daughter lines from Pb-212, Ac-228, and Tl-208",
        },
        "reference_peaks": {
            source: [peak.__dict__ for peak in peaks] for source, peaks in SOURCE_PEAKS.items()
        },
        "source_summary": source_summary,
        "outputs": {
            "peak_positions_csv": str(args.output_dir / "peak_positions.csv"),
            "file_gain_summary_csv": str(args.output_dir / "file_gain_summary.csv"),
            "per_file_spectra_directory": str(args.output_dir / "spectra") if args.write_spectra else None,
            "source_gain_drift_plots": {
                source: str(args.output_dir / f"{source}_gain_drift.png")
                for source in source_summary
            },
        },
    }
    results_path = args.output_dir / "gain_drift_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(results_path), **source_summary}, indent=2))
    return 0 if file_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
