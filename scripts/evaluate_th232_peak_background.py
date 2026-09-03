#!/usr/bin/env python3
"""Score external Th-232 waveforms and compare peak-to-background ratios.

The compact and O2-style late-fusion checkpoints were trained on all selected
Ba-133 photopeaks plus the Na-22 511-keV photopeak.  Score cuts are calibrated
only from the corresponding held-out combined-domain validation positives.
Th-232 is treated as external evaluation data and is not used to select a model
or score threshold.

Peak windows are fitted once on the no-CNN spectrum.  Each peak uses a fixed
+/-2 sigma signal ROI and equal-total-width sidebands at 3--5 sigma on both
sides.  The reported P/B is (ROI counts - estimated background) / estimated
background, where the background is the linearly interpolated sideband count
expected in the ROI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_o2_late_fusion import (  # noqa: E402
    O2LateFusion,
    extract_o2_features,
    sha256_file,
)
from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RawPartition,
    RepresentationConfig,
    apply_channel_statistics,
    representation_config_from_checkpoint,
    build_representation,
)

ACCEPTANCES = (0.99, 0.95, 0.90, 0.60)
REFERENCE_PEAKS_KEV = (238.632, 338.320, 583.187, 911.204, 968.971, 1588.19, 2614.511)
ENERGY_MIN_KEV = 0.0
ENERGY_MAX_KEV = 3200.0
BIN_WIDTH_KEV = 1.0
ENERGY_EDGES = np.arange(ENERGY_MIN_KEV, ENERGY_MAX_KEV + BIN_WIDTH_KEV, BIN_WIDTH_KEV)
ENERGY_CENTERS = 0.5 * (ENERGY_EDGES[:-1] + ENERGY_EDGES[1:])


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


def weighted_acceptance_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    acceptance: float,
) -> float:
    """Return a deterministic lower score cut with requested weighted retention."""
    order = np.argsort(scores)[::-1]
    sorted_scores = np.asarray(scores[order], dtype=np.float64)
    sorted_weights = np.asarray(weights[order], dtype=np.float64)
    target_weight = acceptance * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), target_weight, side="left"))
    return float(sorted_scores[min(index, sorted_scores.size - 1)])


def load_validation_thresholds(scores_path: Path) -> dict[str, Any]:
    with np.load(scores_path) as values:
        labels = np.asarray(values["combined_ba_na511_labels"], dtype=np.int8)
        weights = np.asarray(values["combined_ba_na511_weights"], dtype=np.float64)
        scores = np.asarray(values["combined_ba_na511_scores"], dtype=np.float64)
    positive = labels == 1
    result: dict[str, Any] = {
        "scores_path": scores_path.relative_to(PROJECT_ROOT).as_posix(),
        "scores_sha256": sha256_file(scores_path),
        "positive_event_count": int(np.count_nonzero(positive)),
        "thresholds": {},
    }
    for acceptance in ACCEPTANCES:
        threshold = weighted_acceptance_threshold(
            scores[positive], weights[positive], acceptance
        )
        accepted = scores[positive] >= threshold
        actual = float(np.sum(weights[positive][accepted]) / np.sum(weights[positive]))
        result["thresholds"][f"{int(acceptance * 100)}pct"] = {
            "requested_weighted_acceptance": acceptance,
            "score_threshold": threshold,
            "actual_weighted_acceptance": actual,
            "actual_unweighted_acceptance": float(np.mean(accepted)),
        }
    return result


def load_models(device: torch.device) -> tuple[Any, ...]:
    compact_path = PROJECT_ROOT / "outputs/models/ba_all_na511_cnn/both_ma10_global_t10_w750.pt"
    late_path = PROJECT_ROOT / "outputs/models/ba_all_na511_o2_late_fusion/o2_late_fusion_best.pt"
    compact_checkpoint = torch.load(compact_path, map_location="cpu", weights_only=False)
    late_checkpoint = torch.load(late_path, map_location="cpu", weights_only=False)

    compact_config = representation_config_from_checkpoint(
        compact_checkpoint["representation_config"]
    )
    compact_model = CompactWaveformCNN(
        compact_config.channel_count,
        width=int(compact_checkpoint["model_width"]),
    ).to(device)
    compact_model.load_state_dict(compact_checkpoint["model_state_dict"])
    compact_model.eval()

    late_model = O2LateFusion().to(device)
    late_model.load_state_dict(late_checkpoint["model_state_dict"])
    late_model.eval()
    return (
        compact_path,
        compact_checkpoint,
        compact_config,
        compact_model,
        late_path,
        late_checkpoint,
        late_model,
    )


def score_waveform_batch(
    waveforms: np.ndarray,
    shaped_energy: np.ndarray,
    compact_checkpoint: dict[str, Any],
    compact_config: RepresentationConfig,
    compact_model: torch.nn.Module,
    late_checkpoint: dict[str, Any],
    late_model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dummy = np.zeros(waveforms.shape[0], dtype=np.float32)
    raw = RawPartition(waveforms, shaped_energy, dummy, dummy, np.full(waveforms.shape[0], "th232"))
    compact_values, compact_qc = build_representation(raw, compact_config)
    apply_channel_statistics(compact_values, compact_checkpoint["channel_statistics"])

    charge, current, late_fallback = extract_o2_features(waveforms, shaped_energy)
    statistics = late_checkpoint["feature_statistics"]
    charge -= float(statistics["charge_mean"])
    charge /= float(statistics["charge_std"])
    current -= float(statistics["current_mean"])
    current /= float(statistics["current_std"])

    with torch.no_grad():
        compact_scores = torch.sigmoid(
            compact_model(torch.from_numpy(compact_values).to(device))
        ).cpu().numpy()
        late_scores = torch.sigmoid(
            late_model(
                torch.from_numpy(charge).to(device),
                torch.from_numpy(current).to(device),
            )
        ).cpu().numpy()
    return compact_scores, late_scores, {
        "compact_anchor_fallback_count": int(compact_qc["anchor_fallback_count"]),
        "compact_invalid_scale_count": int(compact_qc["invalid_scale_count"]),
        "late_fusion_t10_fallback_count": int(late_fallback),
    }


def score_th232(
    hdf5_files: list[Path],
    cache_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    (
        compact_path,
        compact_checkpoint,
        compact_config,
        compact_model,
        late_path,
        late_checkpoint,
        late_model,
    ) = load_models(device)
    energies: list[np.ndarray] = []
    compact_scores: list[np.ndarray] = []
    late_scores: list[np.ndarray] = []
    counts = {
        "input_events": 0,
        "admitted_events": 0,
        "rejected_nonfinite_or_outside_0_3200_kev": 0,
        "rejected_nonpositive_shaped_energy": 0,
        "compact_anchor_fallback_count": 0,
        "compact_invalid_scale_count": 0,
        "late_fusion_t10_fallback_count": 0,
    }
    for file_index, path in enumerate(hdf5_files, start=1):
        print(f"Scoring {file_index}/{len(hdf5_files)} {path.name}", flush=True)
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("processing_status")) != "OK":
                raise ValueError(f"Non-OK preprocessing status: {path}")
            event_count = int(handle["waveform"].shape[0])
            counts["input_events"] += event_count
            for start in range(0, event_count, batch_size):
                stop = min(start + batch_size, event_count)
                energy = np.asarray(handle["reconstructed_energy_kev"][start:stop], dtype=np.float32)
                bits = np.asarray(handle["qc_rejection_bits"][start:stop], dtype=np.uint16)
                shaped_all = np.asarray(handle["shaped_energy_unit"][start:stop], dtype=np.float32)
                energy_valid = (
                    np.isfinite(energy)
                    & (energy >= ENERGY_MIN_KEV)
                    & (energy < ENERGY_MAX_KEV)
                    & ((bits & np.uint16(0b111)) == 0)
                )
                shaped_valid = np.isfinite(shaped_all) & (shaped_all > 0.0)
                admitted = energy_valid & shaped_valid
                counts["rejected_nonfinite_or_outside_0_3200_kev"] += int(np.count_nonzero(~energy_valid))
                counts["rejected_nonpositive_shaped_energy"] += int(np.count_nonzero(energy_valid & ~shaped_valid))
                if not np.any(admitted):
                    continue
                waveform = np.asarray(handle["waveform"][start:stop][admitted], dtype=np.float32)
                shaped = shaped_all[admitted]
                batch_compact, batch_late, batch_qc = score_waveform_batch(
                    waveform,
                    shaped,
                    compact_checkpoint,
                    compact_config,
                    compact_model,
                    late_checkpoint,
                    late_model,
                    device,
                )
                energies.append(energy[admitted])
                compact_scores.append(batch_compact.astype(np.float32))
                late_scores.append(batch_late.astype(np.float32))
                counts["admitted_events"] += int(np.count_nonzero(admitted))
                for name, value in batch_qc.items():
                    counts[name] += value
    energy_array = np.concatenate(energies)
    compact_array = np.concatenate(compact_scores)
    late_array = np.concatenate(late_scores)
    np.savez_compressed(
        cache_path,
        energy_kev=energy_array,
        compact_score=compact_array,
        late_fusion_score=late_array,
    )
    return {
        "cache": cache_path.relative_to(PROJECT_ROOT).as_posix(),
        "cache_sha256": sha256_file(cache_path),
        "counts": counts,
        "compact_checkpoint": compact_path.relative_to(PROJECT_ROOT).as_posix(),
        "compact_checkpoint_sha256": sha256_file(compact_path),
        "late_fusion_checkpoint": late_path.relative_to(PROJECT_ROOT).as_posix(),
        "late_fusion_checkpoint_sha256": sha256_file(late_path),
    }


def gaussian_linear(x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float, slope: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * np.square((x - mean) / sigma)) + offset + slope * (x - mean)


def fit_peak_windows(no_cnn_histogram: np.ndarray) -> list[PeakWindow]:
    smoothed = gaussian_filter1d(no_cnn_histogram.astype(np.float64), 1.5)
    windows: list[PeakWindow] = []
    for reference in REFERENCE_PEAKS_KEV:
        predicted = 1.0635 * reference + 0.5
        search_half_width = 12.0 if reference < 1200.0 else 25.0
        search = (ENERGY_CENTERS >= predicted - search_half_width) & (ENERGY_CENTERS <= predicted + search_half_width)
        initial_mean = float(ENERGY_CENTERS[search][np.argmax(smoothed[search])])
        fit_half_width = 13.0 if reference < 1200.0 else 20.0
        selected = (ENERGY_CENTERS >= initial_mean - fit_half_width) & (ENERGY_CENTERS <= initial_mean + fit_half_width)
        x = ENERGY_CENTERS[selected]
        y = no_cnn_histogram[selected].astype(np.float64)
        edge_count = max(2, x.size // 5)
        background = float(np.median(np.concatenate((y[:edge_count], y[-edge_count:]))))
        initial = (max(float(np.max(y) - background), 1.0), initial_mean, 2.0, max(background, 0.0), 0.0)
        lower = (0.0, initial_mean - 4.0, 0.6, 0.0, -np.inf)
        upper = (np.inf, initial_mean + 4.0, 8.0, np.inf, np.inf)
        parameters, _ = curve_fit(gaussian_linear, x, y, p0=initial, bounds=(lower, upper), maxfev=20000)
        mean = float(parameters[1])
        sigma = float(parameters[2])
        windows.append(
            PeakWindow(
                reference_kev=reference,
                centroid_kev=mean,
                sigma_kev=sigma,
                roi_low_kev=mean - 2.0 * sigma,
                roi_high_kev=mean + 2.0 * sigma,
                left_low_kev=mean - 5.0 * sigma,
                left_high_kev=mean - 3.0 * sigma,
                right_low_kev=mean + 3.0 * sigma,
                right_high_kev=mean + 5.0 * sigma,
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


def peak_background_metrics(histogram: np.ndarray, window: PeakWindow) -> dict[str, float]:
    roi_counts, _ = interval_counts(histogram, window.roi_low_kev, window.roi_high_kev)
    left_counts, left_energy = interval_counts(histogram, window.left_low_kev, window.left_high_kev)
    right_counts, right_energy = interval_counts(histogram, window.right_low_kev, window.right_high_kev)
    left_center = left_energy / left_counts if left_counts > 0 else 0.5 * (window.left_low_kev + window.left_high_kev)
    right_center = right_energy / right_counts if right_counts > 0 else 0.5 * (window.right_low_kev + window.right_high_kev)
    roi_width = window.roi_high_kev - window.roi_low_kev
    left_width = window.left_high_kev - window.left_low_kev
    right_width = window.right_high_kev - window.right_low_kev
    left_density = left_counts / left_width
    right_density = right_counts / right_width
    fraction = (window.centroid_kev - left_center) / (right_center - left_center)
    background_density = left_density + fraction * (right_density - left_density)
    background_counts = background_density * roi_width
    net_peak_counts = roi_counts - background_counts
    return {
        "roi_counts": roi_counts,
        "estimated_background_counts": background_counts,
        "net_peak_counts": net_peak_counts,
        "peak_to_background": net_peak_counts / background_counts if background_counts > 0 else float("nan"),
    }


def make_histograms(
    energy: np.ndarray,
    compact_scores: np.ndarray,
    late_scores: np.ndarray,
    threshold_info: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    histograms: dict[str, dict[str, np.ndarray]] = {}
    for model, scores in (("compact_cnn", compact_scores), ("late_fusion", late_scores)):
        model_histograms = {"no_cnn": np.histogram(energy, ENERGY_EDGES)[0]}
        for name, values in threshold_info[model]["thresholds"].items():
            model_histograms[name] = np.histogram(
                energy[scores >= float(values["score_threshold"])], ENERGY_EDGES
            )[0]
        histograms[model] = model_histograms
    return histograms


def write_spectrum_csv(path: Path, histograms: dict[str, dict[str, np.ndarray]]) -> None:
    columns = ["energy_kev_bin_center", "no_cnn"]
    arrays: list[np.ndarray] = [ENERGY_CENTERS, histograms["compact_cnn"]["no_cnn"]]
    for model in ("compact_cnn", "late_fusion"):
        for acceptance in ACCEPTANCES:
            name = f"{int(acceptance * 100)}pct"
            columns.append(f"{model}_{name}")
            arrays.append(histograms[model][name])
    np.savetxt(path, np.column_stack(arrays), delimiter=",", header=",".join(columns), comments="", fmt=["%.1f"] + ["%d"] * 9)


def plot_spectra(
    output_dir: Path,
    histograms: dict[str, dict[str, np.ndarray]],
    windows: list[PeakWindow],
) -> None:
    colors = {"no_cnn": "black", "99pct": "#377eb8", "95pct": "#4daf4a", "90pct": "#ff7f00", "60pct": "#e41a1c"}
    labels = {"no_cnn": "No CNN", "99pct": "99% acceptance", "95pct": "95% acceptance", "90pct": "90% acceptance", "60pct": "60% acceptance"}
    for model, title in (("compact_cnn", "Compact CNN"), ("late_fusion", "Late-fusion CNN")):
        figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
        for name, histogram in histograms[model].items():
            for axis in axes:
                axis.step(ENERGY_CENTERS, histogram, where="mid", linewidth=0.9, color=colors[name], label=labels[name])
        axes[0].set_ylabel("Counts / 1 keV")
        axes[1].set_ylabel("Counts / 1 keV")
        axes[1].set_xlabel("Reconstructed energy (keV; provisional calibration)")
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=0.8)
        axes[0].set_title(f"Th-232 energy spectra after {title} score cuts")
        axes[0].legend(ncol=5, fontsize=9)
        for axis in axes:
            axis.grid(alpha=0.2)
            for window in windows:
                axis.axvline(window.centroid_kev, color="0.6", linewidth=0.5, alpha=0.5)
        figure.savefig(output_dir / f"th232_{model}_energy_spectra.png", dpi=180)
        plt.close(figure)

        figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for axis, window in zip(axes.flat, windows):
            half_width = max(18.0, 6.0 * window.sigma_kev)
            selected = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (ENERGY_CENTERS <= window.centroid_kev + half_width)
            for name, histogram in histograms[model].items():
                axis.step(ENERGY_CENTERS[selected], histogram[selected], where="mid", linewidth=0.9, color=colors[name], label=labels[name])
            axis.axvspan(window.roi_low_kev, window.roi_high_kev, color="#984ea3", alpha=0.10)
            axis.axvspan(window.left_low_kev, window.left_high_kev, color="0.5", alpha=0.08)
            axis.axvspan(window.right_low_kev, window.right_high_kev, color="0.5", alpha=0.08)
            axis.set_title(f"{window.reference_kev:g} keV ref.\nobserved {window.centroid_kev:.2f} keV")
            axis.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(f"Th-232 peak windows: {title}")
        figure.supxlabel("Reconstructed energy (keV)")
        figure.supylabel("Counts / 1 keV")
        figure.savefig(output_dir / f"th232_{model}_peak_zooms.png", dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/waveform_hdf5/th232_evaluation_20260813",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/th232_ba_na511_cnn_evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--reuse-score-cache", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse_score_cache:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_files = sorted(args.hdf5_dir.resolve().glob("*.h5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files under {args.hdf5_dir}")
    cache_path = output_dir / "th232_model_scores.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.reuse_score_cache:
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        scoring = {"cache": cache_path.relative_to(PROJECT_ROOT).as_posix(), "cache_sha256": sha256_file(cache_path), "reused": True}
    else:
        scoring = score_th232(hdf5_files, cache_path, args.batch_size, device)
        scoring["reused"] = False

    compact_validation = load_validation_thresholds(
        PROJECT_ROOT / "outputs/models/ba_all_na511_cnn/frozen_transfer/frozen_transfer_scores.npz"
    )
    late_validation = load_validation_thresholds(
        PROJECT_ROOT / "outputs/models/ba_all_na511_o2_late_fusion/frozen_transfer/frozen_transfer_scores.npz"
    )
    thresholds = {"compact_cnn": compact_validation, "late_fusion": late_validation}
    with np.load(cache_path) as values:
        energy = np.asarray(values["energy_kev"], dtype=np.float32)
        compact_scores = np.asarray(values["compact_score"], dtype=np.float32)
        late_scores = np.asarray(values["late_fusion_score"], dtype=np.float32)
    histograms = make_histograms(energy, compact_scores, late_scores, thresholds)
    windows = fit_peak_windows(histograms["compact_cnn"]["no_cnn"])

    results: dict[str, Any] = {
        "created_utc": utc_now(),
        "status": "EXTERNAL_EVALUATION",
        "protocol": {
            "training_domain": "Ba-133 all selected peaks plus Na-22 511-keV photopeak",
            "threshold_source": "held-out combined-domain validation positive scores",
            "threshold_weighting": "frozen per-peak source weights; equal total weight across five training peaks",
            "th232_used_for_model_or_threshold_selection": False,
            "test_partition_used": False,
            "event_admission": "finite reconstructed energy in [0, 3200) keV, positive finite shaped energy, and QC bits 0--2 clear; noise-bit observations retained",
            "peak_to_background_definition": "(counts in +/-2 sigma ROI - linearly interpolated sideband background) / sideband background; sidebands are 3--5 sigma on each side and have equal total width to ROI",
            "peak_windows_frozen_from": "no-CNN spectrum",
        },
        "inputs": {
            "hdf5_files": [path.relative_to(PROJECT_ROOT).as_posix() for path in hdf5_files],
            "hdf5_file_count": len(hdf5_files),
            "device": str(device),
            "scoring": scoring,
        },
        "thresholds": thresholds,
        "peaks": [],
        "global_retention": {},
        "outputs": {},
    }
    rows: list[dict[str, Any]] = []
    total_events = int(energy.size)
    for model, scores in (("compact_cnn", compact_scores), ("late_fusion", late_scores)):
        results["global_retention"][model] = {}
        for acceptance in ACCEPTANCES:
            name = f"{int(acceptance * 100)}pct"
            threshold = float(thresholds[model]["thresholds"][name]["score_threshold"])
            results["global_retention"][model][name] = float(np.mean(scores >= threshold))
    for window in windows:
        peak_result: dict[str, Any] = {
            "reference_energy_kev": window.reference_kev,
            "observed_centroid_kev": window.centroid_kev,
            "observed_to_reference_ratio": window.centroid_kev / window.reference_kev,
            "sigma_kev": window.sigma_kev,
            "fwhm_kev": 2.354820045 * window.sigma_kev,
            "windows_kev": {
                "roi": [window.roi_low_kev, window.roi_high_kev],
                "left_sideband": [window.left_low_kev, window.left_high_kev],
                "right_sideband": [window.right_low_kev, window.right_high_kev],
            },
            "conditions": {},
        }
        baseline = peak_background_metrics(histograms["compact_cnn"]["no_cnn"], window)
        peak_result["conditions"]["no_cnn"] = baseline
        rows.append({"reference_energy_kev": window.reference_kev, "observed_centroid_kev": window.centroid_kev, "model": "no_cnn", "acceptance": "none", **baseline})
        for model in ("compact_cnn", "late_fusion"):
            peak_result["conditions"][model] = {}
            for acceptance in ACCEPTANCES:
                name = f"{int(acceptance * 100)}pct"
                metrics = peak_background_metrics(histograms[model][name], window)
                metrics["pb_improvement_factor_vs_no_cnn"] = metrics["peak_to_background"] / baseline["peak_to_background"]
                metrics["th232_net_peak_retention_vs_no_cnn"] = metrics["net_peak_counts"] / baseline["net_peak_counts"]
                peak_result["conditions"][model][name] = metrics
                rows.append({"reference_energy_kev": window.reference_kev, "observed_centroid_kev": window.centroid_kev, "model": model, "acceptance": name, **metrics})
        results["peaks"].append(peak_result)

    spectrum_csv = output_dir / "th232_energy_spectra_1kev.csv"
    pb_csv = output_dir / "th232_peak_to_background.csv"
    summary_json = output_dir / "th232_peak_background_summary.json"
    write_spectrum_csv(spectrum_csv, histograms)
    with pb_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) + [
            name for name in rows[-1] if name not in rows[0]
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_spectra(output_dir, histograms, windows)
    results["outputs"] = {
        "summary_json": summary_json.relative_to(PROJECT_ROOT).as_posix(),
        "peak_to_background_csv": pb_csv.relative_to(PROJECT_ROOT).as_posix(),
        "spectrum_csv": spectrum_csv.relative_to(PROJECT_ROOT).as_posix(),
        "compact_spectrum_png": (output_dir / "th232_compact_cnn_energy_spectra.png").relative_to(PROJECT_ROOT).as_posix(),
        "compact_peak_zoom_png": (output_dir / "th232_compact_cnn_peak_zooms.png").relative_to(PROJECT_ROOT).as_posix(),
        "late_fusion_spectrum_png": (output_dir / "th232_late_fusion_energy_spectra.png").relative_to(PROJECT_ROOT).as_posix(),
        "late_fusion_peak_zoom_png": (output_dir / "th232_late_fusion_peak_zooms.png").relative_to(PROJECT_ROOT).as_posix(),
    }
    summary_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Admitted {total_events} Th-232 events", flush=True)
    print(f"Wrote {summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
