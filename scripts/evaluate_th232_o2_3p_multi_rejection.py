#!/usr/bin/env python3
"""Fit multiple Co-60 rejection curves and apply them to cached Th-232 scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_o2_3p_co60_threshold_curve import (  # noqa: E402
    bin_mask,
    closest_constant_pass_threshold,
)
from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    MODEL_NAME,
    fit_peak_windows,
    peak_background_metrics,
    relative,
    sha256_file,
    utc_now,
)

REJECTION_PERCENTAGES = (30, 50, 70, 90, 99)
CO60_EDGES = np.arange(100.0, 1050.0, 50.0)


def fit_threshold_points(energy: np.ndarray, threshold: np.ndarray) -> dict[str, Any]:
    return {
        "family": "shape_preserving_pchip",
        "knot_energy_kev": [float(value) for value in energy],
        "knot_threshold": [float(value) for value in threshold],
        "knot_rmse": 0.0,
        "selection_rule": (
            "shape-preserving interpolation selected from Co-60 only because "
            "high-rejection threshold points are non-monotone"
        ),
        "energy_domain_kev": [100.0, 1000.0],
        "outside_domain_policy": (
            "PCHIP extrapolates from 125-keV and 975-keV knots to the 100-keV "
            "and 1000-keV boundaries; threshold is constant beyond those boundaries"
        ),
    }


def evaluate_fit(energy: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    interpolator = PchipInterpolator(
        np.asarray(fit["knot_energy_kev"]),
        np.asarray(fit["knot_threshold"]),
        extrapolate=True,
    )
    bounded_energy = np.clip(np.asarray(energy, dtype=np.float64), 100.0, 1000.0)
    return np.clip(interpolator(bounded_energy), 0.0, 1.0)


def derive_target(energy, scores, rejection_percentage):
    target_pass = 1.0 - rejection_percentage / 100.0
    rows = []
    centers = 0.5 * (CO60_EDGES[:-1] + CO60_EDGES[1:])
    empirical = []
    for index, (low, high, center) in enumerate(
        zip(CO60_EDGES[:-1], CO60_EDGES[1:], centers)
    ):
        selected = bin_mask(energy, low, high, index == centers.size - 1)
        threshold, passed, passing_fraction = closest_constant_pass_threshold(
            scores[selected], target_pass
        )
        empirical.append(threshold)
        rows.append(
            {
                "rejection_target_percent": rejection_percentage,
                "target_passing_fraction": target_pass,
                "energy_low_kev": low,
                "energy_high_kev": high,
                "energy_center_kev": center,
                "upper_edge_inclusive": index == centers.size - 1,
                "empirical_threshold": threshold,
                "validation_event_count": int(np.count_nonzero(selected)),
                "empirical_passed_count": passed,
                "empirical_passing_fraction": passing_fraction,
            }
        )
    fit = fit_threshold_points(centers, np.asarray(empirical))
    for row in rows:
        row["fitted_threshold_at_center"] = float(
            evaluate_fit(np.asarray([row["energy_center_kev"]]), fit)[0]
        )
    return rows, fit


def make_pb_rows(histograms, windows, fits):
    rows = []
    for window in windows:
        baseline = peak_background_metrics(histograms["no_cut"], window)
        for condition, histogram in histograms.items():
            metrics = peak_background_metrics(histogram, window)
            fit = fits.get(condition)
            rows.append(
                {
                    "reference_energy_kev": window.reference_kev,
                    "observed_centroid_kev": window.centroid_kev,
                    "fwhm_kev": 2.354820045 * window.sigma_kev,
                    "threshold_fit_domain": 100.0 <= window.reference_kev <= 1000.0,
                    "condition": condition,
                    "rejection_target_percent": None if fit is None else fit["rejection_target_percent"],
                    "fitted_threshold_at_centroid": None if fit is None else float(
                        evaluate_fit(np.asarray([window.centroid_kev]), fit)[0]
                    ),
                    **metrics,
                    "pb_improvement_factor_vs_no_cut": metrics["peak_to_background"]
                    / baseline["peak_to_background"],
                    "net_peak_retention_vs_no_cut": metrics["net_peak_counts"]
                    / baseline["net_peak_counts"],
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir, threshold_rows, fits, histograms, pb_rows):
    dense = np.linspace(100.0, 1000.0, 1000)
    figure, axis = plt.subplots(figsize=(10, 6))
    for condition, fit in fits.items():
        target = fit["rejection_target_percent"]
        selected = [row for row in threshold_rows if row["rejection_target_percent"] == target]
        axis.scatter(
            [row["energy_center_kev"] for row in selected],
            [row["empirical_threshold"] for row in selected],
            s=16,
        )
        axis.plot(dense, evaluate_fit(dense, fit), label=f"{target}% rejection")
    axis.set(xlabel="Corrected energy (keV)", ylabel="O2-3P threshold")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "co60_fitted_threshold_curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for condition, histogram in histograms.items():
        label = "No cut" if condition == "no_cut" else condition.replace("rejection_", "").replace("pct", "% rejection")
        for axis in axes:
            axis.step(ENERGY_CENTERS, histogram, where="mid", linewidth=0.8, label=label)
    axes[0].set_ylabel("Counts / 1 keV")
    axes[0].legend(ncol=3)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set(xlabel="Corrected energy (keV)", ylabel="Counts / 1 keV")
    figure.tight_layout()
    figure.savefig(output_dir / "th232_multi_rejection_spectra.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for condition, fit in fits.items():
        selected = [row for row in pb_rows if row["condition"] == condition]
        axis.plot(
            [row["reference_energy_kev"] for row in selected],
            [row["pb_improvement_factor_vs_no_cut"] for row in selected],
            marker="o",
            label=f"{fit['rejection_target_percent']}% rejection",
        )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set(xlabel="Th-232 reference peak (keV)", ylabel="P/B improvement factor vs no cut")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "th232_peak_background_multi_rejection.png", dpi=180)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--co60-validation-scores",
        type=Path,
        default=PROJECT_ROOT / "outputs/co60_continuum_o2_3p_threshold_curve_20260819/validation_scores.h5",
    )
    parser.add_argument(
        "--th232-score-cache",
        type=Path,
        default=PROJECT_ROOT / "outputs/th232_o2_3p_energy_threshold_20260820/th232_o2_3p_scores.h5",
    )
    parser.add_argument(
        "--co60-train-scores",
        type=Path,
        default=PROJECT_ROOT / "outputs/co60_continuum_o2_3p_threshold_curve_20260819/train_scores.h5",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    co60_path = args.co60_validation_scores.resolve()
    with h5py.File(co60_path, "r") as source:
        if source.attrs.get("test_partition_used", False):
            raise ValueError("Co-60 score cache used the locked-test partition")
        co60_energy = np.asarray(source["corrected_energy_kev"], dtype=np.float32)
        co60_scores = np.asarray(source["score"], dtype=np.float32)
    co60_train_path = args.co60_train_scores.resolve()
    with h5py.File(co60_train_path, "r") as source:
        if source.attrs.get("test_partition_used", False):
            raise ValueError("Co-60 train score cache used the locked-test partition")
        co60_train_energy = np.asarray(source["corrected_energy_kev"], dtype=np.float32)
        co60_train_scores = np.asarray(source["score"], dtype=np.float32)

    threshold_rows = []
    fits = {}
    for rejection in REJECTION_PERCENTAGES:
        rows, fit = derive_target(co60_energy, co60_scores, rejection)
        condition = f"rejection_{rejection}pct"
        fit["rejection_target_percent"] = rejection
        fit["target_passing_fraction"] = 1.0 - rejection / 100.0
        fits[condition] = fit
        threshold_rows.extend(rows)

    th232_path = args.th232_score_cache.resolve()
    with h5py.File(th232_path, "r") as source:
        if source.attrs.get("test_partition_used", False):
            raise ValueError("Th-232 score cache used the locked-test partition")
        energy = np.asarray(source["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(source["score"], dtype=np.float32)

    masks = {"no_cut": np.ones(energy.size, dtype=bool)}
    for condition, fit in fits.items():
        masks[condition] = scores >= evaluate_fit(energy, fit)
    histograms = {
        condition: np.histogram(energy[mask], ENERGY_EDGES)[0]
        for condition, mask in masks.items()
    }
    windows = fit_peak_windows(histograms["no_cut"])
    pb_rows = make_pb_rows(histograms, windows, fits)

    for row in threshold_rows:
        condition = f"rejection_{row['rejection_target_percent']}pct"
        selected = bin_mask(
            co60_energy,
            row["energy_low_kev"],
            row["energy_high_kev"],
            row["upper_edge_inclusive"],
        )
        fitted_pass = co60_scores[selected] >= evaluate_fit(co60_energy[selected], fits[condition])
        row["fitted_passed_count"] = int(np.count_nonzero(fitted_pass))
        row["fitted_passing_fraction"] = float(np.mean(fitted_pass))
        train_selected = bin_mask(
            co60_train_energy,
            row["energy_low_kev"],
            row["energy_high_kev"],
            row["upper_edge_inclusive"],
        )
        train_pass = co60_train_scores[train_selected] >= evaluate_fit(
            co60_train_energy[train_selected], fits[condition]
        )
        row["train_event_count"] = int(np.count_nonzero(train_selected))
        row["train_fitted_passed_count"] = int(np.count_nonzero(train_pass))
        row["train_fitted_passing_fraction"] = float(np.mean(train_pass))

    write_csv(output_dir / "co60_threshold_curves_50kev.csv", threshold_rows)
    write_csv(output_dir / "th232_peak_to_background.csv", pb_rows)
    spectrum_values = np.column_stack((ENERGY_CENTERS, *histograms.values()))
    np.savetxt(
        output_dir / "th232_corrected_spectra_1kev.csv",
        spectrum_values,
        delimiter=",",
        header=",".join(["energy_kev_bin_center", *histograms]),
        comments="",
        fmt=["%.1f", *(["%d"] * len(histograms))],
    )
    save_plots(output_dir, threshold_rows, fits, histograms, pb_rows)

    in_domain = (energy >= 100.0) & (energy <= 1000.0)
    conditions = {
        condition: {
            "events": int(np.count_nonzero(mask)),
            "fraction": float(np.mean(mask)),
            "within_100_1000_fraction": float(np.mean(mask[in_domain])),
        }
        for condition, mask in masks.items()
    }
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "model_name": MODEL_NAME,
        "rejection_targets_percent": list(REJECTION_PERCENTAGES),
        "threshold_definition": "score >= threshold passes",
        "fit_selection": (
            "shape-preserving PCHIP through validation-derived 50-keV knots; "
            "selected using Co-60 only"
        ),
        "co60_validation_score_cache": {
            "path": relative(co60_path),
            "sha256": sha256_file(co60_path),
            "event_count": int(co60_energy.size),
        },
        "co60_train_score_cache": {
            "path": relative(co60_train_path),
            "sha256": sha256_file(co60_train_path),
            "event_count": int(co60_train_energy.size),
            "threshold_selection_used": False,
        },
        "th232_score_cache": {
            "path": relative(th232_path),
            "sha256": sha256_file(th232_path),
            "event_count": int(energy.size),
            "rescored": False,
        },
        "fits": fits,
        "co60_threshold_rows": threshold_rows,
        "conditions": conditions,
        "peak_windows": [asdict(window) for window in windows],
        "peak_background_rows": pb_rows,
        "artifacts": {
            name: {
                "path": relative(output_dir / name),
                "sha256": sha256_file(output_dir / name),
            }
            for name in (
                "co60_threshold_curves_50kev.csv",
                "th232_peak_to_background.csv",
                "th232_corrected_spectra_1kev.csv",
                "co60_fitted_threshold_curves.png",
                "th232_multi_rejection_spectra.png",
                "th232_peak_background_multi_rejection.png",
            )
        },
        "test_partition_used": False,
        "scientific_boundary": (
            "Co-60 development validation sets thresholds. Historical corrected Th-232 is confirmation material, "
            "not a newly untouched campaign. Thresholds outside 100-1000 keV are clamped to the nearest fit edge."
        ),
    }
    report_path = output_dir / "th232_o2_3p_multi_rejection_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
