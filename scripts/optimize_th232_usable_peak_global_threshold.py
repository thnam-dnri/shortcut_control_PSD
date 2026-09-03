#!/usr/bin/env python3
"""Re-optimize one global DS-CNN threshold on audited usable Th-232 peaks.

The historical Th-232 score cache is an explicitly authorized deployment-
threshold optimization dataset, not external validation.  This scan excludes
the contaminated 338-keV region, the unresolved 969-keV region, the 2614-keV
line, and its single- and double-escape regions from the P/B objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    PeakWindow,
)
from scripts.plot_th232_revised_pb_threshold_curves import (  # noqa: E402
    PB_ANCHORS,
    fit_anchor_window,
    line_aware_metrics,
)

DEFAULT_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_usable_peak_global_threshold_20260824"
)
SCORE_BIN_COUNT = 2000
RETENTION_FLOORS = (0.90, 0.80, 0.70, 0.50, 0.30, 0.10, 0.00)
MINIMUM_BACKGROUND_COUNTS = 100.0
MINIMUM_NET_PEAK_COUNTS = 1000.0
OLD_GLOBAL_THRESHOLD = 0.4370
SENSITIVITY_EXCLUDED_REFERENCE_KEV = 209.253
NO_BRAINER_POOLED_RETENTION_FLOOR = 0.99


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_windows(baseline_histogram: np.ndarray) -> list[tuple[Any, PeakWindow]]:
    smoothed = gaussian_filter1d(baseline_histogram.astype(np.float64), 1.5)
    return [
        (spec, fit_anchor_window(baseline_histogram, spec.reference_kev, smoothed))
        for spec in PB_ANCHORS
    ]


def reliable_line_metrics(
    histogram: np.ndarray,
    window: PeakWindow,
    sideband_mode: str,
) -> dict[str, Any]:
    metrics = line_aware_metrics(histogram, window, sideband_mode)
    metrics["statistically_reliable"] = bool(
        metrics["estimated_background_counts"] >= MINIMUM_BACKGROUND_COUNTS
        and metrics["net_peak_counts"] >= MINIMUM_NET_PEAK_COUNTS
        and np.isfinite(metrics["peak_to_background"])
        and metrics["peak_to_background"] > 0.0
    )
    return metrics


def geometric_mean(values: list[float], reliable: bool) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not reliable or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        return float("nan")
    return float(np.exp(np.mean(np.log(array))))


def evaluate_histogram(
    histogram: np.ndarray,
    baseline: dict[float, dict[str, Any]],
    windows: list[tuple[Any, PeakWindow]],
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, Any]] = {}
    primary_gains: list[float] = []
    primary_retentions: list[float] = []
    sensitivity_gains: list[float] = []
    sensitivity_retentions: list[float] = []
    primary_reliable = True
    sensitivity_reliable = True
    pooled_baseline_net_counts = 0.0
    pooled_filtered_net_counts = 0.0
    for spec, window in windows:
        metrics = reliable_line_metrics(histogram, window, spec.sideband_mode)
        base = baseline[spec.reference_kev]
        gain = float(metrics["peak_to_background"] / base["peak_to_background"])
        retention = float(metrics["net_peak_counts"] / base["net_peak_counts"])
        per_peak[f"{spec.reference_kev:.3f}"] = {
            **metrics,
            "label": spec.label,
            "sideband_mode": spec.sideband_mode,
            "pb_improvement_factor_vs_no_cut": gain,
            "net_peak_retention_vs_no_cut": retention,
        }
        pooled_baseline_net_counts += float(base["net_peak_counts"])
        pooled_filtered_net_counts += float(metrics["net_peak_counts"])
        primary_gains.append(gain)
        primary_retentions.append(retention)
        primary_reliable = primary_reliable and bool(metrics["statistically_reliable"])
        if spec.reference_kev != SENSITIVITY_EXCLUDED_REFERENCE_KEV:
            sensitivity_gains.append(gain)
            sensitivity_retentions.append(retention)
            sensitivity_reliable = sensitivity_reliable and bool(
                metrics["statistically_reliable"]
            )
    return {
        "geometric_mean_pb_improvement": geometric_mean(
            primary_gains, primary_reliable
        ),
        "minimum_peak_retention": float(np.min(primary_retentions)),
        "mean_peak_retention": float(np.mean(primary_retentions)),
        "pooled_net_peak_retention": float(
            pooled_filtered_net_counts / pooled_baseline_net_counts
        ),
        "all_peak_statistics_reliable": primary_reliable,
        "sensitivity_geometric_mean_pb_improvement_excluding_209": geometric_mean(
            sensitivity_gains, sensitivity_reliable
        ),
        "sensitivity_minimum_peak_retention_excluding_209": float(
            np.min(sensitivity_retentions)
        ),
        "sensitivity_mean_peak_retention_excluding_209": float(
            np.mean(sensitivity_retentions)
        ),
        "sensitivity_statistics_reliable_excluding_209": sensitivity_reliable,
        "per_peak": per_peak,
    }


def evaluate_threshold_grid(
    energy: np.ndarray,
    scores: np.ndarray,
    windows: list[tuple[Any, PeakWindow]],
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    score_edges = np.linspace(0.0, 1.0, SCORE_BIN_COUNT + 1, dtype=np.float64)
    histogram_2d, _, _ = np.histogram2d(
        energy, scores, bins=(ENERGY_EDGES, score_edges)
    )
    cumulative = np.cumsum(histogram_2d[:, ::-1], axis=1)[:, ::-1]
    baseline_histogram = np.histogram(energy, ENERGY_EDGES)[0].astype(np.float64)
    baseline = {
        spec.reference_kev: reliable_line_metrics(
            baseline_histogram, window, spec.sideband_mode
        )
        for spec, window in windows
    }
    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(score_edges[:-1]):
        histogram = cumulative[:, index]
        metrics = evaluate_histogram(histogram, baseline, windows)
        rows.append(
            {
                "threshold": float(threshold),
                "selected_events": int(np.sum(histogram)),
                "selected_fraction": float(np.sum(histogram) / energy.size),
                **metrics,
            }
        )
    return rows, baseline


def select_operating_points(
    rows: list[dict[str, Any]],
    objective_key: str = "geometric_mean_pb_improvement",
    retention_key: str = "minimum_peak_retention",
    reliability_key: str = "all_peak_statistics_reliable",
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for floor in RETENTION_FLOORS:
        eligible = [
            row
            for row in rows
            if row[reliability_key]
            and np.isfinite(row[objective_key])
            and row[retention_key] >= floor
        ]
        if eligible:
            selected[f"minimum_retention_{int(round(100 * floor)):02d}pct"] = max(
                eligible, key=lambda row: row[objective_key]
            )
    return selected


def select_pareto_knee(
    rows: list[dict[str, Any]],
    objective_key: str = "geometric_mean_pb_improvement",
    retention_key: str = "minimum_peak_retention",
    reliability_key: str = "all_peak_statistics_reliable",
) -> dict[str, Any]:
    """Select the knee before the maximum-P/B endpoint without a retention floor.

    Both axes are normalized to their attainable range from the no-cut point to
    the maximum reliable P/B point.  The knee maximizes the fraction of P/B gain
    obtained minus the fraction of minimum-retention loss incurred.
    """
    reliable = [
        row
        for row in rows
        if row[reliability_key]
        and np.isfinite(row[objective_key])
        and np.isfinite(row[retention_key])
    ]
    if not reliable:
        raise ValueError("No reliable rows are available for Pareto-knee selection")
    maximum = max(reliable, key=lambda row: row[objective_key])
    candidates = [row for row in reliable if row["threshold"] <= maximum["threshold"]]
    start = candidates[0]
    gain_span = float(maximum[objective_key] - start[objective_key])
    loss_span = float(start[retention_key] - maximum[retention_key])
    if gain_span <= 0.0 or loss_span <= 0.0:
        raise ValueError("Degenerate P/B-retention Pareto range")
    for row in candidates:
        gain_fraction = float((row[objective_key] - start[objective_key]) / gain_span)
        retention_loss_fraction = float(
            (start[retention_key] - row[retention_key]) / loss_span
        )
        row[f"{objective_key}_normalized_gain_fraction"] = gain_fraction
        row[f"{retention_key}_normalized_loss_fraction"] = retention_loss_fraction
        row[f"{objective_key}_pareto_knee_score"] = gain_fraction - retention_loss_fraction
    return max(
        candidates,
        key=lambda row: (row[f"{objective_key}_pareto_knee_score"], -row["threshold"]),
    )


def select_no_brainer_point(
    rows: list[dict[str, Any]],
    pooled_retention_floor: float = NO_BRAINER_POOLED_RETENTION_FLOOR,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["all_peak_statistics_reliable"]
        and np.isfinite(row["geometric_mean_pb_improvement"])
        and row["pooled_net_peak_retention"] >= pooled_retention_floor
    ]
    if not eligible:
        raise ValueError("No reliable threshold satisfies the pooled-retention floor")
    return max(eligible, key=lambda row: row["geometric_mean_pb_improvement"])


def exact_metrics_at_threshold(
    energy: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    baseline: dict[float, dict[str, Any]],
    windows: list[tuple[Any, PeakWindow]],
) -> dict[str, Any]:
    histogram = np.histogram(energy[scores >= threshold], ENERGY_EDGES)[0]
    return {
        "threshold": float(threshold),
        "selected_events": int(np.count_nonzero(scores >= threshold)),
        "selected_fraction": float(np.mean(scores >= threshold)),
        **evaluate_histogram(histogram, baseline, windows),
    }


def write_scan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "threshold",
        "selected_events",
        "selected_fraction",
        "geometric_mean_pb_improvement",
        "minimum_peak_retention",
        "mean_peak_retention",
        "pooled_net_peak_retention",
        "all_peak_statistics_reliable",
        "sensitivity_geometric_mean_pb_improvement_excluding_209",
        "sensitivity_minimum_peak_retention_excluding_209",
        "sensitivity_mean_peak_retention_excluding_209",
        "sensitivity_statistics_reliable_excluding_209",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_operating_points_csv(
    path: Path,
    primary: dict[str, dict[str, Any]],
    sensitivity: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "anchor_set",
        "operating_point",
        "threshold",
        "selected_fraction",
        "geometric_mean_pb_improvement",
        "minimum_peak_retention",
        "mean_peak_retention",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for anchor_set, selected, objective, minimum, mean in (
            (
                "all_10_usable_peaks",
                primary,
                "geometric_mean_pb_improvement",
                "minimum_peak_retention",
                "mean_peak_retention",
            ),
            (
                "sensitivity_excluding_209_kev",
                sensitivity,
                "sensitivity_geometric_mean_pb_improvement_excluding_209",
                "sensitivity_minimum_peak_retention_excluding_209",
                "sensitivity_mean_peak_retention_excluding_209",
            ),
        ):
            for name, row in selected.items():
                writer.writerow(
                    {
                        "anchor_set": anchor_set,
                        "operating_point": name,
                        "threshold": row["threshold"],
                        "selected_fraction": row["selected_fraction"],
                        "geometric_mean_pb_improvement": row[objective],
                        "minimum_peak_retention": row[minimum],
                        "mean_peak_retention": row[mean],
                    }
                )


def write_per_peak_csv(
    path: Path,
    operating_points: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "operating_point",
        "threshold",
        "reference_energy_kev",
        "label",
        "sideband_mode",
        "base_peak_to_background",
        "filtered_peak_to_background",
        "pb_improvement_factor_vs_no_cut",
        "net_peak_retention_vs_no_cut",
        "statistically_reliable",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, point in operating_points.items():
            for reference, row in point["per_peak"].items():
                writer.writerow(
                    {
                        "operating_point": name,
                        "threshold": point["threshold"],
                        "reference_energy_kev": reference,
                        "label": row["label"],
                        "sideband_mode": row["sideband_mode"],
                        "base_peak_to_background": row["peak_to_background"]
                        / row["pb_improvement_factor_vs_no_cut"],
                        "filtered_peak_to_background": row["peak_to_background"],
                        "pb_improvement_factor_vs_no_cut": row[
                            "pb_improvement_factor_vs_no_cut"
                        ],
                        "net_peak_retention_vs_no_cut": row[
                            "net_peak_retention_vs_no_cut"
                        ],
                        "statistically_reliable": row["statistically_reliable"],
                    }
                )


def write_spectrum_csv(
    path: Path,
    energy: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    no_cut = np.histogram(energy, ENERGY_EDGES)[0]
    selected = np.histogram(energy[scores >= threshold], ENERGY_EDGES)[0]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "energy_kev",
                "no_cut_counts_per_1kev",
                "score_ge_threshold_counts_per_1kev",
            ),
        )
        writer.writeheader()
        for energy_kev, base_count, selected_count in zip(
            ENERGY_CENTERS, no_cut, selected
        ):
            writer.writerow(
                {
                    "energy_kev": float(energy_kev),
                    "no_cut_counts_per_1kev": int(base_count),
                    "score_ge_threshold_counts_per_1kev": int(selected_count),
                }
            )


def write_fpga_presets_csv(
    path: Path,
    sweet_spot: dict[str, Any],
    no_brainer: dict[str, Any],
) -> None:
    fields = (
        "preset",
        "threshold",
        "selection_rule",
        "user_adjustable",
        "geometric_mean_pb_improvement",
        "pooled_net_peak_retention",
        "minimum_peak_retention",
        "mean_peak_retention",
        "selected_event_fraction",
    )
    rows = (
        (
            "sweet_spot",
            sweet_spot,
            "normalized Pareto knee of P/B gain versus worst-peak retention loss",
        ),
        (
            "no_brainer_conservative",
            no_brainer,
            "highest P/B with pooled usable-peak net retention >=99%",
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for preset, point, selection_rule in rows:
            writer.writerow(
                {
                    "preset": preset,
                    "threshold": point["threshold"],
                    "selection_rule": selection_rule,
                    "user_adjustable": True,
                    "geometric_mean_pb_improvement": point[
                        "geometric_mean_pb_improvement"
                    ],
                    "pooled_net_peak_retention": point[
                        "pooled_net_peak_retention"
                    ],
                    "minimum_peak_retention": point["minimum_peak_retention"],
                    "mean_peak_retention": point["mean_peak_retention"],
                    "selected_event_fraction": point["selected_fraction"],
                }
            )


def plot_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    energy: np.ndarray,
    scores: np.ndarray,
    windows: list[tuple[Any, PeakWindow]],
    recommended: dict[str, Any],
    old_global: dict[str, Any],
    maximum_pb: dict[str, Any],
    no_brainer: dict[str, Any],
) -> None:
    thresholds = np.asarray([row["threshold"] for row in rows])
    objective = np.asarray(
        [row["geometric_mean_pb_improvement"] for row in rows]
    )
    minimum_retention = np.asarray([row["minimum_peak_retention"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(thresholds, objective, linewidth=1.3)
    axes[0].set_ylabel("Geometric-mean P/B improvement (10 usable peaks)")
    axes[1].plot(thresholds, minimum_retention, linewidth=1.3)
    axes[1].set_ylabel("Minimum usable-peak net retention")
    for axis in axes:
        axis.axvline(
            old_global["threshold"], color="tab:orange", linestyle="--", label="old 0.4370"
        )
        axis.axvline(
            recommended["threshold"], color="tab:blue", linestyle="--", label="Pareto knee"
        )
        axis.axvline(
            maximum_pb["threshold"], color="tab:red", linestyle=":", label="maximum P/B"
        )
        axis.axvline(
            no_brainer["threshold"],
            color="tab:green",
            linestyle="-.",
            label="no-brainer preset",
        )
        axis.set_xlabel("Global DS-CNN score threshold")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Th-232 global threshold re-optimization on audited usable peaks")
    figure.savefig(output_dir / "threshold_optimization.png", dpi=180)
    plt.close(figure)

    histograms = {
        "No cut": np.histogram(energy, ENERGY_EDGES)[0],
        f"Old 0.4370": np.histogram(
            energy[scores >= old_global["threshold"]], ENERGY_EDGES
        )[0],
        f"Re-optimized {recommended['threshold']:.4f}": np.histogram(
            energy[scores >= recommended["threshold"]], ENERGY_EDGES
        )[0],
    }
    figure, axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)
    for axis, (spec, window) in zip(axes.flat, windows):
        low = window.left_low_kev - window.sigma_kev
        high = window.right_high_kev + window.sigma_kev
        mask = (ENERGY_CENTERS >= low) & (ENERGY_CENTERS <= high)
        for label, histogram in histograms.items():
            axis.step(
                ENERGY_CENTERS[mask], histogram[mask], where="mid", linewidth=0.9, label=label
            )
        no_cut_pb = line_aware_metrics(
            histograms["No cut"], window, spec.sideband_mode
        )["peak_to_background"]
        old_pb = line_aware_metrics(
            histograms["Old 0.4370"], window, spec.sideband_mode
        )["peak_to_background"]
        selected_pb = line_aware_metrics(
            histograms[f"Re-optimized {recommended['threshold']:.4f}"],
            window,
            spec.sideband_mode,
        )["peak_to_background"]
        axis.text(
            0.02,
            0.97,
            "\n".join(
                (
                    f"P/B no cut: {no_cut_pb:.3f}",
                    f"P/B 0.4370: {old_pb:.3f}",
                    f"P/B {recommended['threshold']:.4f}: {selected_pb:.3f}",
                    f"sweet-spot gain: {selected_pb / no_cut_pb:.3f}x",
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        )
        axis.axvspan(window.roi_low_kev, window.roi_high_kev, color="#CC79A7", alpha=0.10)
        axis.axvspan(window.left_low_kev, window.left_high_kev, color="0.5", alpha=0.08)
        axis.axvspan(window.right_low_kev, window.right_high_kev, color="0.5", alpha=0.08)
        axis.set_title(f"{spec.label} ({spec.sideband_mode})")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle("Audited usable Th-232 peak windows: old versus re-optimized global cut")
    figure.savefig(output_dir / "usable_peak_zooms.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)
    for axis, (spec, window) in zip(axes.flat, windows):
        low = window.left_low_kev - window.sigma_kev
        high = window.right_high_kev + window.sigma_kev
        mask = (ENERGY_CENTERS >= low) & (ENERGY_CENTERS <= high)
        local_metrics = {
            label: line_aware_metrics(histogram, window, spec.sideband_mode)
            for label, histogram in histograms.items()
        }
        reference_net_peak = local_metrics["No cut"]["net_peak_counts"]
        for label, histogram in histograms.items():
            net_peak = local_metrics[label]["net_peak_counts"]
            scale = reference_net_peak / net_peak
            axis.step(
                ENERGY_CENTERS[mask],
                histogram[mask] * scale,
                where="mid",
                linewidth=0.9,
                label=label,
            )
        no_cut_pb = local_metrics["No cut"]["peak_to_background"]
        old_pb = local_metrics["Old 0.4370"]["peak_to_background"]
        selected_pb = local_metrics[
            f"Re-optimized {recommended['threshold']:.4f}"
        ]["peak_to_background"]
        axis.text(
            0.02,
            0.97,
            "\n".join(
                (
                    "equal net-peak area",
                    f"P/B no cut: {no_cut_pb:.3f}",
                    f"P/B 0.4370: {old_pb:.3f}",
                    f"P/B {recommended['threshold']:.4f}: {selected_pb:.3f}",
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        )
        axis.axvspan(
            window.roi_low_kev,
            window.roi_high_kev,
            color="#CC79A7",
            alpha=0.10,
        )
        axis.axvspan(
            window.left_low_kev,
            window.left_high_kev,
            color="0.5",
            alpha=0.08,
        )
        axis.axvspan(
            window.right_low_kev,
            window.right_high_kev,
            color="0.5",
            alpha=0.08,
        )
        axis.set_title(f"{spec.label} ({spec.sideband_mode})")
        axis.set_ylabel("Counts / 1 keV (equal net-peak area)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle(
        "Audited usable Th-232 peaks normalized to equal net-peak area"
    )
    figure.savefig(output_dir / "usable_peak_zooms_normalized.png", dpi=180)
    plt.close(figure)

    no_cut = histograms["No cut"]
    selected_label = f"Re-optimized {recommended['threshold']:.4f}"
    selected = histograms[selected_label]
    figure, axes = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True, constrained_layout=True
    )
    for axis in axes:
        axis.step(
            ENERGY_CENTERS,
            no_cut,
            where="mid",
            color="0.25",
            linewidth=0.75,
            label="No score cut",
        )
        axis.step(
            ENERGY_CENTERS,
            selected,
            where="mid",
            color="tab:blue",
            linewidth=0.85,
            label=f"DS-CNN score >= {recommended['threshold']:.4f}",
        )
        for spec, _ in windows:
            axis.axvline(spec.reference_kev, color="tab:green", alpha=0.16, linewidth=0.7)
        axis.set_ylabel("Counts / 1 keV")
        axis.grid(alpha=0.18)
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Linear scale; retained {recommended['selected_fraction']:.2%} of admitted events"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set_title("Log scale")
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[1].set_xlim(float(ENERGY_EDGES[0]), float(ENERGY_EDGES[-1]))
    figure.suptitle(
        "Th-232 energy spectrum with the usable-peak Pareto-knee threshold"
    )
    threshold_tag = f"{recommended['threshold']:.4f}".replace(".", "p")
    figure.savefig(output_dir / f"th232_spectrum_threshold_{threshold_tag}.png", dpi=180)
    plt.close(figure)


def report_markdown(
    recommended: dict[str, Any],
    sensitivity_recommended: dict[str, Any],
    old_global: dict[str, Any],
    maximum_pb: dict[str, Any],
    no_brainer: dict[str, Any],
) -> str:
    lines = [
        "# Th-232 usable-peak global-threshold re-optimization",
        "",
        "## Decision",
        "",
        "- The previous `0.4370` threshold is superseded because its objective used the unreliable 338.3-keV region and unsupported 2614.5-keV P/B region.",
        "- The new objective uses ten audited P/B anchors from 209.3 through 1460.8 keV.",
        "- The 338.3- and 969-keV regions, the 2614-keV single- and double-escape regions, and the 2614.5-keV full-energy line are excluded from selection.",
        "- The 583.2- and 911.2-keV anchors use only their higher-energy sideband.",
        f"- Re-optimized sweet-spot global threshold: `{recommended['threshold']:.4f}`.",
        "- No fixed retention floor is used. The threshold is the normalized Pareto knee between geometric-mean P/B gain and worst-peak retention loss.",
        f"- Geometric-mean P/B improvement over the ten usable peaks: `{recommended['geometric_mean_pb_improvement']:.4f}x`.",
        f"- Minimum/mean usable-peak net retention: `{recommended['minimum_peak_retention']:.2%}` / `{recommended['mean_peak_retention']:.2%}`.",
        f"- Total admitted-event retention: `{recommended['selected_fraction']:.2%}`.",
        "",
        "## FPGA advisory presets",
        "",
        "The deployed comparator threshold remains user-adjustable; these are recommended starting presets, not hard-coded limits.",
        "",
        "| Preset | Threshold | Geometric-mean P/B | Pooled peak retention | Worst-peak retention | Total event retention |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| Sweet spot | {recommended['threshold']:.4f} | {recommended['geometric_mean_pb_improvement']:.4f}x | {recommended['pooled_net_peak_retention']:.2%} | {recommended['minimum_peak_retention']:.2%} | {recommended['selected_fraction']:.2%} |",
        f"| No-brainer conservative | {no_brainer['threshold']:.4f} | {no_brainer['geometric_mean_pb_improvement']:.4f}x | {no_brainer['pooled_net_peak_retention']:.2%} | {no_brainer['minimum_peak_retention']:.2%} | {no_brainer['selected_fraction']:.2%} |",
        "",
        "The no-brainer preset is the highest-P/B threshold with pooled net counts across the ten usable peaks retained at >=99%. This is a pooled criterion, not a claim that every individual peak exceeds 99% retention.",
        "Machine-readable preset metadata is in `fpga_recommended_threshold_presets.csv`.",
        "",
        "## Stability against the weak 209-keV anchor",
        "",
        f"Excluding 209.3 keV and reselecting the Pareto knee gives `{sensitivity_recommended['threshold']:.4f}`. "
        f"Its nine-peak geometric-mean P/B improvement is `{sensitivity_recommended['sensitivity_geometric_mean_pb_improvement_excluding_209']:.4f}x` with minimum retention `{sensitivity_recommended['sensitivity_minimum_peak_retention_excluding_209']:.2%}`.",
        "",
        "## Comparison with the superseded threshold",
        "",
        "| Operating point | Threshold | Geometric-mean P/B | Minimum retention | Mean retention | Total event retention |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| Superseded global | {old_global['threshold']:.4f} | {old_global['geometric_mean_pb_improvement']:.4f}x | {old_global['minimum_peak_retention']:.2%} | {old_global['mean_peak_retention']:.2%} | {old_global['selected_fraction']:.2%} |",
        f"| Re-optimized global | {recommended['threshold']:.4f} | {recommended['geometric_mean_pb_improvement']:.4f}x | {recommended['minimum_peak_retention']:.2%} | {recommended['mean_peak_retention']:.2%} | {recommended['selected_fraction']:.2%} |",
        f"| Maximum reliable P/B | {maximum_pb['threshold']:.4f} | {maximum_pb['geometric_mean_pb_improvement']:.4f}x | {maximum_pb['minimum_peak_retention']:.2%} | {maximum_pb['mean_peak_retention']:.2%} | {maximum_pb['selected_fraction']:.2%} |",
        "",
        "## Re-optimized per-peak result",
        "",
        "| Energy (keV) | Anchor | Sideband | P/B improvement | Net retention |",
        "|---:|:---|:---|---:|---:|",
    ]
    for reference, row in recommended["per_peak"].items():
        lines.append(
            f"| {float(reference):.3f} | {row['label']} | {row['sideband_mode']} | "
            f"{row['pb_improvement_factor_vs_no_cut']:.4f}x | {row['net_peak_retention_vs_no_cut']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Full-spectrum artifacts",
            "",
            f"- Linear/log comparison: `th232_spectrum_threshold_{recommended['threshold']:.4f}` with the decimal point encoded as `p` in the filename.",
            f"- Reproducible 1-keV no-cut and selected counts use the same threshold-coded filename for `{recommended['threshold']:.4f}`.",
            "- Equal-net-peak-area P/B comparison: `usable_peak_zooms_normalized.png`.",
            "",
            "## Claim boundary",
            "",
            "The checkpoint recorded in the input score cache is unchanged during threshold selection. Historical Th-232 events are used directly to select this deployment threshold, so this is in-sample threshold optimization, not external validation. Locked test and Eu-152 remain unopened and unused.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    score_cache = args.score_cache.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(score_cache, "r") as handle:
        if bool(handle.attrs.get("test_partition_used", True)):
            raise ValueError("Score cache is marked as test-contaminated")
        energy = np.asarray(handle["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(handle["score"], dtype=np.float32)
        checkpoint_sha256 = str(handle.attrs["checkpoint_sha256"])
    if energy.size != scores.size or not np.all(np.isfinite(energy)) or not np.all(
        np.isfinite(scores)
    ):
        raise ValueError("Invalid Th-232 score cache arrays")

    baseline_histogram = np.histogram(energy, ENERGY_EDGES)[0].astype(np.float64)
    windows = build_windows(baseline_histogram)
    rows, baseline = evaluate_threshold_grid(energy, scores, windows)
    primary = select_operating_points(rows)
    sensitivity = select_operating_points(
        rows,
        objective_key="sensitivity_geometric_mean_pb_improvement_excluding_209",
        retention_key="sensitivity_minimum_peak_retention_excluding_209",
        reliability_key="sensitivity_statistics_reliable_excluding_209",
    )
    recommended_grid = select_pareto_knee(rows)
    sensitivity_grid = select_pareto_knee(
        rows,
        objective_key="sensitivity_geometric_mean_pb_improvement_excluding_209",
        retention_key="sensitivity_minimum_peak_retention_excluding_209",
        reliability_key="sensitivity_statistics_reliable_excluding_209",
    )
    maximum_grid = primary["minimum_retention_00pct"]
    no_brainer_grid = select_no_brainer_point(rows)
    recommended = exact_metrics_at_threshold(
        energy, scores, recommended_grid["threshold"], baseline, windows
    )
    sensitivity_recommended = exact_metrics_at_threshold(
        energy, scores, sensitivity_grid["threshold"], baseline, windows
    )
    old_global = exact_metrics_at_threshold(
        energy, scores, OLD_GLOBAL_THRESHOLD, baseline, windows
    )
    maximum_pb = exact_metrics_at_threshold(
        energy, scores, maximum_grid["threshold"], baseline, windows
    )
    no_brainer = exact_metrics_at_threshold(
        energy, scores, no_brainer_grid["threshold"], baseline, windows
    )

    scan_csv = output_dir / "threshold_scan.csv"
    operating_csv = output_dir / "operating_points.csv"
    per_peak_csv = output_dir / "per_peak_metrics.csv"
    threshold_tag = f"{recommended['threshold']:.4f}".replace(".", "p")
    spectrum_csv = output_dir / f"th232_spectrum_threshold_{threshold_tag}_1kev.csv"
    fpga_presets_csv = output_dir / "fpga_recommended_threshold_presets.csv"
    report_md = output_dir / "report.md"
    write_scan_csv(scan_csv, rows)
    write_operating_points_csv(operating_csv, primary, sensitivity)
    write_per_peak_csv(
        per_peak_csv,
        {
            "superseded_global_0p437": old_global,
            "reoptimized_global": recommended,
            "sensitivity_excluding_209": sensitivity_recommended,
            "maximum_reliable_pb": maximum_pb,
            "no_brainer_conservative": no_brainer,
        },
    )
    write_spectrum_csv(spectrum_csv, energy, scores, recommended["threshold"])
    write_fpga_presets_csv(fpga_presets_csv, recommended, no_brainer)
    plot_results(
        output_dir,
        rows,
        energy,
        scores,
        windows,
        recommended,
        old_global,
        maximum_pb,
        no_brainer,
    )
    report_md.write_text(
        report_markdown(
            recommended,
            sensitivity_recommended,
            old_global,
            maximum_pb,
            no_brainer,
        ),
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "TH232_USABLE_PEAK_GLOBAL_THRESHOLD_COMPLETE",
        "score_cache": relative(score_cache),
        "score_cache_sha256": sha256_file(score_cache),
        "checkpoint_sha256": checkpoint_sha256,
        "admitted_events": int(energy.size),
        "optimization": {
            "threshold_kind": "one global DS-CNN score threshold",
            "score_bin_count": SCORE_BIN_COUNT,
            "objective": "normalized Pareto knee between geometric mean P/B improvement and minimum usable-peak retention",
            "retention_constraint": None,
            "sweet_spot_definition": (
                "From no cut through the maximum reliable P/B threshold, normalize "
                "attained P/B gain and minimum-retention loss to [0,1]; select the "
                "threshold maximizing gain_fraction - retention_loss_fraction."
            ),
            "fpga_threshold_control": {
                "user_adjustable": True,
                "recommendations_are_advisory": True,
                "score_range": [0.0, 1.0],
            },
            "usable_anchors": [
                {
                    "reference_energy_kev": spec.reference_kev,
                    "label": spec.label,
                    "sideband_mode": spec.sideband_mode,
                }
                for spec in PB_ANCHORS
            ],
            "high_energy_pb_limit_kev": 1460.830,
            "excluded_regions": [
                {"reference_energy_kev": 338.320, "reason": "nearby approximately 328-keV line contaminates background estimation"},
                {"reference_energy_kev": 968.971, "reason": "not retained in the audited isolated-anchor set"},
                {"reference_energy_kev": 1592.0, "reason": "2614-keV double-escape region"},
                {"reference_energy_kev": 2103.5, "reason": "2614-keV single-escape region"},
                {"reference_energy_kev": 2614.533, "reason": "retention-only; no supported higher-energy Compton-background reference"},
            ],
            "statistical_reliability": {
                "minimum_background_counts_per_peak": MINIMUM_BACKGROUND_COUNTS,
                "minimum_net_peak_counts_per_peak": MINIMUM_NET_PEAK_COUNTS,
            },
            "selected_operating_points": primary,
            "sensitivity_selected_operating_points_excluding_209": sensitivity,
            "sweet_spot_grid_selection": {
                "threshold": recommended_grid["threshold"],
                "normalized_gain_fraction": recommended_grid[
                    "geometric_mean_pb_improvement_normalized_gain_fraction"
                ],
                "normalized_retention_loss_fraction": recommended_grid[
                    "minimum_peak_retention_normalized_loss_fraction"
                ],
                "pareto_knee_score": recommended_grid[
                    "geometric_mean_pb_improvement_pareto_knee_score"
                ],
            },
            "sensitivity_sweet_spot_grid_selection_excluding_209": {
                "threshold": sensitivity_grid["threshold"],
                "normalized_gain_fraction": sensitivity_grid[
                    "sensitivity_geometric_mean_pb_improvement_excluding_209_normalized_gain_fraction"
                ],
                "normalized_retention_loss_fraction": sensitivity_grid[
                    "sensitivity_minimum_peak_retention_excluding_209_normalized_loss_fraction"
                ],
                "pareto_knee_score": sensitivity_grid[
                    "sensitivity_geometric_mean_pb_improvement_excluding_209_pareto_knee_score"
                ],
            },
            "recommended_exact": recommended,
            "sensitivity_recommended_exact_excluding_209": sensitivity_recommended,
            "maximum_reliable_pb_exact": maximum_pb,
            "no_brainer_definition": (
                "Highest geometric-mean P/B threshold with pooled net counts "
                "across all ten usable peaks retained at >=99%; this is not a "
                "per-peak >=99% constraint."
            ),
            "no_brainer_grid_selection": {
                "threshold": no_brainer_grid["threshold"],
                "pooled_net_peak_retention": no_brainer_grid[
                    "pooled_net_peak_retention"
                ],
                "geometric_mean_pb_improvement": no_brainer_grid[
                    "geometric_mean_pb_improvement"
                ],
            },
            "no_brainer_exact": no_brainer,
            "superseded_global_0p437_exact": old_global,
        },
        "peak_windows": [
            {**asdict(window), "label": spec.label, "sideband_mode": spec.sideband_mode}
            for spec, window in windows
        ],
        "baseline_peak_metrics": {
            f"{reference:.3f}": metrics for reference, metrics in baseline.items()
        },
        "test_partition_used": False,
        "external_validation": False,
        "claim_boundary": (
            "Historical Th-232 events directly select the deployment threshold; "
            "locked test and Eu-152 remain unopened and unused."
        ),
        "artifacts": {},
    }
    artifact_paths = [
        scan_csv,
        operating_csv,
        per_peak_csv,
        spectrum_csv,
        fpga_presets_csv,
        report_md,
        output_dir / "threshold_optimization.png",
        output_dir / "usable_peak_zooms.png",
        output_dir / "usable_peak_zooms_normalized.png",
        output_dir / f"th232_spectrum_threshold_{threshold_tag}.png",
    ]
    for path in artifact_paths:
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_json = output_dir / "experiment_report.json"
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "recommended": {
                    key: recommended[key]
                    for key in (
                        "threshold",
                        "geometric_mean_pb_improvement",
                        "minimum_peak_retention",
                        "mean_peak_retention",
                        "selected_fraction",
                    )
                },
                "sensitivity_threshold_excluding_209": sensitivity_recommended[
                    "threshold"
                ],
                "report": relative(report_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
