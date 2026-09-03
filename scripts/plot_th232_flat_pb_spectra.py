#!/usr/bin/env python3
"""Generate corrected Th-232 spectra for selected flat P/B improvement targets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    PeakWindow,
    fit_peak_windows,
    peak_background_metrics,
)
from scripts.optimize_th232_all_ba_ds_cnn_threshold import (  # noqa: E402
    PRIMARY_REFERENCE_PEAKS_KEV,
)
from scripts.scan_th232_flat_pb_grid import (  # noqa: E402
    get_peak_target,
    precompute_peak_curves,
)

DEFAULT_TH232_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_flat_pb_grid_scan_20260823/corrected_spectra_20260823"
)
TARGET_GAINS_PCT = [5, 10, 20, 30, 45]

COLORS = {
    "No cut": "black",
    "+5% P/B": "#1f77b4",   # Blue
    "+10% P/B": "#2ca02c",  # Green
    "+20% P/B": "#9467bd",  # Purple
    "+30% P/B": "#ff7f0e",  # Orange
    "+45% P/B": "#d62728",  # Red
}

# These are catalog-constrained line groups that are visibly testable in the
# no-filter Th-232 cache. Close lines are grouped when this spectrum does not
# resolve them separately. The primary threshold-fit lines are marked below;
# the remaining entries are annotation-only diagnostics.
TH232_IDENTIFIABLE_LINE_CATALOG = (
    {"expected_energies_kev": (129.065,), "label": "129.1 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (153.977,), "label": "154.0 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (184.540, 191.353), "label": "184.5/191.4 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (209.253,), "label": "209.3 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (238.632, 240.986), "label": "238.6/241.0 Pb/Ra", "category": "Th-232 chain", "primary": True},
    {"expected_energies_kev": (270.245,), "label": "270.2 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (277.371,), "label": "277.4 Tl-208", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (300.087,), "label": "300.1 Pb-212", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (328.000,), "label": "328.0 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (338.320,), "label": "338.3 Ac-228", "category": "Th-232 chain", "primary": True},
    {"expected_energies_kev": (409.462,), "label": "409.5 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (463.004,), "label": "463.0 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (510.770,), "label": "510.8 Tl-208 / pair", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (562.500,), "label": "562.5 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (572.140,), "label": "572.1 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (583.191,), "label": "583.2 Tl-208", "category": "Th-232 chain", "primary": True},
    {"expected_energies_kev": (701.747,), "label": "701.7 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (727.330,), "label": "727.3 Bi-212", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (755.315,), "label": "755.3 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (763.100,), "label": "763.1 Tl-208", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (772.291,), "label": "772.3 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (782.142, 785.400), "label": "782.1/785.4 Ac/Bi", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (794.947,), "label": "794.9 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (835.710, 840.377), "label": "835.7/840.4 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (860.557,), "label": "860.6 Tl-208", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (911.204,), "label": "911.2 Ac-228", "category": "Th-232 chain", "primary": True},
    {"expected_energies_kev": (964.766, 968.971), "label": "964.8/969.0 Ac-228", "category": "Th-232 chain", "primary": True},
    {"expected_energies_kev": (987.710, 988.630), "label": "987.7/988.6 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1033.248,), "label": "1033.2 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1065.180,), "label": "1065.2 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1095.679,), "label": "1095.7 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1110.610,), "label": "1110.6 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1153.467,), "label": "1153.5 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1247.080,), "label": "1247.1 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1287.680,), "label": "1287.7 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1460.830,), "label": "1460.8 K-40", "category": "background", "primary": False},
    {"expected_energies_kev": (1495.910,), "label": "1495.9 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1588.200,), "label": "1588.2 Ac / DE", "category": "diagnostic", "primary": False},
    {"expected_energies_kev": (1620.500,), "label": "1620.5 Bi-212", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1630.627,), "label": "1630.6 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (1666.523,), "label": "1666.5 Ac-228", "category": "Th-232 chain", "primary": False},
    {"expected_energies_kev": (2103.500,), "label": "2103.5 single escape", "category": "escape", "primary": False},
    {"expected_energies_kev": (2614.533,), "label": "2614.5 Tl-208", "category": "Th-232 chain", "primary": True},
)
IDENTIFIABLE_PEAK_MIN_PROMINENCE = 50.0


def target_label(target_pct: int) -> str:
    return f"+{target_pct}% P/B"


def json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_threshold_curve(
    target_pct: int,
    peak_curves: dict[float, tuple[np.ndarray, np.ndarray, PeakWindow]],
    t_grid: np.ndarray,
) -> tuple[PchipInterpolator, list[dict[str, float]], np.ndarray, np.ndarray]:
    target_gain = 1.0 + target_pct / 100.0
    threshold_rows: list[dict[str, float]] = []
    peak_energies = []
    peak_thresholds = []

    for reference in PRIMARY_REFERENCE_PEAKS_KEV:
        threshold, achieved_gain, retention = get_peak_target(
            reference, target_gain, peak_curves, t_grid
        )
        if not (
            np.isfinite(threshold)
            and np.isfinite(achieved_gain)
            and np.isfinite(retention)
        ):
            raise ValueError(
                f"Non-finite threshold selection for +{target_pct}% at {reference} keV"
            )
        window = peak_curves[reference][2]
        peak_energies.append(float(window.centroid_kev))
        peak_thresholds.append(float(threshold))
        threshold_rows.append(
            {
                "target_pb_gain_percent": float(target_pct),
                "reference_energy_kev": float(reference),
                "observed_centroid_kev": float(window.centroid_kev),
                "target_threshold": float(threshold),
                "achieved_discrete_pb_gain": float(achieved_gain),
                "discrete_net_peak_retention": float(retention),
            }
        )

    order = np.argsort(peak_energies)
    sorted_energies = np.asarray(peak_energies, dtype=np.float64)[order]
    sorted_thresholds = np.asarray(peak_thresholds, dtype=np.float64)[order]
    return (
        PchipInterpolator(sorted_energies, sorted_thresholds),
        threshold_rows,
        sorted_energies,
        sorted_thresholds,
    )


def summarize_continuous_curve(
    target_pct: int,
    energy: np.ndarray,
    scores: np.ndarray,
    selected_histogram: np.ndarray,
    selected_mask: np.ndarray,
    windows: list[PeakWindow],
    base_metrics: dict[float, dict[str, float]],
    threshold_rows: list[dict[str, float]],
) -> dict[str, Any]:
    per_peak: list[dict[str, Any]] = []
    for window in windows:
        if window.reference_kev not in PRIMARY_REFERENCE_PEAKS_KEV:
            continue
        metrics = peak_background_metrics(selected_histogram, window)
        base = base_metrics[window.reference_kev]
        gain = metrics["peak_to_background"] / base["peak_to_background"]
        retention = metrics["net_peak_counts"] / base["net_peak_counts"]
        per_peak.append(
            {
                "reference_energy_kev": float(window.reference_kev),
                "observed_centroid_kev": float(window.centroid_kev),
                "roi_counts": float(metrics["roi_counts"]),
                "estimated_background_counts": float(
                    metrics["estimated_background_counts"]
                ),
                "net_peak_counts": float(metrics["net_peak_counts"]),
                "peak_to_background": float(metrics["peak_to_background"]),
                "continuous_pb_improvement": finite_or_none(gain),
                "continuous_net_peak_retention": finite_or_none(retention),
            }
        )

    gains = [row["continuous_pb_improvement"] for row in per_peak]
    retentions = [row["continuous_net_peak_retention"] for row in per_peak]
    valid_gains = [value for value in gains if value is not None and value > 0.0]
    valid_retentions = [value for value in retentions if value is not None]
    geometric_mean = (
        float(np.exp(np.mean(np.log(valid_gains))))
        if len(valid_gains) == len(gains) and valid_gains
        else None
    )
    return {
        "target_pb_gain_percent": int(target_pct),
        "selected_events": int(np.count_nonzero(selected_mask)),
        "selected_fraction": float(np.mean(selected_mask)),
        "continuous_geometric_mean_pb_improvement": geometric_mean,
        "continuous_minimum_net_peak_retention": (
            float(min(valid_retentions))
            if len(valid_retentions) == len(retentions) and valid_retentions
            else None
        ),
        "continuous_mean_net_peak_retention": (
            float(np.mean(valid_retentions))
            if len(valid_retentions) == len(retentions) and valid_retentions
            else None
        ),
        "thresholds": threshold_rows,
        "per_peak": per_peak,
    }


def identify_th232_no_filter_peaks(histogram: np.ndarray) -> list[dict[str, Any]]:
    """Match visible histogram peaks to a constrained Th-232 line catalog.

    The matching is deliberately catalog-constrained: a generic peak finder
    would also label statistical fluctuations and Compton-structure features.
    The returned peak positions are measured from the current histogram after
    1-keV binning and light smoothing; the catalog energies are only used to
    identify the corresponding line group.
    """
    smoothed = gaussian_filter1d(histogram.astype(np.float64), 1.0)
    candidate_indices, properties = find_peaks(
        smoothed,
        distance=3,
        prominence=IDENTIFIABLE_PEAK_MIN_PROMINENCE,
    )
    candidate_prominences = np.asarray(properties["prominences"], dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for entry in TH232_IDENTIFIABLE_LINE_CATALOG:
        expected = np.asarray(entry["expected_energies_kev"], dtype=np.float64)
        search_low = max(float(ENERGY_CENTERS[0]), float(np.min(expected) - 6.0))
        search_high = min(float(ENERGY_CENTERS[-1]), float(np.max(expected) + 6.0))
        candidate_mask = (
            (ENERGY_CENTERS[candidate_indices] >= search_low)
            & (ENERGY_CENTERS[candidate_indices] <= search_high)
        )
        matching_indices = candidate_indices[candidate_mask]
        matching_prominences = candidate_prominences[candidate_mask]
        if matching_indices.size == 0:
            continue

        selected = int(matching_indices[np.argmax(matching_prominences)])
        prominence = float(
            matching_prominences[np.argmax(matching_prominences)]
        )
        observed = float(ENERGY_CENTERS[selected])
        nearest_expected = float(expected[np.argmin(np.abs(expected - observed))])
        rows.append(
            {
                "label": str(entry["label"]),
                "category": str(entry["category"]),
                "primary_threshold_peak": bool(entry["primary"]),
                "expected_energies_kev": "/".join(f"{value:.3f}" for value in expected),
                "observed_peak_kev": observed,
                "energy_residual_kev": observed - nearest_expected,
                "peak_counts_1kev": int(histogram[selected]),
                "smoothed_prominence_counts": prominence,
                "identification_strength": "strong" if prominence >= 150.0 else "weak",
            }
        )
    return rows


def _annotation_color(row: dict[str, Any]) -> str:
    if row["primary_threshold_peak"]:
        return "#d62728"
    if row["category"] == "background":
        return "#555555"
    if row["category"] in {"escape", "diagnostic"}:
        return "#9467bd"
    return "#1f77b4"


def _plot_annotated_spectrum_axis(
    axis: Any,
    histogram: np.ndarray,
    peak_rows: list[dict[str, Any]],
    x_limits: tuple[float, float],
    title: str,
    log_scale: bool = False,
) -> None:
    visible = (ENERGY_CENTERS >= x_limits[0]) & (ENERGY_CENTERS <= x_limits[1])
    visible_rows = [
        row
        for row in peak_rows
        if x_limits[0] <= row["observed_peak_kev"] <= x_limits[1]
    ]
    visible_maximum = max(int(np.max(histogram[visible])), 1)
    axis.step(
        ENERGY_CENTERS[visible],
        histogram[visible],
        where="mid",
        color="black",
        linewidth=0.85,
        label=f"Th-232 no filter ({int(np.sum(histogram)):,} events)",
    )
    axis.set_xlim(*x_limits)
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel("Counts / 1 keV")
    axis.grid(alpha=0.25, which="both" if log_scale else "major")

    if log_scale:
        axis.set_yscale("log")
        axis.set_ylim(bottom=0.8, top=visible_maximum * 7.0)
        level_factors = (1.25, 1.9, 2.9, 4.4)
    else:
        axis.set_ylim(bottom=0.0, top=visible_maximum * 1.58)
        level_factors = (1.06, 1.18, 1.30, 1.42)

    for row in visible_rows:
        color = _annotation_color(row)
        peak_energy = float(row["observed_peak_kev"])
        peak_counts = max(float(row["peak_counts_1kev"]), 0.8)
        axis.axvline(
            peak_energy,
            color=color,
            linestyle="--",
            linewidth=0.65 if row["identification_strength"] == "strong" else 0.45,
            alpha=0.62,
        )

    for index, row in enumerate(visible_rows):
        color = _annotation_color(row)
        peak_energy = float(row["observed_peak_kev"])
        peak_counts = max(float(row["peak_counts_1kev"]), 0.8)
        factor = level_factors[index % len(level_factors)]
        text_y = visible_maximum * factor
        axis.annotate(
            row["label"],
            xy=(peak_energy, peak_counts),
            xytext=(peak_energy, text_y),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.0 if row["identification_strength"] == "strong" else 6.3,
            color=color,
            arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": color},
            annotation_clip=False,
        )

    axis.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_annotated_no_filter_spectrum(
    output_dir: Path,
    histogram: np.ndarray,
    peak_rows: list[dict[str, Any]],
) -> Path:
    """Write full-range and zoomed no-filter spectrum views with labels."""
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(19, 16),
        gridspec_kw={"height_ratios": (1.0, 1.1, 1.0)},
        constrained_layout=True,
    )
    _plot_annotated_spectrum_axis(
        axes[0],
        histogram,
        peak_rows,
        (0.0, 3000.0),
        "Th-232 no-filter spectrum — identifiable peak map",
    )
    _plot_annotated_spectrum_axis(
        axes[1],
        histogram,
        peak_rows,
        (100.0, 1000.0),
        "Low-energy detail",
    )
    _plot_annotated_spectrum_axis(
        axes[2],
        histogram,
        peak_rows,
        (1000.0, 2700.0),
        "High-energy detail (log scale)",
        log_scale=True,
    )
    axes[2].set_xlabel("Corrected energy (keV)")

    handles = [
        Line2D([0], [0], color="#d62728", linestyle="--", label="Primary threshold-fit peak"),
        Line2D([0], [0], color="#1f77b4", linestyle="--", label="Additional Th-232-chain line"),
        Line2D([0], [0], color="#9467bd", linestyle="--", label="Escape/diagnostic feature"),
        Line2D([0], [0], color="#555555", linestyle="--", label="Likely background line"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(
        "Th-232 no-filter spectrum with catalog-constrained identifiable peaks",
        fontsize=15,
        fontweight="bold",
    )
    output_path = output_dir / "th232_no_filter_identifiable_peaks.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_individual_spectrum(
    output_dir: Path,
    filename: str,
    title_label: str,
    histogram: np.ndarray,
    raw_maximum: int,
    windows: list[PeakWindow],
) -> None:
    color = COLORS[title_label]
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, constrained_layout=True
    )
    for axis in axes:
        axis.step(
            ENERGY_CENTERS,
            histogram,
            where="mid",
            linewidth=0.9,
            color=color,
            label=f"Th-232 {title_label} ({int(histogram.sum()):,} events)",
        )
        for window in windows:
            axis.axvline(
                window.centroid_kev,
                color="0.45",
                linewidth=0.5,
                linestyle="--",
                alpha=0.45,
            )
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, 3000.0)
        axis.legend(loc="upper right", framealpha=0.9)

    axes[0].set_ylabel("Counts / 1 keV")
    axes[0].set_title(f"Th-232 energy spectrum — {title_label}", fontweight="bold")
    axes[1].set_ylabel("Counts / 1 keV")
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8, top=max(raw_maximum, 1) * 2.0)
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_canonical_th232_spectra(
    output_dir: Path,
    histograms: dict[str, np.ndarray],
    windows: list[PeakWindow],
) -> None:
    names = ["No cut", *[target_label(pct) for pct in TARGET_GAINS_PCT]]

    # 1. Two-Panel Full Energy Spectrum (Linear top, Logarithmic bottom)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)

    for name in names:
        hist = histograms[name]
        col = COLORS[name]
        lw = 0.95 if name == "No cut" else 0.85
        alpha = 0.95 if name == "No cut" else 0.85
        axes[0].step(
            ENERGY_CENTERS,
            hist,
            where="mid",
            linewidth=lw,
            color=col,
            alpha=alpha,
            label=name,
        )
        axes[1].step(
            ENERGY_CENTERS,
            hist,
            where="mid",
            linewidth=lw,
            color=col,
            alpha=alpha,
            label=name,
        )

    axes[0].set_ylabel("Counts / 1 keV", fontsize=11)
    targets_text = ", ".join(f"+{pct}%" for pct in TARGET_GAINS_PCT)
    axes[0].set_title(
        f"Th-232 energy spectrum after flat P/B improvement cuts ({targets_text} vs No cut)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(ncol=3, fontsize=9, loc="upper right", framealpha=0.9)
    axes[0].grid(alpha=0.25)
    axes[0].set_xlim(0.0, 3000.0)

    axes[1].set_ylabel("Counts / 1 keV", fontsize=11)
    axes[1].set_xlabel("Corrected energy (keV)", fontsize=11)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8, top=max(histograms["No cut"]) * 2.0)
    axes[1].grid(alpha=0.25, which="both")

    for window in windows:
        axes[0].axvline(window.centroid_kev, color="0.6", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[1].axvline(window.centroid_kev, color="0.6", linewidth=0.5, linestyle="--", alpha=0.5)

    spectrum_png = output_dir / "th232_flat_pb_energy_spectra.png"
    fig.savefig(spectrum_png, dpi=200)
    plt.close(fig)

    # 2. Canonical 2x4 Peak Window Zoom Grid
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    flat_axes = list(axes.flat)

    for axis, window in zip(flat_axes, windows):
        half_width = max(18.0, 6.0 * window.sigma_kev)
        selected = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (
            ENERGY_CENTERS <= window.centroid_kev + half_width
        )
        for name in names:
            axis.step(
                ENERGY_CENTERS[selected],
                histograms[name][selected],
                where="mid",
                linewidth=0.9,
                color=COLORS[name],
                label=name,
            )
        axis.axvspan(
            window.roi_low_kev,
            window.roi_high_kev,
            color="#984ea3",
            alpha=0.12,
            label="Signal ROI (±2σ)",
        )
        axis.axvspan(
            window.left_low_kev,
            window.left_high_kev,
            color="0.5",
            alpha=0.10,
            label="3–5σ sideband",
        )
        axis.axvspan(window.right_low_kev, window.right_high_kev, color="0.5", alpha=0.10)
        axis.set_title(
            f"{window.reference_kev:g} keV ref. (obs {window.centroid_kev:.2f} keV)",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel("Energy (keV)", fontsize=9)
        axis.set_ylabel("Counts / 1 keV", fontsize=9)
        axis.grid(alpha=0.25)

    for axis in flat_axes[len(windows):]:
        axis.axis("off")

    flat_axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        f"Th-232 peak windows after flat P/B improvement score cuts ({targets_text})",
        fontsize=13,
        fontweight="bold",
    )
    zooms_png = output_dir / "th232_flat_pb_peak_zooms.png"
    fig.savefig(zooms_png, dpi=200)
    plt.close(fig)


def make_markdown_report(
    cache_path: Path,
    summaries: list[dict[str, Any]],
    annotated_png: Path,
    annotated_csv: Path,
    annotated_peak_count: int,
) -> str:
    lines = [
        "# Corrected Th-232 flat P/B spectra",
        "",
        "The spectra use finite threshold candidates only. Each target is solved independently at the six primary Th-232 peaks, then the score threshold is interpolated with PCHIP against observed peak centroids and clamped to the calibrated centroid range.",
        "",
        f"- Score cache: `{relative(cache_path)}`",
        f"- Targets: `{', '.join(f'+{pct}%' for pct in TARGET_GAINS_PCT)}`",
        "- Threshold selection: earliest finite, nonnegative-retention threshold crossing on a 0.001 score grid; if the target is not reached, the nearest finite candidate is used with a retention tie-break.",
        f"- Annotated no-filter peak map: `{relative(annotated_png)}` ({annotated_peak_count} catalog-constrained matches).",
        f"- Peak annotation table: `{relative(annotated_csv)}`.",
        "- Claim boundary: thresholds are tuned directly on historical Th-232 data and are not external validation.",
        "",
        "| Target | Events retained | Th-232 retention | Geometric-mean continuous P/B gain | Minimum continuous peak retention | Mean continuous peak retention |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        def fmt_percent(value: float | None) -> str:
            return "undefined" if value is None else f"{value:.2%}"

        geo = summary["continuous_geometric_mean_pb_improvement"]
        lines.append(
            f"| +{summary['target_pb_gain_percent']}% | {summary['selected_events']:,} | {summary['selected_fraction']:.2%} | {geo:.4f}x | {fmt_percent(summary['continuous_minimum_net_peak_retention'])} | {fmt_percent(summary['continuous_mean_net_peak_retention'])} |"
        )
    lines.extend(
        [
            "",
            "The 238-keV line is the limiting peak at aggressive targets. The +45% spectrum is therefore a high-rejection diagnostic, not a general-purpose deployment recommendation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--th232-score-cache", type=Path, default=DEFAULT_TH232_SCORE_CACHE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    th232_path = args.th232_score_cache.resolve()
    with h5py.File(th232_path, "r") as f:
        energy = np.asarray(f["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(f["score"], dtype=np.float32)
        cache_metadata = {str(key): json_scalar(value) for key, value in f.attrs.items()}

    base_hist = np.histogram(energy, ENERGY_EDGES)[0]
    windows = fit_peak_windows(base_hist)
    base_metrics = {
        w.reference_kev: peak_background_metrics(base_hist, w) for w in windows
    }

    t_grid = np.linspace(0.0, 0.90, 901)
    peak_curves = precompute_peak_curves(energy, scores, windows, base_metrics, t_grid)

    histograms: dict[str, np.ndarray] = {"No cut": base_hist}
    selected_masks: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, float]] = []

    for pct in TARGET_GAINS_PCT:
        pchip, threshold_rows, sorted_e, _ = build_threshold_curve(
            pct, peak_curves, t_grid
        )
        all_threshold_rows.extend(threshold_rows)

        min_e, max_e = float(min(sorted_e)), float(max(sorted_e))
        t_curve = np.clip(pchip(np.clip(energy, min_e, max_e)), 0.0, 1.0)
        label = target_label(pct)
        selected_mask = scores >= t_curve
        selected_masks[label] = selected_mask
        selected_histogram = np.histogram(energy[selected_mask], ENERGY_EDGES)[0]
        histograms[label] = selected_histogram
        summaries.append(
            summarize_continuous_curve(
                pct,
                energy,
                scores,
                selected_histogram,
                selected_mask,
                windows,
                base_metrics,
                threshold_rows,
            )
        )

    # Save spectra table CSV
    csv_rows = []
    for idx, e_val in enumerate(ENERGY_CENTERS):
        row = {"energy_kev": float(e_val)}
        for label, h in histograms.items():
            safe_k = label.lower().replace("+", "plus_").replace("%", "").replace(" ", "_")
            row[f"counts_{safe_k}"] = int(h[idx])
        csv_rows.append(row)

    csv_path = output_dir / "th232_flat_pb_energy_spectra_1kev.csv"
    write_csv(csv_path, csv_rows)
    target_csv_path = output_dir / "th232_spectra_5_10_20_30_45pct.csv"
    write_csv(target_csv_path, csv_rows)
    threshold_csv_path = output_dir / "flat_pb_thresholds_by_target.csv"
    write_csv(threshold_csv_path, all_threshold_rows)

    plot_canonical_th232_spectra(output_dir, histograms, windows)
    plot_individual_spectrum(
        output_dir,
        "th232_energy_spectrum_raw.png",
        "No cut",
        histograms["No cut"],
        int(np.max(base_hist)),
        windows,
    )
    for pct in TARGET_GAINS_PCT:
        label = target_label(pct)
        plot_individual_spectrum(
            output_dir,
            f"th232_energy_spectrum_plus_{pct}pct.png",
            label,
            histograms[label],
            int(np.max(base_hist)),
            windows,
        )

    annotated_peak_rows = identify_th232_no_filter_peaks(base_hist)
    annotated_png = plot_annotated_no_filter_spectrum(
        output_dir,
        base_hist,
        annotated_peak_rows,
    )
    annotated_csv = output_dir / "th232_no_filter_identifiable_peaks.csv"
    write_csv(annotated_csv, annotated_peak_rows)

    report_json = output_dir / "th232_flat_pb_spectra_report.json"
    report = {
        "schema_version": 1,
        "status": "TH232_CORRECTED_FLAT_PB_SPECTRA_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_gains_percent": TARGET_GAINS_PCT,
        "threshold_grid": {
            "minimum_score": float(t_grid[0]),
            "maximum_score": float(t_grid[-1]),
            "step": float(t_grid[1] - t_grid[0]),
            "finite_candidate_guard": True,
            "selection_method": (
                "first_finite_crossing_then_nearest_finite_with_retention_tiebreak"
            ),
        },
        "th232_score_cache": relative(th232_path),
        "score_cache_metadata": cache_metadata,
        "no_filter_peak_annotations": {
            "identified_count": len(annotated_peak_rows),
            "smoothing_sigma_bins": 1.0,
            "minimum_smoothed_prominence_counts": IDENTIFIABLE_PEAK_MIN_PROMINENCE,
            "catalog_constrained": True,
            "rows": annotated_peak_rows,
        },
        "summaries": summaries,
        "claim_boundary": (
            "Flat P/B targets were tuned directly against historical Th-232 score data; "
            "the results are deployment-optimization diagnostics, not external validation."
        ),
        "artifacts": {
            "report_md": relative(output_dir / "report.md"),
            "spectrum_csv": relative(csv_path),
            "target_spectrum_csv": relative(target_csv_path),
            "threshold_csv": relative(threshold_csv_path),
            "annotated_no_filter_spectrum_png": relative(annotated_png),
            "annotated_no_filter_peak_csv": relative(annotated_csv),
            "combined_spectrum_png": relative(
                output_dir / "th232_flat_pb_energy_spectra.png"
            ),
            "peak_zooms_png": relative(output_dir / "th232_flat_pb_peak_zooms.png"),
            "raw_spectrum_png": relative(output_dir / "th232_energy_spectrum_raw.png"),
            "target_spectrum_pngs": {
                str(pct): relative(output_dir / f"th232_energy_spectrum_plus_{pct}pct.png")
                for pct in TARGET_GAINS_PCT
            },
        },
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        make_markdown_report(
            th232_path,
            summaries,
            annotated_png,
            annotated_csv,
            len(annotated_peak_rows),
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "artifacts": [
                    "th232_flat_pb_energy_spectra.png",
                    "th232_flat_pb_peak_zooms.png",
                    "th232_flat_pb_energy_spectra_1kev.csv",
                    "th232_spectra_5_10_20_30_45pct.csv",
                    "flat_pb_thresholds_by_target.csv",
                    "th232_no_filter_identifiable_peaks.png",
                    "th232_no_filter_identifiable_peaks.csv",
                    "th232_flat_pb_spectra_report.json",
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
