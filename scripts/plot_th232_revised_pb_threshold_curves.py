#!/usr/bin/env python3
"""Plot revised Th-232 energy-dependent DS-CNN score thresholds.

The P/B objective uses line-aware background windows at usable features from
209.3 through 1460.8 keV. The 583.2-keV and 911.2-keV points use only the
higher-energy sideband to avoid known lower-side features. The 2614.5-keV
line is retained as an ROI-retention diagnostic only; it is not used to fit
or extrapolate the P/B threshold curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    PeakWindow,
    gaussian_linear,
    interval_counts,
)
from scripts.scan_th232_flat_pb_grid import get_peak_target  # noqa: E402

DEFAULT_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409"
)
TARGET_GAINS_PCT = (5, 10, 20, 30, 45)
SCORE_GRID = np.linspace(0.0, 0.90, 901)


@dataclass(frozen=True)
class AnchorSpec:
    reference_kev: float
    label: str
    sideband_mode: str
    use_pb: bool = True


PB_ANCHORS = (
    AnchorSpec(209.253, "209.3 keV Ac-228", "both"),
    AnchorSpec(238.632, "238.6 keV Pb-212", "both"),
    AnchorSpec(300.087, "300.1 keV Pb-212", "both"),
    AnchorSpec(409.462, "409.5 keV Ac-228", "both"),
    AnchorSpec(510.770, "510.8 keV Tl-208/pair", "both"),
    AnchorSpec(583.191, "583.2 keV Tl-208", "higher_only"),
    AnchorSpec(727.330, "727.3 keV Bi-212", "both"),
    AnchorSpec(911.204, "911.2 keV Ac-228", "higher_only"),
    AnchorSpec(1247.080, "1247.1 keV Ac-228", "both"),
    AnchorSpec(1460.830, "1460.8 keV K-40/background", "both"),
)
RETENTION_ONLY_ANCHOR = AnchorSpec(
    2614.533,
    "2614.5 keV Tl-208",
    "diagnostic_only",
    use_pb=False,
)


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_anchor_window(
    histogram: np.ndarray,
    reference_kev: float,
    smoothed: np.ndarray,
) -> PeakWindow:
    search_half_width = 25.0 if reference_kev >= 1200.0 else 15.0
    search = (ENERGY_CENTERS >= reference_kev - search_half_width) & (
        ENERGY_CENTERS <= reference_kev + search_half_width
    )
    initial_mean = float(ENERGY_CENTERS[search][np.argmax(smoothed[search])])
    fit_half_width = 20.0 if reference_kev >= 1200.0 else 13.0
    selected = (ENERGY_CENTERS >= initial_mean - fit_half_width) & (
        ENERGY_CENTERS <= initial_mean + fit_half_width
    )
    x = ENERGY_CENTERS[selected]
    y = histogram[selected].astype(np.float64)
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
    return PeakWindow(
        reference_kev,
        mean,
        sigma,
        mean - 2.0 * sigma,
        mean + 2.0 * sigma,
        mean - 5.0 * sigma,
        mean - 3.0 * sigma,
        mean + 3.0 * sigma,
        mean + 5.0 * sigma,
    )


def line_aware_metrics(
    histogram: np.ndarray,
    window: PeakWindow,
    sideband_mode: str,
) -> dict[str, float]:
    roi_counts, _ = interval_counts(histogram, window.roi_low_kev, window.roi_high_kev)
    right_counts, right_energy = interval_counts(
        histogram, window.right_low_kev, window.right_high_kev
    )
    right_width = window.right_high_kev - window.right_low_kev
    right_density = right_counts / right_width

    if sideband_mode == "higher_only":
        background_density = right_density
    elif sideband_mode == "both":
        left_counts, left_energy = interval_counts(
            histogram, window.left_low_kev, window.left_high_kev
        )
        left_width = window.left_high_kev - window.left_low_kev
        left_density = left_counts / left_width
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
        fraction = (window.centroid_kev - left_center) / (right_center - left_center)
        background_density = left_density + fraction * (right_density - left_density)
    else:
        raise ValueError(f"Unsupported sideband mode: {sideband_mode}")

    background_counts = background_density * (window.roi_high_kev - window.roi_low_kev)
    net_peak_counts = roi_counts - background_counts
    peak_to_background = (
        net_peak_counts / background_counts if background_counts > 0.0 else float("nan")
    )
    return {
        "roi_counts": float(roi_counts),
        "estimated_background_counts": float(background_counts),
        "net_peak_counts": float(net_peak_counts),
        "peak_to_background": float(peak_to_background),
    }


def build_peak_curves(
    energy: np.ndarray,
    scores: np.ndarray,
    anchors: list[tuple[AnchorSpec, PeakWindow]],
    base_histogram: np.ndarray,
) -> dict[float, tuple[np.ndarray, np.ndarray, PeakWindow]]:
    curves: dict[float, tuple[np.ndarray, np.ndarray, PeakWindow]] = {}
    for spec, window in anchors:
        mask = (energy >= window.left_low_kev - 2.0) & (
            energy <= window.right_high_kev + 2.0
        )
        sub_energy = energy[mask]
        sub_scores = scores[mask]
        base = line_aware_metrics(base_histogram, window, spec.sideband_mode)
        gains: list[float] = []
        retentions: list[float] = []
        for threshold in SCORE_GRID:
            selected = sub_scores >= threshold
            histogram = np.histogram(sub_energy[selected], ENERGY_EDGES)[0]
            metrics = line_aware_metrics(histogram, window, spec.sideband_mode)
            gain = metrics["peak_to_background"] / base["peak_to_background"]
            retention = metrics["net_peak_counts"] / base["net_peak_counts"]
            gains.append(float(gain))
            retentions.append(float(retention))
        curves[spec.reference_kev] = (
            np.asarray(gains, dtype=np.float64),
            np.asarray(retentions, dtype=np.float64),
            window,
        )
    return curves


def build_retention_curve(
    energy: np.ndarray,
    scores: np.ndarray,
    window: PeakWindow,
) -> np.ndarray:
    mask = (energy >= window.roi_low_kev) & (energy <= window.roi_high_kev)
    sub_scores = scores[mask]
    baseline_count = float(sub_scores.size)
    if baseline_count <= 0.0:
        return np.full(SCORE_GRID.shape, np.nan, dtype=np.float64)
    return np.asarray(
        [float(np.count_nonzero(sub_scores >= threshold)) / baseline_count for threshold in SCORE_GRID],
        dtype=np.float64,
    )


def make_report(
    cache_path: Path,
    output_dir: Path,
    threshold_rows: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Revised Th-232 P/B threshold curves",
        "",
        "The P/B objective is evaluated from the 209.3-keV anchor through 1460.8 keV. The 583.2-keV and 911.2-keV points use the higher-energy sideband only; 911.2 keV is treated this way because the lower sideband contains the 904.2-keV Ac-228 feature.",
        "",
        f"- Score cache: `{relative(cache_path)}`",
        f"- Curve plot: `{relative(output_dir / 'th232_revised_pb_threshold_curves.png')}`",
        f"- Anchor table: `{relative(output_dir / 'th232_revised_pb_thresholds_by_target.csv')}`",
        f"- 2614-keV retention diagnostic: `{relative(output_dir / 'th232_2614_retention_diagnostic.csv')}`",
        "- 2614.5 keV is not used to calculate P/B or to extrapolate the threshold curve.",
        "",
        "## P/B anchor windows",
        "",
        "| Reference (keV) | Observed centroid (keV) | Sideband method |",
        "|---:|---:|:---|",
    ]
    for row in anchor_rows:
        lines.append(
            f"| {row['reference_energy_kev']:.3f} | {row['observed_centroid_kev']:.3f} | {row['sideband_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Selected thresholds",
            "",
            "| Target | Energy (keV) | Score threshold | Achieved P/B gain | Net retention |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            f"| +{row['target_pb_gain_percent']:.0f}% | {row['reference_energy_kev']:.3f} | {row['target_threshold']:.4f} | {row['achieved_pb_gain']:.4f}x | {row['net_peak_retention']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 2614-keV retention-only diagnostic",
            "",
            "The 2614.5-keV ROI retention is reported against score threshold, without interpreting the sideband as Compton background. The colored markers in the plot show the retention obtained if the 1460.8-keV boundary threshold is held constant above the P/B domain.",
            "",
            "| Target | Boundary threshold at 1460.8 keV | 2614-keV ROI retention |",
            "|---:|---:|---:|",
        ]
    )
    for row in retention_rows:
        lines.append(
            f"| +{row['target_pb_gain_percent']:.0f}% | {row['boundary_threshold']:.4f} | {row['retention_at_boundary']:.2%} |"
        )
    lines.extend(
        [
            "",
            "These are direct optimization diagnostics on the historical Th-232 cache, not external validation.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_curves(
    output_path: Path,
    energy: np.ndarray,
    curve_rows: dict[int, tuple[np.ndarray, np.ndarray]],
    anchor_rows: list[dict[str, Any]],
    retention_curve: np.ndarray,
    retention_rows: list[dict[str, Any]],
) -> None:
    colors = dict(zip(TARGET_GAINS_PCT, plt.cm.viridis(np.linspace(0.1, 0.9, len(TARGET_GAINS_PCT)))))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": (1.25, 1.0)},
        constrained_layout=True,
    )
    threshold_axis, retention_axis = axes
    for target_pct, (energy_grid, threshold_grid) in curve_rows.items():
        color = colors[target_pct]
        threshold_axis.plot(
            energy_grid,
            threshold_grid,
            color=color,
            linewidth=2.0,
            label=f"+{target_pct}% P/B",
        )
        rows = [
            row
            for row in anchor_rows
            if row["target_pb_gain_percent"] == target_pct
        ]
        threshold_axis.scatter(
            [row["observed_centroid_kev"] for row in rows],
            [row["target_threshold"] for row in rows],
            color=color,
            edgecolor="black",
            linewidth=0.45,
            s=28,
            zorder=3,
        )

    threshold_axis.axvline(1460.830, color="0.35", linestyle="--", linewidth=1.0)
    threshold_axis.text(
        1460.830,
        threshold_axis.get_ylim()[1] * 0.96,
        "P/B domain limit",
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        color="0.25",
    )
    threshold_axis.set_xlim(220.0, 1500.0)
    threshold_axis.set_ylim(0.0, 1.0)
    threshold_axis.set_ylabel("DS-CNN score threshold T(E)")
    threshold_axis.set_title(
        "Revised Th-232 energy-dependent thresholds (P/B objective limited to 1460.8 keV)",
        fontweight="bold",
    )
    threshold_axis.grid(alpha=0.25)
    threshold_axis.legend(ncol=3, fontsize=9, loc="upper right")

    retention_axis.plot(
        SCORE_GRID,
        retention_curve,
        color="black",
        linewidth=1.8,
        label="2614.5-keV ROI retention",
    )
    for row in retention_rows:
        color = colors[int(row["target_pb_gain_percent"])]
        threshold = float(row["boundary_threshold"])
        retention = float(row["retention_at_boundary"])
        retention_axis.axvline(threshold, color=color, linestyle="--", linewidth=0.9)
        retention_axis.scatter([threshold], [retention], color=color, s=30, zorder=3)
        retention_axis.annotate(
            f"+{int(row['target_pb_gain_percent'])}%",
            (threshold, retention),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    retention_axis.set_xlim(0.0, 0.90)
    retention_axis.set_ylim(0.0, 1.05)
    retention_axis.set_xlabel("Global score threshold")
    retention_axis.set_ylabel("2614-keV ROI retention")
    retention_axis.set_title(
        "2614.5-keV retention-only diagnostic (not used for P/B fitting)",
        fontweight="bold",
    )
    retention_axis.grid(alpha=0.25)
    retention_axis.legend(fontsize=9, loc="upper right")

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.score_cache.resolve()

    with h5py.File(cache_path, "r") as handle:
        energy = np.asarray(handle["corrected_energy_kev"], dtype=np.float64)
        scores = np.asarray(handle["score"], dtype=np.float64)
        cache_metadata = {
            str(key): json_scalar(value) for key, value in handle.attrs.items()
        }

    base_histogram = np.histogram(energy, ENERGY_EDGES)[0]
    smoothed = gaussian_filter1d(base_histogram.astype(np.float64), 1.5)
    all_specs = [*PB_ANCHORS, RETENTION_ONLY_ANCHOR]
    fitted_anchors = [
        (spec, fit_anchor_window(base_histogram, spec.reference_kev, smoothed))
        for spec in all_specs
    ]
    pb_fitted_anchors = fitted_anchors[: len(PB_ANCHORS)]
    curves = build_peak_curves(energy, scores, pb_fitted_anchors, base_histogram)

    anchor_rows = [
        {
            "reference_energy_kev": float(spec.reference_kev),
            "label": spec.label,
            "observed_centroid_kev": float(window.centroid_kev),
            "sigma_kev": float(window.sigma_kev),
            "sideband_mode": spec.sideband_mode,
        }
        for spec, window in pb_fitted_anchors
    ]
    anchor_centroids = np.asarray(
        [row["observed_centroid_kev"] for row in anchor_rows], dtype=np.float64
    )
    order = np.argsort(anchor_centroids)
    sorted_centroids = anchor_centroids[order]

    threshold_rows: list[dict[str, Any]] = []
    curve_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    interpolators: dict[int, PchipInterpolator] = {}
    boundary_rows: list[dict[str, Any]] = []
    for target_pct in TARGET_GAINS_PCT:
        target_gain = 1.0 + target_pct / 100.0
        selected_points: list[tuple[float, float]] = []
        for spec, window in pb_fitted_anchors:
            threshold, gain, retention = get_peak_target(
                spec.reference_kev,
                target_gain,
                curves,
                SCORE_GRID,
            )
            base = line_aware_metrics(
                base_histogram,
                window,
                spec.sideband_mode,
            )
            threshold_rows.append(
                {
                    "target_pb_gain_percent": float(target_pct),
                    "reference_energy_kev": float(spec.reference_kev),
                    "observed_centroid_kev": float(window.centroid_kev),
                    "sideband_mode": spec.sideband_mode,
                    "target_threshold": float(threshold),
                    "achieved_pb_gain": float(gain),
                    "net_peak_retention": float(retention),
                    "base_peak_to_background": float(base["peak_to_background"]),
                }
            )
            selected_points.append((window.centroid_kev, threshold))
        selected_points.sort()
        point_e = np.asarray([point[0] for point in selected_points], dtype=np.float64)
        point_t = np.asarray([point[1] for point in selected_points], dtype=np.float64)
        interpolator = PchipInterpolator(point_e, point_t)
        interpolators[target_pct] = interpolator
        dense_e = np.linspace(float(point_e.min()), float(point_e.max()), 1000)
        curve_rows[target_pct] = (
            dense_e,
            np.clip(interpolator(dense_e), 0.0, 1.0),
        )

        retention_window = fitted_anchors[-1][1]
        retention_curve = build_retention_curve(energy, scores, retention_window)
        boundary_threshold = float(point_t[np.argmax(point_e)])
        boundary_retention = float(
            retention_curve[np.argmin(np.abs(SCORE_GRID - boundary_threshold))]
        )
        boundary_rows.append(
            {
                "target_pb_gain_percent": float(target_pct),
                "boundary_threshold": boundary_threshold,
                "retention_at_boundary": boundary_retention,
            }
        )

    retention_window = fitted_anchors[-1][1]
    retention_curve = build_retention_curve(energy, scores, retention_window)
    threshold_csv = output_dir / "th232_revised_pb_thresholds_by_target.csv"
    anchor_csv = output_dir / "th232_revised_pb_anchor_windows.csv"
    curve_csv = output_dir / "th232_revised_threshold_curves_1kev.csv"
    retention_csv = output_dir / "th232_2614_retention_diagnostic.csv"
    plot_path = output_dir / "th232_revised_pb_threshold_curves.png"

    write_csv(threshold_csv, threshold_rows)
    write_csv(anchor_csv, anchor_rows)
    write_csv(
        curve_csv,
        [
            {
                "energy_kev": float(energy_value),
                **{
                    f"threshold_plus_{target_pct}pct": float(
                        np.clip(
                            interpolators[target_pct](energy_value),
                            0.0,
                            1.0,
                        )
                    )
                    for target_pct in TARGET_GAINS_PCT
                },
            }
            for energy_value in np.arange(
                float(sorted_centroids.min()),
                float(sorted_centroids.max()) + 1.0,
                1.0,
            )
        ],
    )
    write_csv(
        retention_csv,
        [
            {
                "score_threshold": float(threshold),
                "th232_2614_roi_retention": float(retention),
            }
            for threshold, retention in zip(SCORE_GRID, retention_curve)
        ],
    )
    plot_curves(
        plot_path,
        energy,
        curve_rows,
        [
            {
                **row,
                "target_pb_gain_percent": float(target_pct),
                "target_threshold": float(
                    interpolators[target_pct](row["observed_centroid_kev"])
                ),
            }
            for target_pct in TARGET_GAINS_PCT
            for row in anchor_rows
        ],
        retention_curve,
        boundary_rows,
    )

    report = {
        "schema_version": 1,
        "status": "TH232_REVISED_PB_THRESHOLD_CURVES_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "score_cache": relative(cache_path),
        "score_cache_metadata": cache_metadata,
        "target_gains_percent": list(TARGET_GAINS_PCT),
        "pb_domain_kev": [float(sorted_centroids.min()), float(sorted_centroids.max())],
        "pb_anchor_specs": [spec.__dict__ for spec in PB_ANCHORS],
        "retention_only_anchor": RETENTION_ONLY_ANCHOR.__dict__,
        "anchor_windows": anchor_rows,
        "threshold_rows": threshold_rows,
        "retention_boundary_rows": boundary_rows,
        "artifacts": {
            "plot": relative(plot_path),
            "threshold_csv": relative(threshold_csv),
            "anchor_csv": relative(anchor_csv),
            "curve_csv": relative(curve_csv),
            "retention_csv": relative(retention_csv),
        },
        "claim_boundary": (
            "Thresholds are tuned directly on historical Th-232 score data; this is an optimization diagnostic, not external validation."
        ),
    }
    json_path = output_dir / "th232_revised_pb_threshold_curves_report.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        make_report(cache_path, output_dir, threshold_rows, boundary_rows, anchor_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "artifacts": [
                    plot_path.name,
                    threshold_csv.name,
                    anchor_csv.name,
                    curve_csv.name,
                    retention_csv.name,
                    json_path.name,
                    report_path.name,
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
