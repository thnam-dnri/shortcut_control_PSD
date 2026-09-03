#!/usr/bin/env python3
"""Add bootstrap uncertainty bars and simple fits to revised Th-232 thresholds.

The uncertainty bars are 95% percentile intervals from event-level bootstrap
resampling within each fitted peak ROI plus its sidebands. Peak windows remain
fixed at their no-filter fitted values. The threshold curve is compared with a
three-parameter exponential decay and a second-order polynomial; neither fit is
used to extend the P/B objective beyond the 1460.8-keV anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_th232_revised_pb_threshold_curves import (  # noqa: E402
    DEFAULT_SCORE_CACHE,
    ENERGY_CENTERS,
    ENERGY_EDGES,
    PB_ANCHORS,
    RETENTION_ONLY_ANCHOR,
    SCORE_GRID,
    build_peak_curves,
    build_retention_curve,
    fit_anchor_window,
    get_peak_target,
    json_scalar,
    line_aware_metrics,
    relative,
    write_csv,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409_uncertainty"
)
TARGET_GAINS_PCT = (5, 10, 20, 30, 45)
DEFAULT_BOOTSTRAP_REPLICATES = 200
DEFAULT_SEED = 20260824
BOOTSTRAP_CI = (2.5, 97.5)
FIT_ORIGIN_KEV = 238.0
FIT_SCALE_KEV = 1000.0


def interval_overlap(low: float, high: float) -> np.ndarray:
    """Return fractional 1-keV-bin overlap weights for an energy interval."""

    return np.maximum(
        0.0,
        np.minimum(ENERGY_EDGES[1:], high)
        - np.maximum(ENERGY_EDGES[:-1], low),
    )


def select_first_threshold(
    gains: np.ndarray,
    retentions: np.ndarray,
    target_gain: float,
) -> float:
    """Match the production first-finite-crossing threshold rule."""

    valid = np.isfinite(gains) & np.isfinite(retentions) & (retentions >= 0.0)
    if not np.any(valid):
        return float("nan")
    crossings = np.flatnonzero(valid & (gains >= target_gain))
    if crossings.size:
        return float(SCORE_GRID[int(crossings[0])])
    valid_indices = np.flatnonzero(valid)
    distances = np.abs(gains[valid_indices] - target_gain)
    nearest_distance = float(np.min(distances))
    tied = valid_indices[
        np.isclose(distances, nearest_distance, rtol=0.0, atol=1.0e-12)
    ]
    return float(SCORE_GRID[int(tied[np.argmax(retentions[tied])])])


def bootstrap_thresholds_for_anchor(
    energy: np.ndarray,
    scores: np.ndarray,
    window: Any,
    sideband_mode: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Bootstrap threshold points from the local ROI/sideband event sample."""

    local_mask = (
        np.isfinite(energy)
        & np.isfinite(scores)
        & (energy >= window.left_low_kev - 2.0)
        & (energy <= window.right_high_kev + 2.0)
    )
    local_energy = energy[local_mask]
    local_scores = scores[local_mask]
    if local_energy.size == 0:
        return np.full((replicates, len(TARGET_GAINS_PCT)), np.nan), 0

    local_bins = np.floor(local_energy).astype(np.int64)
    local_bins = np.clip(local_bins, 0, ENERGY_CENTERS.size - 1)
    score_indices = np.searchsorted(SCORE_GRID, local_scores, side="right") - 1
    score_indices = np.clip(score_indices, 0, SCORE_GRID.size - 1).astype(np.int64)

    roi_weights = interval_overlap(window.roi_low_kev, window.roi_high_kev)
    left_weights = interval_overlap(window.left_low_kev, window.left_high_kev)
    right_weights = interval_overlap(window.right_low_kev, window.right_high_kev)
    roi_width = window.roi_high_kev - window.roi_low_kev
    left_width = window.left_high_kev - window.left_low_kev
    right_width = window.right_high_kev - window.right_low_kev

    samples = np.full(
        (replicates, len(TARGET_GAINS_PCT)), np.nan, dtype=np.float64
    )
    event_count = local_energy.size
    for replicate in range(replicates):
        indices = rng.integers(0, event_count, size=event_count)
        bins = local_bins[indices]
        score_bins = score_indices[indices]

        exact_roi = np.bincount(
            score_bins,
            weights=roi_weights[bins],
            minlength=SCORE_GRID.size,
        )
        exact_left = np.bincount(
            score_bins,
            weights=left_weights[bins],
            minlength=SCORE_GRID.size,
        )
        exact_right = np.bincount(
            score_bins,
            weights=right_weights[bins],
            minlength=SCORE_GRID.size,
        )
        exact_left_energy = np.bincount(
            score_bins,
            weights=left_weights[bins] * ENERGY_CENTERS[bins],
            minlength=SCORE_GRID.size,
        )
        exact_right_energy = np.bincount(
            score_bins,
            weights=right_weights[bins] * ENERGY_CENTERS[bins],
            minlength=SCORE_GRID.size,
        )

        selected_roi = np.cumsum(exact_roi[::-1])[::-1]
        selected_left = np.cumsum(exact_left[::-1])[::-1]
        selected_right = np.cumsum(exact_right[::-1])[::-1]
        selected_left_energy = np.cumsum(exact_left_energy[::-1])[::-1]
        selected_right_energy = np.cumsum(exact_right_energy[::-1])[::-1]

        base_roi = float(np.sum(roi_weights[bins]))
        base_left = float(np.sum(left_weights[bins]))
        base_right = float(np.sum(right_weights[bins]))
        base_left_energy = float(np.sum(left_weights[bins] * ENERGY_CENTERS[bins]))
        base_right_energy = float(
            np.sum(right_weights[bins] * ENERGY_CENTERS[bins])
        )

        def metrics(
            roi: np.ndarray | float,
            left: np.ndarray | float,
            right: np.ndarray | float,
            left_energy: np.ndarray | float,
            right_energy: np.ndarray | float,
        ) -> tuple[np.ndarray, np.ndarray]:
            roi_array = np.asarray(roi, dtype=np.float64)
            left_array = np.asarray(left, dtype=np.float64)
            right_array = np.asarray(right, dtype=np.float64)
            left_energy_array = np.asarray(left_energy, dtype=np.float64)
            right_energy_array = np.asarray(right_energy, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                if sideband_mode == "higher_only":
                    background_density = right_array / right_width
                elif sideband_mode == "both":
                    left_center = np.divide(
                        left_energy_array,
                        left_array,
                        out=np.full_like(left_array, 0.5 * (window.left_low_kev + window.left_high_kev)),
                        where=left_array > 0.0,
                    )
                    right_center = np.divide(
                        right_energy_array,
                        right_array,
                        out=np.full_like(right_array, 0.5 * (window.right_low_kev + window.right_high_kev)),
                        where=right_array > 0.0,
                    )
                    fraction = (window.centroid_kev - left_center) / (
                        right_center - left_center
                    )
                    background_density = left_array / left_width + fraction * (
                        right_array / right_width - left_array / left_width
                    )
                else:
                    raise ValueError(f"Unsupported sideband mode: {sideband_mode}")
                background = background_density * roi_width
                net = roi_array - background
                peak_to_background = net / background
            return net, peak_to_background

        base_net_array, base_pb_array = metrics(
            base_roi,
            base_left,
            base_right,
            base_left_energy,
            base_right_energy,
        )
        base_net = float(np.asarray(base_net_array))
        base_pb = float(np.asarray(base_pb_array))
        if not np.isfinite(base_net) or not np.isfinite(base_pb) or base_net <= 0.0:
            continue

        net, peak_to_background = metrics(
            selected_roi,
            selected_left,
            selected_right,
            selected_left_energy,
            selected_right_energy,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            gains = peak_to_background / base_pb
            retentions = net / base_net

        for target_index, target_pct in enumerate(TARGET_GAINS_PCT):
            samples[replicate, target_index] = select_first_threshold(
                gains,
                retentions,
                1.0 + target_pct / 100.0,
            )

    return samples, int(event_count)


def exponential_decay(
    energy_kev: np.ndarray | float,
    asymptote: float,
    amplitude: float,
    tau_kev: float,
    origin_kev: float = FIT_ORIGIN_KEV,
) -> np.ndarray:
    energy = np.asarray(energy_kev, dtype=np.float64)
    return asymptote + amplitude * np.exp(-np.maximum(energy - origin_kev, 0.0) / tau_kev)


def quadratic_scaled(
    energy_kev: np.ndarray | float,
    coefficient_x2: float,
    coefficient_x: float,
    intercept: float,
    origin_kev: float = FIT_ORIGIN_KEV,
    scale_kev: float = FIT_SCALE_KEV,
) -> np.ndarray:
    x = (np.asarray(energy_kev, dtype=np.float64) - origin_kev) / scale_kev
    return coefficient_x2 * x * x + coefficient_x * x + intercept


def fit_models(
    energy: np.ndarray,
    threshold: np.ndarray,
    sigma: np.ndarray,
    target_pct: int,
) -> tuple[list[dict[str, Any]], dict[str, Callable[[np.ndarray], np.ndarray]]]:
    safe_sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 0.001)
    rows: list[dict[str, Any]] = []
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    exp_origin = float(np.min(energy))

    def exponential_for_fit(
        values: np.ndarray,
        asymptote: float,
        amplitude: float,
        tau_kev: float,
    ) -> np.ndarray:
        return exponential_decay(values, asymptote, amplitude, tau_kev, exp_origin)

    try:
        initial_asymptote = float(np.clip(threshold[-1], 0.0, 1.0))
        initial_amplitude = max(float(threshold[0] - initial_asymptote), 0.05)
        parameters, _ = curve_fit(
            exponential_for_fit,
            energy,
            threshold,
            p0=(initial_asymptote, initial_amplitude, 400.0),
            sigma=safe_sigma,
            absolute_sigma=True,
            bounds=((0.0, 0.0, 10.0), (1.0, 2.0, 10000.0)),
            maxfev=100000,
        )
        exp_prediction = exponential_for_fit(energy, *parameters)
        predictors["exponential_decay"] = lambda values: exponential_for_fit(
            values, *parameters
        )
        rows.append(
            fit_diagnostics(
                target_pct,
                "exponential_decay",
                threshold,
                exp_prediction,
                safe_sigma,
                {
                    "asymptote": float(parameters[0]),
                    "amplitude": float(parameters[1]),
                    "tau_kev": float(parameters[2]),
                    "origin_kev": exp_origin,
                },
            )
        )
    except (RuntimeError, ValueError):
        rows.append(
            failed_fit_row(target_pct, "exponential_decay", "curve_fit_failed")
        )

    x = (energy - FIT_ORIGIN_KEV) / FIT_SCALE_KEV
    polynomial_parameters = np.polyfit(x, threshold, 2, w=1.0 / safe_sigma)
    polynomial_prediction = np.polyval(polynomial_parameters, x)
    predictors["quadratic"] = lambda values: quadratic_scaled(
        values,
        float(polynomial_parameters[0]),
        float(polynomial_parameters[1]),
        float(polynomial_parameters[2]),
        FIT_ORIGIN_KEV,
        FIT_SCALE_KEV,
    )
    rows.append(
        fit_diagnostics(
            target_pct,
            "quadratic",
            threshold,
            polynomial_prediction,
            safe_sigma,
            {
                "coefficient_x2": float(polynomial_parameters[0]),
                "coefficient_x": float(polynomial_parameters[1]),
                "intercept": float(polynomial_parameters[2]),
                "origin_kev": FIT_ORIGIN_KEV,
                "scale_kev": FIT_SCALE_KEV,
            },
        )
    )
    return rows, predictors


def failed_fit_row(target_pct: int, model: str, reason: str) -> dict[str, Any]:
    return {
        "target_pb_gain_percent": float(target_pct),
        "model": model,
        "status": "FAIL",
        "failure_reason": reason,
        "rmse": float("nan"),
        "weighted_rmse": float("nan"),
        "mae": float("nan"),
        "max_absolute_error": float("nan"),
        "chi2": float("nan"),
        "aic": float("nan"),
        "bic": float("nan"),
        "parameter_json": "{}",
    }


def fit_diagnostics(
    target_pct: int,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
    parameters: dict[str, float],
) -> dict[str, Any]:
    residual = observed - predicted
    standardized = residual / sigma
    chi2 = float(np.sum(np.square(standardized)))
    parameter_count = 3
    sample_count = observed.size
    return {
        "target_pb_gain_percent": float(target_pct),
        "model": model,
        "status": "PASS",
        "failure_reason": "",
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "weighted_rmse": float(np.sqrt(np.mean(np.square(standardized)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "chi2": chi2,
        "aic": float(chi2 + 2.0 * parameter_count),
        "bic": float(chi2 + parameter_count * np.log(sample_count)),
        "parameter_json": json.dumps(parameters, sort_keys=True),
    }


def plot_results(
    output_path: Path,
    point_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    predictors_by_target: dict[int, dict[str, Callable[[np.ndarray], np.ndarray]]],
    retention_curve: np.ndarray,
    retention_rows: list[dict[str, Any]],
) -> None:
    colors = dict(
        zip(
            TARGET_GAINS_PCT,
            plt.cm.viridis(np.linspace(0.1, 0.9, len(TARGET_GAINS_PCT))),
        )
    )
    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    retention_axis = figure.add_subplot(grid[1, :])

    model_titles = {
        "exponential_decay": "Simple exponential decay",
        "quadratic": "Second-order polynomial",
    }
    dense_energy = np.linspace(220.0, 1500.0, 1200)
    for axis, model in zip(axes, ("exponential_decay", "quadratic")):
        for target_pct in TARGET_GAINS_PCT:
            color = colors[target_pct]
            rows = [
                row
                for row in point_rows
                if int(row["target_pb_gain_percent"]) == target_pct
            ]
            rows.sort(key=lambda row: row["observed_centroid_kev"])
            energies = np.asarray([row["observed_centroid_kev"] for row in rows])
            thresholds = np.asarray([row["target_threshold"] for row in rows])
            lower = np.asarray([row["bootstrap_ci_low"] for row in rows])
            upper = np.asarray([row["bootstrap_ci_high"] for row in rows])
            axis.errorbar(
                energies,
                thresholds,
                yerr=np.vstack((thresholds - lower, upper - thresholds)),
                fmt="o",
                markersize=4.5,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.5,
                alpha=0.75,
                label="_nolegend_",
                zorder=3,
            )
            prediction = predictors_by_target[target_pct][model](dense_energy)
            axis.plot(
                dense_energy,
                np.clip(prediction, 0.0, 1.0),
                color=color,
                linewidth=2.0,
                label=f"+{target_pct}% P/B",
            )
        axis.axvline(1460.830, color="0.35", linestyle="--", linewidth=1.0)
        axis.set_xlim(220.0, 1500.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Corrected energy (keV)")
        axis.set_ylabel("DS-CNN score threshold T(E)")
        axis.set_title(model_titles[model], fontweight="bold")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8, loc="upper right")

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
    figure.suptitle(
        "Th-232 threshold points with 95% bootstrap intervals and simple fits",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def make_report(
    cache_path: Path,
    output_dir: Path,
    point_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> str:
    lines = [
        "# Th-232 threshold uncertainty and simple fits",
        "",
        "P/B anchor points are used from 209.3 through 1460.8 keV. The 2614.5-keV line remains retention-only.",
        "",
        f"- Score cache: `{relative(cache_path)}`",
        f"- Plot: `{relative(output_dir / 'th232_threshold_error_bars_simple_fits.png')}`",
        f"- Bootstrap replicates: {replicates} (seed {seed})",
        "- Error bars: 95% percentile intervals from local event-level bootstrap resampling; fitted peak windows are fixed.",
        "- The uncertainty bars represent finite-event statistical uncertainty only; calibration, sideband-model, and source-systematic uncertainties are not included.",
        "",
        "## Fit definitions",
        "",
        "- Exponential: `T(E) = c + a exp(-(E - E0)/tau)` with positive `a` and `tau`.",
        "- Quadratic: `T(E) = a2 x^2 + a1 x + c`, where `x = (E - 238 keV)/1000 keV`.",
        "- Fits are descriptive within the 209.3--1460.4-keV observed-centroid domain; they are not used to extrapolate to 2614 keV.",
        "",
        "## Fit comparison",
        "",
        "| Target | Model | RMSE | Weighted RMSE | AIC | BIC |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for row in fit_rows:
        if row["status"] != "PASS":
            lines.append(
                f"| +{row['target_pb_gain_percent']:.0f}% | {row['model']} | failed | failed | failed | failed |"
            )
        else:
            lines.append(
                f"| +{row['target_pb_gain_percent']:.0f}% | {row['model']} | {row['rmse']:.4f} | {row['weighted_rmse']:.2f} | {row['aic']:.2f} | {row['bic']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 2614-keV retention-only boundary values",
            "",
            "| Target | Threshold at 1460.8 keV | 2614-keV retention |",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_replicates < 20:
        raise ValueError("Use at least 20 bootstrap replicates for percentile intervals")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.score_cache.resolve()
    rng = np.random.default_rng(args.seed)

    with h5py.File(cache_path, "r") as handle:
        energy = np.asarray(handle["corrected_energy_kev"], dtype=np.float64)
        scores = np.asarray(handle["score"], dtype=np.float64)
        cache_metadata = {
            str(key): json_scalar(value) for key, value in handle.attrs.items()
        }

    base_histogram = np.histogram(energy, ENERGY_EDGES)[0]
    smoothed = np.asarray(base_histogram, dtype=np.float64)
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(smoothed, 1.5)
    fitted_anchors = [
        (spec, fit_anchor_window(base_histogram, spec.reference_kev, smoothed))
        for spec in [*PB_ANCHORS, RETENTION_ONLY_ANCHOR]
    ]
    pb_fitted_anchors = fitted_anchors[: len(PB_ANCHORS)]
    curves = build_peak_curves(energy, scores, pb_fitted_anchors, base_histogram)

    point_rows: list[dict[str, Any]] = []
    predictors_by_target: dict[int, dict[str, Callable[[np.ndarray], np.ndarray]]] = {}
    fit_rows: list[dict[str, Any]] = []
    dense_energy = np.arange(
        float(min(window.centroid_kev for _, window in pb_fitted_anchors)),
        float(max(window.centroid_kev for _, window in pb_fitted_anchors)) + 1.0,
        1.0,
    )
    dense_rows: list[dict[str, Any]] = [
        {"energy_kev": float(value)} for value in dense_energy
    ]

    for target_pct in TARGET_GAINS_PCT:
        target_point_rows: list[dict[str, Any]] = []
        for spec, window in pb_fitted_anchors:
            threshold, achieved_gain, retention = get_peak_target(
                spec.reference_kev,
                1.0 + target_pct / 100.0,
                curves,
                SCORE_GRID,
            )
            bootstrap_samples, local_event_count = bootstrap_thresholds_for_anchor(
                energy,
                scores,
                window,
                spec.sideband_mode,
                args.bootstrap_replicates,
                rng,
            )
            target_index = TARGET_GAINS_PCT.index(target_pct)
            samples = bootstrap_samples[:, target_index]
            finite_samples = samples[np.isfinite(samples)]
            if finite_samples.size:
                ci_low, ci_high = np.percentile(finite_samples, BOOTSTRAP_CI)
                bootstrap_median = float(np.median(finite_samples))
            else:
                ci_low = ci_high = bootstrap_median = float("nan")
            base = line_aware_metrics(base_histogram, window, spec.sideband_mode)
            row = {
                "target_pb_gain_percent": float(target_pct),
                "reference_energy_kev": float(spec.reference_kev),
                "label": spec.label,
                "observed_centroid_kev": float(window.centroid_kev),
                "sideband_mode": spec.sideband_mode,
                "target_threshold": float(threshold),
                "achieved_pb_gain": float(achieved_gain),
                "net_peak_retention": float(retention),
                "base_peak_to_background": float(base["peak_to_background"]),
                "local_event_count": int(local_event_count),
                "bootstrap_valid_replicates": int(finite_samples.size),
                "bootstrap_median": bootstrap_median,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
            }
            point_rows.append(row)
            target_point_rows.append(row)

        target_point_rows.sort(key=lambda row: row["observed_centroid_kev"])
        fit_energy = np.asarray(
            [row["observed_centroid_kev"] for row in target_point_rows],
            dtype=np.float64,
        )
        fit_threshold = np.asarray(
            [row["target_threshold"] for row in target_point_rows],
            dtype=np.float64,
        )
        fit_sigma = np.asarray(
            [
                max(
                    (row["bootstrap_ci_high"] - row["bootstrap_ci_low"]) / 2.0,
                    0.001,
                )
                for row in target_point_rows
            ],
            dtype=np.float64,
        )
        target_fit_rows, predictors = fit_models(
            fit_energy,
            fit_threshold,
            fit_sigma,
            target_pct,
        )
        fit_rows.extend(target_fit_rows)
        predictors_by_target[target_pct] = predictors
        for model in ("exponential_decay", "quadratic"):
            prediction = predictors[model](dense_energy)
            for row, value in zip(dense_rows, prediction):
                row[f"{model}_plus_{target_pct}pct"] = float(value)

    retention_window = fitted_anchors[-1][1]
    retention_curve = build_retention_curve(energy, scores, retention_window)
    boundary_rows = []
    for target_pct in TARGET_GAINS_PCT:
        boundary_row = max(
            (
                row
                for row in point_rows
                if int(row["target_pb_gain_percent"]) == target_pct
            ),
            key=lambda row: row["observed_centroid_kev"],
        )
        boundary_threshold = float(boundary_row["target_threshold"])
        retention = float(
            retention_curve[np.argmin(np.abs(SCORE_GRID - boundary_threshold))]
        )
        boundary_rows.append(
            {
                "target_pb_gain_percent": float(target_pct),
                "boundary_threshold": boundary_threshold,
                "retention_at_boundary": retention,
            }
        )

    anchor_csv = output_dir / "th232_threshold_anchor_uncertainty.csv"
    fit_csv = output_dir / "th232_threshold_fit_comparison.csv"
    curve_csv = output_dir / "th232_threshold_simple_fits_1kev.csv"
    retention_csv = output_dir / "th232_2614_retention_diagnostic.csv"
    plot_path = output_dir / "th232_threshold_error_bars_simple_fits.png"
    report_json_path = output_dir / "th232_threshold_error_bars_simple_fits_report.json"
    report_path = output_dir / "report.md"

    write_csv(anchor_csv, point_rows)
    write_csv(fit_csv, fit_rows)
    write_csv(curve_csv, dense_rows)
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
    plot_results(
        plot_path,
        point_rows,
        fit_rows,
        predictors_by_target,
        retention_curve,
        boundary_rows,
    )

    report = {
        "schema_version": 1,
        "status": "TH232_THRESHOLD_ERROR_BARS_SIMPLE_FITS_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "score_cache": relative(cache_path),
        "score_cache_metadata": cache_metadata,
        "target_gains_percent": list(TARGET_GAINS_PCT),
        "pb_domain_kev": [
            float(min(window.centroid_kev for _, window in pb_fitted_anchors)),
            float(max(window.centroid_kev for _, window in pb_fitted_anchors)),
        ],
        "bootstrap": {
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "confidence_interval_percent": list(BOOTSTRAP_CI),
            "resampling_scope": "events in each fitted ROI plus sidebands",
            "peak_windows_fixed": True,
            "systematics_included": False,
        },
        "fit_definitions": {
            "exponential_decay": "c + a * exp(-(E - E0) / tau), with a >= 0 and tau > 0",
            "quadratic": "a2 * x^2 + a1 * x + c, x = (E - 238 keV) / 1000 keV",
        },
        "point_rows": point_rows,
        "fit_rows": fit_rows,
        "retention_boundary_rows": boundary_rows,
        "artifacts": {
            "plot": relative(plot_path),
            "anchor_csv": relative(anchor_csv),
            "fit_csv": relative(fit_csv),
            "curve_csv": relative(curve_csv),
            "retention_csv": relative(retention_csv),
        },
        "claim_boundary": "Bootstrap intervals quantify finite-event uncertainty on direct historical Th-232 optimization points; they are not external validation or systematic uncertainty intervals.",
    }
    report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        make_report(
            cache_path,
            output_dir,
            point_rows,
            fit_rows,
            boundary_rows,
            args.bootstrap_replicates,
            args.seed,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "artifacts": [
                    plot_path.name,
                    anchor_csv.name,
                    fit_csv.name,
                    curve_csv.name,
                    retention_csv.name,
                    report_json_path.name,
                    report_path.name,
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
