#!/usr/bin/env python3
"""Optimize one DS-CNN score threshold per audited Th-232 spectral line.

The primary objective is a relative, spectrum-conditioned detection limit.  It
uses Poisson variance from the peak ROI and the scaled sideband estimator, a
95% false-positive/95% detection-probability construction, and explicitly
divides by retained net-photopeak efficiency.  It is not an absolute MDA in Bq.
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
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_EDGES,
    PeakWindow,
)
from scripts.optimize_th232_usable_peak_global_threshold import (  # noqa: E402
    MINIMUM_BACKGROUND_COUNTS,
    MINIMUM_NET_PEAK_COUNTS,
)
from scripts.plot_th232_revised_pb_threshold_curves import (  # noqa: E402
    PB_ANCHORS,
    fit_anchor_window,
)

DEFAULT_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_peak_relative_detection_limit_20260824"
)
SCORE_BIN_COUNT = 2000
DEFAULT_BOOTSTRAP_REPLICATES = 500
DEFAULT_BOOTSTRAP_SEED = 20260824
GLOBAL_COMPARISON_THRESHOLDS = (0.2660, 0.3880)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reverse_cumulative(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values[..., ::-1], axis=-1)[..., ::-1]


def score_category_counts(
    energy: np.ndarray,
    scores: np.ndarray,
    source_index: np.ndarray,
    source_count: int,
    window: PeakWindow,
) -> dict[str, np.ndarray]:
    """Return per-source score histograms for ROI and both sidebands."""

    score_bin = np.searchsorted(
        np.linspace(0.0, 1.0, SCORE_BIN_COUNT + 1), scores, side="right"
    ) - 1
    score_bin = np.clip(score_bin, 0, SCORE_BIN_COUNT - 1)
    flat_size = source_count * SCORE_BIN_COUNT
    flat_index = source_index.astype(np.int64) * SCORE_BIN_COUNT + score_bin

    categories: dict[str, np.ndarray] = {}
    intervals = {
        "roi": (window.roi_low_kev, window.roi_high_kev),
        "left": (window.left_low_kev, window.left_high_kev),
        "right": (window.right_low_kev, window.right_high_kev),
    }
    for name, (low, high) in intervals.items():
        selected = (energy >= low) & (energy < high)
        counts = np.bincount(flat_index[selected], minlength=flat_size)
        categories[name] = counts.reshape(source_count, SCORE_BIN_COUNT).astype(
            np.float64
        )
        if name != "roi":
            energy_sum = np.bincount(
                flat_index[selected], weights=energy[selected], minlength=flat_size
            )
            categories[f"{name}_energy"] = energy_sum.reshape(
                source_count, SCORE_BIN_COUNT
            )
    return categories


def detection_limit_counts(
    background: np.ndarray,
    background_estimator_variance: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return critical and detection levels for ROI-minus-sideband net counts.

    The detection level solves
    ``Ld = Lc + k_beta * sqrt(Ld + B + Var(B_hat))``.
    """

    variance_null = np.maximum(background + background_estimator_variance, 0.0)
    k_alpha = float(norm.ppf(1.0 - alpha))
    k_beta = float(norm.ppf(1.0 - beta))
    critical = k_alpha * np.sqrt(variance_null)
    root = 0.5 * (
        k_beta
        + np.sqrt(k_beta**2 + 4.0 * (variance_null + critical))
    )
    detection = critical + k_beta * root
    return critical, detection


def metric_curve(
    category_histograms: dict[str, np.ndarray],
    window: PeakWindow,
    sideband_mode: str,
    source_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Calculate threshold curves, optionally for a bootstrap source resample."""

    if source_weights is None:
        aggregate = {key: value.sum(axis=0) for key, value in category_histograms.items()}
    else:
        aggregate = {
            key: np.tensordot(source_weights, value, axes=(0, 0))
            for key, value in category_histograms.items()
        }
    cumulative = {key: reverse_cumulative(value) for key, value in aggregate.items()}
    roi = cumulative["roi"]
    left = cumulative["left"]
    right = cumulative["right"]
    left_energy = cumulative["left_energy"]
    right_energy = cumulative["right_energy"]
    roi_width = window.roi_high_kev - window.roi_low_kev
    left_width = window.left_high_kev - window.left_low_kev
    right_width = window.right_high_kev - window.right_low_kev

    with np.errstate(divide="ignore", invalid="ignore"):
        if sideband_mode == "higher_only":
            right_coefficient = np.full_like(right, roi_width / right_width)
            left_coefficient = np.zeros_like(left)
        elif sideband_mode == "both":
            left_center = np.divide(
                left_energy,
                left,
                out=np.full_like(left, 0.5 * (window.left_low_kev + window.left_high_kev)),
                where=left > 0.0,
            )
            right_center = np.divide(
                right_energy,
                right,
                out=np.full_like(
                    right, 0.5 * (window.right_low_kev + window.right_high_kev)
                ),
                where=right > 0.0,
            )
            fraction = np.divide(
                window.centroid_kev - left_center,
                right_center - left_center,
                out=np.full_like(left_center, 0.5),
                where=right_center != left_center,
            )
            left_coefficient = roi_width * (1.0 - fraction) / left_width
            right_coefficient = roi_width * fraction / right_width
        else:
            raise ValueError(f"Unsupported sideband mode: {sideband_mode}")

        background = left_coefficient * left + right_coefficient * right
        background_variance = (
            np.square(left_coefficient) * left
            + np.square(right_coefficient) * right
        )
        net_peak = roi - background
        peak_to_background = net_peak / background
        raw_net_peak_retention = net_peak / net_peak[0]
        signal_efficiency = np.clip(raw_net_peak_retention, 0.0, 1.0)
        critical, detection = detection_limit_counts(background, background_variance)
        relative_detection_limit = detection / (signal_efficiency * detection[0])
        variance_aware_background_approximation = np.sqrt(
            (background + background_variance)
            / (background[0] + background_variance[0])
        ) / signal_efficiency

    reliable = (
        np.isfinite(relative_detection_limit)
        & np.isfinite(peak_to_background)
        & (background >= MINIMUM_BACKGROUND_COUNTS)
        & (net_peak >= MINIMUM_NET_PEAK_COUNTS)
        & (signal_efficiency > 0.0)
    )
    return {
        "roi_counts": roi,
        "estimated_background_counts": background,
        "background_estimator_variance": background_variance,
        "net_peak_counts": net_peak,
        "peak_to_background": peak_to_background,
        "raw_net_peak_retention": raw_net_peak_retention,
        "signal_efficiency": signal_efficiency,
        "critical_level_counts": critical,
        "detection_limit_counts": detection,
        "relative_detection_limit": relative_detection_limit,
        "background_limited_relative_detection_limit": variance_aware_background_approximation,
        "reliable": reliable,
    }


def select_optimum(curve: dict[str, np.ndarray]) -> int:
    eligible = np.flatnonzero(curve["reliable"])
    if eligible.size == 0:
        raise ValueError("No statistically reliable threshold exists")
    objective = curve["relative_detection_limit"][eligible]
    return int(eligible[int(np.nanargmin(objective))])


def bootstrap_optimum(
    category_histograms: dict[str, np.ndarray],
    window: PeakWindow,
    sideband_mode: str,
    selected_index: int,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    source_count = next(iter(category_histograms.values())).shape[0]
    selected_thresholds: list[float] = []
    fixed_relative_limits: list[float] = []
    thresholds = np.arange(SCORE_BIN_COUNT, dtype=np.float64) / SCORE_BIN_COUNT
    for _ in range(replicates):
        sampled = rng.integers(0, source_count, size=source_count)
        weights = np.bincount(sampled, minlength=source_count).astype(np.float64)
        curve = metric_curve(category_histograms, window, sideband_mode, weights)
        try:
            optimum = select_optimum(curve)
        except ValueError:
            continue
        selected_thresholds.append(float(thresholds[optimum]))
        fixed_value = float(curve["relative_detection_limit"][selected_index])
        if np.isfinite(fixed_value):
            fixed_relative_limits.append(fixed_value)
    if not selected_thresholds or not fixed_relative_limits:
        raise ValueError("Bootstrap produced no finite relative detection limits")
    return {
        "valid_replicates": len(selected_thresholds),
        "threshold_median": float(np.median(selected_thresholds)),
        "threshold_ci95_low": float(np.percentile(selected_thresholds, 2.5)),
        "threshold_ci95_high": float(np.percentile(selected_thresholds, 97.5)),
        "fixed_threshold_relative_detection_limit_median": float(
            np.median(fixed_relative_limits)
        ),
        "fixed_threshold_relative_detection_limit_ci95_low": float(
            np.percentile(fixed_relative_limits, 2.5)
        ),
        "fixed_threshold_relative_detection_limit_ci95_high": float(
            np.percentile(fixed_relative_limits, 97.5)
        ),
    }


def value_at(curve: dict[str, np.ndarray], threshold: float, key: str) -> float:
    index = int(round(threshold * SCORE_BIN_COUNT))
    index = min(max(index, 0), SCORE_BIN_COUNT - 1)
    return float(curve[key][index])


def plot_curves(
    output_dir: Path,
    results: list[dict[str, Any]],
    curves: dict[float, dict[str, np.ndarray]],
) -> list[Path]:
    thresholds = np.arange(SCORE_BIN_COUNT, dtype=np.float64) / SCORE_BIN_COUNT
    figure, axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)
    for axis, result in zip(axes.flat, results):
        curve = curves[result["reference_energy_kev"]]
        valid = curve["reliable"] & (thresholds <= 0.75)
        axis.plot(
            thresholds[valid],
            curve["relative_detection_limit"][valid],
            color="tab:blue",
            label="finite-count relative DL",
        )
        axis.plot(
            thresholds[valid],
            curve["background_limited_relative_detection_limit"][valid],
            color="0.45",
            linestyle="--",
            label="background-limited approximation",
        )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.axvline(
            result["optimal_threshold"], color="tab:red", linestyle=":", label="minimum"
        )
        axis.set_title(result["label"])
        axis.set_xlabel("DS-CNN score threshold")
        axis.set_ylabel("Relative detection limit (lower is better)")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Per-peak Th-232 spectrum-conditioned relative detection limits")
    curve_path = output_dir / "relative_detection_limit_curves.png"
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    energy = np.asarray([row["reference_energy_kev"] for row in results])
    threshold = np.asarray([row["optimal_threshold"] for row in results])
    low = np.asarray([row["bootstrap_threshold_ci95_low"] for row in results])
    high = np.asarray([row["bootstrap_threshold_ci95_high"] for row in results])
    relative_limit = np.asarray([row["relative_detection_limit"] for row in results])
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].errorbar(
        energy,
        threshold,
        yerr=np.vstack((threshold - low, high - threshold)),
        fmt="o",
        linestyle="none",
        capsize=3,
    )
    axes[0].set_ylabel("Optimal score threshold")
    axes[1].plot(energy, relative_limit, marker="o")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Minimum relative detection limit")
    axes[1].set_xlabel("Peak energy (keV)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_title("Peak-specific score thresholds (file-bootstrap 95% intervals)")
    summary_path = output_dir / "optimal_thresholds_vs_energy.png"
    figure.savefig(summary_path, dpi=180)
    plt.close(figure)
    return [curve_path, summary_path]


def report_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Th-232 peak-specific relative detection-limit optimization",
        "",
        "## Decision",
        "",
        "Each audited spectral line is optimized independently by minimizing a finite-count relative detection limit. Lower values are better; 1.0 is the unfiltered spectrum. The objective includes both sideband-estimation variance and net-photopeak retention.",
        "",
        "| Energy (keV) | Anchor | Threshold | Relative DL | Change | Peak retained | P/B gain | Bootstrap supports improvement | Bootstrap threshold 95% interval |",
        "|---:|:---|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in results:
        lines.append(
            f"| {row['reference_energy_kev']:.3f} | {row['label']} | "
            f"{row['optimal_threshold']:.4f} | {row['relative_detection_limit']:.4f} | "
            f"{row['relative_detection_limit_change_percent']:+.2f}% | "
            f"{row['signal_efficiency']:.2%} | {row['pb_improvement_factor']:.4f}x | "
            f"{'yes' if row['bootstrap_supports_improvement'] else 'no'} | "
            f"[{row['bootstrap_threshold_ci95_low']:.4f}, {row['bootstrap_threshold_ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Statistical definition",
            "",
            "For net counts `N_ROI - B_hat`, the null variance is approximated as `B_hat + Var(B_hat)`, where `Var(B_hat)` is propagated from Poisson sideband counts and their ROI scaling/interpolation coefficients. With one-sided alpha=beta=0.05, the detection limit solves `Ld = Lc + k_beta sqrt(Ld + B_hat + Var(B_hat))`. The reported objective is `(Ld_cut / peak_retention) / Ld_no_cut`. Because a cut cannot physically retain more than all signal events, net-area retention estimates slightly above 1.0 are conservatively capped at 1.0 only in this denominator; the uncapped estimate remains in the CSV and JSON.",
            "",
            "The uncertainty intervals use a 30-file cluster bootstrap. They describe acquisition-segment stability and are not an external-validation interval.",
            "",
            "## Scope and claim boundary",
            "",
            "This is a relative, spectrum-conditioned detection limit, not an absolute minimum detectable activity in Bq. Absolute MDA additionally requires calibrated full-energy efficiency, emission probability, live time/dead-time correction, source geometry, and attenuation corrections. The 1460.8-keV anchor is K-40/background and is not a Th-232-chain MDA claim.",
            "",
            "Historical Th-232 events directly select these thresholds. Locked test and Eu-152 remain unopened and unused.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
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
        energy = np.asarray(handle["corrected_energy_kev"], dtype=np.float64)
        scores = np.asarray(handle["score"], dtype=np.float64)
        source_index = np.asarray(handle["source_file_index"], dtype=np.int64)
        source_count = int(handle["source_files/path"].shape[0])
        checkpoint_sha256 = str(handle.attrs["checkpoint_sha256"])
    if not (
        energy.size == scores.size == source_index.size
        and np.all(np.isfinite(energy))
        and np.all(np.isfinite(scores))
    ):
        raise ValueError("Invalid Th-232 score-cache arrays")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    baseline_histogram = np.histogram(energy, ENERGY_EDGES)[0].astype(np.float64)
    smoothed = gaussian_filter1d(baseline_histogram, 1.5)
    windows = [
        (spec, fit_anchor_window(baseline_histogram, spec.reference_kev, smoothed))
        for spec in PB_ANCHORS
    ]
    rng = np.random.default_rng(args.bootstrap_seed)
    thresholds = np.arange(SCORE_BIN_COUNT, dtype=np.float64) / SCORE_BIN_COUNT
    results: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    curves: dict[float, dict[str, np.ndarray]] = {}
    for spec, window in windows:
        categories = score_category_counts(
            energy, scores, source_index, source_count, window
        )
        curve = metric_curve(categories, window, spec.sideband_mode)
        curves[spec.reference_kev] = curve
        optimum = select_optimum(curve)
        bootstrap = bootstrap_optimum(
            categories,
            window,
            spec.sideband_mode,
            optimum,
            args.bootstrap_replicates,
            rng,
        )
        base_pb = float(curve["peak_to_background"][0])
        result = {
            "reference_energy_kev": spec.reference_kev,
            "label": spec.label,
            "sideband_mode": spec.sideband_mode,
            "optimal_threshold": float(thresholds[optimum]),
            "relative_detection_limit": float(curve["relative_detection_limit"][optimum]),
            "relative_detection_limit_change_percent": float(
                100.0 * (curve["relative_detection_limit"][optimum] - 1.0)
            ),
            "background_limited_relative_detection_limit": float(
                curve["background_limited_relative_detection_limit"][optimum]
            ),
            "raw_net_peak_retention": float(curve["raw_net_peak_retention"][optimum]),
            "signal_efficiency": float(curve["signal_efficiency"][optimum]),
            "background_retention": float(
                curve["estimated_background_counts"][optimum]
                / curve["estimated_background_counts"][0]
            ),
            "peak_to_background": float(curve["peak_to_background"][optimum]),
            "pb_improvement_factor": float(
                curve["peak_to_background"][optimum] / base_pb
            ),
            "critical_level_counts": float(curve["critical_level_counts"][optimum]),
            "detection_limit_counts": float(curve["detection_limit_counts"][optimum]),
            "no_cut_detection_limit_counts": float(curve["detection_limit_counts"][0]),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_valid_replicates": bootstrap["valid_replicates"],
            "bootstrap_threshold_median": bootstrap["threshold_median"],
            "bootstrap_threshold_ci95_low": bootstrap["threshold_ci95_low"],
            "bootstrap_threshold_ci95_high": bootstrap["threshold_ci95_high"],
            "bootstrap_fixed_threshold_relative_detection_limit_median": bootstrap[
                "fixed_threshold_relative_detection_limit_median"
            ],
            "bootstrap_fixed_threshold_relative_detection_limit_ci95_low": bootstrap[
                "fixed_threshold_relative_detection_limit_ci95_low"
            ],
            "bootstrap_fixed_threshold_relative_detection_limit_ci95_high": bootstrap[
                "fixed_threshold_relative_detection_limit_ci95_high"
            ],
            "bootstrap_supports_improvement": bool(
                bootstrap[
                    "fixed_threshold_relative_detection_limit_ci95_high"
                ]
                < 1.0
            ),
        }
        for threshold in GLOBAL_COMPARISON_THRESHOLDS:
            suffix = str(threshold).replace(".", "p")
            result[f"relative_detection_limit_at_{suffix}"] = value_at(
                curve, threshold, "relative_detection_limit"
            )
        results.append(result)
        for index in np.flatnonzero(curve["reliable"]):
            scan_rows.append(
                {
                    "reference_energy_kev": spec.reference_kev,
                    "label": spec.label,
                    "sideband_mode": spec.sideband_mode,
                    "threshold": float(thresholds[index]),
                    "relative_detection_limit": float(curve["relative_detection_limit"][index]),
                    "background_limited_relative_detection_limit": float(
                        curve["background_limited_relative_detection_limit"][index]
                    ),
                    "raw_net_peak_retention": float(
                        curve["raw_net_peak_retention"][index]
                    ),
                    "signal_efficiency": float(curve["signal_efficiency"][index]),
                    "background_retention": float(
                        curve["estimated_background_counts"][index]
                        / curve["estimated_background_counts"][0]
                    ),
                    "pb_improvement_factor": float(
                        curve["peak_to_background"][index] / base_pb
                    ),
                    "critical_level_counts": float(curve["critical_level_counts"][index]),
                    "detection_limit_counts": float(curve["detection_limit_counts"][index]),
                }
            )

    summary_csv = output_dir / "per_peak_optimal_thresholds.csv"
    scan_csv = output_dir / "relative_detection_limit_scan.csv"
    report_md = output_dir / "report.md"
    write_csv(summary_csv, results)
    write_csv(scan_csv, scan_rows)
    report_md.write_text(report_markdown(results), encoding="utf-8")
    plot_paths = plot_curves(output_dir, results, curves)

    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "TH232_PEAK_RELATIVE_DETECTION_LIMIT_OPTIMIZATION_COMPLETE",
        "score_cache": relative(score_cache),
        "score_cache_sha256": sha256_file(score_cache),
        "checkpoint_sha256": checkpoint_sha256,
        "admitted_events": int(energy.size),
        "source_file_count": source_count,
        "optimization": {
            "threshold_kind": "independent DS-CNN score threshold per audited peak",
            "score_bin_count": SCORE_BIN_COUNT,
            "objective": "minimum finite-count relative spectrum-conditioned detection limit",
            "alpha": 0.05,
            "beta": 0.05,
            "signal_efficiency": "net photopeak retention relative to no cut",
            "background_uncertainty": "Poisson propagation of scaled/interpolated sideband counts",
            "bootstrap_unit": "source acquisition file",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "absolute_mda_bq": False,
            "results": results,
        },
        "peak_windows": [
            {**asdict(window), "label": spec.label, "sideband_mode": spec.sideband_mode}
            for spec, window in windows
        ],
        "excluded_regions": [
            {"reference_energy_kev": 338.320, "reason": "nearby 328-keV line contaminates background estimation"},
            {"reference_energy_kev": 968.971, "reason": "not retained in the audited isolated-anchor set"},
            {"reference_energy_kev": 1592.0, "reason": "2614-keV double-escape region"},
            {"reference_energy_kev": 2103.5, "reason": "2614-keV single-escape region"},
            {"reference_energy_kev": 2614.533, "reason": "no supported higher-energy Compton-background reference"},
        ],
        "test_partition_used": False,
        "external_validation": False,
        "claim_boundary": (
            "Historical Th-232 events directly select the thresholds. Results are "
            "relative spectrum-conditioned detection limits, not absolute MDA in Bq."
        ),
        "artifacts": {},
    }
    for path in [summary_csv, scan_csv, report_md, *plot_paths]:
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
                "results": [
                    {
                        "energy_kev": row["reference_energy_kev"],
                        "threshold": row["optimal_threshold"],
                        "relative_detection_limit": row["relative_detection_limit"],
                    }
                    for row in results
                ],
                "report": relative(report_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
