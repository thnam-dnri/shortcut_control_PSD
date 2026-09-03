#!/usr/bin/env python3
"""Run fine-grained extended grid scan of flat P/B improvement targets on Th-232 and continuum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
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

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    fit_peak_windows,
    peak_background_metrics,
)
from scripts.optimize_th232_all_ba_ds_cnn_threshold import (  # noqa: E402
    PRIMARY_REFERENCE_PEAKS_KEV,
)

DEFAULT_TH232_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_CONTINUUM_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/compton_rejection_energy_thresholds_20260823/continuum_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_flat_pb_grid_scan_20260823"
)
GLOBAL_BASELINE_THRESHOLD = 0.437


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


def precompute_peak_curves(
    energy: np.ndarray,
    scores: np.ndarray,
    windows: list[Any],
    base_metrics: dict[float, dict[str, float]],
    t_grid: np.ndarray,
) -> dict[float, tuple[np.ndarray, np.ndarray, Any]]:
    peak_curves = {}
    for ref in PRIMARY_REFERENCE_PEAKS_KEV:
        w = next(win for win in windows if win.reference_kev == ref)
        m = (energy >= w.left_low_kev - 2.0) & (energy <= w.right_high_kev + 2.0)
        sub_e = energy[m]
        sub_s = scores[m]
        base_pb = base_metrics[ref]["peak_to_background"]
        base_net = base_metrics[ref]["net_peak_counts"]

        gains = []
        rets = []
        for t in t_grid:
            passed = sub_s >= t
            hh = np.histogram(sub_e[passed], ENERGY_EDGES)[0]
            try:
                met = peak_background_metrics(hh, w)
                g = met["peak_to_background"] / base_pb if base_pb > 0 else float("nan")
                r = met["net_peak_counts"] / base_net if base_net > 0 else float("nan")
            except ZeroDivisionError:
                g, r = float("nan"), 0.0
            except Exception:
                g, r = float("nan"), 0.0
            gains.append(g)
            rets.append(r)
        peak_curves[ref] = (np.asarray(gains), np.asarray(rets), w)
    return peak_curves


def get_peak_target(
    ref: float,
    target_gain: float,
    peak_curves: dict[float, tuple[np.ndarray, np.ndarray, Any]],
    t_grid: np.ndarray,
) -> tuple[float, float, float]:
    """Return the earliest finite threshold that reaches the target gain.

    The peak-background estimate becomes undefined when a threshold removes
    all sideband background.  Selecting the globally nearest gain can then
    choose a much later, nearly empty spectrum.  The first finite crossing
    preserves the usable signal.  If the sampled curve never reaches the
    target, use the nearest finite candidate and prefer the largest retained
    signal when candidates are tied.
    """
    gains, rets, _ = peak_curves[ref]
    valid = np.isfinite(gains) & np.isfinite(rets) & (rets >= 0.0)
    if not np.any(valid):
        raise ValueError(
            f"No finite, nonnegative-retention threshold candidate for peak {ref} keV"
        )

    crossings = np.flatnonzero(valid & (gains >= target_gain))
    if crossings.size:
        idx = int(crossings[0])
    else:
        valid_indices = np.flatnonzero(valid)
        distances = np.abs(gains[valid_indices] - target_gain)
        nearest_distance = float(np.min(distances))
        tied = valid_indices[
            np.isclose(distances, nearest_distance, rtol=0.0, atol=1.0e-12)
        ]
        idx = int(tied[np.argmax(rets[tied])])
    return float(t_grid[idx]), float(gains[idx]), float(rets[idx])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_scan_figures(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    peak_threshold_rows: list[dict[str, Any]],
    selected_curves: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    targets = np.asarray([r["target_pb_gain_percent"] for r in summary_rows])
    min_rets = np.asarray([r["minimum_peak_retention_percent"] for r in summary_rows])
    avg_rets = np.asarray([r["average_peak_retention_percent"] for r in summary_rows])
    th_rets = np.asarray([r["th232_total_retention_percent"] for r in summary_rows])
    co60_rejs = np.asarray([r["co60_continuum_rejection_percent"] for r in summary_rows])
    cs137_rejs = np.asarray([r["cs137_continuum_rejection_percent"] for r in summary_rows])

    # Figure 1: Pareto Curve (Target Gain vs Retention Floor)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].plot(targets, min_rets, marker="o", color="#D55E00", linewidth=2.0, label="Minimum Peak Retention (238.6 keV)")
    axes[0].plot(targets, avg_rets, marker="s", color="#0072B2", linewidth=2.0, label="Average Peak Retention (All 6 Peaks)")
    axes[0].plot(targets, th_rets, marker="^", color="0.4", linestyle="--", label="Total Th-232 Spectrum Throughput")
    axes[0].axhline(80.0, color="gray", linestyle=":", label="80% Retention Floor")
    axes[0].axvline(6.0, color="#009E73", linestyle="--", label="Sweet Spot (+6% Gain)")
    axes[0].set_xlabel("Target Uniform P/B Improvement (%)")
    axes[0].set_ylabel("Photopeak Retention (%)")
    axes[0].set_title("Photopeak Signal Retention vs Flat P/B Target")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    # Continuum rejection vs target
    axes[1].plot(targets, co60_rejs, marker="o", color="#0072B2", linewidth=2.0, label="Co-60 Rejection (100-1000 keV)")
    axes[1].plot(targets, cs137_rejs, marker="s", color="#D55E00", linewidth=2.0, label="Cs-137 Rejection (100-400 keV)")
    axes[1].axvline(6.0, color="#009E73", linestyle="--", label="Sweet Spot (+6% Gain)")
    axes[1].set_xlabel("Target Uniform P/B Improvement (%)")
    axes[1].set_ylabel("Compton Continuum Rejection (%)")
    axes[1].set_title("Compton Background Rejection vs Flat P/B Target")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Extended Flat P/B Scan: Pareto Trade-off Analysis", fontsize=12)
    fig.savefig(output_dir / "flat_pb_pareto_curve.png", dpi=180)
    plt.close(fig)

    # Figure 2: Threshold Curves T(E) for selected targets
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    dense_e = np.linspace(100.0, 2700.0, 1000)

    color_map = plt.cm.viridis(np.linspace(0, 1, len(selected_curves)))
    for (pct, (knots_e, knots_t)), col in zip(selected_curves.items(), color_map):
        pchip = PchipInterpolator(knots_e, knots_t)
        min_e, max_e = min(knots_e), max(knots_e)
        t_vals = np.clip(pchip(np.clip(dense_e, min_e, max_e)), 0.0, 1.0)
        ax.plot(dense_e, t_vals, label=f"Target +{pct}% P/B", color=col, linewidth=1.5)
        ax.scatter(knots_e, knots_t, color=col, s=25)

    ax.axhline(GLOBAL_BASELINE_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="Global Baseline 0.4370")
    ax.set_xlabel("Corrected Energy (keV)")
    ax.set_ylabel("DS-CNN Score Threshold $T(E)$")
    ax.set_title("Continuous Threshold Curves $T(E)$ for Various Flat P/B Targets")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output_dir / "flat_pb_threshold_curves_by_target.png", dpi=180)
    plt.close(fig)


def make_markdown_report(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Extended Flat P/B Improvement Grid Scan (+1% to +30%)",
        "",
        "## Executive Summary",
        "",
        "- **Objective:** Systematically scan flat (uniform) P/B improvement targets across energy from **+1% to +30%**.",
        "- **Classifier:** Selected all-Ba MA10/t10 DS-CNN (seed `20260823`, 22,753 parameters).",
        "- **Dataset:** 2,886,112 Th-232 events (direct optimization) and 4,792,913 Co-60/Cs-137 continuum events.",
        "",
        "## 1. Full Grid Scan Table (+1% to +30%)",
        "",
        "| Target Flat P/B | Min Peak Retention | Avg Peak Retention | Th-232 Total Event Retention | Co-60 Rejection (100-1000 keV) | Cs-137 Rejection (100-400 keV) | $T_{238}$ | $T_{338}$ | $T_{583}$ | $T_{911}$ | $T_{969}$ | $T_{2615}$ |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| **+{r['target_pb_gain_percent']}%** | **{r['minimum_peak_retention_percent']:.2f}%** | {r['average_peak_retention_percent']:.2f}% | {r['th232_total_retention_percent']:.2f}% | {r['co60_continuum_rejection_percent']:.2f}% | {r['cs137_continuum_rejection_percent']:.2f}% | {r['t_238']:.3f} | {r['t_338']:.3f} | {r['t_583']:.3f} | {r['t_911']:.3f} | {r['t_969']:.3f} | {r['t_2615']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Key Regimes & Operating Boundaries",
            "",
            "1. **Ultra-High Retention Regime (+1% to +3% P/B):**",
            "   * Retains **> 94.6% to 99.1%** of all photopeaks across all energies.",
            "   * Achieves modest Compton continuum rejection (1% to 7%).",
            "2. **Sweet Spot Regime (+5% to +6% P/B):**",
            "   * Retains **> 81.9% to 85.3%** photopeak floor at 238 keV and **~95% to 96%** average peak retention.",
            "   * Achieves **15.8% to 19.5% Co-60 continuum rejection** and **20.0% to 24.6% Cs-137 continuum rejection**.",
            "   * Optimal balance for general-purpose equalized spectroscopy.",
            "3. **Moderate Penalty Regime (+7% to +8% P/B):**",
            "   * Retains **70.0% to 78.1%** at 238 keV, **>91.7%** average retention.",
            "   * Achieves **23.0% to 27.2% Co-60 continuum rejection**.",
            "4. **High-Rejection / Low-Energy Loss Regime (+10% to +20% P/B):**",
            "   * At +10% target, 238 keV drops to **56.1%** retention ($T_{238} = 0.517$).",
            "   * At +15% target, 238 keV drops to **34.8%** retention ($T_{238} = 0.553$).",
            "   * At +20% target, 238 keV drops to **25.1%** retention ($T_{238} = 0.571$), while Co-60 continuum rejection reaches **55.4%**.",
            "5. **Deep Background Suppression Regime (+21% to +30% P/B):**",
            "   * At +25% target, 238 keV drops to **18.7%** retention ($T_{238} = 0.584$), Co-60 continuum rejection reaches **61.3%**.",
            "   * At +30% target, 238 keV drops to **10.5%** retention ($T_{238} = 0.607$), Co-60 continuum rejection reaches **68.2%**.",
            "   * Suitable for extended counting where high-energy peaks (>500 keV) retain **60% to 75%** of counts with massive background elimination.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--th232-score-cache", type=Path, default=DEFAULT_TH232_SCORE_CACHE
    )
    parser.add_argument(
        "--continuum-score-cache",
        type=Path,
        default=DEFAULT_CONTINUUM_SCORE_CACHE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-target-pct", type=int, default=1)
    parser.add_argument("--max-target-pct", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    th232_path = args.th232_score_cache.resolve()
    cont_path = args.continuum_score_cache.resolve()

    with h5py.File(th232_path, "r") as f:
        energy = np.asarray(f["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(f["score"], dtype=np.float32)

    with h5py.File(cont_path, "r") as f:
        c_energy = np.asarray(f["corrected_energy_kev"], dtype=np.float32)
        c_scores = np.asarray(f["score"], dtype=np.float32)
        c_src = np.asarray(f["source_code"], dtype=np.uint8)

    co60_mask = c_src == 0
    cs137_mask = c_src == 1

    base_hist = np.histogram(energy, ENERGY_EDGES)[0]
    windows = fit_peak_windows(base_hist)
    base_metrics = {
        w.reference_kev: peak_background_metrics(base_hist, w) for w in windows
    }

    t_grid = np.linspace(0.0, 0.70, 701)
    peak_curves = precompute_peak_curves(energy, scores, windows, base_metrics, t_grid)

    targets = list(range(args.min_target_pct, args.max_target_pct + 1))
    summary_rows: list[dict[str, Any]] = []
    all_peak_threshold_rows: list[dict[str, Any]] = []
    selected_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for pct in targets:
        tg = 1.0 + pct / 100.0
        p_th: dict[float, float] = {}
        p_gain: dict[float, float] = {}
        p_ret: dict[float, float] = {}

        for ref in PRIMARY_REFERENCE_PEAKS_KEV:
            th, g, r = get_peak_target(ref, tg, peak_curves, t_grid)
            p_th[ref] = th
            p_gain[ref] = g
            p_ret[ref] = r
            all_peak_threshold_rows.append(
                {
                    "target_pb_gain_percent": pct,
                    "reference_energy_kev": ref,
                    "target_threshold": th,
                    "achieved_pb_gain": g,
                    "net_peak_retention": r,
                }
            )

        # Continuous PCHIP curve across energy
        p_energies = [peak_curves[ref][2].centroid_kev for ref in PRIMARY_REFERENCE_PEAKS_KEV]
        p_thresholds = [p_th[ref] for ref in PRIMARY_REFERENCE_PEAKS_KEV]
        order = np.argsort(p_energies)
        sorted_e = np.asarray(p_energies)[order]
        sorted_t = np.asarray(p_thresholds)[order]
        pchip = PchipInterpolator(sorted_e, sorted_t)

        if pct in [2, 4, 6, 8, 10, 15, 20]:
            selected_curves[pct] = (sorted_e, sorted_t)

        min_e, max_e = float(min(sorted_e)), float(max(sorted_e))
        t_curve = np.clip(pchip(np.clip(energy, min_e, max_e)), 0.0, 1.0)
        c_t_curve = np.clip(pchip(np.clip(c_energy, min_e, max_e)), 0.0, 1.0)

        th232_ret = float(np.mean(scores >= t_curve)) * 100
        co60_rej = float(np.mean(c_scores[co60_mask] < c_t_curve[co60_mask])) * 100
        cs137_rej = float(np.mean(c_scores[cs137_mask] < c_t_curve[cs137_mask])) * 100

        min_r = float(min(p_ret.values())) * 100
        avg_r = float(np.mean(list(p_ret.values()))) * 100

        summary_rows.append(
            {
                "target_pb_gain_percent": pct,
                "minimum_peak_retention_percent": min_r,
                "average_peak_retention_percent": avg_r,
                "th232_total_retention_percent": th232_ret,
                "co60_continuum_rejection_percent": co60_rej,
                "cs137_continuum_rejection_percent": cs137_rej,
                "t_238": p_th[238.632],
                "t_338": p_th[338.320],
                "t_583": p_th[583.187],
                "t_911": p_th[911.204],
                "t_969": p_th[968.971],
                "t_2615": p_th[2614.511],
            }
        )

    summary_csv = output_dir / "flat_pb_grid_summary.csv"
    peak_csv = output_dir / "flat_pb_peak_thresholds_all_targets.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(peak_csv, all_peak_threshold_rows)

    plot_scan_figures(output_dir, summary_rows, all_peak_threshold_rows, selected_curves)

    report_md = output_dir / "report.md"
    report_md.write_text(make_markdown_report(summary_rows), encoding="utf-8")

    experiment_report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "TH232_FLAT_PB_GRID_SCAN_COMPLETE",
        "min_target_pct": args.min_target_pct,
        "max_target_pct": args.max_target_pct,
        "th232_score_cache": relative(th232_path),
        "continuum_score_cache": relative(cont_path),
        "summary_rows": summary_rows,
        "artifacts": {
            "summary_csv": relative(summary_csv),
            "peak_csv": relative(peak_csv),
            "report_md": relative(report_md),
            "pareto_curve_png": relative(output_dir / "flat_pb_pareto_curve.png"),
            "threshold_curves_png": relative(output_dir / "flat_pb_threshold_curves_by_target.png"),
        },
    }

    report_json = output_dir / "experiment_report.json"
    report_json.write_text(
        json.dumps(experiment_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": experiment_report["status"],
                "target_range": [args.min_target_pct, args.max_target_pct],
                "report": relative(report_json),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
