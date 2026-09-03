#!/usr/bin/env python3
"""Create a simple comparison report for all-Ba t10 DS-CNN training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
PEAK_LABELS = {
    "ba133_276kev": "Ba 276",
    "ba133_303kev": "Ba 303",
    "ba133_356kev": "Ba 356",
    "ba133_384kev": "Ba 384",
    "na22_511kev": "Na 511",
    "cs137_662kev": "Cs 662",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_all_ba_t10_20260823"
        / "experiment_report.json",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822"
        / "experiment_report.json",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_all_ba_t10_20260823"
        / "cache_manifest.json",
    )
    return parser


def peak_statistics(
    report: dict[str, Any], partition: str, peak: str
) -> tuple[float, float, list[float]]:
    values = [
        float(run["evaluation"][partition]["per_peak"][peak]["auroc"])
        for run in report["runs"]
    ]
    return float(np.mean(values)), float(np.std(values, ddof=1)), values


def main() -> int:
    args = build_parser().parse_args()
    candidate_path = args.candidate_report.resolve()
    baseline_path = args.baseline_report.resolve()
    cache_path = args.cache_manifest.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    output_dir = candidate_path.parent

    all_peaks = tuple(
        candidate["runs"][0]["evaluation"]["relaxed_file_validation"]["per_peak"]
    )
    rows: list[dict[str, Any]] = []
    for partition in ("relaxed_file_validation", "strict_internal"):
        peaks = all_peaks if partition == "relaxed_file_validation" else COMMON_PEAKS
        for peak in peaks:
            candidate_mean, candidate_sd, _ = peak_statistics(
                candidate, partition, peak
            )
            if peak in COMMON_PEAKS:
                baseline_mean, baseline_sd, _ = peak_statistics(
                    baseline, partition, peak
                )
                delta: float | str = candidate_mean - baseline_mean
            else:
                baseline_mean = baseline_sd = float("nan")
                delta = ""
            rows.append(
                {
                    "partition": partition,
                    "peak_id": peak,
                    "baseline_mean_auroc": baseline_mean,
                    "baseline_seed_sd": baseline_sd,
                    "all_ba_mean_auroc": candidate_mean,
                    "all_ba_seed_sd": candidate_sd,
                    "delta": delta,
                }
            )
    with (output_dir / "all_ba_t10_peak_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gate = candidate["all_ba_comparison_gate"]
    paired_deltas = []
    for baseline_run, candidate_run in zip(baseline["runs"], candidate["runs"]):
        if baseline_run["seed"] != candidate_run["seed"]:
            raise ValueError("Seed ordering mismatch")
        baseline_macro = float(
            baseline_run["evaluation"]["strict_internal"]["macro_auroc"]
        )
        candidate_macro = float(
            candidate_run["evaluation"]["strict_internal"][
                "common_three_macro_auroc"
            ]
        )
        paired_deltas.append(candidate_macro - baseline_macro)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(COMMON_PEAKS))
    width = 0.36
    baseline_means = [gate["baseline_per_peak"][peak] for peak in COMMON_PEAKS]
    candidate_means = [gate["candidate_per_peak"][peak] for peak in COMMON_PEAKS]
    axes[0].bar(x - width / 2, baseline_means, width, label="3-peak t10")
    axes[0].bar(x + width / 2, candidate_means, width, label="all-Ba t10")
    axes[0].set_xticks(x, [PEAK_LABELS[peak] for peak in COMMON_PEAKS])
    axes[0].set_ylabel("Strict energy-matched AUROC")
    axes[0].set_title("Like-for-like strict comparison")
    axes[0].set_ylim(0.60, 0.72)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    relaxed_means = [
        peak_statistics(candidate, "relaxed_file_validation", peak)[0]
        for peak in all_peaks
    ]
    axes[1].bar(
        np.arange(len(all_peaks)),
        relaxed_means,
        color=["tab:blue" if peak.startswith("ba133") else "tab:orange" for peak in all_peaks],
    )
    axes[1].set_xticks(
        np.arange(len(all_peaks)),
        [PEAK_LABELS[peak] for peak in all_peaks],
        rotation=30,
        ha="right",
    )
    axes[1].set_ylabel("Relaxed file-validation AUROC")
    axes[1].set_title("All six training peaks")
    axes[1].set_ylim(0.57, 0.72)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("MA10/t10 DS-CNN with all Ba-133 peaks")
    figure.tight_layout()
    figure.savefig(output_dir / "all_ba_t10_comparison.png", dpi=180)
    plt.close(figure)

    train_counts = {
        row["peak_id"]: row["selected_events"] for row in cache["train_counts"]
    }
    validation_counts = {
        row["peak_id"]: row["selected_events"]
        for row in cache["relaxed_file_validation_counts"]
    }
    relaxed_metrics = {
        peak: peak_statistics(candidate, "relaxed_file_validation", peak)[0]
        for peak in all_peaks
    }
    report = f"""# All-Ba t10 DS-CNN

## Simple result

- Decision: `{candidate['decision']}`
- Representation: MA10 charge/current, shared t10 anchor, 750 samples.
- Training events: {cache['cache']['train']['event_count']:,} across six peaks.
- Relaxed file-validation events: {cache['cache']['relaxed_file_validation']['event_count']:,}.
- Strict comparison events: {cache['cache']['strict_internal']['event_count']:,}; identical 356/511/662-keV rows to the three-peak baseline.
- Three-peak baseline strict macro AUROC: {gate['baseline_macro_auroc']:.6f}
- All-Ba training strict macro AUROC: {gate['candidate_macro_auroc']:.6f}
- Change: {gate['macro_auroc_delta']:+.6f}; adoption gate: at least +0.004.
- Paired seed changes: {', '.join(f'{value:+.6f}' for value in paired_deltas)}.

## Strict result by common peak

| Peak | 3-peak t10 | all-Ba t10 | change |
|---|---:|---:|---:|
"""
    for peak in COMMON_PEAKS:
        report += (
            f"| {PEAK_LABELS[peak]} keV | {gate['baseline_per_peak'][peak]:.6f} | "
            f"{gate['candidate_per_peak'][peak]:.6f} | "
            f"{gate['per_peak_deltas'][peak]:+.6f} |\n"
        )
    report += """

## All six relaxed-validation peaks

| Peak | Training events | Validation events | AUROC |
|---|---:|---:|---:|
"""
    for peak in all_peaks:
        report += (
            f"| {PEAK_LABELS[peak]} keV | {train_counts[peak]:,} | "
            f"{validation_counts[peak]:,} | {relaxed_metrics[peak]:.6f} |\n"
        )
    report += f"""

## Interpretation

All three paired seeds improved the common strict macro AUROC, but the mean gain
of {gate['macro_auroc_delta']:+.6f} narrowly missed the predeclared +0.004 gate.
Na-511 and Cs-662 improved, while Ba-356 decreased slightly. Treat the expanded
Ba training pool as promising but not adopted over the current three-peak t10
baseline.

Locked test, Th-232, and Eu-152 were not used.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": candidate["decision"],
                "strict_macro_delta": gate["macro_auroc_delta"],
                "paired_seed_deltas": paired_deltas,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
