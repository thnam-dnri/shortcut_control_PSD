#!/usr/bin/env python3
"""Summarize the controlled t10/t50/t90 joint DS-CNN anchor ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ("t10", "t50", "t90")
PARTITIONS = ("relaxed_file_validation", "strict_internal")
PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t10-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822"
        / "experiment_report.json",
    )
    parser.add_argument(
        "--t50-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_anchor_t50_20260822"
        / "experiment_report.json",
    )
    parser.add_argument(
        "--t90-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_anchor_t90_20260822"
        / "experiment_report.json",
    )
    parser.add_argument(
        "--t10-cache",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_20260822"
        / "cache_manifest.json",
    )
    parser.add_argument(
        "--t50-cache",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_anchor_t50_20260822"
        / "cache_manifest.json",
    )
    parser.add_argument(
        "--t90-cache",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_anchor_t90_20260822"
        / "cache_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/ds_cnn_anchor_ablation_20260822",
    )
    parser.add_argument("--minimum-strict-gain", type=float, default=0.004)
    parser.add_argument("--maximum-per-peak-loss", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = {
        "t10": args.t10_report.resolve(),
        "t50": args.t50_report.resolve(),
        "t90": args.t90_report.resolve(),
    }
    cache_paths = {
        "t10": args.t10_cache.resolve(),
        "t50": args.t50_cache.resolve(),
        "t90": args.t90_cache.resolve(),
    }
    reports = {
        anchor: json.loads(path.read_text(encoding="utf-8"))
        for anchor, path in report_paths.items()
    }
    caches = {
        anchor: json.loads(path.read_text(encoding="utf-8"))
        for anchor, path in cache_paths.items()
    }
    for anchor in ANCHORS:
        if caches[anchor]["representation_config"]["anchor"] != anchor:
            raise ValueError(f"Cache anchor mismatch for {anchor}")
    for partition in ("train",) + PARTITIONS:
        hashes = {
            caches[anchor]["cache"][partition]["metadata_sha256"]
            for anchor in ANCHORS
        }
        if len(hashes) != 1:
            raise ValueError(f"Metadata mismatch for {partition}")

    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for partition in PARTITIONS:
            macro_values = np.asarray(
                [
                    run["evaluation"][partition]["macro_auroc"]
                    for run in reports[anchor]["runs"]
                ],
                dtype=np.float64,
            )
            row: dict[str, Any] = {
                "anchor": anchor,
                "partition": partition,
                "macro_auroc_mean": float(np.mean(macro_values)),
                "macro_auroc_seed_sd": float(np.std(macro_values, ddof=1)),
                "worst_peak_auroc_mean": float(
                    np.mean(
                        [
                            run["evaluation"][partition]["worst_peak_auroc"]
                            for run in reports[anchor]["runs"]
                        ]
                    )
                ),
                "pooled_auroc_mean": float(
                    np.mean(
                        [
                            run["evaluation"][partition]["pooled_auroc"]
                            for run in reports[anchor]["runs"]
                        ]
                    )
                ),
            }
            for peak in PEAKS:
                row[f"{peak}_auroc_mean"] = float(
                    np.mean(
                        [
                            run["evaluation"][partition]["per_peak"][peak]["auroc"]
                            for run in reports[anchor]["runs"]
                        ]
                    )
                )
            rows.append(row)
            for run in reports[anchor]["runs"]:
                seed_rows.append(
                    {
                        "anchor": anchor,
                        "partition": partition,
                        "seed": int(run["seed"]),
                        "macro_auroc": float(
                            run["evaluation"][partition]["macro_auroc"]
                        ),
                    }
                )

    by_anchor_partition = {
        (row["anchor"], row["partition"]): row for row in rows
    }
    baseline = by_anchor_partition[("t10", "strict_internal")]
    decisions: dict[str, Any] = {}
    for anchor in ("t50", "t90"):
        candidate = by_anchor_partition[(anchor, "strict_internal")]
        strict_gain = candidate["macro_auroc_mean"] - baseline["macro_auroc_mean"]
        per_peak_deltas = {
            peak: candidate[f"{peak}_auroc_mean"]
            - baseline[f"{peak}_auroc_mean"]
            for peak in PEAKS
        }
        supported = strict_gain >= args.minimum_strict_gain and all(
            delta >= -args.maximum_per_peak_loss
            for delta in per_peak_deltas.values()
        )
        decisions[anchor] = {
            "strict_macro_delta_vs_t10": strict_gain,
            "strict_per_peak_delta_vs_t10": per_peak_deltas,
            "supported": supported,
        }
    decision = (
        "LATER_ANCHOR_SUPPORTED"
        if any(row["supported"] for row in decisions.values())
        else "T50_T90_NOT_SUPPORTED_T10_OBSERVED_LEADER"
    )

    summary_csv = output_dir / "anchor_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    seed_csv = output_dir / "anchor_seed_metrics.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(ANCHORS))
    width = 0.36
    for offset, partition, label in (
        (-width / 2, "relaxed_file_validation", "Relaxed file validation"),
        (width / 2, "strict_internal", "Strict energy matched"),
    ):
        values = [
            by_anchor_partition[(anchor, partition)]["macro_auroc_mean"]
            for anchor in ANCHORS
        ]
        errors = [
            by_anchor_partition[(anchor, partition)]["macro_auroc_seed_sd"]
            for anchor in ANCHORS
        ]
        axes[0].bar(x + offset, values, width, yerr=errors, capsize=4, label=label)
    axes[0].set_xticks(x, ANCHORS)
    axes[0].set_ylabel("Macro AUROC (mean +/- seed SD)")
    axes[0].set_title("Overall anchor comparison")
    axes[0].set_ylim(0.61, 0.705)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    peak_x = np.arange(len(PEAKS))
    peak_width = 0.24
    for index, anchor in enumerate(ANCHORS):
        row = by_anchor_partition[(anchor, "strict_internal")]
        axes[1].bar(
            peak_x + (index - 1) * peak_width,
            [row[f"{peak}_auroc_mean"] for peak in PEAKS],
            peak_width,
            label=anchor,
        )
    axes[1].set_xticks(peak_x, ("356 keV", "511 keV", "662 keV"))
    axes[1].set_ylabel("Strict internal AUROC")
    axes[1].set_title("Strict energy-matched result by peak")
    axes[1].set_ylim(0.60, 0.705)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Joint three-peak MA10 DS-CNN anchor ablation")
    figure.tight_layout()
    figure_path = output_dir / "anchor_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "primary_metric": "three-seed mean strict-internal macro AUROC",
        "gate": {
            "minimum_strict_macro_gain_vs_t10": args.minimum_strict_gain,
            "maximum_allowed_per_peak_loss": args.maximum_per_peak_loss,
        },
        "anchor_decisions": decisions,
        "summary": rows,
        "contract": {
            "only_variable": "shared first-rising charge crossing anchor",
            "fixed": (
                "MA10 charge, MA10-derived current, 250 pre/500 post samples, "
                "global-RMS normalization, train-only z-score, identical event "
                "rows, DS-CNN width 24, AdamW, six epochs, three fixed seeds"
            ),
        },
        "inputs": {
            anchor: {
                "training_report": report_paths[anchor].relative_to(PROJECT_ROOT).as_posix(),
                "training_report_sha256": sha256_file(report_paths[anchor]),
                "cache_manifest": cache_paths[anchor].relative_to(PROJECT_ROOT).as_posix(),
                "cache_manifest_sha256": sha256_file(cache_paths[anchor]),
            }
            for anchor in ANCHORS
        },
        "artifacts": {
            "summary_csv": summary_csv.relative_to(PROJECT_ROOT).as_posix(),
            "summary_csv_sha256": sha256_file(summary_csv),
            "seed_csv": seed_csv.relative_to(PROJECT_ROOT).as_posix(),
            "seed_csv_sha256": sha256_file(seed_csv),
            "figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
            "figure_sha256": sha256_file(figure_path),
        },
        "claim_boundary": (
            "Development anchor ablation only; no locked test, Th-232, Eu-152, "
            "or independent external interaction-truth campaign used."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "decisions": decisions, "summary": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
