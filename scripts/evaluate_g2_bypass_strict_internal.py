#!/usr/bin/env python3
"""Apply frozen global and Group-2-bypass cuts to strict internal data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_group_specific_thresholds import (
    GROUPS,
    PEAKS,
    apply_thresholds,
    global_thresholds,
)


SEED = 20260822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    natural_cache = (
        PROJECT_ROOT / "processed_data/group_fusion_natural_validation_ma20_20260822"
    )
    strict_cache = (
        PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_ma20_20260822"
    )
    group_cache = PROJECT_ROOT / "processed_data/relaxed_continuum_six_group_20260822"
    score_dir = (
        PROJECT_ROOT
        / "outputs/experiments/shared_six_group_ds_cnn_ma10_vs_ma20_20260822"
    )
    parser.add_argument(
        "--natural-metadata",
        type=Path,
        default=natural_cache / "natural_file_validation_metadata.npz",
    )
    parser.add_argument(
        "--natural-assignments",
        type=Path,
        default=natural_cache / "natural_file_validation_assignments.npz",
    )
    parser.add_argument(
        "--natural-scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/group_ds_cnn_fusion_20260822"
        / "ma20_shared_ensemble_scores.npy",
    )
    parser.add_argument(
        "--strict-metadata",
        type=Path,
        default=strict_cache / "strict_internal_metadata.npz",
    )
    parser.add_argument(
        "--strict-assignments",
        type=Path,
        default=group_cache / "strict_internal_assignments.npz",
    )
    parser.add_argument(
        "--strict-score",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each frozen MA20 seed score. Defaults to all three seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/g2_bypass_strict_internal_20260822",
    )
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(
        default_score_paths=[
            score_dir / f"ma20_seed_{seed}_strict_scores.npy"
            for seed in (20260822, 20260823, 20260824)
        ]
    )
    return parser


def selected_arrays(
    metadata_path: Path, assignments_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = np.load(metadata_path)
    assignments = np.load(assignments_path)
    selected = assignments["selected"].astype(bool)
    groups = assignments["assignment"].astype(np.int16) + 1
    keep = selected & np.isin(groups, np.asarray(GROUPS))
    return (
        metadata["label"][keep].astype(np.int8),
        metadata["peak_id"][keep].astype(str),
        groups[keep],
        metadata["hdf5"][keep].astype(str),
        np.flatnonzero(keep).astype(np.int64),
    )


def acceptance_metrics(
    labels: np.ndarray,
    peaks: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for peak in PEAKS:
        positive = (labels == 1) & (peaks == peak)
        negative = (labels == 0) & (peaks == peak)
        positive_retention = float(np.mean(accepted[positive]))
        continuum_acceptance = float(np.mean(accepted[negative]))
        result[peak] = {
            "positive_events": int(np.count_nonzero(positive)),
            "continuum_events": int(np.count_nonzero(negative)),
            "positive_retention": positive_retention,
            "continuum_acceptance": continuum_acceptance,
            "peak_to_background_efficiency_ratio": (
                positive_retention / continuum_acceptance
                if continuum_acceptance > 0.0
                else float("inf")
            ),
        }
    positive = labels == 1
    negative = labels == 0
    positive_retention = float(np.mean(accepted[positive]))
    continuum_acceptance = float(np.mean(accepted[negative]))
    return {
        "positive_retention": positive_retention,
        "continuum_acceptance": continuum_acceptance,
        "peak_to_background_efficiency_ratio": (
            positive_retention / continuum_acceptance
            if continuum_acceptance > 0.0
            else float("inf")
        ),
        "worst_peak_positive_retention": float(
            min(row["positive_retention"] for row in result.values())
        ),
        "per_peak": result,
    }


def clustered_difference_ci(
    files: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    unique_files, inverse = np.unique(files, return_inverse=True)
    totals = np.bincount(inverse).astype(np.float64)
    first_counts = np.bincount(inverse, weights=first.astype(np.float64))
    second_counts = np.bincount(inverse, weights=second.astype(np.float64))
    differences = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 500):
        stop = min(start + 500, replicates)
        indices = rng.integers(
            0, unique_files.size, size=(stop - start, unique_files.size)
        )
        denominator = np.sum(totals[indices], axis=1)
        differences[start:stop] = (
            np.sum(second_counts[indices], axis=1)
            - np.sum(first_counts[indices], axis=1)
        ) / denominator
    low, high = np.quantile(differences, (0.025, 0.975))
    return {
        "difference": float(np.mean(second) - np.mean(first)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "files": int(unique_files.size),
    }


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.target < 1.0:
        raise ValueError("--target must lie between zero and one")
    if args.bootstrap_replicates < 1000:
        raise ValueError("--bootstrap-replicates must be at least 1000")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    natural_labels, natural_peaks, natural_groups, _, _ = selected_arrays(
        args.natural_metadata.resolve(), args.natural_assignments.resolve()
    )
    natural_scores = np.load(args.natural_scores.resolve()).astype(np.float64)
    if natural_scores.shape != natural_labels.shape:
        raise ValueError("Natural validation scores are not aligned")
    global_cut = global_thresholds(
        natural_labels,
        natural_peaks,
        natural_groups,
        natural_scores,
        args.target,
        False,
    )
    bypass_cut = global_thresholds(
        natural_labels,
        natural_peaks,
        natural_groups,
        natural_scores,
        args.target,
        True,
    )

    strict_labels, strict_peaks, strict_groups, strict_files, _ = selected_arrays(
        args.strict_metadata.resolve(), args.strict_assignments.resolve()
    )
    score_paths = (
        [path.resolve() for path in args.strict_score]
        if args.strict_score
        else [path.resolve() for path in args.default_score_paths]
    )
    strict_scores = np.mean(
        np.stack([np.load(path).astype(np.float64) for path in score_paths]), axis=0
    )
    if strict_scores.shape != strict_labels.shape:
        raise ValueError("Strict internal scores are not aligned")

    global_accepted = apply_thresholds(strict_groups, strict_scores, global_cut)
    bypass_accepted = apply_thresholds(strict_groups, strict_scores, bypass_cut)
    global_metrics = acceptance_metrics(strict_labels, strict_peaks, global_accepted)
    bypass_metrics = acceptance_metrics(strict_labels, strict_peaks, bypass_accepted)

    rng = np.random.default_rng(SEED)
    uncertainty: dict[str, Any] = {}
    for scope, scope_mask in {
        "pooled": np.ones(strict_labels.size, dtype=bool),
        **{peak: strict_peaks == peak for peak in PEAKS},
    }.items():
        uncertainty[scope] = {}
        for class_name, class_label in (("photopeak", 1), ("continuum", 0)):
            mask = scope_mask & (strict_labels == class_label)
            uncertainty[scope][class_name] = clustered_difference_ci(
                strict_files[mask],
                global_accepted[mask],
                bypass_accepted[mask],
                args.bootstrap_replicates,
                rng,
            )

    rows: list[dict[str, Any]] = []
    for scope in ("pooled",) + PEAKS:
        if scope == "pooled":
            global_row = global_metrics
            bypass_row = bypass_metrics
        else:
            global_row = global_metrics["per_peak"][scope]
            bypass_row = bypass_metrics["per_peak"][scope]
        rows.append(
            {
                "scope": scope,
                "global_positive_retention": global_row["positive_retention"],
                "bypass_positive_retention": bypass_row["positive_retention"],
                "positive_retention_difference": (
                    bypass_row["positive_retention"] - global_row["positive_retention"]
                ),
                "positive_difference_ci_95_low": uncertainty[scope]["photopeak"][
                    "ci_95_low"
                ],
                "positive_difference_ci_95_high": uncertainty[scope]["photopeak"][
                    "ci_95_high"
                ],
                "global_continuum_acceptance": global_row["continuum_acceptance"],
                "bypass_continuum_acceptance": bypass_row["continuum_acceptance"],
                "continuum_acceptance_difference": (
                    bypass_row["continuum_acceptance"]
                    - global_row["continuum_acceptance"]
                ),
                "continuum_difference_ci_95_low": uncertainty[scope]["continuum"][
                    "ci_95_low"
                ],
                "continuum_difference_ci_95_high": uncertainty[scope]["continuum"][
                    "ci_95_high"
                ],
                "global_pb_efficiency_ratio": global_row[
                    "peak_to_background_efficiency_ratio"
                ],
                "bypass_pb_efficiency_ratio": bypass_row[
                    "peak_to_background_efficiency_ratio"
                ],
            }
        )
    summary_csv = output_dir / "strict_internal_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    peak_labels = ["356 keV", "511 keV", "662 keV", "Pooled"]
    plot_rows = rows[1:] + rows[:1]
    x = np.arange(len(plot_rows))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, key, title in (
        (axes[0], "positive_retention", "Photopeak retained"),
        (axes[1], "continuum_acceptance", "Continuum accepted"),
    ):
        global_values = [100.0 * row[f"global_{key}"] for row in plot_rows]
        bypass_values = [100.0 * row[f"bypass_{key}"] for row in plot_rows]
        axis.bar(x - width / 2, global_values, width, label="Global threshold")
        axis.bar(x + width / 2, bypass_values, width, label="G2 bypass")
        axis.set_xticks(x, peak_labels)
        axis.set_ylabel("Events (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Frozen G2-bypass rule on strict energy-matched internal data")
    figure.tight_layout()
    figure_path = output_dir / "strict_internal_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    continuum_delta = uncertainty["pooled"]["continuum"]
    positive_delta = uncertainty["pooled"]["photopeak"]
    worst_peak_not_lower = (
        bypass_metrics["worst_peak_positive_retention"]
        >= global_metrics["worst_peak_positive_retention"]
    )
    decision = (
        "G2_BYPASS_STRICT_INTERNAL_SUPPORTED_EXTERNAL_CONFIRMATION_REQUIRED"
        if continuum_delta["difference"] < 0.0
        and continuum_delta["ci_95_high"] < 0.0
        and worst_peak_not_lower
        else "G2_BYPASS_STRICT_INTERNAL_NOT_SUPPORTED"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "calibration": {
            "partition": "natural development file-validation candidates",
            "target_minimum_positive_retention_per_peak": args.target,
            "global_threshold": global_cut[1],
            "group_2_bypass_common_threshold": bypass_cut[1],
            "group_2_threshold": bypass_cut[2],
        },
        "strict_internal": {
            "events": int(strict_scores.size),
            "global_threshold": global_metrics,
            "group_2_bypass": bypass_metrics,
            "paired_file_bootstrap_difference_bypass_minus_global": uncertainty,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "artifacts": {
            "summary_csv": summary_csv.relative_to(PROJECT_ROOT).as_posix(),
            "summary_csv_sha256": sha256_file(summary_csv),
            "figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
            "figure_sha256": sha256_file(figure_path),
        },
        "inputs": {
            "natural_metadata_sha256": sha256_file(args.natural_metadata.resolve()),
            "natural_assignments_sha256": sha256_file(
                args.natural_assignments.resolve()
            ),
            "natural_scores_sha256": sha256_file(args.natural_scores.resolve()),
            "strict_metadata_sha256": sha256_file(args.strict_metadata.resolve()),
            "strict_assignments_sha256": sha256_file(
                args.strict_assignments.resolve()
            ),
            "strict_score_files": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in score_paths
            ],
        },
        "claim_boundary": (
            "Strict energy-matched internal development check. Thresholds were "
            "not selected on this partition, but this is not independent external "
            "interaction-truth validation."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "calibration": report["calibration"],
                "global": global_metrics,
                "group_2_bypass": bypass_metrics,
                "pooled_difference_bypass_minus_global": {
                    "photopeak": positive_delta,
                    "continuum": continuum_delta,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
