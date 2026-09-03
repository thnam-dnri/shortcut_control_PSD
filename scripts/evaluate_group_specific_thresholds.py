#!/usr/bin/env python3
"""Cross-validate group-specific DS-CNN thresholds at fixed peak retention."""

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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = tuple(range(1, 7))
PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
METHODS = ("global_threshold", "group_2_bypass", "six_group_thresholds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cache = (
        PROJECT_ROOT / "processed_data/group_fusion_natural_validation_ma20_20260822"
    )
    fusion = PROJECT_ROOT / "outputs/experiments/group_ds_cnn_fusion_20260822"
    parser.add_argument(
        "--metadata",
        type=Path,
        default=cache / "natural_file_validation_metadata.npz",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=cache / "natural_file_validation_assignments.npz",
    )
    parser.add_argument(
        "--scores", type=Path, default=fusion / "ma20_shared_ensemble_scores.npy"
    )
    parser.add_argument(
        "--fold-ids", type=Path, default=fusion / "fusion_fold_id.npy"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/group_specific_thresholds_20260822",
    )
    parser.add_argument(
        "--retention-targets", type=float, nargs="+", default=(0.90, 0.95, 0.99)
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def threshold_for_required_positive_count(
    scores: np.ndarray, required: int
) -> float:
    if required <= 0:
        return float(np.nextafter(1.0, np.inf))
    if required > scores.size:
        raise ValueError(f"Required {required} positives but only {scores.size} exist")
    ordered = np.sort(scores)[::-1]
    return float(ordered[required - 1])


def global_thresholds(
    labels: np.ndarray,
    peaks: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    target: float,
    bypass_group_2: bool,
) -> dict[int, float]:
    peak_thresholds: list[float] = []
    for peak in PEAKS:
        positive = (labels == 1) & (peaks == peak)
        required_total = int(np.ceil(target * np.count_nonzero(positive)))
        if bypass_group_2:
            accepted_by_bypass = int(np.count_nonzero(positive & (groups == 2)))
            eligible = positive & (groups != 2)
            required = max(0, required_total - accepted_by_bypass)
        else:
            eligible = positive
            required = required_total
        peak_thresholds.append(
            threshold_for_required_positive_count(scores[eligible], required)
        )
    common = min(peak_thresholds)
    result = {group: common for group in GROUPS}
    if bypass_group_2:
        result[2] = 0.0
    return result


def candidate_states(
    labels: np.ndarray,
    peaks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_scores = scores[labels == 1]
    thresholds = np.concatenate(
        (
            np.asarray([np.nextafter(1.0, np.inf)]),
            np.unique(positive_scores)[::-1],
        )
    )
    negative_sorted = np.sort(scores[labels == 0])
    false_positive = negative_sorted.size - np.searchsorted(
        negative_sorted, thresholds, side="left"
    )
    true_positive = np.zeros((len(PEAKS), thresholds.size), dtype=np.int64)
    for index, peak in enumerate(PEAKS):
        peak_sorted = np.sort(scores[(labels == 1) & (peaks == peak)])
        true_positive[index] = peak_sorted.size - np.searchsorted(
            peak_sorted, thresholds, side="left"
        )

    # If several thresholds accept the same number of continuum events, retain
    # only the lowest threshold because it accepts at least as many positives.
    reverse = false_positive[::-1]
    _, reverse_indices = np.unique(reverse, return_index=True)
    keep = np.sort(false_positive.size - 1 - reverse_indices)
    return thresholds[keep], false_positive[keep], true_positive[:, keep]


def optimize_six_thresholds(
    labels: np.ndarray,
    peaks: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    target: float,
) -> dict[int, float]:
    thresholds_by_group: list[np.ndarray] = []
    false_positive_by_group: list[np.ndarray] = []
    true_positive_by_group: list[np.ndarray] = []
    for group in GROUPS:
        mask = groups == group
        thresholds, false_positive, true_positive = candidate_states(
            labels[mask], peaks[mask], scores[mask]
        )
        thresholds_by_group.append(thresholds)
        false_positive_by_group.append(false_positive)
        true_positive_by_group.append(true_positive)

    offsets = np.cumsum(
        [0] + [thresholds.size for thresholds in thresholds_by_group]
    )
    variable_count = int(offsets[-1])
    constraint_count = len(GROUPS) + len(PEAKS)
    matrix = lil_matrix((constraint_count, variable_count), dtype=np.float64)
    objective = np.empty(variable_count, dtype=np.float64)
    for group_index in range(len(GROUPS)):
        start, stop = offsets[group_index], offsets[group_index + 1]
        matrix[group_index, start:stop] = 1.0
        true_positive_total = np.sum(
            true_positive_by_group[group_index], axis=0
        ).astype(np.float64)
        objective[start:stop] = (
            false_positive_by_group[group_index].astype(np.float64)
            - 1.0e-7 * true_positive_total
        )
        for peak_index in range(len(PEAKS)):
            matrix[len(GROUPS) + peak_index, start:stop] = (
                true_positive_by_group[group_index][peak_index]
            )

    lower = np.concatenate(
        (
            np.ones(len(GROUPS)),
            np.asarray(
                [
                    np.ceil(target * np.count_nonzero((labels == 1) & (peaks == peak)))
                    for peak in PEAKS
                ],
                dtype=np.float64,
            ),
        )
    )
    upper = np.concatenate(
        (np.ones(len(GROUPS)), np.full(len(PEAKS), np.inf, dtype=np.float64))
    )
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Threshold MILP failed: {result.message}")
    selected: dict[int, float] = {}
    for group_index, group in enumerate(GROUPS):
        start, stop = offsets[group_index], offsets[group_index + 1]
        choice = int(np.argmax(result.x[start:stop]))
        selected[group] = float(thresholds_by_group[group_index][choice])
    return selected


def apply_thresholds(
    groups: np.ndarray, scores: np.ndarray, thresholds: dict[int, float]
) -> np.ndarray:
    accepted = np.zeros(scores.size, dtype=bool)
    for group in GROUPS:
        mask = groups == group
        accepted[mask] = scores[mask] >= thresholds[group]
    return accepted


def evaluate(
    labels: np.ndarray,
    peaks: np.ndarray,
    groups: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    positive = labels == 1
    negative = labels == 0
    per_peak = {
        peak: float(np.mean(accepted[positive & (peaks == peak)])) for peak in PEAKS
    }
    per_group = {}
    for group in GROUPS:
        group_mask = groups == group
        per_group[f"group_{group}"] = {
            "positive_retention": float(np.mean(accepted[group_mask & positive])),
            "continuum_acceptance": float(np.mean(accepted[group_mask & negative])),
            "positive_events": int(np.count_nonzero(group_mask & positive)),
            "continuum_events": int(np.count_nonzero(group_mask & negative)),
        }
    return {
        "positive_retention": float(np.mean(accepted[positive])),
        "continuum_acceptance": float(np.mean(accepted[negative])),
        "continuum_rejection": float(1.0 - np.mean(accepted[negative])),
        "per_peak_positive_retention": per_peak,
        "worst_peak_positive_retention": float(min(per_peak.values())),
        "per_group": per_group,
        "positive_events": int(np.count_nonzero(positive)),
        "continuum_events": int(np.count_nonzero(negative)),
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(sorted(set(float(value) for value in args.retention_targets)))
    if any(value <= 0.0 or value >= 1.0 for value in targets):
        raise ValueError("Retention targets must lie strictly between 0 and 1")

    metadata = np.load(args.metadata.resolve())
    assignments = np.load(args.assignments.resolve())
    selected = assignments["selected"].astype(bool)
    one_based_group = assignments["assignment"].astype(np.int16) + 1
    keep = selected & np.isin(one_based_group, np.asarray(GROUPS))
    labels = metadata["label"][keep].astype(np.int8)
    peaks = metadata["peak_id"][keep].astype(str)
    groups = one_based_group[keep]
    scores = np.load(args.scores.resolve()).astype(np.float64)
    fold_ids = np.load(args.fold_ids.resolve()).astype(np.int8)
    if not (labels.shape == peaks.shape == groups.shape == scores.shape == fold_ids.shape):
        raise ValueError("Input arrays are not aligned")
    if set(np.unique(peaks)) != set(PEAKS):
        raise ValueError(f"Unexpected peak IDs: {np.unique(peaks)}")

    fold_rows: list[dict[str, Any]] = []
    oof_acceptance = {
        (target, method): np.zeros(labels.size, dtype=bool)
        for target in targets
        for method in METHODS
    }
    for fold in sorted(np.unique(fold_ids).tolist()):
        fit = fold_ids != fold
        held = fold_ids == fold
        for target in targets:
            thresholds_by_method = {
                "global_threshold": global_thresholds(
                    labels[fit], peaks[fit], groups[fit], scores[fit], target, False
                ),
                "group_2_bypass": global_thresholds(
                    labels[fit], peaks[fit], groups[fit], scores[fit], target, True
                ),
                "six_group_thresholds": optimize_six_thresholds(
                    labels[fit], peaks[fit], groups[fit], scores[fit], target
                ),
            }
            for method, thresholds in thresholds_by_method.items():
                accepted_fit = apply_thresholds(groups[fit], scores[fit], thresholds)
                accepted_held = apply_thresholds(groups[held], scores[held], thresholds)
                oof_acceptance[(target, method)][held] = accepted_held
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "target": target,
                        "method": method,
                        "thresholds": {str(key): value for key, value in thresholds.items()},
                        "fit": evaluate(
                            labels[fit], peaks[fit], groups[fit], accepted_fit
                        ),
                        "held": evaluate(
                            labels[held], peaks[held], groups[held], accepted_held
                        ),
                    }
                )

    pooled: dict[str, Any] = {}
    for target in targets:
        target_key = f"{int(round(target * 100))}_percent"
        pooled[target_key] = {}
        for method in METHODS:
            pooled[target_key][method] = evaluate(
                labels, peaks, groups, oof_acceptance[(target, method)]
            )

    csv_rows: list[dict[str, Any]] = []
    for target in targets:
        target_key = f"{int(round(target * 100))}_percent"
        for method in METHODS:
            row = pooled[target_key][method]
            csv_rows.append(
                {
                    "target": target,
                    "method": method,
                    "positive_retention": row["positive_retention"],
                    "worst_peak_positive_retention": row[
                        "worst_peak_positive_retention"
                    ],
                    "continuum_acceptance": row["continuum_acceptance"],
                    "continuum_rejection": row["continuum_rejection"],
                    **{
                        f"{peak}_positive_retention": row[
                            "per_peak_positive_retention"
                        ][peak]
                        for peak in PEAKS
                    },
                }
            )
    summary_csv = output_dir / "threshold_method_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    thresholds_csv = output_dir / "thresholds_by_fold.csv"
    with thresholds_csv.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["fold", "target", "method"] + [
            f"group_{group}_threshold" for group in GROUPS
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(
                {
                    "fold": row["fold"],
                    "target": row["target"],
                    "method": row["method"],
                    **{
                        f"group_{group}_threshold": row["thresholds"][str(group)]
                        for group in GROUPS
                    },
                }
            )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {
        "global_threshold": "#4c78a8",
        "group_2_bypass": "#f58518",
        "six_group_thresholds": "#54a24b",
    }
    labels_by_method = {
        "global_threshold": "Global threshold",
        "group_2_bypass": "G2 bypass",
        "six_group_thresholds": "Six thresholds",
    }
    for method in METHODS:
        x = [100.0 * target for target in targets]
        positive_retention = [
            100.0
            * pooled[f"{int(round(target * 100))}_percent"][method][
                "positive_retention"
            ]
            for target in targets
        ]
        continuum_acceptance = [
            100.0
            * pooled[f"{int(round(target * 100))}_percent"][method][
                "continuum_acceptance"
            ]
            for target in targets
        ]
        axes[0].plot(
            x, positive_retention, marker="o", color=colors[method], label=labels_by_method[method]
        )
        axes[1].plot(
            x, continuum_acceptance, marker="o", color=colors[method], label=labels_by_method[method]
        )
    axes[0].plot([100.0 * targets[0], 100.0 * targets[-1]], [100.0 * targets[0], 100.0 * targets[-1]], "k--", linewidth=1, label="Target")
    axes[0].set_ylabel("Out-of-fold photopeak retention (%)")
    axes[1].set_ylabel("Out-of-fold continuum accepted (%)")
    for axis in axes:
        axis.set_xlabel("Tuning target: minimum retention per development peak (%)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Cross-validated morphology-group threshold optimization")
    figure.tight_layout()
    figure_path = output_dir / "threshold_method_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "DO_NOT_ADOPT_SIX_THRESHOLDS_GROUP2_BYPASS_PROVISIONAL",
        "protocol": {
            "score": "frozen three-seed shared MA20/t10 DS-CNN ensemble",
            "folds": "three complete-HDF5-file folds inherited from the prior fusion audit",
            "threshold_fit": "two folds only; evaluated on the third fold",
            "objective": "minimize accepted continuum subject to the requested positive-retention floor in every development peak",
            "targets": targets,
            "six_threshold_optimizer": "binary linear optimization over all nondominated positive-score thresholds in each group",
        },
        "pooled_out_of_fold": pooled,
        "fold_results": fold_rows,
        "artifacts": {
            "summary_csv": summary_csv.relative_to(PROJECT_ROOT).as_posix(),
            "summary_csv_sha256": sha256_file(summary_csv),
            "thresholds_csv": thresholds_csv.relative_to(PROJECT_ROOT).as_posix(),
            "thresholds_csv_sha256": sha256_file(thresholds_csv),
            "comparison_figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
            "comparison_figure_sha256": sha256_file(figure_path),
        },
        "inputs": {
            "metadata_sha256": sha256_file(args.metadata.resolve()),
            "assignments_sha256": sha256_file(args.assignments.resolve()),
            "scores_sha256": sha256_file(args.scores.resolve()),
            "fold_ids_sha256": sha256_file(args.fold_ids.resolve()),
        },
        "claim_boundary": (
            "Development file-validation proxy only. Thresholds are evaluated "
            "out of fold but are not deployment thresholds. Candidate class ratios "
            "reflect source exposure and ROI construction."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pooled_out_of_fold": pooled}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
