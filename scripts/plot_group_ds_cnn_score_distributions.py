#!/usr/bin/env python3
"""Plot shared MA20 DS-CNN score distributions for six morphology groups."""

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
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = tuple(range(1, 7))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/group_fusion_natural_validation_ma20_20260822"
        / "natural_file_validation_metadata.npz",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/group_fusion_natural_validation_ma20_20260822"
        / "natural_file_validation_assignments.npz",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/group_ds_cnn_fusion_20260822"
        / "ma20_shared_ensemble_scores.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/group_ds_cnn_score_distributions_20260822",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def summarize_group(
    group: int,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    positive = scores[labels == 1]
    continuum = scores[labels == 0]
    return {
        "group": group,
        "positive_events": int(positive.size),
        "continuum_events": int(continuum.size),
        "positive_score_mean": float(np.mean(positive)),
        "positive_score_median": float(np.median(positive)),
        "continuum_score_mean": float(np.mean(continuum)),
        "continuum_score_median": float(np.median(continuum)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
    }


def main() -> int:
    args = build_parser().parse_args()
    metadata_path = args.metadata.resolve()
    assignments_path = args.assignments.resolve()
    scores_path = args.scores.resolve()
    output_dir = args.output_dir.resolve()
    if args.bins < 10:
        raise ValueError("--bins must be at least 10")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = np.load(metadata_path)
    assignments = np.load(assignments_path)
    selected = assignments["selected"].astype(bool)
    one_based_group = assignments["assignment"].astype(np.int16) + 1
    keep = selected & np.isin(one_based_group, np.asarray(GROUPS))
    labels = metadata["label"][keep].astype(np.int8)
    groups = one_based_group[keep]
    scores = np.load(scores_path).astype(np.float64)
    if not (labels.shape == groups.shape == scores.shape):
        raise ValueError(
            f"Alignment mismatch: labels={labels.shape}, groups={groups.shape}, "
            f"scores={scores.shape}"
        )
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Scores must be finite probabilities in [0, 1]")

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    summaries: list[dict[str, Any]] = []
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for group, axis in zip(GROUPS, axes.flat):
        mask = groups == group
        group_labels = labels[mask]
        group_scores = scores[mask]
        row = summarize_group(group, group_labels, group_scores)
        summaries.append(row)
        for label, color, name in (
            (1, "#1f77b4", "Photopeak"),
            (0, "#d62728", "Continuum"),
        ):
            class_scores = group_scores[group_labels == label]
            counts, _ = np.histogram(class_scores, bins=edges)
            percent = 100.0 * counts / class_scores.size
            axis.step(centers, percent, where="mid", color=color, linewidth=1.8, label=name)
            axis.axvline(
                np.median(class_scores), color=color, linestyle="--", linewidth=1.0, alpha=0.8
            )
        axis.set_title(
            f"Group {group}  |  AUROC {row['pooled_auroc']:.3f}\n"
            f"photopeak {row['positive_events']:,}, continuum {row['continuum_events']:,}"
        )
        axis.grid(alpha=0.2)
        axis.set_xlim(0.0, 1.0)
    axes[0, 0].legend(frameon=False)
    figure.supxlabel("Shared MA20 DS-CNN score (higher = more photopeak-like)")
    figure.supylabel("Events in score bin (% of that class)")
    figure.suptitle("DS-CNN score distribution within each frozen morphology group", fontsize=14)
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    figure_path = output_dir / "group_score_distributions.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    summary_path = output_dir / "group_score_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": int(scores.size),
        "score": "mean probability from the three frozen shared MA20/t10 DS-CNN seeds",
        "distribution_normalization": "each class independently sums to 100 percent within each group",
        "summary": summaries,
        "inputs": {
            "metadata": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
            "assignments": assignments_path.relative_to(PROJECT_ROOT).as_posix(),
            "assignments_sha256": sha256_file(assignments_path),
            "scores": scores_path.relative_to(PROJECT_ROOT).as_posix(),
            "scores_sha256": sha256_file(scores_path),
        },
        "artifacts": {
            "figure": figure_path.relative_to(PROJECT_ROOT).as_posix(),
            "figure_sha256": sha256_file(figure_path),
            "summary_csv": summary_path.relative_to(PROJECT_ROOT).as_posix(),
            "summary_csv_sha256": sha256_file(summary_path),
        },
        "claim_boundary": (
            "Development file-validation candidates only. No threshold selection, "
            "locked test, Th-232, or Eu-152 data used."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(figure_path), "summary": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
