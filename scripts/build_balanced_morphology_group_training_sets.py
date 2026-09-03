#!/usr/bin/env python3
"""Build deterministic class- and peak-balanced fit selections per group."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path

SEED = 20260822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_indices_by_peak(
    labels: np.ndarray,
    peak_ids: np.ndarray,
    candidates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select equal positive/negative counts inside every represented peak."""

    selected: list[np.ndarray] = []
    for peak in sorted(np.unique(peak_ids[candidates]).astype(str)):
        peak_candidates = candidates[peak_ids[candidates].astype(str) == peak]
        positive = peak_candidates[labels[peak_candidates] == 1]
        negative = peak_candidates[labels[peak_candidates] == 0]
        target = min(positive.size, negative.size)
        if target == 0:
            raise ValueError(f"Peak {peak} lacks one class")
        selected.extend(
            (
                rng.choice(positive, size=target, replace=False),
                rng.choice(negative, size=target, replace=False),
            )
        )
    return np.sort(np.concatenate(selected).astype(np.int64))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/morphology_catalogue_minimum_1000_1500_20260821",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/morphology_catalogue_minimum_1000_1500_merged_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/morphology_group_balanced_minimum_1000_1500_20260822",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    feature_dir = args.feature_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (feature_dir, experiment_dir, output_dir):
        assert_no_forbidden_path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = feature_dir / "fit_features.npz"
    assignment_path = experiment_dir / "catalogue/fit_assignments.npz"
    features = np.load(feature_path)
    assignments = np.load(assignment_path)
    valid = features["valid"].astype(bool) & assignments["valid"].astype(bool)
    labels = features["label"].astype(np.int8)
    peak_ids = features["peak_id"].astype(str)
    group_count = assignments["probability"].shape[1]
    rng = np.random.default_rng(args.seed)

    selected_parts: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        candidates = np.flatnonzero(
            valid & (assignments["assignment"] == group_index)
        )
        selected = balanced_indices_by_peak(labels, peak_ids, candidates, rng)
        selected_parts.append(selected)
        np.savez_compressed(
            output_dir / f"group_{group_index + 1}_balanced_selection.npz",
            fit_cache_index=selected,
            original_event_index=features["original_event_index"][selected],
            label=labels[selected],
            peak_id=features["peak_id"][selected],
            pair_index=features["pair_index"][selected],
            pair_member=features["pair_member"][selected],
        )
        for peak in sorted(np.unique(peak_ids[selected])):
            mask = peak_ids[selected] == peak
            peak_labels = labels[selected][mask]
            summary_rows.append(
                {
                    "group": group_index + 1,
                    "peak_id": peak,
                    "positive_events": int(np.count_nonzero(peak_labels == 1)),
                    "negative_events": int(np.count_nonzero(peak_labels == 0)),
                    "total_events": int(peak_labels.size),
                }
            )

    combined = np.sort(np.concatenate(selected_parts).astype(np.int64))
    combined_group = assignments["assignment"][combined].astype(np.int16) + 1
    np.savez_compressed(
        output_dir / "balanced_training_selection.npz",
        fit_cache_index=combined,
        original_event_index=features["original_event_index"][combined],
        group=combined_group,
        label=labels[combined],
        peak_id=features["peak_id"][combined],
        pair_index=features["pair_index"][combined],
        pair_member=features["pair_member"][combined],
    )
    manifest = pd.DataFrame(
        {
            "fit_cache_index": combined,
            "original_train_event_index": features["original_event_index"][combined],
            "group": combined_group,
            "label": labels[combined],
            "class_name": np.where(labels[combined] == 1, "positive", "negative"),
            "peak_id": features["peak_id"][combined],
            "source": features["source"][combined],
            "energy_kev": features["energy_kev"][combined],
            "hdf5": features["hdf5"][combined],
            "source_row": features["source_row"][combined],
            "pair_index": features["pair_index"][combined],
            "pair_member": features["pair_member"][combined],
            "raw_minimum_index": features["raw_minimum_index"][combined],
        }
    )
    manifest.to_csv(output_dir / "balanced_training_manifest.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    totals = (
        summary.groupby("group", as_index=False)[
            ["positive_events", "negative_events", "total_events"]
        ]
        .sum()
        .assign(positive_percent=50.0, negative_percent=50.0)
    )
    summary.to_csv(output_dir / "balance_by_group_and_peak.csv", index=False)
    totals.to_csv(output_dir / "balance_by_group.csv", index=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FIT_ONLY_BALANCED_GROUP_TRAINING_SELECTIONS",
        "seed": args.seed,
        "selection_rule": (
            "Within each merged morphology group and peak_id, retain all events "
            "from the smaller class and deterministically downsample the larger "
            "class to the same count."
        ),
        "total_selected_events": int(combined.size),
        "groups": totals.to_dict(orient="records"),
        "input": {
            "fit_features": feature_path.relative_to(PROJECT_ROOT).as_posix(),
            "fit_features_sha256": sha256_file(feature_path),
            "fit_assignments": assignment_path.relative_to(PROJECT_ROOT).as_posix(),
            "fit_assignments_sha256": sha256_file(assignment_path),
        },
        "internal_partition_used": False,
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "balance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(totals.to_string(index=False), flush=True)
    print(f"total_selected_events={combined.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
