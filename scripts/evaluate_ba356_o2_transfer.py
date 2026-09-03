#!/usr/bin/env python3
"""Evaluate the frozen Ba-356-trained O2-style CNN across Ba/Na/Cs domains."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_source_ablation import cnn_scores, grouped_metrics  # noqa: E402
from train_o2_late_fusion import build_partition_features, sha256_file  # noqa: E402


def main() -> int:
    checkpoint_path = PROJECT_ROOT / "outputs/models/ba356_o2_late_fusion/o2_late_fusion_best.pt"
    output_dir = PROJECT_ROOT / "outputs/models/ba356_o2_late_fusion/frozen_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    event_store_dir = PROJECT_ROOT / "processed_data/event_store"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training_positive_domain": "Ba-133 356-keV photopeak only",
            "negative_sources": ["co60", "cs137", "na22"],
            "checkpoint_frozen_on": "Ba-356 internal validation AUROC",
            "target_domains_used_for_selection": False,
            "test_partition_used": False,
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
        "domains": {},
    }
    score_arrays: dict[str, np.ndarray] = {}
    csv_paths = {
        "ba356": PROJECT_ROOT / "outputs/labels/ba356_positive/label_pairs_validation.csv",
        "na22": PROJECT_ROOT / "outputs/labels/source_ablation/na22_positive/label_pairs_validation.csv",
        "cs137": PROJECT_ROOT / "outputs/labels/source_ablation/cs137_positive/label_pairs_validation.csv",
    }
    for source, csv_path in csv_paths.items():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        data = build_partition_features(csv_path, event_store_dir=event_store_dir)
        scores = cnn_scores(checkpoint_path, data, device)
        grouped = grouped_metrics(pair_rows, data, scores)
        result["domains"][source] = {
            "validation_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(csv_path),
            **grouped,
        }
        score_arrays[f"{source}_labels"] = data.labels.astype(np.int8)
        score_arrays[f"{source}_weights"] = data.weights
        score_arrays[f"{source}_scores"] = scores
        print(source, grouped["overall"], flush=True)
    metrics_path = output_dir / "frozen_transfer_metrics.json"
    scores_path = output_dir / "frozen_transfer_scores.npz"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(scores_path, **score_arrays)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
