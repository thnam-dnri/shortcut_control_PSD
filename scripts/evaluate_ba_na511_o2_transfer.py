#!/usr/bin/env python3
"""Evaluate frozen Ba-all-peaks plus Na-511 late-fusion CNN by domain."""

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
    checkpoint_path = PROJECT_ROOT / "outputs/models/ba_all_na511_o2_late_fusion/o2_late_fusion_best.pt"
    output_dir = PROJECT_ROOT / "outputs/models/ba_all_na511_o2_late_fusion/frozen_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    event_store_dir = PROJECT_ROOT / "processed_data/event_store"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_paths = {
        "combined_ba_na511": PROJECT_ROOT / "outputs/labels/ba_all_na511_positive/label_pairs_validation.csv",
        "ba133": PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive/label_pairs_validation.csv",
        "na22": PROJECT_ROOT / "outputs/labels/source_ablation/na22_positive/label_pairs_validation.csv",
        "cs137": PROJECT_ROOT / "outputs/labels/source_ablation/cs137_positive/label_pairs_validation.csv",
    }
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training_domain": "Ba-133 all peaks plus Na-22 511-keV positives",
            "checkpoint_frozen_on": "combined five-peak-balanced internal validation AUROC",
            "cs137_used_for_model_selection": False,
            "test_partition_used": False,
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
        "domains": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for domain, csv_path in csv_paths.items():
        print(f"Loading domain={domain}", flush=True)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        data = build_partition_features(csv_path, event_store_dir=event_store_dir)
        scores = cnn_scores(checkpoint_path, data, device)
        grouped = grouped_metrics(pair_rows, data, scores)
        result["domains"][domain] = {
            "validation_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(csv_path),
            **grouped,
        }
        arrays[f"{domain}_labels"] = data.labels.astype(np.int8)
        arrays[f"{domain}_weights"] = data.weights
        arrays[f"{domain}_scores"] = scores
        print(domain, grouped["overall"], flush=True)
    metrics_path = output_dir / "frozen_transfer_metrics.json"
    scores_path = output_dir / "frozen_transfer_scores.npz"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(scores_path, **arrays)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
