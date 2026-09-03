#!/usr/bin/env python3
"""Rescore source-ablation models overall and by negative continuum source."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_boosting_baselines import tabular_features  # noqa: E402
from train_o2_late_fusion import (  # noqa: E402
    O2LateFusion,
    PartitionData,
    build_partition_features,
    sha256_file,
)

SOURCES = ("ba133", "na22", "cs137")


def metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "event_count": int(labels.size),
        "pair_count": int(labels.size // 2),
    }


def cnn_scores(
    checkpoint_path: Path,
    data: PartitionData,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    statistics = checkpoint["feature_statistics"]
    charge = (data.charge - statistics["charge_mean"]) / statistics["charge_std"]
    current = (data.current - statistics["current_mean"]) / statistics["current_std"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(charge), torch.from_numpy(current)),
        batch_size=512,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    model = O2LateFusion().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for charge_batch, current_batch in loader:
            logits = model(
                charge_batch.to(device, non_blocking=True),
                current_batch.to(device, non_blocking=True),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def grouped_metrics(
    pair_rows: list[dict[str, str]],
    data: PartitionData,
    scores: np.ndarray,
) -> dict[str, Any]:
    if len(pair_rows) * 2 != scores.size:
        raise ValueError("Pair CSV and event-score lengths disagree")
    if not np.array_equal(data.labels.reshape(-1, 2), np.asarray([[1.0, 0.0]] * len(pair_rows))):
        raise ValueError("Expected positive/negative pair ordering")
    result: dict[str, Any] = {
        "overall": metrics(data.labels, scores, data.weights),
        "by_negative_source": {},
        "by_peak": {},
        "by_peak_and_negative_source": {},
    }
    negative_sources = np.asarray([row["negative_source"] for row in pair_rows])
    peak_ids = np.asarray([row["peak_id"] for row in pair_rows])

    def event_indices(pair_mask: np.ndarray) -> np.ndarray:
        pair_indices = np.flatnonzero(pair_mask)
        return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)

    for negative_source in sorted(set(negative_sources.tolist())):
        indices = event_indices(negative_sources == negative_source)
        result["by_negative_source"][negative_source] = metrics(
            data.labels[indices], scores[indices], data.weights[indices]
        )
    for peak_id in sorted(set(peak_ids.tolist())):
        indices = event_indices(peak_ids == peak_id)
        result["by_peak"][peak_id] = metrics(
            data.labels[indices], scores[indices], data.weights[indices]
        )
        result["by_peak_and_negative_source"][peak_id] = {}
        for negative_source in sorted(set(negative_sources[peak_ids == peak_id].tolist())):
            mask = (peak_ids == peak_id) & (negative_sources == negative_source)
            indices = event_indices(mask)
            result["by_peak_and_negative_source"][peak_id][negative_source] = metrics(
                data.labels[indices], scores[indices], data.weights[indices]
            )
    return result


def main() -> int:
    label_root = PROJECT_ROOT / "outputs/labels/source_ablation"
    model_root = PROJECT_ROOT / "outputs/models/source_ablation"
    output_path = model_root / "source_ablation_comparison.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "validation",
        "test_partition_used": False,
        "experiments": {},
    }
    for source in SOURCES:
        labels_dir = label_root / f"{source}_positive"
        experiment_root = model_root / f"{source}_positive"
        csv_path = labels_dir / "label_pairs_validation.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        print(f"Loading {source} validation features ...", flush=True)
        data = build_partition_features(csv_path)
        features = tabular_features(data)
        experiment: dict[str, Any] = {
            "validation_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(csv_path),
            "models": {},
        }
        for model_name in ("hist_gradient_boosting", "xgboost"):
            model_path = experiment_root / "boosting" / f"{model_name}.joblib"
            model = joblib.load(model_path)
            scores = model.predict_proba(features)[:, 1]
            experiment["models"][model_name] = grouped_metrics(
                pair_rows, data, scores
            )
        checkpoint_path = experiment_root / "o2_late_fusion/o2_late_fusion_best.pt"
        scores = cnn_scores(checkpoint_path, data, device)
        experiment["models"]["o2_late_fusion"] = grouped_metrics(
            pair_rows, data, scores
        )
        result["experiments"][source] = experiment
        print(
            source,
            {
                name: round(values["overall"]["auroc"], 6)
                for name, values in experiment["models"].items()
            },
            flush=True,
        )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
