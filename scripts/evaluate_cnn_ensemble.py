#!/usr/bin/env python3
"""Evaluate and freeze a validation-selected ensemble of trained CNNs.

The scan combines predictions from the provisional late-fusion CNN and the
registered multiscale residual CNN.  It uses cached train/validation features and
never opens the test pair manifest or test waveform rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_multiscale_cnn import MultiscaleResidualCNN  # noqa: E402
from train_o2_late_fusion import O2LateFusion, per_peak_metrics, sha256_file  # noqa: E402


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak_metrics(labels, scores, weights, peak_ids),
    }


def predict_late_fusion(
    checkpoint_path: Path,
    charge: np.ndarray,
    current: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    statistics = checkpoint["feature_statistics"]
    charge_values = (charge - statistics["charge_mean"]) / statistics["charge_std"]
    current_values = (current - statistics["current_mean"]) / statistics["current_std"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(charge_values), torch.from_numpy(current_values)),
        batch_size=batch_size,
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


def predict_registered(
    checkpoint_path: Path,
    charge: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    statistics = checkpoint["feature_statistics"]
    registered_current = np.gradient(charge, 4.0, axis=1).astype(np.float32)
    values = np.stack((charge, registered_current), axis=1)
    means = np.asarray(statistics["channel_mean"], dtype=np.float32)
    standard_deviations = np.asarray(
        statistics["channel_standard_deviation"], dtype=np.float32
    )
    values -= means[None, :, None]
    values /= standard_deviations[None, :, None]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    model = MultiscaleResidualCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            scores.append(torch.sigmoid(model(batch.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(scores)


def load_partition(cache_dir: Path, partition: str) -> dict[str, np.ndarray]:
    with np.load(cache_dir / f"o2_features_{partition}.npz") as data:
        return {name: data[name] for name in ("charge", "current", "labels", "weights", "peak_ids")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/model_features",
    )
    parser.add_argument(
        "--late-fusion-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/o2_late_fusion/o2_late_fusion_best.pt",
    )
    parser.add_argument(
        "--registered-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/multiscale_registered_cnn/multiscale_registered_cnn_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/cnn_ensemble",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    late_checkpoint = args.late_fusion_checkpoint.resolve()
    registered_checkpoint = args.registered_checkpoint.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        cache_dir / "feature_cache_manifest.json",
        late_checkpoint,
        registered_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partitions: dict[str, dict[str, np.ndarray]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for partition in ("train", "validation"):
        print(f"Loading cached {partition} features ...", flush=True)
        data = load_partition(cache_dir, partition)
        partitions[partition] = data
        print(f"Scoring {partition} with both CNNs ...", flush=True)
        predictions[partition] = {
            "late_fusion": predict_late_fusion(
                late_checkpoint,
                data["charge"],
                data["current"],
                args.batch_size,
                device,
            ),
            "registered_residual": predict_registered(
                registered_checkpoint,
                data["charge"],
                args.batch_size,
                device,
            ),
        }

    validation = partitions["validation"]
    scan: list[dict[str, float]] = []
    for late_weight in np.linspace(0.0, 1.0, 101):
        scores = (
            late_weight * predictions["validation"]["late_fusion"]
            + (1.0 - late_weight) * predictions["validation"]["registered_residual"]
        )
        scan.append(
            {
                "late_fusion_weight": float(late_weight),
                "registered_residual_weight": float(1.0 - late_weight),
                "auroc": float(roc_auc_score(validation["labels"], scores)),
                "weighted_auroc": float(
                    roc_auc_score(
                        validation["labels"], scores, sample_weight=validation["weights"]
                    )
                ),
            }
        )
    selected = max(scan, key=lambda row: (row["weighted_auroc"], row["auroc"]))
    late_weight = selected["late_fusion_weight"]
    metrics: dict[str, Any] = {"selection_metric": "validation_weighted_auroc"}
    for partition in ("train", "validation"):
        data = partitions[partition]
        ensemble_scores = (
            late_weight * predictions[partition]["late_fusion"]
            + (1.0 - late_weight) * predictions[partition]["registered_residual"]
        )
        metrics[partition] = {
            "late_fusion": metric_summary(
                data["labels"], predictions[partition]["late_fusion"], data["weights"], data["peak_ids"]
            ),
            "registered_residual": metric_summary(
                data["labels"], predictions[partition]["registered_residual"], data["weights"], data["peak_ids"]
            ),
            "ensemble": metric_summary(
                data["labels"], ensemble_scores, data["weights"], data["peak_ids"]
            ),
        }
    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selected,
        "blend": "weighted arithmetic mean of sigmoid probabilities",
        "scan_step": 0.01,
        "late_fusion_checkpoint": late_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "late_fusion_checkpoint_sha256": sha256_file(late_checkpoint),
        "registered_checkpoint": registered_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "registered_checkpoint_sha256": sha256_file(registered_checkpoint),
        "feature_cache_manifest_sha256": sha256_file(cache_dir / "feature_cache_manifest.json"),
        "test_partition_used": False,
        "caveats": [
            "The ensemble weight was selected on internal validation and is model-selection biased.",
            "Positive and negative labels still originate from different isotope sources.",
            "Exploratory manifests retain WARN/FAIL/UNKNOWN QC files.",
        ],
    }
    save_json(output_dir / "internal_metrics.json", metrics)
    save_json(output_dir / "ensemble_config.json", configuration)
    save_json(output_dir / "weight_scan.json", scan)
    np.savez(
        output_dir / "validation_scores.npz",
        labels=validation["labels"],
        weights=validation["weights"],
        peak_ids=validation["peak_ids"],
        late_fusion_scores=predictions["validation"]["late_fusion"],
        registered_residual_scores=predictions["validation"]["registered_residual"],
        ensemble_scores=(
            late_weight * predictions["validation"]["late_fusion"]
            + (1.0 - late_weight) * predictions["validation"]["registered_residual"]
        ),
    )
    print(
        f"selected_late_weight={late_weight:.2f} "
        f"validation_auroc={metrics['validation']['ensemble']['auroc']:.6f} "
        f"validation_weighted_auroc={metrics['validation']['ensemble']['weighted_auroc']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
