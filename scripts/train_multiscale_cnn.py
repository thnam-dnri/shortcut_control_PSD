#!/usr/bin/env python3
"""Train a multiscale residual CNN on aligned charge and its derivative.

This is a validation-only alternative to the provisional late-fusion network.  It
uses a single two-channel stream so charge morphology and current-like edges remain
temporally registered.  Only train and validation manifests are opened.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_o2_late_fusion import (  # noqa: E402
    PartitionData,
    build_partition_features,
    per_peak_metrics,
    sha256_file,
)

SEED = 20260811


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_registered_inputs(
    train: PartitionData,
    validation: PartitionData,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Create charge/current channels and apply train-only channel scaling."""
    train_current = np.gradient(train.charge, 4.0, axis=1).astype(np.float32)
    validation_current = np.gradient(validation.charge, 4.0, axis=1).astype(np.float32)
    train_values = np.stack((train.charge, train_current), axis=1)
    validation_values = np.stack((validation.charge, validation_current), axis=1)
    means = np.mean(train_values, axis=(0, 2), dtype=np.float64)
    standard_deviations = np.std(train_values, axis=(0, 2), dtype=np.float64)
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0):
        raise ValueError("Invalid train-only channel standard deviation")
    train_values -= means[None, :, None]
    train_values /= standard_deviations[None, :, None]
    validation_values -= means[None, :, None]
    validation_values /= standard_deviations[None, :, None]
    return train_values, validation_values, {
        "channel_mean": means.tolist(),
        "channel_standard_deviation": standard_deviations.tolist(),
    }


def normalization(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        padding = 3 * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            normalization(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            normalization(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.layers(values) + self.skip(values))


class MultiscaleResidualCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=15, stride=2, padding=7, bias=False),
            normalization(32),
            nn.GELU(),
            ResidualBlock(32, 32, stride=1, dilation=1),
            ResidualBlock(32, 64, stride=2, dilation=1),
            ResidualBlock(64, 64, stride=1, dilation=2),
            ResidualBlock(64, 96, stride=2, dilation=1),
            ResidualBlock(96, 96, stride=1, dilation=2),
            ResidualBlock(96, 96, stride=1, dilation=4),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        features = self.features(values)
        pooled = torch.cat(
            (
                torch.mean(features, dim=2),
                torch.amax(features, dim=2),
            ),
            dim=1,
        )
        return self.head(pooled).squeeze(1)


def make_loader(
    values: np.ndarray,
    data: PartitionData,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    dataset = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(data.labels),
        torch.from_numpy(data.weights),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        for values, target, sample_weight in loader:
            values = values.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            logits = model(values)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss_sum += float((losses * sample_weight).sum().item())
            labels.append(target.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            weights.append(sample_weight.cpu().numpy())
    label_array = np.concatenate(labels)
    score_array = np.concatenate(scores)
    weight_array = np.concatenate(weights)
    return (
        loss_sum / float(np.sum(weight_array)),
        metrics(label_array, score_array, weight_array),
        label_array,
        score_array,
        weight_array,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    weight_sum = 0.0
    for values, target, sample_weight in loader:
        values = values.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        loss = (losses * sample_weight).sum() / sample_weight.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += float((losses * sample_weight).sum().item())
        weight_sum += float(sample_weight.sum().item())
    return loss_sum / weight_sum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "outputs/labels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/multiscale_registered_cnn",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-events-per-partition", type=int, default=None)
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=None,
        help="Optional consolidated event-store directory for faster waveform reads.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if min(args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError("epochs, batch-size, and patience must be positive")
    set_seed(SEED)

    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    dataset_manifest = labels_dir / "label_dataset_manifest.json"
    partition_manifest = labels_dir / "file_partition_manifest.json"
    for path in (train_csv, validation_csv, dataset_manifest, partition_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading train representations ...", flush=True)
    event_store_dir = (
        args.event_store_dir.resolve() if args.event_store_dir is not None else None
    )
    train_data = build_partition_features(
        train_csv,
        args.max_events_per_partition,
        event_store_dir=event_store_dir,
    )
    print("Loading validation representations ...", flush=True)
    validation_data = build_partition_features(
        validation_csv,
        args.max_events_per_partition,
        event_store_dir=event_store_dir,
    )
    train_values, validation_values, statistics = build_registered_inputs(
        train_data, validation_data
    )
    train_loader = make_loader(
        train_values, train_data, args.batch_size, True, SEED
    )
    train_eval_loader = make_loader(
        train_values, train_data, args.batch_size, False, SEED
    )
    validation_loader = make_loader(
        validation_values, validation_data, args.batch_size, False, SEED
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiscaleResidualCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.0e-4, weight_decay=3.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = output_dir / "multiscale_registered_cnn_best.pt"
    history: list[dict[str, Any]] = []
    best_metric = -np.inf
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        _, train_metrics, _, _, _ = evaluate(model, train_eval_loader, device)
        validation_loss, validation_metrics, _, _, _ = evaluate(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        current = validation_metrics["weighted_auroc"]
        print(
            f"epoch={epoch:02d} train_auroc={train_metrics['auroc']:.5f} "
            f"val_auroc={validation_metrics['auroc']:.5f} "
            f"val_weighted_auroc={current:.5f}",
            flush=True,
        )
        if current > best_metric + 1.0e-4:
            best_metric = current
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "feature_statistics": statistics,
                    "seed": SEED,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop=epoch_{epoch}", flush=True)
                break
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_loss, train_metrics, train_labels, train_scores, train_weights = evaluate(
        model, train_eval_loader, device
    )
    validation_loss, validation_metrics, validation_labels, validation_scores, validation_weights = evaluate(
        model, validation_loader, device
    )
    internal_metrics = {
        "selection_metric": "validation_weighted_auroc",
        "best_epoch": best_epoch,
        "train": {
            **train_metrics,
            "loss": train_loss,
            "per_peak": per_peak_metrics(
                train_labels, train_scores, train_weights, train_data.peak_ids
            ),
        },
        "validation": {
            **validation_metrics,
            "loss": validation_loss,
            "per_peak": per_peak_metrics(
                validation_labels,
                validation_scores,
                validation_weights,
                validation_data.peak_ids,
            ),
        },
    }
    run = {
        "created_utc": utc_now(),
        "model": {
            "name": "multiscale_registered_charge_current_residual_cnn",
            "parameter_count": parameter_count,
            "input_shape": [2, 750],
            "normalization": "GroupNorm",
            "pooling": "concatenated global mean and maximum",
        },
        "representation": {
            "charge": "frozen t10-aligned, energy-normalized, smoothed 750-sample charge window",
            "current": "4 ns numerical derivative of the aligned charge channel",
            "registration": "both channels share the charge t10 time axis",
            "train_only_statistics": statistics,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 6.0e-4,
            "weight_decay": 3.0e-4,
            "scheduler": "CosineAnnealingLR",
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "seed": SEED,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "input": {
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(validation_csv),
            "test_csv": "outputs/labels/label_pairs_test.csv",
            "test_partition_used": False,
            "label_dataset_manifest_sha256": sha256_file(dataset_manifest),
            "file_partition_manifest_sha256": sha256_file(partition_manifest),
            "train_event_count": int(train_data.labels.size),
            "validation_event_count": int(validation_data.labels.size),
            "max_events_per_partition": args.max_events_per_partition,
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix()
            if event_store_dir is not None
            else None,
        },
        "caveats": [
            "Internal file-disjoint validation metric used for model selection.",
            "Positive and negative classes originate from different isotope sources.",
            "Exploratory manifests retain WARN/FAIL/UNKNOWN QC files.",
            "The test partition was not opened or evaluated.",
        ],
    }
    save_json(output_dir / "internal_metrics.json", internal_metrics)
    save_json(output_dir / "training_history.json", history)
    save_json(output_dir / "training_run.json", run)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
