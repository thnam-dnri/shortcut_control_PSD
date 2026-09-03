#!/usr/bin/env python3
"""Train energy-specialist DS-CNN classifiers on individual photopeak domains.

Each specialist (356A, 511A, 661A) is an identical DS-CNN (input_channels=2, width=24)
trained strictly on its own energy domain from the frozen PASS+WARN three-peak split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
    set_seed,
    train_epoch,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


SEED = 20260820
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/peak_specialist_ds_cnn_20260820"
DEFAULT_REFERENCE_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/three_peak_positive_polarity_20260820/compact_cnn_best.pt"
)

SPECIALISTS = (
    {
        "name": "356A",
        "peak_id": "ba133_356kev",
        "short_name": "ba356",
        "nominal_energy_kev": 356.0129,
        "source": "ba133",
    },
    {
        "name": "511A",
        "peak_id": "na22_511kev",
        "short_name": "na511",
        "nominal_energy_kev": 510.99895,
        "source": "na22",
    },
    {
        "name": "661A",
        "peak_id": "cs137_662kev",
        "short_name": "cs662",
        "nominal_energy_kev": 661.657,
        "source": "cs137",
    },
)

ALL_PEAK_IDS = [spec["peak_id"] for spec in SPECIALISTS]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def make_loader(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(values),
            torch.from_numpy(labels.astype(np.float32, copy=False)),
            torch.from_numpy(weights.astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def predict(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            logits = model(values.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def evaluate_metrics(labels: np.ndarray, scores: np.ndarray, peak_ids: np.ndarray) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in ALL_PEAK_IDS:
        mask = peak_ids == peak_id
        if np.count_nonzero(mask) == 0:
            continue
        sub_labels = labels[mask]
        sub_scores = scores[mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(sub_labels, sub_scores)),
            "average_precision": float(average_precision_score(sub_labels, sub_scores)),
            "event_count": int(np.count_nonzero(mask)),
            "pair_count": int(np.count_nonzero(mask) // 2),
        }
    peak_aurocs = [float(item["auroc"]) for item in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(peak_aurocs)) if peak_aurocs else 0.0,
        "worst_peak_auroc": float(np.min(peak_aurocs)) if peak_aurocs else 0.0,
        "pooled_auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "pooled_average_precision": float(average_precision_score(labels, scores))
        if len(np.unique(labels)) > 1
        else 0.0,
        "per_peak": per_peak,
    }


def load_reference_contract(
    checkpoint_path: Path,
) -> tuple[RepresentationConfig, dict[str, list[float]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    return config, checkpoint["feature_statistics"], checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.labels_dir / "label_pairs_train.csv"
    split_path = args.labels_dir / "train_internal_split_indices.npz"
    label_manifest_path = args.labels_dir / "label_dataset_manifest.json"

    assert_development_csv(train_csv)
    assert_no_forbidden_path(train_csv)
    assert_no_forbidden_path(split_path)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    print(f"Using compute device: {device}")

    config, feature_stats, ref_ckpt = load_reference_contract(args.reference_checkpoint)
    print(f"Loaded reference representation contract: {config.name} (pulse_polarity={config.pulse_polarity})")

    # Load raw partition
    print(f"Loading raw training events from {train_csv}...")
    raw = load_raw_partition(train_csv, args.event_store_dir)
    print(f"Building normalized representation for {raw.labels.size} events...")
    values, stats = build_representation(raw, config)
    apply_channel_statistics(values, feature_stats)
    print(f"Representation ready: shape={values.shape}, fallbacks={stats.get('fallback_count', 0)}")

    # Load split indices
    split = np.load(split_path)
    fit_pairs = split["fit_pair_indices"]
    internal_pairs = split["internal_pair_indices"]
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)

    train_labels = raw.labels
    train_peak_ids = raw.peak_ids
    weights_uniform = np.ones(train_labels.size, dtype=np.float32)

    # Save dataset manifest and experiment config
    dataset_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(train_csv.relative_to(PROJECT_ROOT)),
        "train_csv_sha256": sha256_file(train_csv),
        "split_path": str(split_path.relative_to(PROJECT_ROOT)),
        "split_path_sha256": sha256_file(split_path),
        "label_manifest_path": str(label_manifest_path.relative_to(PROJECT_ROOT)),
        "event_store_dir": str(args.event_store_dir.relative_to(PROJECT_ROOT)),
        "total_train_pairs": len(fit_pairs) + len(internal_pairs),
        "fit_pair_count": len(fit_pairs),
        "internal_pair_count": len(internal_pairs),
        "specialists": SPECIALISTS,
        "test_partition_used": False,
        "held_out_partition_loaded": False,
    }
    save_json(args.output_dir / "dataset_manifest.json", dataset_manifest)

    experiment_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "DSCNN",
        "model_width": 24,
        "input_channels": 2,
        "window_length": 750,
        "representation_config": config.as_dict(),
        "feature_statistics": feature_stats,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "specialists": [spec["name"] for spec in SPECIALISTS],
    }
    save_json(args.output_dir / "experiment_config.json", experiment_config)

    # Train each specialist
    for spec in SPECIALISTS:
        spec_name = spec["name"]
        peak_id = spec["peak_id"]
        print(f"\n=======================================================")
        print(f"Training Specialist {spec_name} for domain {peak_id} ({spec['nominal_energy_kev']} keV)")
        print(f"=======================================================")

        spec_seed = args.seed + int(spec["nominal_energy_kev"])
        set_seed(spec_seed)

        # Slice fit events and internal events strictly for this peak domain
        spec_fit_mask = train_peak_ids[fit_events] == peak_id
        spec_fit_events = fit_events[spec_fit_mask]

        spec_internal_mask = train_peak_ids[internal_events] == peak_id
        spec_internal_events = internal_events[spec_internal_mask]

        print(f"{spec_name} data: {len(spec_fit_events)//2} fit pairs ({len(spec_fit_events)} events), "
              f"{len(spec_internal_events)//2} internal pairs ({len(spec_internal_events)} events)")

        fit_loader = make_loader(
            values[spec_fit_events],
            train_labels[spec_fit_events],
            weights_uniform[spec_fit_events],
            args.batch_size,
            True,
            spec_seed,
        )

        # Loader for in-domain internal validation
        spec_internal_loader = make_loader(
            values[spec_internal_events],
            train_labels[spec_internal_events],
            weights_uniform[spec_internal_events],
            args.batch_size,
            False,
            spec_seed,
        )

        # Loader for all internal validation events (to track cross-domain performance during training)
        all_internal_loader = make_loader(
            values[internal_events],
            train_labels[internal_events],
            weights_uniform[internal_events],
            args.batch_size,
            False,
            spec_seed,
        )

        model = DSCNN(input_channels=2, width=24).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{spec_name} DS-CNN parameters: {param_count:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_in_domain_auroc = -np.inf
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        best_metrics: dict[str, Any] | None = None
        training_history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, fit_loader, optimizer, device)
            scheduler.step()

            # Predict on in-domain internal validation
            in_domain_scores = predict(model, spec_internal_loader, device)
            in_domain_auroc = float(roc_auc_score(train_labels[spec_internal_events], in_domain_scores))
            in_domain_ap = float(average_precision_score(train_labels[spec_internal_events], in_domain_scores))

            # Predict across all internal validation for tracking
            all_internal_scores = predict(model, all_internal_loader, device)
            all_internal_metrics = evaluate_metrics(
                train_labels[internal_events],
                all_internal_scores,
                train_peak_ids[internal_events],
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": scheduler.get_last_lr()[0],
                "in_domain_auroc": in_domain_auroc,
                "in_domain_ap": in_domain_ap,
                "all_internal_metrics": all_internal_metrics,
            }
            training_history.append(epoch_record)

            print(
                f"{spec_name} Epoch {epoch}/{args.epochs}: train_loss={train_loss:.6f} "
                f"in_domain_auroc={in_domain_auroc:.6f} "
                f"all_internal_macro_auroc={all_internal_metrics['macro_auroc']:.6f}",
                flush=True,
            )

            if in_domain_auroc > best_in_domain_auroc:
                best_in_domain_auroc = in_domain_auroc
                best_epoch = epoch
                best_metrics = epoch_record
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None or best_metrics is None or best_epoch < 1:
            raise RuntimeError(f"Training failed to select a valid checkpoint for {spec_name}")

        print(f"\n{spec_name} Best Epoch: {best_epoch} (In-domain AUROC: {best_in_domain_auroc:.6f})")

        # Save checkpoint
        checkpoint_path = args.output_dir / f"model_{spec_name}_checkpoint.pt"
        checkpoint_payload = {
            "schema_version": "1",
            "model_kind": "ds_cnn",
            "specialist_name": spec_name,
            "target_peak_id": peak_id,
            "target_nominal_energy_kev": spec["nominal_energy_kev"],
            "model_width": 24,
            "parameter_count": param_count,
            "representation_config": config.as_dict(),
            "feature_statistics": feature_stats,
            "selection_metric": "in_domain_internal_auroc",
            "best_epoch": best_epoch,
            "best_internal_in_domain_auroc": best_in_domain_auroc,
            "best_internal_metrics": best_metrics,
            "model_state_dict": best_state,
            "seed": spec_seed,
            "test_partition_used": False,
            "held_out_partition_loaded": False,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"Saved {checkpoint_path} (SHA-256: {sha256_file(checkpoint_path)})")

        # Save training JSON log
        log_payload = {
            "specialist_name": spec_name,
            "target_peak_id": peak_id,
            "target_nominal_energy_kev": spec["nominal_energy_kev"],
            "best_epoch": best_epoch,
            "best_in_domain_auroc": best_in_domain_auroc,
            "best_metrics": best_metrics,
            "history": training_history,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        save_json(args.output_dir / f"training_{spec_name}.json", log_payload)

        del model, optimizer, scheduler, fit_loader, spec_internal_loader, all_internal_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll 3 specialists trained successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
