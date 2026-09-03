#!/usr/bin/env python3
"""Train the fit-only Stage-1 DS-CNN used for cascade development mining.

The production baseline checkpoint is refit on the complete development
manifest.  It is therefore not suitable for defining an independent internal
ambiguous subset.  This script trains a deterministic six-epoch development
copy on the fit pairs only, with channel statistics also fit on fit events only.
The internal partition is scored for diagnosis but never used for training or
epoch selection.
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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
    representation_config_from_checkpoint,
    set_seed,
    train_epoch,
)
from src.cascade_refinement import (  # noqa: E402
    SELECTED_PEAK_WEIGHTS,
    event_indices,
    make_event_weights,
    save_json,
    sha256_file,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


EXPERIMENT_ID = "cascaded_ambiguous_refinement_ds_cnn_20260821"
SEED = 20260821
EPOCHS = 6
BATCH_SIZE = 256
LEARNING_RATE = 8.0e-4
WEIGHT_DECAY = 3.0e-4
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_REFERENCE_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID / "stage1_dev"


def validate_split(fit_pairs: np.ndarray, internal_pairs: np.ndarray, total_pairs: int) -> None:
    fit_pairs = np.asarray(fit_pairs, dtype=np.int64)
    internal_pairs = np.asarray(internal_pairs, dtype=np.int64)
    if np.intersect1d(fit_pairs, internal_pairs).size:
        raise ValueError("Fit and internal pair splits overlap")
    combined = np.sort(np.concatenate((fit_pairs, internal_pairs)))
    if not np.array_equal(combined, np.arange(total_pairs, dtype=np.int64)):
        raise ValueError("Fit and internal pair splits do not cover the train manifest")


def make_loader(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
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
        batch_size=BATCH_SIZE,
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
            scores.append(
                torch.sigmoid(model(values.to(device, non_blocking=True))).cpu().numpy()
            )
    return np.concatenate(scores)


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(peak_ids.tolist())):
        mask = peak_ids == peak_id
        if np.unique(labels[mask]).size < 2:
            raise ValueError(f"Internal metric stratum lacks both classes: {peak_id}")
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
        }
    aurocs = [float(value["auroc"]) for value in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "worst_peak_auroc": float(np.min(aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def load_stage1_contract(checkpoint_path: Path) -> tuple[RepresentationConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("Reference checkpoint is not a DS-CNN checkpoint")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Reference checkpoint has an invalid test boundary")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError("Reference checkpoint loaded held-out data")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.name != "both_ma10_global_t10_w750_positive_polarity":
        raise ValueError(f"Unexpected Stage-1 representation: {config.name}")
    return config, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    reference_checkpoint = args.reference_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    for path in (labels_dir, event_store_dir, reference_checkpoint, train_csv, split_path):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(SEED)
    config, reference = load_stage1_contract(reference_checkpoint)

    print(f"device={device}", flush=True)
    print(f"loading={train_csv}", flush=True)
    raw = load_raw_partition(train_csv, event_store_dir)
    split = np.load(split_path)
    fit_pairs = np.asarray(split["fit_pair_indices"], dtype=np.int64)
    internal_pairs = np.asarray(split["internal_pair_indices"], dtype=np.int64)
    validate_split(fit_pairs, internal_pairs, raw.labels.size // 2)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)
    weights = make_event_weights(raw.peak_ids)

    print("building Stage-1 representation", flush=True)
    values, representation_qc = build_representation(raw, config)
    fit_statistics = fit_channel_statistics(values[fit_events])
    apply_channel_statistics(values, fit_statistics)
    if not np.all(np.isfinite(values)):
        raise ValueError("Stage-1 development representation contains nonfinite values")

    model = DSCNN(input_channels=2, width=24).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fit_loader = make_loader(
        values[fit_events], raw.labels[fit_events], weights[fit_events], True, SEED
    )
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, fit_loader, optimizer, device)
        print(f"stage1_dev epoch={epoch} fit_loss={loss:.6f}", flush=True)

    internal_loader = make_loader(
        values[internal_events], raw.labels[internal_events], weights[internal_events], False, SEED
    )
    internal_scores = predict(model, internal_loader, device)
    internal_metrics = metric_summary(
        raw.labels[internal_events],
        internal_scores,
        weights[internal_events],
        raw.peak_ids[internal_events],
    )
    checkpoint_path = output_dir / "stage1_dev_best.pt"
    checkpoint = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "model_kind": "ds_cnn",
        "model_role": "stage1_development_only",
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_width": 24,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "representation_config": config.as_dict(),
        "feature_statistics": fit_statistics,
        "selected_peak_weights": SELECTED_PEAK_WEIGHTS,
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "selection_rule": "fixed six epochs; internal metrics diagnostic only",
        },
        "partition": {
            "total_pair_count": int(raw.labels.size // 2),
            "fit_pair_count": int(fit_pairs.size),
            "internal_pair_count": int(internal_pairs.size),
            "fit_only_training": True,
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "target_data_used_for_selection": False,
        },
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
        "representation_qc": representation_qc,
        "internal_diagnostic_metrics": internal_metrics,
        "reference_checkpoint": reference_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "reference_checkpoint_training_metadata": {
            "model_kind": reference.get("model_kind"),
            "refit_epochs": reference.get("refit_epochs"),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": checkpoint["parameter_count"],
        "partition": checkpoint["partition"],
        "representation_config": config.as_dict(),
        "feature_statistics": fit_statistics,
        "representation_qc": representation_qc,
        "internal_diagnostic_metrics": internal_metrics,
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
    }
    save_json(output_dir / "stage1_dev_training.json", result)
    np.savez_compressed(
        output_dir / "stage1_dev_internal_scores.npz",
        labels=raw.labels[internal_events],
        peak_ids=raw.peak_ids[internal_events],
        weights=weights[internal_events],
        scores=internal_scores,
        event_indices=internal_events,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
