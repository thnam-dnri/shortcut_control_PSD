#!/usr/bin/env python3
"""Run a controlled corrected-input Compact CNN versus DS-CNN comparison.

The runner deliberately loads only the development training manifest.  It uses
the frozen positive-polarity Compact representation and fixed three-peak loss
weights so that the first comparison changes the model topology, not the data
contract or objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
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
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820"
DEFAULT_REFERENCE_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/three_peak_positive_polarity_20260820/compact_cnn_best.pt"
)
DEFAULT_EVENT_STORE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/event_store/architecture_pass_warn_20260815_source_ablation/event_store_manifest.json"
)

MODEL_NAMES = ("compact_cnn", "ds_cnn")
PEAK_WEIGHT_KEYS = {
    "ba133_356kev": "ba356",
    "na22_511kev": "na511",
    "cs137_662kev": "cs662",
}
SELECTED_PEAK_WEIGHTS = {"ba356": 0.4, "na511": 0.4, "cs662": 0.2}
EXPECTED_REPRESENTATION_NAME = "both_ma10_global_t10_w750_positive_polarity"


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


def validate_split(fit_pairs: np.ndarray, internal_pairs: np.ndarray, total_pairs: int) -> None:
    fit_pairs = np.asarray(fit_pairs, dtype=np.int64)
    internal_pairs = np.asarray(internal_pairs, dtype=np.int64)
    if np.intersect1d(fit_pairs, internal_pairs).size:
        raise ValueError("Fit and internal pair splits overlap")
    combined = np.sort(np.concatenate((fit_pairs, internal_pairs)))
    expected = np.arange(total_pairs, dtype=np.int64)
    if not np.array_equal(combined, expected):
        raise ValueError("Fit and internal pair splits do not cover the training manifest")


def make_event_weights(peak_ids: np.ndarray, peak_weights: dict[str, float]) -> np.ndarray:
    peak_ids = np.asarray(peak_ids)
    if peak_ids.size % 2:
        raise ValueError("Expected an even number of positive/negative events")
    pair_peak_ids = peak_ids[::2]
    if not np.array_equal(pair_peak_ids, peak_ids[1::2]):
        raise ValueError("Positive and negative members of a pair have different peak IDs")
    counts = Counter(pair_peak_ids.tolist())
    if set(counts) != set(PEAK_WEIGHT_KEYS):
        raise ValueError(f"Unexpected peak IDs: {dict(counts)}")
    pair_weights = np.asarray(
        [peak_weights[PEAK_WEIGHT_KEYS[peak_id]] / counts[peak_id] for peak_id in pair_peak_ids],
        dtype=np.float32,
    )
    return np.repeat(pair_weights, 2)


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


def build_model(name: str) -> nn.Module:
    if name == "compact_cnn":
        return CompactWaveformCNN(input_channels=2, width=24)
    if name == "ds_cnn":
        return DSCNN(input_channels=2, width=24)
    raise ValueError(f"Unknown model name: {name}")


def predict(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            logits = model(values.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(PEAK_WEIGHT_KEYS)):
        mask = peak_ids == peak_id
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
            "pair_count": int(np.count_nonzero(mask) // 2),
        }
    peak_aurocs = [float(item["auroc"]) for item in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(peak_aurocs)),
        "worst_peak_auroc": float(np.min(peak_aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak,
    }


def load_reference_contract(
    checkpoint_path: Path,
) -> tuple[RepresentationConfig, dict[str, list[float]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "compact_cnn":
        raise ValueError("Reference checkpoint is not a Compact CNN checkpoint")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Reference checkpoint has an unexpected test-data flag")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.name != EXPECTED_REPRESENTATION_NAME:
        raise ValueError(f"Unexpected reference representation: {config.name}")
    if config.pulse_polarity != "negative_to_positive":
        raise ValueError("Reference checkpoint does not use corrected polarity")
    if config.input_mode != "both" or config.window_length != 750:
        raise ValueError("Reference checkpoint does not use the frozen [2,750] contract")
    if checkpoint.get("selected_peak_weights") != SELECTED_PEAK_WEIGHTS:
        raise ValueError("Reference checkpoint peak weights do not match the frozen experiment")
    return config, checkpoint["feature_statistics"], checkpoint


def validate_representation_contract(values: np.ndarray, config: RepresentationConfig) -> None:
    expected_shape = (2, config.window_length)
    if values.ndim != 3 or tuple(values.shape[1:]) != expected_shape:
        raise ValueError(f"Unexpected representation shape {values.shape}; expected [N,{expected_shape[0]},{expected_shape[1]}]")
    if not np.all(np.isfinite(values)):
        raise ValueError("Representation contains nonfinite values")


def train_and_refit(
    name: str,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_peak_ids: np.ndarray,
    fit_events: np.ndarray,
    internal_events: np.ndarray,
    all_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    fit_weights = all_weights[fit_events]
    internal_weights = all_weights[internal_events]
    fit_loader = make_loader(
        train_values[fit_events],
        train_labels[fit_events],
        fit_weights,
        args.batch_size,
        True,
        args.seed,
    )
    internal_loader = make_loader(
        train_values[internal_events],
        train_labels[internal_events],
        internal_weights,
        args.batch_size,
        False,
        args.seed,
    )
    model = build_model(name).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_epoch = -1
    best_metric = -np.inf
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, fit_loader, optimizer, device)
        internal_scores = predict(model, internal_loader, device)
        internal_metrics = metric_summary(
            train_labels[internal_events],
            internal_scores,
            internal_weights,
            train_peak_ids[internal_events],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "internal": internal_metrics,
            }
        )
        current = float(internal_metrics["macro_auroc"])
        print(
            f"{name} epoch={epoch} train_loss={train_loss:.6f} "
            f"internal_macro={current:.6f} "
            f"internal_worst={internal_metrics['worst_peak_auroc']:.6f}",
            flush=True,
        )
        if current > best_metric + 1.0e-4:
            best_metric = current
            best_epoch = epoch
            best_metrics = internal_metrics
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None or best_metrics is None or best_epoch < 1:
        raise RuntimeError(f"No internal checkpoint selected for {name}")
    del model, optimizer, fit_loader, internal_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    refit_seed = args.seed + 100000
    set_seed(refit_seed)
    refit_loader = make_loader(
        train_values,
        train_labels,
        all_weights,
        args.batch_size,
        True,
        refit_seed,
    )
    refit_model = build_model(name).to(device)
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    refit_history: list[float] = []
    for _epoch in range(best_epoch):
        refit_history.append(train_epoch(refit_model, refit_loader, refit_optimizer, device))
    refit_internal_loader = make_loader(
        train_values[internal_events],
        train_labels[internal_events],
        internal_weights,
        args.batch_size,
        False,
        refit_seed,
    )
    refit_internal_scores = predict(refit_model, refit_internal_loader, device)
    refit_internal_metrics = metric_summary(
        train_labels[internal_events],
        refit_internal_scores,
        internal_weights,
        train_peak_ids[internal_events],
    )
    checkpoint_path = output_dir / name / f"{name}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "1",
        "model_kind": name,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in refit_model.state_dict().items()
        },
        "model_width": 24,
        "parameter_count": parameter_count,
        "representation_config": args.representation_config,
        "feature_statistics": args.feature_statistics,
        "selected_peak_weights": SELECTED_PEAK_WEIGHTS,
        "selection_metric": "internal_equal_peak_macro_auroc",
        "scan_best_epoch": best_epoch,
        "scan_best_internal_metrics": best_metrics,
        "scan_history": history,
        "refit_seed": refit_seed,
        "refit_epochs": best_epoch,
        "refit_training_loss_history": refit_history,
        "refit_internal_metrics": refit_internal_metrics,
        "test_partition_used": False,
        "held_out_partition_loaded": False,
        "target_data_used_for_selection": False,
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "scan_best_epoch": best_epoch,
        "scan_best_internal_metrics": best_metrics,
        "refit_epochs": best_epoch,
        "refit_internal_metrics": refit_internal_metrics,
        "scan_history": history,
        "refit_training_loss_history": refit_history,
    }
    save_json(output_dir / name / "training_result.json", result)
    del refit_model, refit_optimizer, refit_loader, refit_internal_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--event-store-manifest", type=Path, default=DEFAULT_EVENT_STORE_MANIFEST)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def main() -> int:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    event_store_manifest = args.event_store_manifest.resolve()
    reference_checkpoint = args.reference_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    for path in (labels_dir, event_store_dir, event_store_manifest, reference_checkpoint, train_csv, split_path):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config, feature_statistics, reference_checkpoint_data = load_reference_contract(reference_checkpoint)
    args.representation_config = config.as_dict()
    args.feature_statistics = feature_statistics
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"loading development train manifest={train_csv}", flush=True)
    raw = load_raw_partition(train_csv, event_store_dir)
    if raw.labels.size % 2 or raw.labels.size != raw.peak_ids.size:
        raise ValueError("Invalid pair/event layout in training partition")
    split = np.load(split_path)
    fit_pairs = np.asarray(split["fit_pair_indices"], dtype=np.int64)
    internal_pairs = np.asarray(split["internal_pair_indices"], dtype=np.int64)
    validate_split(fit_pairs, internal_pairs, raw.labels.size // 2)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)
    all_weights = make_event_weights(raw.peak_ids, SELECTED_PEAK_WEIGHTS)

    print("building corrected Compact representation from development train data", flush=True)
    values, representation_qc = build_representation(raw, config)
    validate_representation_contract(values, config)
    recomputed_statistics = fit_channel_statistics(values)
    if not np.allclose(
        np.asarray(recomputed_statistics["means"]),
        np.asarray(feature_statistics["means"]),
        rtol=1.0e-5,
        atol=1.0e-7,
    ) or not np.allclose(
        np.asarray(recomputed_statistics["standard_deviations"]),
        np.asarray(feature_statistics["standard_deviations"]),
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("Frozen checkpoint statistics do not match the current train partition")
    apply_channel_statistics(values, feature_statistics)
    validate_representation_contract(values, config)
    args.representation_qc = representation_qc

    summary: dict[str, Any] = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "corrected Compact CNN versus DS-CNN performance comparison",
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "gradient_clip_norm": 5.0,
        },
        "representation_config": args.representation_config,
        "feature_statistics": feature_statistics,
        "representation_qc": representation_qc,
        "selected_peak_weights": SELECTED_PEAK_WEIGHTS,
        "input": {
            "labels_dir": labels_dir.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "split_path": split_path.relative_to(PROJECT_ROOT).as_posix(),
            "split_sha256": sha256_file(split_path),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
            "event_store_manifest": event_store_manifest.relative_to(PROJECT_ROOT).as_posix(),
            "event_store_manifest_sha256": sha256_file(event_store_manifest),
            "reference_checkpoint": reference_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
            "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
            "reference_checkpoint_model_kind": reference_checkpoint_data.get("model_kind"),
        },
        "partition": {
            "total_pair_count": int(raw.labels.size // 2),
            "fit_pair_count": int(fit_pairs.size),
            "internal_pair_count": int(internal_pairs.size),
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "target_data_used_for_selection": False,
        },
        "models": {},
    }
    save_json(output_dir / "experiment_config.json", summary)

    for model_index, name in enumerate(MODEL_NAMES):
        set_seed(args.seed + model_index)
        print(f"starting model={name}", flush=True)
        summary["models"][name] = train_and_refit(
            name,
            values,
            raw.labels,
            raw.peak_ids,
            fit_events,
            internal_events,
            all_weights,
            args,
            device,
            output_dir,
        )
        save_json(output_dir / "experiment_config.json", summary)

    print(json.dumps(summary["models"], indent=2, sort_keys=True), flush=True)
    print(f"wrote={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
