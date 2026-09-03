#!/usr/bin/env python3
"""Scan Ba-356/Na-511/Cs-662 training weights for compact and late-fusion CNNs.

The current PASS+WARN source-ablation train partition is split into a fitting
subset and an internal validation subset by peak.  The existing validation file
partition remains untouched until the final held-out evaluation.  Weight
selection uses equal-peak macro-AUROC, not a weight-dependent pooled metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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

from scripts.train_o2_late_fusion import (  # noqa: E402
    O2LateFusion,
    PartitionData,
    build_partition_features,
    evaluate as evaluate_late,
    train_epoch as train_epoch_late,
)
from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RawPartition,
    apply_channel_statistics,
    build_representation,
    evaluate_model,
    fit_channel_statistics,
    load_raw_partition,
    train_epoch as train_epoch_compact,
)

PEAKS = (
    ("ba356", "ba133", "ba133_356kev", 356.0129),
    ("na511", "na22", "na22_511kev", 510.99895),
    ("cs662", "cs137", "cs137_662kev", 661.657),
)
PEAK_IDS = tuple(item[2] for item in PEAKS)
PEAK_WEIGHT_KEYS = {peak_id: short_name for short_name, _source, peak_id, _energy in PEAKS}
COMPACT_REPRESENTATION = {
    "name": "both_ma10_global_t10_w750_positive_polarity",
    "input_mode": "both",
    "moving_average": 10,
    "normalization": "global",
    "anchor": "t10",
    "pre_samples": 250,
    "post_samples": 500,
    "pulse_polarity": "negative_to_positive",
    "standardization": "train_zscore",
    "downsample": 1,
}
COMPACT_WIDTH = 24
COMPACT_LEARNING_RATE = 8.0e-4
COMPACT_WEIGHT_DECAY = 3.0e-4
LATE_LEARNING_RATE = 1.0e-3
LATE_WEIGHT_DECAY = 1.0e-4
WARNING_STATUS = "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


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


def read_selected_rows(label_root: Path, partition: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for short_name, source, peak_id, _energy in PEAKS:
        csv_path = label_root / f"{source}_positive" / f"label_pairs_{partition}.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            source_rows = [
                dict(row)
                for row in csv.DictReader(stream)
                if row["peak_id"] == peak_id
            ]
        for index, row in enumerate(source_rows):
            row["pair_id"] = f"{short_name}_{partition}_{index:08d}"
            row["source_weight"] = "1.0"
            rows.append(row)
    if not rows:
        raise ValueError(f"No selected three-peak rows found for {partition}")
    return rows


def write_combined_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty manifest: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def peak_pair_ids(rows: list[dict[str, str]]) -> np.ndarray:
    values = np.asarray([row["peak_id"] for row in rows], dtype="U64")
    return values


def split_train_pairs(
    rows: list[dict[str, str]],
    internal_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < internal_fraction < 1.0:
        raise ValueError("internal-fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    pair_peaks = peak_pair_ids(rows)
    fit: list[int] = []
    internal: list[int] = []
    for peak_id in PEAK_IDS:
        available = np.flatnonzero(pair_peaks == peak_id)
        if available.size < 2:
            raise ValueError(f"Too few pairs for {peak_id}")
        shuffled = available.copy()
        rng.shuffle(shuffled)
        internal_count = max(1, int(round(available.size * internal_fraction)))
        internal.extend(shuffled[:internal_count].tolist())
        fit.extend(shuffled[internal_count:].tolist())
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(internal), dtype=np.int64)


def select_scan_pairs(
    fit_pair_indices: np.ndarray,
    rows: list[dict[str, str]],
    limit_per_peak: int,
    seed: int,
) -> np.ndarray:
    if limit_per_peak < 1:
        raise ValueError("scan-pair-limit-per-peak must be positive")
    rng = np.random.default_rng(seed)
    pair_peaks = peak_pair_ids(rows)
    selected: list[int] = []
    for peak_id in PEAK_IDS:
        available = fit_pair_indices[pair_peaks[fit_pair_indices] == peak_id]
        count = min(limit_per_peak, available.size)
        selected.extend(rng.choice(available, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def pair_count_by_peak(rows: list[dict[str, str]]) -> dict[str, int]:
    return {peak_id: int(sum(row["peak_id"] == peak_id for row in rows)) for peak_id in PEAK_IDS}


def make_event_weights(
    pair_peaks: np.ndarray,
    peak_weights: dict[str, float],
) -> np.ndarray:
    counts = Counter(pair_peaks.tolist())
    if any(counts[peak_id] == 0 for peak_id in PEAK_IDS):
        raise ValueError(f"Missing peak in weight partition: {counts}")
    pair_weights = np.asarray(
        [
            peak_weights[PEAK_WEIGHT_KEYS[peak_id]] / counts[peak_id]
            for peak_id in pair_peaks
        ],
        dtype=np.float32,
    )
    return np.repeat(pair_weights, 2)


def weight_grid(step: float) -> list[dict[str, float]]:
    steps = int(round(1.0 / step))
    if steps < 1 or not np.isclose(steps * step, 1.0, atol=1.0e-8):
        raise ValueError("grid-step must divide one exactly")
    grid: list[dict[str, float]] = []
    for ba_index in range(steps + 1):
        for na_index in range(steps - ba_index + 1):
            cs_index = steps - ba_index - na_index
            grid.append(
                {
                    "ba356": ba_index / steps,
                    "na511": na_index / steps,
                    "cs662": cs_index / steps,
                }
            )
    return grid


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in PEAK_IDS:
        mask = peak_ids == peak_id
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
            "pair_count": int(np.count_nonzero(mask) // 2),
        }
    macro_auroc = float(np.mean([item["auroc"] for item in per_peak.values()]))
    return {
        "macro_auroc": macro_auroc,
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak,
    }


def make_loader(
    kind: str,
    features: tuple[np.ndarray, ...],
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    tensors = [torch.from_numpy(feature) for feature in features]
    tensors.extend((torch.from_numpy(labels), torch.from_numpy(weights)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def predict(
    kind: str,
    model: nn.Module,
    features: tuple[np.ndarray, ...],
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(kind, features, labels, weights, batch_size, False, 0)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = [item.to(device, non_blocking=True) for item in batch[:-2]]
            logits = model(*inputs) if kind == "late_fusion" else model(inputs[0])
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def build_model(kind: str, device: torch.device) -> nn.Module:
    if kind == "compact_cnn":
        return CompactWaveformCNN(2, width=COMPACT_WIDTH).to(device)
    if kind == "late_fusion":
        return O2LateFusion().to(device)
    raise ValueError(f"Unknown model kind: {kind}")


def train_model_epoch(
    kind: str,
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    if kind == "compact_cnn":
        return train_epoch_compact(model, loader, optimizer, device)
    return train_epoch_late(model, loader, optimizer, device)


def scan_architecture(
    kind: str,
    train_features: tuple[np.ndarray, ...],
    train_labels: np.ndarray,
    train_peak_ids: np.ndarray,
    internal_features: tuple[np.ndarray, ...],
    internal_labels: np.ndarray,
    internal_peak_ids: np.ndarray,
    combinations: list[dict[str, float]],
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    learning_rate = COMPACT_LEARNING_RATE if kind == "compact_cnn" else LATE_LEARNING_RATE
    weight_decay = COMPACT_WEIGHT_DECAY if kind == "compact_cnn" else LATE_WEIGHT_DECAY
    train_pair_peaks = train_peak_ids[::2]
    internal_pair_peaks = internal_peak_ids[::2]
    for combination_index, peak_weights in enumerate(combinations):
        print(
            f"scan model={kind} combination={combination_index + 1}/{len(combinations)} "
            f"weights={peak_weights}",
            flush=True,
        )
        set_seed(seed)
        train_weights = make_event_weights(train_pair_peaks, peak_weights)
        internal_weights = make_event_weights(internal_pair_peaks, peak_weights)
        train_loader = make_loader(
            kind,
            train_features,
            train_labels,
            train_weights,
            batch_size,
            True,
            seed,
        )
        internal_loader = make_loader(
            kind,
            internal_features,
            internal_labels,
            internal_weights,
            batch_size,
            False,
            seed,
        )
        model = build_model(kind, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_macro = -np.inf
        best_epoch = -1
        best_metrics: dict[str, Any] | None = None
        stale = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            train_loss = train_model_epoch(kind, model, train_loader, optimizer, device)
            if kind == "compact_cnn":
                _metrics, internal_scores = evaluate_model(model, internal_loader, device)
            else:
                _loss, _metrics, _labels, internal_scores = evaluate_late(
                    model, internal_loader, device
                )
            internal_metrics = metric_summary(
                internal_labels,
                internal_scores,
                internal_weights,
                internal_peak_ids,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "internal_macro_auroc": internal_metrics["macro_auroc"],
                    "internal_pooled_auroc": internal_metrics["pooled_auroc"],
                }
            )
            current = float(internal_metrics["macro_auroc"])
            print(
                f"  epoch={epoch} internal_macro={current:.6f} "
                f"internal_pooled={internal_metrics['pooled_auroc']:.6f}",
                flush=True,
            )
            if current > best_macro + 1.0e-4:
                best_macro = current
                best_epoch = epoch
                best_metrics = internal_metrics
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_epoch < 1:
            raise RuntimeError("No internal checkpoint epoch was selected")
        if best_metrics is None:
            raise RuntimeError("Missing metrics for selected internal epoch")
        results.append(
            {
                "weights": peak_weights,
                "best_epoch": best_epoch,
                "selection_metric": "internal_equal_peak_macro_auroc",
                "internal": best_metrics,
                "history": history,
            }
        )
        del model, optimizer, train_loader, internal_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results.sort(
        key=lambda item: (
            item["internal"]["macro_auroc"],
            item["internal"]["pooled_auroc"],
        ),
        reverse=True,
    )
    return results


def refit_selected(
    kind: str,
    train_features: tuple[np.ndarray, ...],
    train_labels: np.ndarray,
    train_peak_ids: np.ndarray,
    selected: dict[str, Any],
    batch_size: int,
    seed: int,
    device: torch.device,
    output_path: Path,
    feature_statistics: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed + 100000)
    peak_weights = selected["weights"]
    train_weights = make_event_weights(train_peak_ids[::2], peak_weights)
    loader = make_loader(
        kind,
        train_features,
        train_labels,
        train_weights,
        batch_size,
        True,
        seed + 100000,
    )
    model = build_model(kind, device)
    learning_rate = COMPACT_LEARNING_RATE if kind == "compact_cnn" else LATE_LEARNING_RATE
    weight_decay = COMPACT_WEIGHT_DECAY if kind == "compact_cnn" else LATE_WEIGHT_DECAY
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    refit_epochs = int(selected["best_epoch"])
    history: list[float] = []
    for _epoch in range(refit_epochs):
        history.append(train_model_epoch(kind, model, loader, optimizer, device))
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_kind": kind,
        "selected_peak_weights": peak_weights,
        "refit_epochs": refit_epochs,
        "selection_metric": selected["selection_metric"],
        "selection_internal_metrics": selected["internal"],
        "feature_statistics": feature_statistics,
        "seed": seed + 100000,
        "test_partition_used": False,
        "training_loss_history": history,
    }
    if kind == "compact_cnn":
        checkpoint.update(
            {
                "representation_config": COMPACT_REPRESENTATION,
                "model_width": COMPACT_WIDTH,
            }
        )
    else:
        checkpoint["architecture"] = "O2_style_charge_current_late_fusion"
    torch.save(checkpoint, output_path)
    return model, {
        "checkpoint": relative(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "selected_peak_weights": peak_weights,
        "refit_epochs": refit_epochs,
        "training_loss_history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--output-label-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_weight_scan_20260819",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/three_peak_weight_scan_20260819",
    )
    parser.add_argument("--internal-fraction", type=float, default=0.20)
    parser.add_argument("--scan-pair-limit-per-peak", type=int, default=2500)
    parser.add_argument("--grid-step", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    label_root = args.label_root.resolve()
    event_store_dir = args.event_store_dir.resolve()
    output_label_dir = args.output_label_dir.resolve()
    output_model_dir = args.output_model_dir.resolve()
    for path in (output_label_dir, output_model_dir):
        if path.exists() and any(path.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch-size, epochs, and patience must be positive")

    train_rows = read_selected_rows(label_root, "train")
    external_rows = read_selected_rows(label_root, "validation")
    train_csv = output_label_dir / "label_pairs_train.csv"
    external_csv = output_label_dir / "label_pairs_validation.csv"
    write_combined_csv(train_csv, train_rows)
    write_combined_csv(external_csv, external_rows)
    fit_pairs, internal_pairs = split_train_pairs(
        train_rows, args.internal_fraction, args.seed
    )
    scan_pairs = select_scan_pairs(
        fit_pairs, train_rows, args.scan_pair_limit_per_peak, args.seed + 1
    )
    np.savez_compressed(
        output_label_dir / "train_internal_split_indices.npz",
        fit_pair_indices=fit_pairs,
        internal_pair_indices=internal_pairs,
        scan_pair_indices=scan_pairs,
    )
    source_manifests = {}
    for _short_name, source, _peak_id, _energy in PEAKS:
        manifest_path = label_root / f"{source}_positive" / "label_dataset_manifest.json"
        source_manifests[source] = {
            "path": relative(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    label_manifest = {
        "created_utc": utc_now(),
        "positive_peaks": [
            {
                "short_name": short_name,
                "source": source,
                "peak_id": peak_id,
                "nominal_energy_kev": energy,
            }
            for short_name, source, peak_id, energy in PEAKS
        ],
        "source_manifests": source_manifests,
        "train_csv_sha256": sha256_file(train_csv),
        "external_validation_csv_sha256": sha256_file(external_csv),
        "partitions": {
            "train": {
                "pair_count": len(train_rows),
                "pair_count_by_peak": pair_count_by_peak(train_rows),
            },
            "external_validation": {
                "pair_count": len(external_rows),
                "pair_count_by_peak": pair_count_by_peak(external_rows),
                "source_partition": "existing validation file partition",
            },
        },
        "internal_split": {
            "fit_pair_count": int(fit_pairs.size),
            "internal_pair_count": int(internal_pairs.size),
            "fit_pair_count_by_peak": pair_count_by_peak(
                [train_rows[index] for index in fit_pairs]
            ),
            "internal_pair_count_by_peak": pair_count_by_peak(
                [train_rows[index] for index in internal_pairs]
            ),
            "internal_fraction": args.internal_fraction,
            "seed": args.seed,
        },
        "scan_subset": {
            "pair_count": int(scan_pairs.size),
            "pair_count_by_peak": pair_count_by_peak(
                [train_rows[index] for index in scan_pairs]
            ),
            "limit_per_peak": args.scan_pair_limit_per_peak,
            "seed": args.seed + 1,
        },
        "test_partition_used": False,
        "external_evaluation_warning": (
            "Held-out validation files are source/file-disjoint from the training "
            "partition but are not an independent isotope or session campaign."
        ),
    }
    save_json(output_label_dir / "label_dataset_manifest.json", label_manifest)

    combinations = weight_grid(args.grid_step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} combinations={len(combinations)}", flush=True)

    # Compact features and statistics are built from train plus held-out validation;
    # validation is not used to fit normalization statistics or select weights.
    # The representation builder expects the project dataclass, so import lazily here.
    from src.ba133_cnn import RepresentationConfig  # noqa: PLC0415

    compact_config = RepresentationConfig(**COMPACT_REPRESENTATION)
    compact_raw_train = load_raw_partition(train_csv, event_store_dir)
    compact_raw_external = load_raw_partition(external_csv, event_store_dir)
    compact_values_train, train_qc = build_representation(compact_raw_train, compact_config)
    compact_values_external, external_qc = build_representation(
        compact_raw_external, compact_config
    )
    compact_statistics = fit_channel_statistics(compact_values_train)
    apply_channel_statistics(compact_values_train, compact_statistics)
    apply_channel_statistics(compact_values_external, compact_statistics)
    compact_train_peaks = compact_raw_train.peak_ids
    compact_external_peaks = compact_raw_external.peak_ids
    compact_fit_events = event_indices(fit_pairs)
    compact_internal_events = event_indices(internal_pairs)
    compact_scan_events = event_indices(scan_pairs)
    compact_train_features = (compact_values_train[compact_scan_events],)
    compact_internal_features = (compact_values_train[compact_internal_events],)
    compact_all_train_features = (compact_values_train,)
    compact_external_features = (compact_values_external,)
    compact_scan_labels = compact_raw_train.labels[compact_scan_events]
    compact_internal_labels = compact_raw_train.labels[compact_internal_events]
    compact_all_train_labels = compact_raw_train.labels
    compact_external_labels = compact_raw_external.labels

    compact_scan_results = scan_architecture(
        "compact_cnn",
        compact_train_features,
        compact_scan_labels,
        compact_train_peaks[compact_scan_events],
        compact_internal_features,
        compact_internal_labels,
        compact_train_peaks[compact_internal_events],
        combinations,
        args.epochs,
        args.patience,
        args.batch_size,
        args.seed,
        device,
    )
    compact_selected = compact_scan_results[0]
    compact_checkpoint_path = output_model_dir / "compact_cnn_best.pt"
    compact_model, compact_refit = refit_selected(
        "compact_cnn",
        compact_all_train_features,
        compact_all_train_labels,
        compact_train_peaks,
        compact_selected,
        args.batch_size,
        args.seed,
        device,
        compact_checkpoint_path,
        compact_statistics,
    )
    compact_external_weights = make_event_weights(
        compact_external_peaks[::2], compact_selected["weights"]
    )
    compact_external_scores = predict(
        "compact_cnn",
        compact_model,
        compact_external_features,
        compact_external_labels,
        compact_external_weights,
        args.batch_size,
        device,
    )
    compact_external_metrics = metric_summary(
        compact_external_labels,
        compact_external_scores,
        compact_external_weights,
        compact_external_peaks,
    )
    del compact_model, compact_values_train, compact_values_external
    del compact_raw_train, compact_raw_external
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    late_train_data = build_partition_features(train_csv, event_store_dir=event_store_dir)
    late_external_data = build_partition_features(
        external_csv, event_store_dir=event_store_dir
    )
    late_train_stats: dict[str, float] = {}
    for name in ("charge", "current"):
        values = getattr(late_train_data, name)
        mean = float(np.mean(values, dtype=np.float64))
        std = float(np.std(values, dtype=np.float64))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Invalid late-fusion standard deviation for {name}")
        getattr(late_train_data, name)[:] = (values - mean) / std
        validation_values = getattr(late_external_data, name)
        validation_values[:] = (validation_values - mean) / std
        late_train_stats[f"{name}_mean"] = mean
        late_train_stats[f"{name}_std"] = std
    late_train_peaks = late_train_data.peak_ids
    late_external_peaks = late_external_data.peak_ids
    late_scan_data = PartitionData(
        late_train_data.charge[compact_scan_events],
        late_train_data.current[compact_scan_events],
        late_train_data.labels[compact_scan_events],
        late_train_data.weights[compact_scan_events],
        late_train_data.peak_ids[compact_scan_events],
        0,
    )
    late_internal_data = PartitionData(
        late_train_data.charge[compact_internal_events],
        late_train_data.current[compact_internal_events],
        late_train_data.labels[compact_internal_events],
        late_train_data.weights[compact_internal_events],
        late_train_data.peak_ids[compact_internal_events],
        0,
    )
    late_all_train_features = (late_train_data.charge, late_train_data.current)
    late_scan_features = (late_scan_data.charge, late_scan_data.current)
    late_internal_features = (late_internal_data.charge, late_internal_data.current)
    late_external_features = (late_external_data.charge, late_external_data.current)
    late_scan_results = scan_architecture(
        "late_fusion",
        late_scan_features,
        late_scan_data.labels,
        late_scan_data.peak_ids,
        late_internal_features,
        late_internal_data.labels,
        late_internal_data.peak_ids,
        combinations,
        args.epochs,
        args.patience,
        args.batch_size,
        args.seed,
        device,
    )
    late_selected = late_scan_results[0]
    late_checkpoint_path = output_model_dir / "late_fusion_best.pt"
    late_model, late_refit = refit_selected(
        "late_fusion",
        late_all_train_features,
        late_train_data.labels,
        late_train_peaks,
        late_selected,
        args.batch_size,
        args.seed,
        device,
        late_checkpoint_path,
        late_train_stats,
    )
    late_external_weights = make_event_weights(
        late_external_peaks[::2], late_selected["weights"]
    )
    late_external_scores = predict(
        "late_fusion",
        late_model,
        late_external_features,
        late_external_data.labels,
        late_external_weights,
        args.batch_size,
        device,
    )
    late_external_metrics = metric_summary(
        late_external_data.labels,
        late_external_scores,
        late_external_weights,
        late_external_peaks,
    )

    scan_summary = {
        "created_utc": utc_now(),
        "warning_status": WARNING_STATUS,
        "protocol": {
            "positive_peaks": [
                {"source": source, "peak_id": peak_id, "nominal_energy_kev": energy}
                for _short_name, source, peak_id, energy in PEAKS
            ],
            "training_partition": relative(train_csv),
            "internal_selection_partition": "deterministic 20% split of train pairs by peak",
            "held_out_external_partition": relative(external_csv),
            "weight_grid_step": args.grid_step,
            "weight_combinations": len(combinations),
            "selection_metric": "internal_equal_peak_macro_auroc",
            "target_data_used_for_selection": False,
            "test_partition_used": False,
        },
        "labels": {
            "manifest": relative(output_label_dir / "label_dataset_manifest.json"),
            "manifest_sha256": sha256_file(output_label_dir / "label_dataset_manifest.json"),
            "train_csv_sha256": sha256_file(train_csv),
            "external_csv_sha256": sha256_file(external_csv),
        },
        "training": {
            "device": str(device),
            "epochs_per_scan_combination": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "scan_pair_limit_per_peak": args.scan_pair_limit_per_peak,
            "seed": args.seed,
        },
        "architectures": {
            "compact_cnn": {
                "representation": COMPACT_REPRESENTATION,
                "scan_results_ranked": compact_scan_results,
                "selected": compact_selected,
                "refit": compact_refit,
                "external": compact_external_metrics,
                "representation_qc": {"train": train_qc, "external": external_qc},
            },
            "late_fusion": {
                "scan_results_ranked": late_scan_results,
                "selected": late_selected,
                "refit": late_refit,
                "external": late_external_metrics,
                "feature_statistics": late_train_stats,
                "representation_qc": {
                    "train_t10_fallback_count": late_train_data.t10_fallback_count,
                    "external_t10_fallback_count": late_external_data.t10_fallback_count,
                },
            },
        },
    }
    save_json(output_model_dir / "three_peak_weight_scan_summary.json", scan_summary)
    np.savez_compressed(
        output_model_dir / "held_out_external_scores.npz",
        compact_scores=compact_external_scores,
        late_fusion_scores=late_external_scores,
        labels=compact_external_labels.astype(np.int8),
        peak_ids=compact_external_peaks,
        compact_weights=compact_external_weights,
        late_fusion_weights=late_external_weights,
    )
    print(
        json.dumps(
            {
                "compact_selected_weights": compact_selected["weights"],
                "compact_internal_macro_auroc": compact_selected["internal"]["macro_auroc"],
                "compact_external_macro_auroc": compact_external_metrics["macro_auroc"],
                "late_selected_weights": late_selected["weights"],
                "late_internal_macro_auroc": late_selected["internal"]["macro_auroc"],
                "late_external_macro_auroc": late_external_metrics["macro_auroc"],
                "test_partition_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
