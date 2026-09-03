#!/usr/bin/env python3
"""Repeat strict energy-matched DS-CNN training under frozen configurations."""

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
    RawPartition,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
    set_seed,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


REPRESENTATION = RepresentationConfig(
    name="both_ma10_global_t10_w750_positive_polarity",
    input_mode="both",
    moving_average=10,
    normalization="global",
    anchor="t10",
    pre_samples=250,
    post_samples=500,
    pulse_polarity="negative_to_positive",
    standardization="train_zscore",
)

CONFIGS: dict[str, dict[str, Any]] = {
    "six_peak_equal_weight": {
        "labels_dir": PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815",
        "event_store_dir": PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815",
        "peaks": (
            "ba133_276kev",
            "ba133_303kev",
            "ba133_356kev",
            "ba133_384kev",
            "na22_511kev",
            "cs137_662kev",
        ),
        "weights": {
            "ba133_276kev": 1.0 / 6.0,
            "ba133_303kev": 1.0 / 6.0,
            "ba133_356kev": 1.0 / 6.0,
            "ba133_384kev": 1.0 / 6.0,
            "na22_511kev": 1.0 / 6.0,
            "cs137_662kev": 1.0 / 6.0,
        },
        "split_mode": "derive_stratified_20_percent_internal",
    },
    "three_peak_manuscript_weight": {
        "labels_dir": PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
        "event_store_dir": PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
        "peaks": ("ba133_356kev", "na22_511kev", "cs137_662kev"),
        "weights": {
            "ba133_356kev": 0.4,
            "na22_511kev": 0.4,
            "cs137_662kev": 0.2,
        },
        "split_mode": "reuse_frozen_training_internal_split",
    },
    "three_peak_equal_weight": {
        "labels_dir": PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
        "event_store_dir": PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
        "peaks": ("ba133_356kev", "na22_511kev", "cs137_662kev"),
        "weights": {
            "ba133_356kev": 1.0 / 3.0,
            "na22_511kev": 1.0 / 3.0,
            "cs137_662kev": 1.0 / 3.0,
        },
        "split_mode": "reuse_frozen_training_internal_split",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_manifest(path: Path, expected_peaks: tuple[str, ...]) -> int:
    if "label_pairs_test" in path.name or "test" in path.parts:
        raise ValueError(f"Test manifest is forbidden: {path}")
    assert_development_csv(path)
    expected = set(expected_peaks)
    count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            if row["peak_id"] not in expected:
                raise ValueError(f"Unexpected peak {row['peak_id']} in {path}")
            if abs(float(row["positive_energy_kev"]) - float(row["negative_energy_kev"])) >= 0.5:
                raise ValueError(f"Pair exceeds strict 0.5-keV match: {row['pair_id']}")
    if count == 0:
        raise ValueError(f"Empty development manifest: {path}")
    return count


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pairs, 2 * pairs + 1)).reshape(-1)


def derive_split(pair_peak_ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit: list[int] = []
    internal: list[int] = []
    for peak_id in sorted(set(pair_peak_ids.tolist())):
        available = np.flatnonzero(pair_peak_ids == peak_id)
        shuffled = available.copy()
        rng.shuffle(shuffled)
        internal_count = max(1, int(round(available.size * fraction)))
        internal.extend(shuffled[:internal_count].tolist())
        fit.extend(shuffled[internal_count:].tolist())
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(internal), dtype=np.int64)


def load_split(labels_dir: Path, pair_peak_ids: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    split_path = labels_dir / "train_internal_split_indices.npz"
    if mode == "reuse_frozen_training_internal_split":
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        split = np.load(split_path)
        fit = np.asarray(split["fit_pair_indices"], dtype=np.int64)
        internal = np.asarray(split["internal_pair_indices"], dtype=np.int64)
    else:
        fit, internal = derive_split(pair_peak_ids, 0.20, seed)
    combined = np.sort(np.concatenate((fit, internal)))
    if not np.array_equal(combined, np.arange(pair_peak_ids.size)):
        raise ValueError("Internal split does not cover the training manifest exactly")
    return fit, internal


def make_event_weights(peak_ids: np.ndarray, peak_weights: dict[str, float]) -> np.ndarray:
    peak_ids = np.asarray(peak_ids)
    if peak_ids.size % 2 or not np.array_equal(peak_ids[::2], peak_ids[1::2]):
        raise ValueError("Pair layout is not positive/negative alternating")
    counts = Counter(peak_ids[::2].tolist())
    if set(counts) != set(peak_weights):
        raise ValueError(f"Unexpected peak IDs for weights: {dict(counts)}")
    pair_weights = np.asarray(
        [peak_weights[peak_id] / counts[peak_id] for peak_id in peak_ids[::2]],
        dtype=np.float32,
    )
    return np.repeat(pair_weights, 2)


def make_loader(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(values[indices]),
            torch.from_numpy(labels[indices].astype(np.float32, copy=False)),
            torch.from_numpy(weights[indices].astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
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
    for values, labels, weights in loader:
        values = values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = nn.functional.binary_cross_entropy_with_logits(
            model(values), labels, reduction="none"
        )
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += float((losses * weights).sum().item())
        weight_sum += float(weights.sum().item())
    return loss_sum / weight_sum


def predict(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            scores.append(torch.sigmoid(model(values.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(scores)


def metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, peak_ids: np.ndarray, peaks: tuple[str, ...]) -> dict[str, Any]:
    per_peak: dict[str, Any] = {}
    for peak in peaks:
        mask = peak_ids == peak
        per_peak[peak] = {
            "event_count": int(mask.sum()),
            "pair_count": int(mask.sum() // 2),
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
        }
    values = [item["auroc"] for item in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(values)),
        "worst_peak_auroc": float(np.min(values)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "per_peak": per_peak,
    }


def subset_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
    subset: tuple[str, ...],
) -> dict[str, Any]:
    mask = np.isin(peak_ids, subset)
    return metrics(labels[mask], scores[mask], weights[mask], peak_ids[mask], subset)


def run_seed(
    values: np.ndarray,
    labels: np.ndarray,
    peak_ids: np.ndarray,
    train_weights: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    validation_peak_ids: np.ndarray,
    validation_weights: np.ndarray,
    fit_events: np.ndarray,
    internal_events: np.ndarray,
    seed: int,
    output_dir: Path,
    config_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    feature_statistics: dict[str, list[float]],
) -> dict[str, Any]:
    set_seed(seed)
    fit_loader = make_loader(values, labels, train_weights, fit_events, args.batch_size, True, seed)
    internal_loader = make_loader(values, labels, train_weights, internal_events, args.batch_size, False, seed)
    model = DSCNN(input_channels=2, width=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_epoch = -1
    best_metric = -np.inf
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, fit_loader, optimizer, device)
        scores = predict(model, internal_loader, device)
        current = metrics(
            labels[internal_events], scores, train_weights[internal_events], peak_ids[internal_events], config["peaks"]
        )
        history.append({"epoch": epoch, "train_loss": loss, "internal": current})
        print(
            f"{config_name} seed={seed} epoch={epoch} train_loss={loss:.6f} "
            f"internal_macro={current['macro_auroc']:.6f}",
            flush=True,
        )
        if current["macro_auroc"] > best_metric + 1.0e-4:
            best_metric = current["macro_auroc"]
            best_epoch = epoch
            best_metrics = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No internal checkpoint was selected")
    del model, optimizer, fit_loader, internal_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    refit_seed = seed + 100000
    set_seed(refit_seed)
    all_events = np.arange(labels.size, dtype=np.int64)
    refit_loader = make_loader(values, labels, train_weights, all_events, args.batch_size, True, refit_seed)
    refit_model = DSCNN(input_channels=2, width=24).to(device)
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    refit_losses = [
        train_epoch(refit_model, refit_loader, refit_optimizer, device)
        for _epoch in range(best_epoch)
    ]
    refit_internal_loader = make_loader(
        values, labels, train_weights, internal_events, args.batch_size, False, refit_seed
    )
    refit_internal_scores = predict(refit_model, refit_internal_loader, device)
    refit_internal = metrics(
        labels[internal_events],
        refit_internal_scores,
        train_weights[internal_events],
        peak_ids[internal_events],
        config["peaks"],
    )
    validation_loader = make_loader(
        validation_values,
        validation_labels,
        validation_weights,
        np.arange(validation_labels.size, dtype=np.int64),
        args.batch_size,
        False,
        refit_seed,
    )
    validation_scores = predict(refit_model, validation_loader, device)
    validation_metrics = metrics(
        validation_labels, validation_scores, validation_weights, validation_peak_ids, config["peaks"]
    )
    primary = tuple(peak for peak in ("ba133_356kev", "na22_511kev", "cs137_662kev") if peak in config["peaks"])
    if primary != config["peaks"]:
        validation_metrics["primary_three_peak_subset"] = subset_metrics(
            validation_labels, validation_scores, validation_weights, validation_peak_ids, primary
        )

    checkpoint_path = output_dir / f"seed_{seed}" / "ds_cnn_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "model_kind": "ds_cnn",
        "model_state_dict": {key: value.detach().cpu() for key, value in refit_model.state_dict().items()},
        "model_width": 24,
        "parameter_count": sum(parameter.numel() for parameter in refit_model.parameters()),
        "representation_config": REPRESENTATION.as_dict(),
        "feature_statistics": feature_statistics,
        "selected_peak_weights": config["weights"],
        "selection_metric": "internal_equal_peak_macro_auroc",
        "scan_best_epoch": best_epoch,
        "scan_best_internal_metrics": best_metrics,
        "refit_seed": refit_seed,
        "refit_epochs": best_epoch,
        "refit_internal_metrics": refit_internal,
        "training_config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": 5.0,
        },
        "test_partition_used": False,
        "held_out_partition_loaded": False,
        "target_data_used_for_selection": False,
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "config": config_name,
        "seed": seed,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": checkpoint["parameter_count"],
        "scan_best_epoch": best_epoch,
        "scan_best_internal_metrics": best_metrics,
        "refit_internal_metrics": refit_internal,
        "validation_metrics": validation_metrics,
        "refit_training_loss_history": refit_losses,
        "scan_history": history,
    }
    save_json(checkpoint_path.parent / "training_result.json", result)
    del refit_model, refit_optimizer, refit_loader, refit_internal_loader, validation_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=tuple(CONFIGS), nargs="+", default=list(CONFIGS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/experiments/strict_ds_cnn_reproducibility_20260825")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def prepare_config(config_name: str, output_root: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = CONFIGS[config_name]
    labels_dir = config["labels_dir"].resolve()
    event_store_dir = config["event_store_dir"].resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    for path in (labels_dir, event_store_dir, train_csv, validation_csv):
        assert_no_forbidden_path(path)
    train_count = validate_manifest(train_csv, config["peaks"])
    validation_count = validate_manifest(validation_csv, config["peaks"])
    if train_count < 2 or validation_count < 2:
        raise ValueError("Development manifests are too small")
    config_dir = output_root / config_name
    if config_dir.exists() and any(config_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {config_dir}")
    config_dir.mkdir(parents=True, exist_ok=True)
    return config, {
        "labels_dir": labels_dir,
        "event_store_dir": event_store_dir,
        "train_csv": train_csv,
        "validation_csv": validation_csv,
        "config_dir": config_dir,
    }


def run_config(config_name: str, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    config, paths = prepare_config(config_name, args.output_dir.resolve(), args)
    print(f"loading {config_name} train={paths['train_csv']}", flush=True)
    train_raw = load_raw_partition(paths["train_csv"], paths["event_store_dir"])
    validation_raw = load_raw_partition(paths["validation_csv"], paths["event_store_dir"])
    if set(train_raw.peak_ids[::2]) != set(config["peaks"]):
        raise ValueError("Loaded training peak set does not match configuration")
    fit_pairs, internal_pairs = load_split(
        paths["labels_dir"], train_raw.peak_ids[::2], config["split_mode"], args.seeds[0]
    )
    np.savez_compressed(
        paths["config_dir"] / "train_internal_split_indices.npz",
        fit_pair_indices=fit_pairs,
        internal_pair_indices=internal_pairs,
    )
    print(f"building {config_name} representation", flush=True)
    train_values, train_qc = build_representation(train_raw, REPRESENTATION)
    validation_values, validation_qc = build_representation(validation_raw, REPRESENTATION)
    feature_statistics = fit_channel_statistics(train_values)
    apply_channel_statistics(train_values, feature_statistics)
    apply_channel_statistics(validation_values, feature_statistics)
    train_weights = make_event_weights(train_raw.peak_ids, config["weights"])
    validation_weights = make_event_weights(validation_raw.peak_ids, config["weights"])
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_name,
        "representation": REPRESENTATION.as_dict(),
        "feature_statistics": feature_statistics,
        "representation_qc": {"train": train_qc, "validation": validation_qc},
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seeds": args.seeds,
        },
        "peak_weights": config["weights"],
        "train_csv": paths["train_csv"].relative_to(PROJECT_ROOT).as_posix(),
        "train_csv_sha256": sha256_file(paths["train_csv"]),
        "validation_csv": paths["validation_csv"].relative_to(PROJECT_ROOT).as_posix(),
        "validation_csv_sha256": sha256_file(paths["validation_csv"]),
        "train_pair_count": int(train_raw.labels.size // 2),
        "validation_pair_count": int(validation_raw.labels.size // 2),
        "fit_pair_count": int(fit_pairs.size),
        "internal_pair_count": int(internal_pairs.size),
        "strict_match_rule": "absolute positive-negative corrected-energy difference < 0.5 keV",
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(paths["config_dir"] / "experiment_config.json", metadata)
    results = []
    for seed in args.seeds:
        results.append(
            run_seed(
                train_values,
                train_raw.labels,
                train_raw.peak_ids,
                train_weights,
                validation_values,
                validation_raw.labels,
                validation_raw.peak_ids,
                validation_weights,
                event_indices(fit_pairs),
                event_indices(internal_pairs),
                seed,
                paths["config_dir"],
                config_name,
                config,
                args,
                device,
                feature_statistics,
            )
        )
    save_json(paths["config_dir"] / "summary.json", {"config": metadata, "runs": results})
    del train_values, validation_values, train_raw, validation_raw
    return results


def main() -> int:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid training configuration")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    all_results: list[dict[str, Any]] = []
    for config_name in args.config:
        all_results.extend(run_config(config_name, args, device))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": all_results}
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir.resolve() / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
