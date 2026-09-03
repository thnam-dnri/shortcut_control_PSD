#!/usr/bin/env python3
"""Train the provisional FPGA-oriented architecture candidates.

This entrypoint uses only the frozen development train/validation manifests and
their versioned event stores. It implements a common two-channel, 750-sample
representation for DS-CNN, TCN, Multi-Rate HPGe, and CNN-GRU so that the current
architecture phase can proceed with an explicit shortcut warning. The results
are provisional and must not be presented as claim-grade physics ranking before
the frozen external AUROC and spectral P/B/retention evaluation.
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

from src.architecture_candidates import build_candidate  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    SCREEN_CONFIGS,
    RawPartition,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
    set_seed,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


CANDIDATES = ("ds_cnn", "tcn", "multi_rate_hpge", "cnn_gru")
SEED = 20260816
WARNING_STATUS = "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED"
REPRESENTATION_NAME = "both_ma10_energy_t10_w750"
BALANCED_OUTPUT_DIR_NAME = "architecture_candidates_warning_balanced_20260816"
REPRESENTATION = next(
    config for config in SCREEN_CONFIGS if config.name == REPRESENTATION_NAME
)


def warning_metadata(candidate: str | None = None) -> dict[str, Any]:
    """Return the explicit non-fatal warning and its later return criteria."""

    metadata: dict[str, Any] = {
        "status": WARNING_STATUS,
        "test_partition_used": False,
        "external_return_metrics": [
            "external_auroc",
            "external_spectral_peak_to_background",
            "external_photopeak_retention",
            "external_energy_coverage",
            "uncertainty_and_failure_checks",
        ],
        "action_if_inconsistent": (
            "return_to_shortcut_warning_audit_before_claim_grade_interpretation"
        ),
    }
    if candidate is not None:
        metadata["candidate"] = candidate
    return metadata


def select_peak_balanced_subset(
    raw: RawPartition,
    max_events: int | None,
    seed: int,
) -> tuple[RawPartition, dict[str, Any]]:
    """Select a reproducible equal-by-peak pair subset without changing labels."""

    if max_events is None or max_events >= raw.labels.size:
        return raw, {
            "mode": "full_partition",
            "requested_event_count": max_events,
            "selected_event_count": int(raw.labels.size),
            "selected_pair_count": int(raw.labels.size // 2),
            "peak_pair_counts": {
                peak: int(np.count_nonzero(raw.peak_ids[::2] == peak))
                for peak in sorted(set(raw.peak_ids[::2].tolist()))
            },
        }
    if max_events < 2:
        raise ValueError("max_events must be at least 2 for a balanced subset")

    target_pairs = max_events // 2
    pair_peak_ids = raw.peak_ids[::2]
    peaks = sorted(set(pair_peak_ids.tolist()))
    if target_pairs < len(peaks):
        raise ValueError(
            f"max_events={max_events} cannot allocate one pair to {len(peaks)} peaks"
        )
    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(target_pairs, len(peaks))
    selected_pairs: list[int] = []
    selected_counts: dict[str, int] = {}
    available: dict[str, np.ndarray] = {}
    for peak in peaks:
        available[peak] = np.flatnonzero(pair_peak_ids == peak)
        count = min(base_count, available[peak].size)
        selected_counts[peak] = count
        selected_pairs.extend(
            rng.choice(available[peak], size=count, replace=False).tolist()
        )

    remaining = target_pairs - len(selected_pairs)
    while remaining > 0:
        candidates = [
            peak for peak in peaks if selected_counts[peak] < available[peak].size
        ]
        if not candidates:
            break
        candidates.sort(key=lambda peak: (selected_counts[peak], peak))
        peak = candidates[0]
        chosen = set(selected_pairs)
        unselected = np.asarray(
            [index for index in available[peak].tolist() if index not in chosen],
            dtype=np.int64,
        )
        selected_pairs.append(int(rng.choice(unselected)))
        selected_counts[peak] += 1
        remaining -= 1

    if len(selected_pairs) != target_pairs:
        raise ValueError(
            f"Could not select {target_pairs} balanced pairs; selected {len(selected_pairs)}"
        )
    selected_pairs.sort()
    event_indices = np.column_stack(
        (2 * np.asarray(selected_pairs), 2 * np.asarray(selected_pairs) + 1)
    ).reshape(-1)
    selected = RawPartition(
        waveforms=raw.waveforms[event_indices],
        shaped_energy=raw.shaped_energy[event_indices],
        labels=raw.labels[event_indices],
        weights=raw.weights[event_indices],
        peak_ids=raw.peak_ids[event_indices],
    )
    return selected, {
        "mode": "equal_peak_pair_subset",
        "requested_event_count": int(max_events),
        "selected_event_count": int(selected.labels.size),
        "selected_pair_count": int(selected.labels.size // 2),
        "seed": int(seed),
        "peak_pair_counts": {
            peak: int(selected_counts[peak]) for peak in peaks
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
    }


def per_peak_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for peak_id in sorted({str(value) for value in peak_ids}):
        mask = peak_ids == peak_id
        if np.unique(labels[mask]).size < 2:
            result[peak_id] = {
                "event_count": int(np.count_nonzero(mask)),
                "auroc": None,
                "weighted_auroc": None,
            }
            continue
        result[peak_id] = {
            "event_count": int(np.count_nonzero(mask)),
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "weighted_auroc": float(
                roc_auc_score(labels[mask], scores[mask], sample_weight=weights[mask])
            ),
        }
    return result


def make_loader(
    values: np.ndarray,
    raw: RawPartition,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(raw.labels),
        torch.from_numpy(raw.weights),
    )
    return DataLoader(
        dataset,
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
    for values, target, sample_weight in loader:
        values = values.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        loss = (losses * sample_weight).sum() / sample_weight.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += float((losses * sample_weight).sum().item())
        weight_sum += float(sample_weight.sum().item())
    return loss_sum / weight_sum


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    loss_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for values, target, sample_weight in loader:
            values = values.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            logits = model(values)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
            loss_sum += float((losses * sample_weight).sum().item())
            weight_sum += float(sample_weight.sum().item())
            labels.append(target.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            weights.append(sample_weight.cpu().numpy())
    label_array = np.concatenate(labels)
    score_array = np.concatenate(scores)
    weight_array = np.concatenate(weights)
    metrics = metric_summary(label_array, score_array, weight_array)
    metrics["loss"] = loss_sum / weight_sum
    return metrics, label_array, score_array, weight_array


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def train_candidate(
    name: str,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    train_data: RawPartition,
    validation_data: RawPartition,
    statistics: dict[str, list[float]],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    candidate_dir = output_dir / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_values, train_data, args.batch_size, True, seed)
    train_eval_loader = make_loader(train_values, train_data, args.batch_size, False, seed)
    validation_loader = make_loader(
        validation_values,
        validation_data,
        args.batch_size,
        False,
        seed,
    )

    model = build_candidate(name, input_channels=2, width=args.width).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    checkpoint_path = candidate_dir / f"{name}_best.pt"
    best_metric = -np.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_metrics, _, _, _ = evaluate(model, train_eval_loader, device)
        validation_metrics, _, _, _ = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        current = validation_metrics["weighted_auroc"]
        print(
            f"{name} epoch={epoch:02d} train_auroc={train_metrics['auroc']:.5f} "
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
                    "architecture": name,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "input_shape": [2, REPRESENTATION.window_length],
                    "representation": REPRESENTATION.as_dict(),
                    "feature_statistics": statistics,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "warning_status": WARNING_STATUS,
                    "test_partition_used": False,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"{name} early_stop=epoch_{epoch}", flush=True)
                break
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics, train_labels, train_scores, train_weights = evaluate(
        model,
        train_eval_loader,
        device,
    )
    validation_metrics, validation_labels, validation_scores, validation_weights = evaluate(
        model,
        validation_loader,
        device,
    )
    internal_metrics = {
        "status": "PROVISIONAL_SHORTCUT_WARNING",
        "selection_metric": "validation_weighted_auroc",
        "warning_status": WARNING_STATUS,
        "warning": warning_metadata(name),
        "best_epoch": best_epoch,
        "train": {
            **train_metrics,
            "per_peak": per_peak_metrics(
                train_labels,
                train_scores,
                train_weights,
                train_data.peak_ids,
            ),
        },
        "validation": {
            **validation_metrics,
            "per_peak": per_peak_metrics(
                validation_labels,
                validation_scores,
                validation_weights,
                validation_data.peak_ids,
            ),
        },
    }
    save_json(candidate_dir / "internal_metrics.json", internal_metrics)
    save_json(candidate_dir / "training_history.json", history)
    return {
        "architecture": name,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "validation": validation_metrics,
        "status": "PROVISIONAL_SHORTCUT_WARNING",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / f"outputs/models/{BALANCED_OUTPUT_DIR_NAME}",
    )
    parser.add_argument("--candidate", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=6.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--max-events-per-partition", type=int, default=None)
    parser.add_argument(
        "--balanced-subset",
        action="store_true",
        help="Select an equal-by-peak pair subset instead of taking the CSV prefix.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.width) < 1:
        raise ValueError("epochs, patience, batch-size, and width must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.max_events_per_partition is not None and args.max_events_per_partition < 2:
        raise ValueError("max-events-per-partition must be at least 2")

    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    dataset_manifest = labels_dir / "label_dataset_manifest.json"
    partition_manifest = labels_dir / "file_partition_manifest.json"
    event_store_manifest = output_dir.parent.parent / "event_store" / "architecture_pass_warn_20260815" / "event_store_manifest.json"
    for path in (train_csv, validation_csv, dataset_manifest, partition_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (train_csv, validation_csv, labels_dir, event_store_dir):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    assert_development_csv(validation_csv)
    if not args.overwrite and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"representation={REPRESENTATION_NAME} input_shape=[2,{REPRESENTATION.window_length}]", flush=True)
    print("Loading train event store ...", flush=True)
    train_data = load_raw_partition(
        train_csv,
        event_store_dir,
        None if args.balanced_subset else args.max_events_per_partition,
    )
    print("Loading validation event store ...", flush=True)
    validation_data = load_raw_partition(
        validation_csv,
        event_store_dir,
        None if args.balanced_subset else args.max_events_per_partition,
    )
    train_selection: dict[str, Any]
    validation_selection: dict[str, Any]
    if args.balanced_subset:
        train_data, train_selection = select_peak_balanced_subset(
            train_data,
            args.max_events_per_partition,
            args.seed,
        )
        validation_data, validation_selection = select_peak_balanced_subset(
            validation_data,
            args.max_events_per_partition,
            args.seed + 1,
        )
        print(f"train_selection={train_selection}", flush=True)
        print(f"validation_selection={validation_selection}", flush=True)
    else:
        train_selection = {
            "mode": "csv_prefix_or_full",
            "requested_event_count": args.max_events_per_partition,
            "selected_event_count": int(train_data.labels.size),
        }
        validation_selection = {
            "mode": "csv_prefix_or_full",
            "requested_event_count": args.max_events_per_partition,
            "selected_event_count": int(validation_data.labels.size),
        }
    print("Building train representation ...", flush=True)
    train_values, train_representation_stats = build_representation(train_data, REPRESENTATION)
    print("Building validation representation ...", flush=True)
    validation_values, validation_representation_stats = build_representation(
        validation_data,
        REPRESENTATION,
    )
    statistics = fit_channel_statistics(train_values)
    apply_channel_statistics(train_values, statistics)
    apply_channel_statistics(validation_values, statistics)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PROVISIONAL_ARCHITECTURE_COMPARISON_WITH_SHORTCUT_WARNING",
        "warning_status": WARNING_STATUS,
        "warning": warning_metadata(),
        "created_utc": utc_now(),
        "device": str(device),
        "representation": REPRESENTATION.as_dict(),
        "representation_statistics": {
            "train": train_representation_stats,
            "validation": validation_representation_stats,
            "channel_statistics_fit_on": "train_only",
            "channel_statistics": statistics,
        },
        "input": {
            "labels_dir": labels_dir.relative_to(PROJECT_ROOT).as_posix(),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "validation_csv_sha256": sha256_file(validation_csv),
            "label_dataset_manifest_sha256": sha256_file(dataset_manifest),
            "file_partition_manifest_sha256": sha256_file(partition_manifest),
            "event_store_manifest": event_store_manifest.relative_to(PROJECT_ROOT).as_posix()
            if event_store_manifest.is_file()
            else None,
            "train_event_count": int(train_data.labels.size),
            "validation_event_count": int(validation_data.labels.size),
            "max_events_per_partition": args.max_events_per_partition,
            "balanced_subset": args.balanced_subset,
            "train_selection": train_selection,
            "validation_selection": validation_selection,
            "test_partition_used": False,
        },
        "training": {
            "candidates": list(args.candidate),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "base_seed": args.seed,
            "torch_version": torch.__version__,
        },
        "candidates": {},
        "caveats": [
            "Internal validation is file-disjoint but remains shortcut-sensitive.",
            "Co-60 Compton-continuum is an operational spectral-background label; no unique microscopic event history is asserted.",
            "External AUROC and spectral P/B/retention remain the scientific return gate.",
            "Locked test, Th-232, and Eu-152 data were not opened.",
        ],
    }
    for index, name in enumerate(args.candidate):
        result["candidates"][name] = train_candidate(
            name,
            train_values,
            validation_values,
            train_data,
            validation_data,
            statistics,
            output_dir,
            args,
            device,
            args.seed + index,
        )
    save_json(output_dir / "comparison.json", result)
    print(json.dumps(result["candidates"], indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output_dir.relative_to(PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
