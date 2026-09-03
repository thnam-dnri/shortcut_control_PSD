#!/usr/bin/env python3
"""Scan strict Ba356/Na511/Cs662 DS-CNN loss weights.

The normalized weight simplex is selected on the frozen internal development
partition with equal-peak macro-AUROC. The existing file-held-out validation
partition is opened only after the best weight vector is selected.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_strict_ds_cnn_reproducibility import (
    REPRESENTATION,
    event_indices,
    load_split,
    make_event_weights,
    metrics,
    run_seed,
    set_seed,
    train_epoch,
    validate_manifest,
)
from src.ba133_cnn import (
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
)
from src.architecture_candidates import DSCNN
from src.data_access_guards import assert_no_forbidden_path
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
EVENT_STORE_DIR = PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                    "ba133_356kev": ba_index / steps,
                    "na22_511kev": na_index / steps,
                    "cs137_662kev": cs_index / steps,
                }
            )
    return grid


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


def predict(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            scores.append(torch.sigmoid(model(values.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(scores)


def scan_one_weight_vector(
    values: np.ndarray,
    labels: np.ndarray,
    peak_ids: np.ndarray,
    fit_events: np.ndarray,
    internal_events: np.ndarray,
    weights: dict[str, float],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    index: int,
    total: int,
) -> dict[str, Any]:
    print(f"scan {index}/{total} weights={weights}", flush=True)
    set_seed(seed)
    event_weights = make_event_weights(peak_ids, weights)
    fit_loader = make_loader(values, labels, event_weights, fit_events, batch_size, True, seed)
    internal_loader = make_loader(values, labels, event_weights, internal_events, batch_size, False, seed)
    model = DSCNN(input_channels=2, width=24).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch = -1
    best_metric = -np.inf
    best_metrics: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, fit_loader, optimizer, device)
        scores = predict(model, internal_loader, device)
        current = metrics(
            labels[internal_events],
            scores,
            event_weights[internal_events],
            peak_ids[internal_events],
            PEAKS,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "internal": current})
        print(
            f"  epoch={epoch} train_loss={train_loss:.6f} "
            f"internal_macro={current['macro_auroc']:.6f}",
            flush=True,
        )
        if current["macro_auroc"] > best_metric + 1.0e-4:
            best_metric = current["macro_auroc"]
            best_epoch = epoch
            best_metrics = current
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch < 1 or best_metrics is None:
        raise RuntimeError("No internal checkpoint epoch was selected")
    del model, optimizer, fit_loader, internal_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "weights": weights,
        "scan_seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": "internal_equal_peak_macro_auroc",
        "internal": best_metrics,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/strict_three_peak_ds_cnn_weight_scan_20260826",
    )
    parser.add_argument("--grid-step", type=float, default=0.20)
    parser.add_argument("--scan-seed", type=int, default=20260828)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
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
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def main() -> int:
    args = build_parser().parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    if not args.confirmation_seeds:
        raise ValueError("At least one confirmation seed is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = LABELS_DIR / "label_pairs_train.csv"
    validation_csv = LABELS_DIR / "label_pairs_validation.csv"
    for path in (LABELS_DIR, EVENT_STORE_DIR, train_csv, validation_csv):
        assert_no_forbidden_path(path)
    train_pair_count = validate_manifest(train_csv, PEAKS)
    validation_pair_count = validate_manifest(validation_csv, PEAKS)

    print(f"device={resolve_device(args.device)}", flush=True)
    device = resolve_device(args.device)
    train_raw = load_raw_partition(train_csv, EVENT_STORE_DIR)
    if set(train_raw.peak_ids[::2]) != set(PEAKS):
        raise ValueError("Training manifest does not contain exactly the selected three peaks")

    fit_pairs, internal_pairs = load_split(
        LABELS_DIR,
        train_raw.peak_ids[::2],
        "reuse_frozen_training_internal_split",
        args.scan_seed,
    )
    train_values, train_qc = build_representation(train_raw, REPRESENTATION)
    feature_statistics = fit_channel_statistics(train_values)
    apply_channel_statistics(train_values, feature_statistics)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)

    combinations = weight_grid(args.grid_step)
    scan_results = [
        scan_one_weight_vector(
            train_values,
            train_raw.labels,
            train_raw.peak_ids,
            fit_events,
            internal_events,
            weights,
            args.scan_seed,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            device,
            index,
            len(combinations),
        )
        for index, weights in enumerate(combinations, start=1)
    ]
    scan_results.sort(
        key=lambda result: (result["internal"]["macro_auroc"], result["internal"]["pooled_auroc"]),
        reverse=True,
    )
    selected = scan_results[0]
    save_json(output_dir / "weight_matrix_scan.json", {"results": scan_results})

    validation_raw = load_raw_partition(validation_csv, EVENT_STORE_DIR)
    validation_values, validation_qc = build_representation(validation_raw, REPRESENTATION)
    apply_channel_statistics(validation_values, feature_statistics)

    confirmation_dir = output_dir / "selected_confirmation"
    confirmation_config = {
        "peaks": PEAKS,
        "weights": selected["weights"],
    }
    confirmation_args = argparse.Namespace(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    confirmations = []
    for seed in args.confirmation_seeds:
        confirmations.append(
            run_seed(
                train_values,
                train_raw.labels,
                train_raw.peak_ids,
                make_event_weights(train_raw.peak_ids, selected["weights"]),
                validation_values,
                validation_raw.labels,
                validation_raw.peak_ids,
                make_event_weights(validation_raw.peak_ids, selected["weights"]),
                fit_events,
                internal_events,
                seed,
                confirmation_dir,
                "selected_weight_confirmation",
                confirmation_config,
                confirmation_args,
                device,
                feature_statistics,
            )
        )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "labels_dir": LABELS_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "event_store_dir": EVENT_STORE_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
        "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
        "train_pair_count": train_pair_count,
        "validation_pair_count": validation_pair_count,
        "fit_pair_count": int(fit_pairs.size),
        "internal_pair_count": int(internal_pairs.size),
        "strict_match_rule": "absolute positive-negative corrected-energy difference < 0.5 keV",
        "representation": REPRESENTATION.as_dict(),
        "representation_qc": {"train": train_qc, "validation": validation_qc},
        "feature_statistics": feature_statistics,
        "grid_step": args.grid_step,
        "grid_size": len(combinations),
        "scan_seed": args.scan_seed,
        "confirmation_seeds": args.confirmation_seeds,
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "selection_metric": "scan internal equal-peak macro-AUROC",
        "selected_weights": selected["weights"],
        "selected_scan_internal": selected["internal"],
        "confirmation_validation": confirmations,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_dir / "experiment_config.json", metadata)
    save_json(output_dir / "summary.json", metadata)
    print(json.dumps({"selected_weights": selected["weights"], "selected_scan_internal": selected["internal"], "confirmation_runs": len(confirmations)}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
