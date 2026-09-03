#!/usr/bin/env python3
"""Optimize configurable compact 1D CNNs on approved development labels.

Target-only label manifests are deliberately not accepted by this script. Use a
separate frozen-checkpoint transfer evaluator after model selection.
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

from src.ba133_cnn import (  # noqa: E402
    SCREEN_CONFIGS,
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    evaluate_model,
    fit_channel_statistics,
    load_raw_partition,
    make_loader,
    set_seed,
    train_epoch,
)
from scripts.train_o2_late_fusion import per_peak_metrics, sha256_file  # noqa: E402

SEED = 20260813


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_configs(value: str | None) -> list[RepresentationConfig]:
    if value is None:
        return list(SCREEN_CONFIGS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    by_name = {config.name: config for config in SCREEN_CONFIGS}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown configurations: {unknown}")
    return [by_name[name] for name in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/ba133_cnn_optimization/screening",
    )
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--max-train-events", type=int, default=40000)
    parser.add_argument("--max-validation-events", type=int, default=16000)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use every Ba-133 train and validation event, overriding subset limits.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    allowed_label_dirs = {
        (PROJECT_ROOT / "outputs/labels").resolve():
            "all_training_peaks_internal_validation_only",
        (PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive").resolve():
            "ba133_all_peaks_internal_validation_only",
        (PROJECT_ROOT / "outputs/labels/ba356_positive").resolve():
            "ba133_356kev_internal_validation_only",
        (PROJECT_ROOT / "outputs/labels/ba_all_na511_positive").resolve():
            "ba133_all_peaks_plus_na22_511kev_internal_validation_only",
        (PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815").resolve():
            "frozen_pass_warn_primary_internal_validation_only",
    }
    peak_combination_root = (PROJECT_ROOT / "outputs/labels/peak_combinations").resolve()
    if labels_dir.parent == peak_combination_root:
        allowed_label_dirs[labels_dir] = (
            f"peak_combination_{labels_dir.name}_internal_validation_only"
        )
    pass_warn_source_ablation_root = (
        PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815_source_ablation"
    ).resolve()
    if labels_dir == pass_warn_source_ablation_root / "ba356_positive":
        allowed_label_dirs[labels_dir] = (
            "ba133_356kev_pass_warn_internal_validation_only"
        )
    if labels_dir not in allowed_label_dirs:
        raise ValueError(
            "labels-dir must be an approved development manifest: "
            f"{sorted(str(path) for path in allowed_label_dirs)}"
        )
    selection_domain = allowed_label_dirs[labels_dir]
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if min(args.epochs, args.patience, args.batch_size, args.model_width) < 1:
        raise ValueError("Training arguments must be positive")

    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    event_store_dir = args.event_store_dir.resolve()
    max_train_events = None if args.full_data else args.max_train_events
    max_validation_events = None if args.full_data else args.max_validation_events
    print("Loading train raw waveforms ...", flush=True)
    train_raw = load_raw_partition(train_csv, event_store_dir, max_train_events)
    print("Loading validation raw waveforms ...", flush=True)
    validation_raw = load_raw_partition(
        validation_csv, event_store_dir, max_validation_events
    )
    configs = parse_configs(args.configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_domain": selection_domain,
        "selection_metric": "validation_weighted_auroc",
        "target_domains_opened": False,
        "test_partition_used": False,
        "input": {
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(validation_csv),
            "train_event_count": int(train_raw.labels.size),
            "validation_event_count": int(validation_raw.labels.size),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "model_width": args.model_width,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "trials": [],
    }

    for trial_index, config in enumerate(configs, start=1):
        print(f"trial={trial_index}/{len(configs)} config={config.name}", flush=True)
        set_seed(args.seed)
        train_values, train_representation_qc = build_representation(train_raw, config)
        validation_values, validation_representation_qc = build_representation(
            validation_raw, config
        )
        statistics = fit_channel_statistics(train_values)
        apply_channel_statistics(train_values, statistics)
        apply_channel_statistics(validation_values, statistics)
        train_loader = make_loader(
            train_values, train_raw, args.batch_size, True, args.seed
        )
        validation_loader = make_loader(
            validation_values, validation_raw, args.batch_size, False, args.seed
        )
        model = CompactWaveformCNN(
            config.channel_count, width=args.model_width
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_metric = -np.inf
        best_epoch = -1
        stale_epochs = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, Any]] = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            validation_metrics, _ = evaluate_model(model, validation_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation": validation_metrics,
                }
            )
            current = validation_metrics["weighted_auroc"]
            print(
                f"  epoch={epoch} val_auroc={validation_metrics['auroc']:.6f} "
                f"val_weighted={current:.6f}",
                flush=True,
            )
            if current > best_metric + 1.0e-4:
                best_metric = current
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError("No checkpoint selected")
        model.load_state_dict(best_state)
        validation_metrics, validation_scores = evaluate_model(
            model, validation_loader, device
        )
        validation_metrics["per_peak"] = per_peak_metrics(
            validation_raw.labels,
            validation_scores,
            validation_raw.weights,
            validation_raw.peak_ids,
        )
        checkpoint_path = output_dir / f"{config.name}.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "representation_config": config.as_dict(),
                "channel_statistics": statistics,
                "model_width": args.model_width,
                "parameter_count": parameter_count,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "selection_domain": selection_domain,
                "target_domains_opened": False,
            },
            checkpoint_path,
        )
        result["trials"].append(
            {
                "config": config.as_dict(),
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "validation": validation_metrics,
                "channel_statistics": statistics,
                "representation_qc": {
                    "train": train_representation_qc,
                    "validation": validation_representation_qc,
                },
                "history": history,
            }
        )
        del train_values, validation_values, model, optimizer

    result["trials"].sort(
        key=lambda item: item["validation"]["weighted_auroc"], reverse=True
    )
    result["ranking"] = [
        {
            "rank": index,
            "config": trial["config"]["name"],
            "validation_auroc": trial["validation"]["auroc"],
            "validation_weighted_auroc": trial["validation"]["weighted_auroc"],
        }
        for index, trial in enumerate(result["trials"], start=1)
    ]
    save_json(output_dir / "optimization_results.json", result)
    print("ranking", result["ranking"], flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
