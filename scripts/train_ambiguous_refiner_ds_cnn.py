#!/usr/bin/env python3
"""Train the five predeclared high-resolution Stage-2 DS-CNN refiners."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import train_epoch  # noqa: E402
from src.cascade_refinement import (  # noqa: E402
    CANDIDATES,
    CANDIDATE_ORDER,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_LEARNING_RATE,
    STAGE2_SEED,
    STAGE2_WEIGHT_DECAY,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    save_json,
    set_seed,
    sha256_file,
    validate_representation,
)


EXPERIMENT_ID = "cascaded_ambiguous_refinement_ds_cnn_20260821"
DEFAULT_MINING_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID / "mining"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID / "stage2_candidates"


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype="U64",
    )


def load_subset(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        result = {
            "waveforms": np.asarray(handle["waveform"], dtype=np.float32),
            "labels": np.asarray(handle["labels"], dtype=np.float32),
            "weights": np.asarray(handle["weights"], dtype=np.float32),
            "stage1_score": np.asarray(handle["stage1_score"], dtype=np.float32),
            "peak_ids": decode_strings(np.asarray(handle["peak_ids"])),
            "source_event_index": np.asarray(handle["source_event_index"], dtype=np.int64),
        }
    if not np.all(np.isfinite(result["waveforms"])):
        raise ValueError(f"Nonfinite waveforms in {path}")
    if not np.all(np.isfinite(result["weights"])) or np.any(result["weights"] <= 0.0):
        raise ValueError(f"Invalid weights in {path}")
    return result


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
            torch.from_numpy(labels),
            torch.from_numpy(weights),
        ),
        batch_size=STAGE2_BATCH_SIZE,
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
    return np.concatenate(scores).astype(np.float32, copy=False)


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
            raise ValueError(f"Ambiguous internal stratum lacks both labels: {peak_id}")
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
            "positive_count": int(np.count_nonzero(mask & (labels == 1.0))),
            "negative_count": int(np.count_nonzero(mask & (labels == 0.0))),
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


def train_candidate(
    candidate_name: str,
    fit_subset: dict[str, np.ndarray],
    internal_subset: dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    representation = CANDIDATES[candidate_name]
    set_seed(STAGE2_SEED)
    print(f"building candidate={candidate_name}", flush=True)
    fit_values, fit_qc = build_representation(fit_subset["waveforms"], representation)
    internal_values, internal_qc = build_representation(
        internal_subset["waveforms"], representation
    )
    validate_representation(fit_values, representation)
    validate_representation(internal_values, representation)
    statistics = fit_channel_statistics(fit_values)
    apply_channel_statistics(fit_values, statistics)
    apply_channel_statistics(internal_values, statistics)
    validate_representation(fit_values, representation)
    validate_representation(internal_values, representation)

    fit_loader = make_loader(
        fit_values,
        fit_subset["labels"],
        fit_subset["weights"],
        True,
        STAGE2_SEED,
    )
    internal_loader = make_loader(
        internal_values,
        internal_subset["labels"],
        internal_subset["weights"],
        False,
        STAGE2_SEED,
    )
    model = DSCNN(input_channels=representation.channel_count, width=24).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STAGE2_LEARNING_RATE, weight_decay=STAGE2_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS, eta_min=0.0
    )
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = -1
    best_metric = -np.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss = train_epoch(model, fit_loader, optimizer, device)
        internal_scores = predict(model, internal_loader, device)
        internal_metrics = metric_summary(
            internal_subset["labels"],
            internal_scores,
            internal_subset["weights"],
            internal_subset["peak_ids"],
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "internal_metrics": internal_metrics,
            }
        )
        current = float(internal_metrics["macro_auroc"])
        print(
            f"candidate={candidate_name} epoch={epoch} loss={train_loss:.6f} "
            f"internal_macro={current:.6f} "
            f"internal_worst={internal_metrics['worst_peak_auroc']:.6f}",
            flush=True,
        )
        if current > best_metric + 1.0e-7:
            best_metric = current
            best_epoch = epoch
            best_metrics = internal_metrics
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None or best_metrics is None or best_epoch < 1:
        raise RuntimeError(f"No checkpoint selected for {candidate_name}")
    model.load_state_dict(best_state)
    best_internal_scores = predict(model, internal_loader, device)
    checkpoint_path = output_dir / candidate_name / f"{candidate_name}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "model_kind": "ds_cnn",
        "model_role": "stage2_ambiguous_refiner_development",
        "candidate_name": candidate_name,
        "model_state_dict": best_state,
        "model_width": 24,
        "parameter_count": parameter_count,
        "representation": representation.as_dict(),
        "feature_statistics": statistics,
        "training": {
            "epochs": STAGE2_EPOCHS,
            "batch_size": STAGE2_BATCH_SIZE,
            "learning_rate": STAGE2_LEARNING_RATE,
            "weight_decay": STAGE2_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR(T_max=8, eta_min=0)",
            "seed": STAGE2_SEED,
            "selection_metric": "internal_ambiguous_macro_auroc",
            "selected_epoch": best_epoch,
        },
        "partition": {
            "fit_event_count": int(fit_subset["labels"].size),
            "internal_event_count": int(internal_subset["labels"].size),
            "event_level_selection": True,
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "target_data_used_for_selection": False,
        },
        "fit_representation_qc": fit_qc,
        "internal_representation_qc": internal_qc,
        "scan_best_internal_metrics": best_metrics,
        "candidate_history": history,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    torch.save(checkpoint, checkpoint_path)
    np.savez_compressed(
        output_dir / candidate_name / "internal_scores.npz",
        labels=internal_subset["labels"],
        peak_ids=internal_subset["peak_ids"],
        weights=internal_subset["weights"],
        stage1_scores=internal_subset["stage1_score"],
        stage2_scores=best_internal_scores,
        source_event_index=internal_subset["source_event_index"],
    )
    result = {
        "candidate_name": candidate_name,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "representation": representation.as_dict(),
        "feature_statistics": statistics,
        "fit_representation_qc": fit_qc,
        "internal_representation_qc": internal_qc,
        "selected_epoch": best_epoch,
        "scan_best_internal_metrics": best_metrics,
        "history": history,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_dir / candidate_name / "training_result.json", result)
    del model, optimizer, scheduler, fit_values, internal_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining-dir", type=Path, default=DEFAULT_MINING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mining_dir = args.mining_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    for path in (
        mining_dir,
        mining_dir / "mining_summary.json",
        mining_dir / "ambiguous_fit_events.h5",
        mining_dir / "ambiguous_internal_events.h5",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    summary = json.loads((mining_dir / "mining_summary.json").read_text(encoding="utf-8"))
    if summary.get("held_out_partition_loaded") is not False or summary.get("test_partition_used") is not False:
        raise ValueError("Mining manifest has an invalid data boundary")
    fit_subset = load_subset(mining_dir / "ambiguous_fit_events.h5")
    internal_subset = load_subset(mining_dir / "ambiguous_internal_events.h5")
    if not np.all((internal_subset["stage1_score"] >= 0.4) & (internal_subset["stage1_score"] <= 0.6)):
        raise ValueError("Internal subset contains events outside the fixed ambiguous region")
    print(f"device={device}", flush=True)
    print(
        f"fit_ambiguous={fit_subset['labels'].size} internal_ambiguous={internal_subset['labels'].size}",
        flush=True,
    )
    results: dict[str, Any] = {}
    for candidate_name in CANDIDATE_ORDER:
        results[candidate_name] = train_candidate(
            candidate_name, fit_subset, internal_subset, output_dir, device
        )
    ordered = sorted(
        results.values(),
        key=lambda value: (
            float(value["scan_best_internal_metrics"]["macro_auroc"]),
            float(value["scan_best_internal_metrics"]["worst_peak_auroc"]),
        ),
        reverse=True,
    )
    selection = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "selection_rule": "highest internal ambiguous macro-AUROC; worst-peak AUROC tie-break",
        "selected_candidate": ordered[0]["candidate_name"],
        "ranking": [
            {
                "candidate_name": value["candidate_name"],
                "macro_auroc": value["scan_best_internal_metrics"]["macro_auroc"],
                "worst_peak_auroc": value["scan_best_internal_metrics"]["worst_peak_auroc"],
                "selected_epoch": value["selected_epoch"],
            }
            for value in ordered
        ],
        "candidate_results": results,
        "mining_summary": str((mining_dir / "mining_summary.json").relative_to(PROJECT_ROOT)),
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
    }
    save_json(output_dir / "candidate_selection.json", selection)
    save_json(
        output_dir / "stage2_experiment_config.json",
        {
            "schema_version": "1",
            "experiment_id": EXPERIMENT_ID,
            "candidate_order": list(CANDIDATE_ORDER),
            "candidates": {name: CANDIDATES[name].as_dict() for name in CANDIDATE_ORDER},
            "training": {
                "epochs": STAGE2_EPOCHS,
                "batch_size": STAGE2_BATCH_SIZE,
                "learning_rate": STAGE2_LEARNING_RATE,
                "weight_decay": STAGE2_WEIGHT_DECAY,
                "seed": STAGE2_SEED,
            },
            "mining_summary_sha256": sha256_file(mining_dir / "mining_summary.json"),
            "held_out_partition_loaded": False,
            "test_partition_used": False,
        },
    )
    print(json.dumps(selection["ranking"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
