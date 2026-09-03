#!/usr/bin/env python3
"""Finalize and evaluate the cascaded ambiguous-region DS-CNN experiment.

The script follows the frozen order of operations for this experiment:

1. compare the predeclared fusion rules on the fit-only-Stage-1 internal
   development partition;
2. refit the selected Stage-2 candidate on all development events routed by
   the frozen production Stage-1 model;
3. freeze the fusion rule and positive-event score thresholds;
4. report the same-domain held-out partition; and
5. only then score the corrected Th-232 cache.

The existing corrected full-development DS-CNN checkpoint is never changed.
It remains the Stage-1 baseline for held-out and Th-232 comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    ENERGY_MAX_KEV,
    ENERGY_MIN_KEV,
    REFERENCE_PEAKS_KEV,
    PeakWindow,
    fit_peak_windows,
    peak_background_metrics,
    th232_admission_mask,
)
from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    apply_channel_statistics as apply_stage1_statistics,
    build_representation as build_stage1_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
    train_epoch,
)
from src.cascade_refinement import (  # noqa: E402
    CANDIDATES,
    STAGE2_BATCH_SIZE,
    STAGE2_EPOCHS,
    STAGE2_LEARNING_RATE,
    STAGE2_SEED,
    STAGE2_WEIGHT_DECAY,
    TAU_HIGH,
    TAU_LOW,
    BivariateIsotonicCalibrator,
    apply_channel_statistics as apply_stage2_statistics,
    build_representation as build_stage2_representation,
    event_indices,
    fit_channel_statistics as fit_stage2_statistics,
    isotonic_fusion,
    make_event_weights,
    metric_summary,
    piecewise_fusion,
    save_json,
    set_seed,
    sha256_file,
    soft_gate_fusion,
    weighted_acceptance_threshold,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


EXPERIMENT_ID = "cascaded_ambiguous_refinement_ds_cnn_20260821"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_STAGE1_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID
DEFAULT_TH232_DIR = PROJECT_ROOT / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813"
ACCEPTANCES = (0.99, 0.95, 0.90, 0.80, 0.50, 0.30, 0.10)
SOFT_GATE_TEMPERATURES = (0.01, 0.02, 0.05, 0.10)
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def threshold_name(acceptance: float) -> str:
    return f"{int(round(100.0 * acceptance))}pct"


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


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
            torch.from_numpy(values.astype(np.float32, copy=False)),
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


def predict_values(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values.astype(np.float32, copy=False))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            chunks.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False))
    scores = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced nonfinite scores")
    return scores


def load_stage1_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("Stage-1 checkpoint is not a DS-CNN checkpoint")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Stage-1 checkpoint is marked as test-contaminated")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError("Stage-1 checkpoint loaded held-out data")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.name != "both_ma10_global_t10_w750_positive_polarity":
        raise ValueError(f"Unexpected frozen Stage-1 representation: {config.name}")
    if config.channel_count != 2 or config.window_length != 750:
        raise ValueError("Stage-1 representation must be [2,750]")
    width = int(checkpoint.get("model_width", 24))
    if width != 24:
        raise ValueError(f"Unexpected Stage-1 width: {width}")
    model = DSCNN(input_channels=config.channel_count, width=width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "model_kind": checkpoint["model_kind"],
        "model_role": "frozen_production_stage1_baseline",
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": int(checkpoint["parameter_count"]),
        "model_width": width,
        "representation_config": config.as_dict(),
        "feature_statistics": checkpoint["feature_statistics"],
        "selected_peak_weights": checkpoint["selected_peak_weights"],
        "scan_best_epoch": int(checkpoint["scan_best_epoch"]),
        "refit_epochs": int(checkpoint["refit_epochs"]),
        "internal_selection_metrics": checkpoint["scan_best_internal_metrics"],
        "test_partition_used": checkpoint["test_partition_used"],
        "held_out_partition_loaded": checkpoint["held_out_partition_loaded"],
    }
    return model, metadata, config


def score_stage1_raw(
    raw: RawPartition,
    model: nn.Module,
    config: Any,
    statistics: dict[str, list[float]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    values, qc = build_stage1_representation(raw, config)
    apply_stage1_statistics(values, statistics)
    if not np.all(np.isfinite(values)):
        raise ValueError("Stage-1 representation contains nonfinite values")
    scores = predict_values(model, values, device, batch_size)
    del values
    return scores, qc


def score_stage2_raw(
    waveforms: np.ndarray,
    model: nn.Module,
    representation: Any,
    statistics: dict[str, list[float]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    values, qc = build_stage2_representation(waveforms, representation)
    apply_stage2_statistics(values, statistics)
    scores = predict_values(model, values, device, batch_size)
    del values
    return scores, qc


def load_stage1_development_scores(
    mining_dir: Path,
    stage2_dir: Path,
    selected_candidate: str,
) -> dict[str, np.ndarray]:
    with np.load(mining_dir / "stage1_train_scores.npz", allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "scores",
        "labels",
        "peak_ids",
        "weights",
        "internal_event_mask",
        "ambiguous_mask",
    }
    if not required.issubset(data):
        raise ValueError(f"Stage-1 mining artifact lacks keys: {sorted(required - set(data))}")
    internal_mask = data["internal_event_mask"].astype(bool, copy=False)
    internal_ambiguous = internal_mask & data["ambiguous_mask"].astype(bool, copy=False)
    with np.load(
        stage2_dir / selected_candidate / "internal_scores.npz", allow_pickle=False
    ) as source:
        stage2_source = {name: np.asarray(source[name]) for name in source.files}
    source_indices = stage2_source["source_event_index"].astype(np.int64, copy=False)
    if not np.array_equal(np.sort(source_indices), np.flatnonzero(internal_ambiguous)):
        raise ValueError("Stage-2 internal score artifact does not match the mined internal route")
    if not np.allclose(
        stage2_source["stage1_scores"], data["scores"][source_indices], rtol=0.0, atol=1.0e-6
    ):
        raise ValueError("Stage-2 internal artifact has inconsistent Stage-1 scores")
    stage2_scores = np.full(data["scores"].shape, np.nan, dtype=np.float64)
    stage2_scores[source_indices] = stage2_source["stage2_scores"].astype(np.float64)
    if not np.all(np.isfinite(stage2_scores[internal_ambiguous])):
        raise ValueError("Missing Stage-2 scores in the internal ambiguous route")
    return {
        "stage1_scores": data["scores"].astype(np.float64),
        "stage2_scores": stage2_scores,
        "labels": data["labels"].astype(np.float32),
        "weights": data["weights"].astype(np.float64),
        "peak_ids": data["peak_ids"],
        "internal_mask": internal_mask,
        "ambiguous_mask": internal_ambiguous,
        "event_indices": data.get(
            "event_indices", np.arange(data["scores"].size, dtype=np.int64)
        ).astype(np.int64),
    }


def apply_selected_fusion(
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    choice: dict[str, Any],
    calibrator: BivariateIsotonicCalibrator | None = None,
) -> np.ndarray:
    method = str(choice["method"])
    if method == "piecewise":
        return piecewise_fusion(stage1_scores, stage2_scores)
    if method == "soft_gate":
        return soft_gate_fusion(
            stage1_scores,
            stage2_scores,
            float(choice["temperature"]),
        )
    if method == "bivariate_isotonic":
        if calibrator is None:
            raise ValueError("Bivariate isotonic fusion requires a fitted calibrator")
        result = np.asarray(stage1_scores, dtype=np.float64).copy()
        ambiguous = np.isfinite(stage2_scores) & (stage1_scores >= TAU_LOW) & (
            stage1_scores <= TAU_HIGH
        )
        result[ambiguous] = calibrator.predict(
            stage1_scores[ambiguous], stage2_scores[ambiguous]
        )
        return result
    raise ValueError(f"Unknown fusion method: {method}")


def select_internal_fusion(
    mining_dir: Path,
    stage2_dir: Path,
    candidate_selection: dict[str, Any],
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selected_candidate = str(candidate_selection["selected_candidate"])
    arrays = load_stage1_development_scores(mining_dir, stage2_dir, selected_candidate)
    internal = arrays["internal_mask"]
    stage1 = arrays["stage1_scores"][internal]
    stage2 = arrays["stage2_scores"][internal]
    labels = arrays["labels"][internal]
    weights = arrays["weights"][internal]
    peak_ids = arrays["peak_ids"][internal]
    baseline_metrics = metric_summary(labels, stage1, weights, peak_ids)

    method_results: list[dict[str, Any]] = []
    piecewise_scores = piecewise_fusion(stage1, stage2)
    method_results.append(
        {
            "method": "piecewise",
            "description": "tails=Stage-1; ambiguous=0.4+0.2*Stage-2 probability",
            "metrics": metric_summary(labels, piecewise_scores, weights, peak_ids),
        }
    )
    for temperature in SOFT_GATE_TEMPERATURES:
        scores = soft_gate_fusion(stage1, stage2, temperature)
        method_results.append(
            {
                "method": "soft_gate",
                "temperature": temperature,
                "description": "temperature-smoothed blend inside the fixed [0.4,0.6] route",
                "metrics": metric_summary(labels, scores, weights, peak_ids),
            }
        )
    isotonic_scores, isotonic_calibrator = isotonic_fusion(
        stage1, stage2, labels, weights, bins=12
    )
    method_results.append(
        {
            "method": "bivariate_isotonic",
            "bins": 12,
            "description": "coordinatewise monotone 2-D isotonic map fit on internal ambiguous events",
            "metrics": metric_summary(labels, isotonic_scores, weights, peak_ids),
            "calibrator": isotonic_calibrator.as_dict(),
        }
    )
    method_priority = {
        "piecewise": 0,
        "soft_gate": 1,
        "bivariate_isotonic": 2,
    }
    ordered = sorted(
        method_results,
        key=lambda item: (
            float(item["metrics"]["macro_auroc"]),
            float(item["metrics"]["worst_peak_auroc"]),
            -method_priority[str(item["method"])],
        ),
        reverse=True,
    )
    selected = dict(ordered[0])
    selected.pop("metrics", None)
    selection = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "selected_candidate": selected_candidate,
        "baseline_stage1_metrics": baseline_metrics,
        "fusion_candidates": method_results,
        "ranking": [
            {
                "method": item["method"],
                "temperature": item.get("temperature"),
                "macro_auroc": item["metrics"]["macro_auroc"],
                "worst_peak_auroc": item["metrics"]["worst_peak_auroc"],
            }
            for item in ordered
        ],
        "selected_fusion": selected,
        "selection_rule": (
            "highest full-internal macro-AUROC; worst-peak AUROC tie-break; "
            "piecewise preferred only for exact ties"
        ),
        "temperature_grid": list(SOFT_GATE_TEMPERATURES),
        "internal_partition_only": True,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_path, selection)
    return selection, arrays


def fit_final_stage2(
    raw: RawPartition,
    ambiguous_indices: np.ndarray,
    representation: Any,
    stage1_checkpoint_metadata: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, Path, dict[str, Any], dict[str, Any], np.ndarray]:
    if ambiguous_indices.size == 0:
        raise ValueError("Production Stage-1 routed no development events to Stage-2")
    values, representation_qc = build_stage2_representation(
        raw.waveforms[ambiguous_indices], representation
    )
    statistics = fit_stage2_statistics(values)
    apply_stage2_statistics(values, statistics)
    labels = raw.labels[ambiguous_indices]
    weights = make_event_weights(raw.peak_ids)[ambiguous_indices]
    loader = make_loader(values, labels, weights, batch_size, True, STAGE2_SEED + 100000)
    set_seed(STAGE2_SEED + 100000)
    model = DSCNN(input_channels=representation.channel_count, width=24).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STAGE2_LEARNING_RATE, weight_decay=STAGE2_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS, eta_min=0.0
    )
    losses: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        losses.append({"epoch": epoch, "train_loss": loss, "learning_rate": learning_rate})
        scheduler.step()
        print(
            f"stage2_final epoch={epoch} train_loss={loss:.6f} lr={learning_rate:.8g}",
            flush=True,
        )
    model.eval()
    stage2_scores = predict_values(model, values, device, batch_size)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = output_dir / "selected_stage2_best.pt"
    checkpoint = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "model_kind": "ds_cnn",
        "model_role": "stage2_ambiguous_refiner_final_development_refit",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_width": 24,
        "parameter_count": parameter_count,
        "representation": representation.as_dict(),
        "feature_statistics": statistics,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": STAGE2_LEARNING_RATE,
            "weight_decay": STAGE2_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR(T_max=8, eta_min=0)",
            "seed": STAGE2_SEED + 100000,
            "epoch_count_frozen_from_candidate_scan": epochs,
            "training_scope": "all development events routed by frozen production Stage-1",
        },
        "partition": {
            "development_event_count": int(raw.labels.size),
            "ambiguous_event_count": int(ambiguous_indices.size),
            "event_level_routing": True,
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "target_data_used_for_selection": False,
        },
        "stage1_checkpoint": stage1_checkpoint_metadata,
        "representation_qc": representation_qc,
        "training_loss_history": losses,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    result = {
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "representation": representation.as_dict(),
        "feature_statistics": statistics,
        "representation_qc": representation_qc,
        "training": checkpoint["training"],
        "partition": checkpoint["partition"],
        "training_loss_history": losses,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_dir / "training_result.json", result)
    return model, checkpoint_path, statistics, result, stage2_scores


def fit_final_internal_calibrator(
    choice: dict[str, Any],
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    raw: RawPartition,
    internal_events: np.ndarray,
) -> BivariateIsotonicCalibrator | None:
    if choice["method"] != "bivariate_isotonic":
        return None
    internal_ambiguous = np.isin(internal_events, np.flatnonzero(np.isfinite(stage2_scores)))
    selected_events = internal_events[internal_ambiguous]
    if selected_events.size == 0:
        raise ValueError("Final internal partition has no routed events for isotonic calibration")
    weights = make_event_weights(raw.peak_ids)
    calibrator = BivariateIsotonicCalibrator(bins=int(choice.get("bins", 12))).fit(
        stage1_scores[selected_events],
        stage2_scores[selected_events],
        raw.labels[selected_events],
        weights[selected_events],
    )
    return calibrator


def calibrate_threshold_grid(
    stage1_scores: np.ndarray,
    cascade_scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    internal_events: np.ndarray,
) -> dict[str, Any]:
    positive = labels[internal_events] == 1.0
    internal_weights = weights[internal_events]
    stage1_internal = stage1_scores[internal_events]
    cascade_internal = cascade_scores[internal_events]
    result: dict[str, Any] = {
        "source": "final internal development positive events",
        "selection_complete_before_held_out": True,
        "th232_used": False,
        "stage1": {},
        "cascade": {},
    }
    for acceptance in ACCEPTANCES:
        name = threshold_name(acceptance)
        result["stage1"][name] = weighted_acceptance_threshold(
            stage1_internal[positive], internal_weights[positive], acceptance
        )
        result["cascade"][name] = weighted_acceptance_threshold(
            cascade_internal[positive], internal_weights[positive], acceptance
        )
    return result


def evaluate_final_internal(
    raw: RawPartition,
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    cascade_scores: np.ndarray,
    internal_events: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = make_event_weights(raw.peak_ids)
    internal = internal_events
    metrics = {
        "stage1": metric_summary(
            raw.labels[internal],
            stage1_scores[internal],
            weights[internal],
            raw.peak_ids[internal],
        ),
        "cascade": metric_summary(
            raw.labels[internal],
            cascade_scores[internal],
            weights[internal],
            raw.peak_ids[internal],
        ),
    }
    routed = np.isfinite(stage2_scores)
    if np.count_nonzero(routed & np.isin(np.arange(raw.labels.size), internal)):
        route_internal = internal[np.isfinite(stage2_scores[internal])]
        metrics["stage2_ambiguous"] = metric_summary(
            raw.labels[route_internal],
            stage2_scores[route_internal],
            weights[route_internal],
            raw.peak_ids[route_internal],
        )
    thresholds = calibrate_threshold_grid(
        stage1_scores, cascade_scores, raw.labels, weights, internal_events
    )
    np.savez_compressed(
        output_dir / "internal_scores.npz",
        labels=raw.labels[internal],
        peak_ids=raw.peak_ids[internal],
        weights=weights[internal],
        stage1_scores=stage1_scores[internal].astype(np.float32),
        stage2_scores=stage2_scores[internal].astype(np.float32),
        cascade_scores=cascade_scores[internal].astype(np.float32),
        event_indices=internal.astype(np.int64),
    )
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "partition": "full-development internal event subset",
        "event_count": int(internal.size),
        "routed_event_count": int(np.count_nonzero(np.isfinite(stage2_scores[internal]))),
        "metrics": metrics,
        "thresholds": thresholds,
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_dir / "internal_evaluation.json", report)
    return report, thresholds


def evaluate_held_out(
    labels_dir: Path,
    event_store_dir: Path,
    stage1_model: nn.Module,
    stage1_metadata: dict[str, Any],
    stage1_config: Any,
    stage2_model: nn.Module,
    stage2_representation: Any,
    stage2_statistics: dict[str, list[float]],
    fusion_choice: dict[str, Any],
    calibrator: BivariateIsotonicCalibrator | None,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    validation_csv = labels_dir / "label_pairs_validation.csv"
    assert_no_forbidden_path(validation_csv)
    assert_development_csv(validation_csv)
    raw = load_raw_partition(validation_csv, event_store_dir)
    weights = make_event_weights(raw.peak_ids)
    stage1_scores, stage1_qc = score_stage1_raw(
        raw,
        stage1_model,
        stage1_config,
        stage1_metadata["feature_statistics"],
        device,
        batch_size,
    )
    ambiguous = (stage1_scores >= TAU_LOW) & (stage1_scores <= TAU_HIGH)
    stage2_scores = np.full(stage1_scores.shape, np.nan, dtype=np.float64)
    stage2_qc: dict[str, int] = {}
    if np.any(ambiguous):
        selected_scores, stage2_qc = score_stage2_raw(
            raw.waveforms[ambiguous],
            stage2_model,
            stage2_representation,
            stage2_statistics,
            device,
            batch_size,
        )
        stage2_scores[ambiguous] = selected_scores
    cascade_scores = apply_selected_fusion(
        stage1_scores.astype(np.float64), stage2_scores, fusion_choice, calibrator
    )
    metrics = {
        "stage1": metric_summary(raw.labels, stage1_scores, weights, raw.peak_ids),
        "cascade": metric_summary(raw.labels, cascade_scores, weights, raw.peak_ids),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "held_out_scores.npz"
    np.savez_compressed(
        score_path,
        labels=raw.labels,
        peak_ids=raw.peak_ids,
        weights=weights,
        stage1_scores=stage1_scores.astype(np.float32),
        stage2_scores=stage2_scores.astype(np.float32),
        cascade_scores=cascade_scores.astype(np.float32),
        ambiguous_mask=ambiguous,
    )
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "partition": "same-domain development held-out validation file partition",
        "event_count": int(raw.labels.size),
        "pair_count": int(raw.labels.size // 2),
        "routed_event_count": int(np.count_nonzero(ambiguous)),
        "routed_fraction": float(np.mean(ambiguous)),
        "stage1_checkpoint": stage1_metadata,
        "stage2_representation": stage2_representation.as_dict(),
        "stage2_feature_statistics": stage2_statistics,
        "fusion": fusion_choice,
        "metrics": metrics,
        "representation_qc": {"stage1": stage1_qc, "stage2": stage2_qc},
        "score_artifact": {
            "path": relative(score_path),
            "sha256": sha256_file(score_path),
        },
        "held_out_partition_already_consumed_for_prior_baseline_comparison": True,
        "held_out_scores_used_for_this_cascade_selection": False,
        "selection_complete_before_held_out": True,
        "test_partition_used": False,
        "scientific_boundary": (
            "Same-domain file-disjoint development holdout; not an independent isotope "
            "or session campaign and not the locked test partition."
        ),
    }
    save_json(output_dir / "held_out_evaluation.json", report)
    del raw
    return report


def resize_th232_cache(handle: h5py.File, new_size: int) -> None:
    for name in (
        "corrected_energy_kev",
        "stage1_score",
        "stage2_score",
        "cascade_score",
        "qc_rejection_bits",
        "source_file_index",
        "source_row",
        "event_id",
    ):
        handle[name].resize((new_size,))


def create_th232_cache(handle: h5py.File, file_count: int, chunk_size: int) -> h5py.Group:
    for name, dtype in (
        ("corrected_energy_kev", np.float32),
        ("stage1_score", np.float32),
        ("stage2_score", np.float32),
        ("cascade_score", np.float32),
        ("qc_rejection_bits", np.uint16),
        ("source_file_index", np.uint16),
        ("source_row", np.int64),
        ("event_id", np.int64),
    ):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=dtype,
        )
    source_files = handle.create_group("source_files")
    source_files.create_dataset("path", shape=(file_count,), dtype=STRING_DTYPE)
    source_files.create_dataset("sha256", shape=(file_count,), dtype=STRING_DTYPE)
    source_files.create_dataset("input_event_count", shape=(file_count,), dtype=np.int64)
    source_files.create_dataset("admitted_event_count", shape=(file_count,), dtype=np.int64)
    return source_files


def score_th232(
    files: list[Path],
    checkpoint_path: Path,
    stage1_model: nn.Module,
    stage1_metadata: dict[str, Any],
    stage1_config: Any,
    stage2_model: nn.Module,
    stage2_representation: Any,
    stage2_statistics: dict[str, list[float]],
    fusion_choice: dict[str, Any],
    calibrator: BivariateIsotonicCalibrator | None,
    output_path: Path,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    if output_path.exists():
        output_path.unlink()
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    counts = {
        "input_events": 0,
        "admitted_events": 0,
        "rejected_energy_or_qc_bits_0_to_2": 0,
        "rejected_nonpositive_shaped_energy": 0,
        "stage1_ambiguous_events": 0,
        "stage1_anchor_fallback_count": 0,
        "stage1_invalid_scale_count": 0,
        "stage2_anchor_fallback_count": 0,
        "stage2_rise_invalid_event_count": 0,
        "stage2_ae_invalid_scale_event_count": 0,
        "stage2_global_scale_invalid_event_count": 0,
    }
    records: list[dict[str, Any]] = []
    next_index = 0
    try:
        with h5py.File(partial, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "experiment_id": EXPERIMENT_ID,
                    "stage1_checkpoint": relative(checkpoint_path),
                    "stage1_checkpoint_sha256": stage1_metadata["checkpoint_sha256"],
                    "stage2_representation": json.dumps(
                        stage2_representation.as_dict(), sort_keys=True
                    ),
                    "stage2_feature_statistics": json.dumps(
                        stage2_statistics, sort_keys=True
                    ),
                    "fusion": json.dumps(fusion_choice, sort_keys=True),
                    "energy_dataset": "corrected_energy_kev",
                    "created_utc": utc_now(),
                    "test_partition_used": False,
                    "retention_policy": "all source events remain in the corrected input HDF5; this score cache retains admitted-event provenance",
                }
            )
            source_table = create_th232_cache(output, len(files), batch_size)
            for file_index, path in enumerate(files):
                digest = sha256_file(path)
                admitted_for_file = 0
                with h5py.File(path, "r") as source:
                    if str(source.attrs.get("processing_status")) != "OK":
                        raise ValueError(f"Non-OK preprocessing status: {path}")
                    if str(source.attrs.get("source_label", "")).lower() != "th232":
                        raise ValueError(f"Unexpected source label: {path}")
                    event_count = int(source["waveform"].shape[0])
                    if source["waveform"].shape != (event_count, 4500):
                        raise ValueError(f"Unexpected waveform shape: {path}")
                    counts["input_events"] += event_count
                    for start in range(0, event_count, batch_size):
                        stop = min(start + batch_size, event_count)
                        energy = np.asarray(
                            source["corrected_energy_kev"][start:stop], dtype=np.float32
                        )
                        shaped = np.asarray(
                            source["shaped_energy_unit"][start:stop], dtype=np.float32
                        )
                        bits = np.asarray(
                            source["qc_rejection_bits"][start:stop], dtype=np.uint16
                        )
                        admitted, energy_valid, shaped_valid = th232_admission_mask(
                            energy, shaped, bits
                        )
                        counts["rejected_energy_or_qc_bits_0_to_2"] += int(
                            np.count_nonzero(~energy_valid)
                        )
                        counts["rejected_nonpositive_shaped_energy"] += int(
                            np.count_nonzero(energy_valid & ~shaped_valid)
                        )
                        if not np.any(admitted):
                            continue
                        selected_rows = start + np.flatnonzero(admitted)
                        waveforms = np.asarray(
                            source["waveform"][selected_rows], dtype=np.float32
                        )
                        selected_shaped = shaped[admitted]
                        raw = RawPartition(
                            waveforms=waveforms,
                            shaped_energy=selected_shaped,
                            labels=np.zeros(waveforms.shape[0], dtype=np.float32),
                            weights=np.ones(waveforms.shape[0], dtype=np.float32),
                            peak_ids=np.full(waveforms.shape[0], "th232", dtype="U16"),
                        )
                        stage1_values, stage1_qc = build_stage1_representation(
                            raw, stage1_config
                        )
                        apply_stage1_statistics(
                            stage1_values, stage1_metadata["feature_statistics"]
                        )
                        stage1_scores = predict_values(
                            stage1_model, stage1_values, device, batch_size
                        ).astype(np.float64)
                        counts["stage1_anchor_fallback_count"] += int(
                            stage1_qc["anchor_fallback_count"]
                        )
                        counts["stage1_invalid_scale_count"] += int(
                            stage1_qc["invalid_scale_count"]
                        )
                        ambiguous = (stage1_scores >= TAU_LOW) & (
                            stage1_scores <= TAU_HIGH
                        )
                        counts["stage1_ambiguous_events"] += int(np.count_nonzero(ambiguous))
                        stage2_scores = np.full(stage1_scores.shape, np.nan, dtype=np.float64)
                        if np.any(ambiguous):
                            stage2_values, stage2_qc = build_stage2_representation(
                                waveforms[ambiguous], stage2_representation
                            )
                            apply_stage2_statistics(stage2_values, stage2_statistics)
                            selected_stage2 = predict_values(
                                stage2_model, stage2_values, device, batch_size
                            )
                            stage2_scores[ambiguous] = selected_stage2
                            for key in (
                                "anchor_fallback_count",
                                "rise_invalid_event_count",
                                "ae_invalid_scale_event_count",
                                "global_scale_invalid_event_count",
                            ):
                                counts[f"stage2_{key}"] += int(stage2_qc[key])
                        cascade_scores = apply_selected_fusion(
                            stage1_scores, stage2_scores, fusion_choice, calibrator
                        ).astype(np.float32)
                        count = int(selected_rows.size)
                        end_index = next_index + count
                        resize_th232_cache(output, end_index)
                        output["corrected_energy_kev"][next_index:end_index] = energy[
                            admitted
                        ]
                        output["stage1_score"][next_index:end_index] = stage1_scores
                        output["stage2_score"][next_index:end_index] = stage2_scores
                        output["cascade_score"][next_index:end_index] = cascade_scores
                        output["qc_rejection_bits"][next_index:end_index] = bits[admitted]
                        output["source_file_index"][next_index:end_index] = file_index
                        output["source_row"][next_index:end_index] = selected_rows
                        if "event_id" in source:
                            event_ids = np.asarray(
                                source["event_id"][selected_rows], dtype=np.int64
                            )
                        else:
                            event_ids = selected_rows.astype(np.int64)
                        output["event_id"][next_index:end_index] = event_ids
                        next_index = end_index
                        admitted_for_file += count
                        counts["admitted_events"] += count
                source_table["path"][file_index] = relative(path)
                source_table["sha256"][file_index] = digest
                source_table["input_event_count"][file_index] = event_count
                source_table["admitted_event_count"][file_index] = admitted_for_file
                records.append(
                    {
                        "path": relative(path),
                        "sha256": digest,
                        "input_events": event_count,
                        "admitted_events": admitted_for_file,
                    }
                )
                print(
                    f"cascade Th232 file {file_index + 1}/{len(files)} "
                    f"admitted={admitted_for_file} cumulative={next_index}",
                    flush=True,
                )
            output.attrs["event_count"] = next_index
            output.attrs["input_event_count"] = counts["input_events"]
            output.attrs["admitted_event_count"] = counts["admitted_events"]
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "score_cache": relative(output_path),
        "score_cache_sha256": sha256_file(output_path),
        "checkpoint": stage1_metadata["checkpoint"],
        "checkpoint_sha256": stage1_metadata["checkpoint_sha256"],
        "counts": counts,
        "files": records,
    }


def write_spectrum_csv(
    path: Path,
    histograms: dict[str, np.ndarray],
) -> None:
    names = list(histograms)
    columns = ["energy_kev_bin_center", *names]
    np.savetxt(
        path,
        np.column_stack([ENERGY_CENTERS, *[histograms[name] for name in names]]),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt=["%.1f"] + ["%d"] * len(names),
    )


def write_peak_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_th232_reports(
    energy: np.ndarray,
    stage1_scores: np.ndarray,
    cascade_scores: np.ndarray,
    thresholds: dict[str, Any],
    output_dir: Path,
) -> tuple[list[PeakWindow], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    windows = fit_peak_windows(np.histogram(energy, ENERGY_EDGES)[0])
    histograms: dict[str, np.ndarray] = {
        "no_cut": np.histogram(energy, ENERGY_EDGES)[0],
    }
    for acceptance in ACCEPTANCES:
        name = threshold_name(acceptance)
        stage1_mask = stage1_scores >= float(thresholds["stage1"][name]["score_threshold"])
        cascade_mask = cascade_scores >= float(
            thresholds["cascade"][name]["score_threshold"]
        )
        histograms[f"ds_cnn_{name}"] = np.histogram(energy[stage1_mask], ENERGY_EDGES)[0]
        histograms[f"cascade_{name}"] = np.histogram(
            energy[cascade_mask], ENERGY_EDGES
        )[0]

    rows: list[dict[str, Any]] = []
    for window in windows:
        baseline = peak_background_metrics(histograms["no_cut"], window)
        rows.append(
            {
                "reference_energy_kev": float(window.reference_kev),
                "observed_centroid_kev": float(window.centroid_kev),
                "model": "no_cut",
                "acceptance": "none",
                "score_threshold": "",
                "events": int(energy.size),
                **{key: float(value) for key, value in baseline.items()},
                "pb_improvement_factor_vs_no_cut": 1.0,
                "net_peak_retention_vs_no_cut": 1.0,
            }
        )
        for model_name, threshold_key in (("ds_cnn", "stage1"), ("cascade", "cascade")):
            for acceptance in ACCEPTANCES:
                name = threshold_name(acceptance)
                score = stage1_scores if model_name == "ds_cnn" else cascade_scores
                threshold = float(thresholds[threshold_key][name]["score_threshold"])
                mask = score >= threshold
                metrics = peak_background_metrics(
                    histograms[f"{model_name}_{name}"], window
                )
                rows.append(
                    {
                        "reference_energy_kev": float(window.reference_kev),
                        "observed_centroid_kev": float(window.centroid_kev),
                        "model": model_name,
                        "acceptance": name,
                        "score_threshold": threshold,
                        "requested_weighted_acceptance": float(
                            thresholds[threshold_key][name][
                                "requested_weighted_acceptance"
                            ]
                        ),
                        "actual_weighted_acceptance": float(
                            thresholds[threshold_key][name]["actual_weighted_acceptance"]
                        ),
                        "events": int(np.count_nonzero(mask)),
                        **{key: float(value) for key, value in metrics.items()},
                        "pb_improvement_factor_vs_no_cut": float(
                            metrics["peak_to_background"]
                            / baseline["peak_to_background"]
                        ),
                        "net_peak_retention_vs_no_cut": float(
                            metrics["net_peak_counts"] / baseline["net_peak_counts"]
                        ),
                    }
                )

    at_90: list[dict[str, Any]] = []
    for window in windows:
        ds = next(
            row
            for row in rows
            if row["model"] == "ds_cnn"
            and row["acceptance"] == "90pct"
            and row["reference_energy_kev"] == float(window.reference_kev)
        )
        cascade = next(
            row
            for row in rows
            if row["model"] == "cascade"
            and row["acceptance"] == "90pct"
            and row["reference_energy_kev"] == float(window.reference_kev)
        )
        pb_ratio = float(cascade["peak_to_background"] / ds["peak_to_background"])
        retention_delta = float(
            cascade["net_peak_retention_vs_no_cut"]
            - ds["net_peak_retention_vs_no_cut"]
        )
        at_90.append(
            {
                "reference_energy_kev": float(window.reference_kev),
                "ds_cnn_peak_to_background": float(ds["peak_to_background"]),
                "cascade_peak_to_background": float(cascade["peak_to_background"]),
                "cascade_over_ds_pb_ratio": pb_ratio,
                "ds_cnn_net_peak_retention": float(
                    ds["net_peak_retention_vs_no_cut"]
                ),
                "cascade_net_peak_retention": float(
                    cascade["net_peak_retention_vs_no_cut"]
                ),
                "cascade_minus_ds_retention": retention_delta,
                "pb_gate_pass": bool(
                    np.isfinite(pb_ratio) and cascade["peak_to_background"] >= ds["peak_to_background"]
                ),
                "retention_gate_pass": bool(retention_delta >= -0.01),
            }
        )
    comparison = {
        "acceptance": "90pct",
        "per_peak": at_90,
        "all_peak_pb_gate_pass": all(row["pb_gate_pass"] for row in at_90),
        "all_peak_retention_gate_pass": all(row["retention_gate_pass"] for row in at_90),
        "retention_tolerance_absolute": 0.01,
        "overall_gate_pass": all(
            row["pb_gate_pass"] and row["retention_gate_pass"] for row in at_90
        ),
    }
    return windows, histograms, rows, comparison


def plot_th232_results(
    output_dir: Path,
    histograms: dict[str, np.ndarray],
    windows: list[PeakWindow],
    rows: list[dict[str, Any]],
) -> None:
    names = list(histograms)
    cascade_names = ["no_cut", *[f"cascade_{threshold_name(a)}" for a in ACCEPTANCES]]
    colors = {"no_cut": "black"}
    palette = plt.cm.viridis(np.linspace(0.08, 0.94, len(ACCEPTANCES)))
    colors.update(
        {f"cascade_{threshold_name(a)}": color for a, color in zip(ACCEPTANCES, palette)}
    )
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for name in cascade_names:
        for axis in axes:
            axis.step(
                ENERGY_CENTERS,
                histograms[name],
                where="mid",
                linewidth=0.8,
                color=colors[name],
                label=name if axis is axes[0] else "_nolegend_",
            )
    axes[0].set_ylabel("Counts / 1 keV")
    axes[1].set_ylabel("Counts / 1 keV")
    axes[1].set_xlabel("Corrected energy (keV; preliminary calibration)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[0].set_title("Th-232 spectrum after cascaded DS-CNN score cuts")
    axes[0].legend(ncol=4, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        for window in windows:
            axis.axvline(window.centroid_kev, color="0.6", linewidth=0.45, alpha=0.5)
    figure.savefig(output_dir / "th232_cascade_energy_spectra.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for model_name, marker in (("ds_cnn", "o"), ("cascade", "s")):
        for acceptance in ACCEPTANCES:
            selected = [
                row
                for row in rows
                if row["model"] == model_name and row["acceptance"] == threshold_name(acceptance)
            ]
            selected.sort(key=lambda row: row["reference_energy_kev"])
            axes[0].plot(
                [row["reference_energy_kev"] for row in selected],
                [row["peak_to_background"] for row in selected],
                marker=marker,
                linewidth=0.8,
                label=f"{model_name} {threshold_name(acceptance)[:-3]}%",
            )
    axes[0].set_xlabel("Reference peak energy (keV)")
    axes[0].set_ylabel("Peak / background")
    axes[0].set_title("Th-232 P/B across retention thresholds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    selected_90 = [
        row
        for row in rows
        if row["acceptance"] == "90pct" and row["model"] in {"ds_cnn", "cascade"}
    ]
    for model_name, color in (("ds_cnn", "tab:blue"), ("cascade", "tab:orange")):
        values = [
            row for row in selected_90 if row["model"] == model_name
        ]
        values.sort(key=lambda row: row["reference_energy_kev"])
        axes[1].plot(
            [row["reference_energy_kev"] for row in values],
            [row["net_peak_retention_vs_no_cut"] for row in values],
            marker="o",
            color=color,
            label=model_name,
        )
    axes[1].axhline(0.0, color="0.5", linewidth=0.6)
    axes[1].set_xlabel("Reference peak energy (keV)")
    axes[1].set_ylabel("Net peak retention vs no cut")
    axes[1].set_title("90% operating point: net peak retention")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_dir / "th232_cascade_peak_metrics.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    flat_axes = list(axes.flat)
    zoom_names = ["no_cut", "cascade_99pct", "cascade_95pct", "cascade_90pct"]
    zoom_colors = ["black", "#440154", "#31688e", "#35b779"]
    for axis, window in zip(flat_axes, windows):
        half_width = max(18.0, 6.0 * window.sigma_kev)
        selected = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (
            ENERGY_CENTERS <= window.centroid_kev + half_width
        )
        for name, color in zip(zoom_names, zoom_colors):
            axis.step(
                ENERGY_CENTERS[selected],
                histograms[name][selected],
                where="mid",
                linewidth=0.8,
                color=color,
                label=name,
            )
        axis.axvspan(window.roi_low_kev, window.roi_high_kev, color="#984ea3", alpha=0.10)
        axis.set_title(
            f"{window.reference_kev:g} keV ref.\nobserved {window.centroid_kev:.2f} keV"
        )
        axis.grid(alpha=0.2)
    for axis in flat_axes[len(windows) :]:
        axis.axis("off")
    flat_axes[0].legend(fontsize=7)
    figure.suptitle("Th-232 peak windows: cascaded DS-CNN score cuts")
    figure.supxlabel("Corrected energy (keV)")
    figure.supylabel("Counts / 1 keV")
    figure.savefig(output_dir / "th232_cascade_peak_zooms.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--th232-dir", type=Path, default=DEFAULT_TH232_DIR)
    parser.add_argument("--expected-th232-files", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.expected_th232_files < 1:
        raise ValueError("batch-size and expected-th232-files must be positive")
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    stage1_checkpoint = args.stage1_checkpoint.resolve()
    experiment_dir = args.experiment_dir.resolve()
    th232_dir = args.th232_dir.resolve()
    mining_dir = experiment_dir / "mining"
    stage2_dir = experiment_dir / "stage2_candidates"
    fusion_path = experiment_dir / "fusion_selection.json"
    final_refit_dir = experiment_dir / "final_refit"
    internal_dir = experiment_dir / "final_internal"
    held_out_dir = experiment_dir / "held_out"
    th232_output_dir = experiment_dir / "th232"
    for path in (
        labels_dir,
        event_store_dir,
        stage1_checkpoint,
        mining_dir / "mining_summary.json",
        mining_dir / "stage1_train_scores.npz",
        stage2_dir / "candidate_selection.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    assert_no_forbidden_path(train_csv)
    assert_no_forbidden_path(split_path)
    assert_development_csv(train_csv)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    stage1_model, stage1_metadata, stage1_config = load_stage1_model(
        stage1_checkpoint, device
    )
    candidate_selection = json.loads(
        (stage2_dir / "candidate_selection.json").read_text(encoding="utf-8")
    )
    if candidate_selection.get("held_out_partition_loaded") is not False:
        raise ValueError("Candidate selection has an invalid held-out boundary")
    selected_candidate = str(candidate_selection["selected_candidate"])
    if selected_candidate not in CANDIDATES:
        raise ValueError(f"Unknown selected candidate: {selected_candidate}")
    selected_result = json.loads(
        (stage2_dir / selected_candidate / "training_result.json").read_text(
            encoding="utf-8"
        )
    )
    selected_epoch = int(selected_result["selected_epoch"])
    if not 1 <= selected_epoch <= STAGE2_EPOCHS:
        raise ValueError(f"Invalid selected Stage-2 epoch: {selected_epoch}")

    print("selecting fusion on the internal development partition", flush=True)
    fusion_selection, _development_arrays = select_internal_fusion(
        mining_dir, stage2_dir, candidate_selection, fusion_path
    )
    fusion_choice = dict(fusion_selection["selected_fusion"])
    print(
        f"selected_candidate={selected_candidate} "
        f"selected_fusion={fusion_choice['method']} "
        f"temperature={fusion_choice.get('temperature')}",
        flush=True,
    )

    print("loading the full development manifest for the frozen final refit", flush=True)
    train_raw = load_raw_partition(train_csv, event_store_dir)
    split = np.load(split_path)
    fit_pairs = np.asarray(split["fit_pair_indices"], dtype=np.int64)
    internal_pairs = np.asarray(split["internal_pair_indices"], dtype=np.int64)
    internal_events = event_indices(internal_pairs)
    if np.intersect1d(fit_pairs, internal_pairs).size:
        raise ValueError("Training split pairs overlap")
    if not np.array_equal(
        np.sort(np.concatenate((fit_pairs, internal_pairs))),
        np.arange(train_raw.labels.size // 2, dtype=np.int64),
    ):
        raise ValueError("Training split does not cover the development manifest")
    train_weights = make_event_weights(train_raw.peak_ids)
    print("scoring the frozen production Stage-1 model on development events", flush=True)
    train_stage1_scores, train_stage1_qc = score_stage1_raw(
        train_raw,
        stage1_model,
        stage1_config,
        stage1_metadata["feature_statistics"],
        device,
        args.batch_size,
    )
    train_ambiguous = (train_stage1_scores >= TAU_LOW) & (
        train_stage1_scores <= TAU_HIGH
    )
    ambiguous_indices = np.flatnonzero(train_ambiguous)
    print(
        f"final_refit_development_events={train_raw.labels.size} "
        f"final_refit_ambiguous_events={ambiguous_indices.size}",
        flush=True,
    )
    stage2_representation = CANDIDATES[selected_candidate]
    stage2_model, stage2_checkpoint_path, stage2_statistics, stage2_result, train_stage2_scores = fit_final_stage2(
        train_raw,
        ambiguous_indices,
        stage2_representation,
        stage1_metadata,
        final_refit_dir,
        device,
        selected_epoch,
        STAGE2_BATCH_SIZE,
    )
    train_stage2_full = np.full(train_stage1_scores.shape, np.nan, dtype=np.float64)
    train_stage2_full[ambiguous_indices] = train_stage2_scores.astype(np.float64)
    final_calibrator = fit_final_internal_calibrator(
        fusion_choice,
        train_stage1_scores.astype(np.float64),
        train_stage2_full,
        train_raw,
        internal_events,
    )
    train_cascade_scores = apply_selected_fusion(
        train_stage1_scores.astype(np.float64),
        train_stage2_full,
        fusion_choice,
        final_calibrator,
    )
    final_internal_report, thresholds = evaluate_final_internal(
        train_raw,
        train_stage1_scores.astype(np.float64),
        train_stage2_full,
        train_cascade_scores,
        internal_events,
        internal_dir,
    )
    final_protocol = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_BEFORE_HELD_OUT_AND_TH232",
        "stage1": stage1_metadata,
        "stage1_representation_qc_on_development": train_stage1_qc,
        "stage2_candidate_selection": {
            "selected_candidate": selected_candidate,
            "selected_epoch": selected_epoch,
            "candidate_selection_path": relative(stage2_dir / "candidate_selection.json"),
            "candidate_selection_sha256": sha256_file(stage2_dir / "candidate_selection.json"),
            "candidate_checkpoint_sha256": selected_result["checkpoint_sha256"],
        },
        "stage2_final_refit": stage2_result,
        "fusion_selection": fusion_selection,
        "selected_fusion": fusion_choice,
        "final_internal_evaluation": final_internal_report,
        "thresholds": thresholds,
        "partition": {
            "development_event_count": int(train_raw.labels.size),
            "internal_event_count": int(internal_events.size),
            "final_refit_ambiguous_event_count": int(ambiguous_indices.size),
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "target_data_used_for_selection": False,
        },
        "calibrator": final_calibrator.as_dict() if final_calibrator is not None else None,
        "scientific_boundary": (
            "Internal development metrics and thresholds are provisional under the "
            "scalar-shortcut warning; no locked test, Eu-152, or Th-232 information was used."
        ),
    }
    save_json(experiment_dir / "final_protocol.json", final_protocol)
    print(
        json.dumps(final_internal_report["metrics"], indent=2, sort_keys=True),
        flush=True,
    )
    print("final protocol frozen; evaluating same-domain held-out partition", flush=True)

    held_out_report = evaluate_held_out(
        labels_dir,
        event_store_dir,
        stage1_model,
        stage1_metadata,
        stage1_config,
        stage2_model,
        stage2_representation,
        stage2_statistics,
        fusion_choice,
        final_calibrator,
        held_out_dir,
        device,
        args.batch_size,
    )
    final_protocol["held_out_partition_loaded"] = True
    final_protocol["held_out_scores_used_for_selection"] = False
    final_protocol["held_out_evaluation"] = {
        "path": relative(held_out_dir / "held_out_evaluation.json"),
        "sha256": sha256_file(held_out_dir / "held_out_evaluation.json"),
    }
    save_json(experiment_dir / "final_protocol.json", final_protocol)
    print(json.dumps(held_out_report["metrics"], indent=2, sort_keys=True), flush=True)

    print("held-out report complete; opening corrected Th-232 files", flush=True)
    th232_files = sorted(th232_dir.glob("*.h5"))
    if len(th232_files) != args.expected_th232_files:
        raise ValueError(
            f"Expected {args.expected_th232_files} Th-232 files, found {len(th232_files)}"
        )
    th232_output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = th232_output_dir / "th232_cascade_scores.h5"
    scoring = score_th232(
        th232_files,
        stage1_checkpoint,
        stage1_model,
        stage1_metadata,
        stage1_config,
        stage2_model,
        stage2_representation,
        stage2_statistics,
        fusion_choice,
        final_calibrator,
        cache_path,
        args.batch_size,
        device,
        args.overwrite,
    )
    with h5py.File(cache_path, "r") as cache:
        energy = np.asarray(cache["corrected_energy_kev"], dtype=np.float32)
        stage1_scores = np.asarray(cache["stage1_score"], dtype=np.float32)
        cascade_scores = np.asarray(cache["cascade_score"], dtype=np.float32)
    windows, histograms, peak_rows, comparison = build_th232_reports(
        energy,
        stage1_scores,
        cascade_scores,
        thresholds,
        th232_output_dir,
    )
    spectrum_path = th232_output_dir / "th232_cascade_energy_spectra_1kev.csv"
    peak_path = th232_output_dir / "th232_cascade_peak_to_background.csv"
    write_spectrum_csv(spectrum_path, histograms)
    write_peak_csv(peak_path, peak_rows)
    plot_th232_results(th232_output_dir, histograms, windows, peak_rows)
    th232_report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "status": "EXTERNAL_EVALUATION",
        "model_frozen_before_th232": True,
        "selection_complete_before_held_out": True,
        "held_out_evaluation_complete_before_th232": True,
        "stage1": stage1_metadata,
        "stage2": stage2_result,
        "selected_fusion": fusion_choice,
        "threshold_calibration": thresholds,
        "th232_scoring": scoring,
        "admission": {
            "energy_dataset": "corrected_energy_kev",
            "energy_range_kev": [ENERGY_MIN_KEV, ENERGY_MAX_KEV],
            "qc_rule": "reject qc_rejection_bits 0-2; retain noise bit 3 and pulse bit 4",
            "shaped_energy_rule": "finite and positive",
            "raw_events_retained_in_input_cache": True,
        },
        "global_event_count": int(energy.size),
        "global_retention": {
            "stage1": {
                threshold_name(a): {
                    "score_threshold": float(
                        thresholds["stage1"][threshold_name(a)]["score_threshold"]
                    ),
                    "events": int(
                        np.count_nonzero(
                            stage1_scores
                            >= thresholds["stage1"][threshold_name(a)]["score_threshold"]
                        )
                    ),
                }
                for a in ACCEPTANCES
            },
            "cascade": {
                threshold_name(a): {
                    "score_threshold": float(
                        thresholds["cascade"][threshold_name(a)]["score_threshold"]
                    ),
                    "events": int(
                        np.count_nonzero(
                            cascade_scores
                            >= thresholds["cascade"][threshold_name(a)]["score_threshold"]
                        )
                    ),
                }
                for a in ACCEPTANCES
            },
        },
        "peak_windows": [asdict(window) for window in windows],
        "peak_background_definition": (
            "(counts in observed centroid +/-2 sigma ROI - linearly interpolated "
            "3--5 sigma sideband background) / estimated background"
        ),
        "peak_background_rows": peak_rows,
        "comparison_at_90_percent": comparison,
        "scientific_boundary": (
            "Historical corrected Th-232 cache and same-domain development-positive "
            "threshold calibration; no independent isotope/session claim. Th-232 "
            "was not used to select the candidate, fusion, checkpoint, or score cuts, "
            "and the locked test partition was not used."
        ),
        "artifacts": {},
    }
    artifact_paths = [
        cache_path,
        spectrum_path,
        peak_path,
        th232_output_dir / "th232_cascade_energy_spectra.png",
        th232_output_dir / "th232_cascade_peak_metrics.png",
        th232_output_dir / "th232_cascade_peak_zooms.png",
    ]
    for path in artifact_paths:
        th232_report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    th232_report_path = th232_output_dir / "th232_cascade_evaluation.json"
    th232_report["artifacts"][th232_report_path.name] = {
        "path": relative(th232_report_path)
    }
    save_json(th232_report_path, th232_report)
    print(
        f"th232_admitted_events={energy.size} "
        f"90pct_gate_pass={comparison['overall_gate_pass']}",
        flush=True,
    )
    print(f"report={relative(th232_report_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
