#!/usr/bin/env python3
"""Train the positive-polarity dual-anchor Compact CNN without test leakage."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
    train_epoch,
)

PEAK_IDS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
HIGH_ENERGY_PEAK_IDS = ("na22_511kev", "cs137_662kev")
DUAL_ANCHOR_REPRESENTATION = RepresentationConfig(
    name="both_ma10_independent_peak_dual_anchor_w501",
    input_mode="both",
    moving_average=10,
    normalization="independent_positive_peak",
    anchor="dual_t10_current_peak",
    pre_samples=250,
    post_samples=250,
    pulse_polarity="negative_to_positive",
    standardization="none",
    downsample=1,
    representation_schema_version=2,
    endpoint_inclusive=True,
    current_search_start=1100,
    current_search_stop=1500,
    clip_charge_to_unit_interval=True,
)
SHARED_T10_REPRESENTATION = RepresentationConfig(
    name="both_ma10_charge_peak_shared_t10_w501",
    input_mode="both",
    moving_average=10,
    normalization="charge_peak_shared",
    anchor="t10",
    pre_samples=250,
    post_samples=250,
    pulse_polarity="negative_to_positive",
    standardization="fixed_current_peak_scale",
    downsample=1,
    representation_schema_version=2,
    endpoint_inclusive=True,
    clip_charge_to_unit_interval=False,
)
SHARED_T10_RAW_ZSCORE_REPRESENTATION = RepresentationConfig(
    name="both_raw_global_t10_w501_positive_polarity",
    input_mode="both",
    moving_average=1,
    normalization="global",
    anchor="t10",
    pre_samples=250,
    post_samples=250,
    pulse_polarity="negative_to_positive",
    standardization="train_zscore",
    downsample=1,
    representation_schema_version=2,
    endpoint_inclusive=True,
    clip_charge_to_unit_interval=False,
)
REPRESENTATIONS = {
    "dual_anchor": DUAL_ANCHOR_REPRESENTATION,
    "shared_t10_relaxed": SHARED_T10_REPRESENTATION,
    "shared_t10_raw_zscore": SHARED_T10_RAW_ZSCORE_REPRESENTATION,
}
WIDTH = 24
LEARNING_RATE = 8.0e-4
WEIGHT_DECAY = 3.0e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pair_peaks(csv_path: Path) -> np.ndarray:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return np.asarray([row["peak_id"] for row in csv.DictReader(stream)], dtype="U64")


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def equal_peak_event_weights(selected_pair_peaks: np.ndarray) -> np.ndarray:
    unique = sorted(set(selected_pair_peaks.tolist()))
    if not unique:
        raise ValueError("Cannot weight an empty partition")
    pair_weights = np.empty(selected_pair_peaks.size, dtype=np.float32)
    for peak_id in unique:
        mask = selected_pair_peaks == peak_id
        pair_weights[mask] = 1.0 / (len(unique) * int(np.count_nonzero(mask)))
    return np.repeat(pair_weights, 2)


def make_loader(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(values),
            torch.from_numpy(labels),
            torch.from_numpy(weights),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for values, target, _weight in loader:
            logits = model(values.to(device, non_blocking=True))
            labels.append(target.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def metrics(labels: np.ndarray, scores: np.ndarray, event_peaks: np.ndarray) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(event_peaks.tolist())):
        mask = event_peaks == peak_id
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "pair_count": int(np.count_nonzero(mask) // 2),
        }
    aurocs = [float(value["auroc"]) for value in per_peak.values()]
    return {
        "worst_peak_auroc": float(min(aurocs)),
        "macro_auroc": float(np.mean(aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "per_peak": per_peak,
    }


def stratified_stage1_split(
    fit_pairs: np.ndarray,
    all_pair_peaks: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for peak_id in HIGH_ENERGY_PEAK_IDS:
        available = fit_pairs[all_pair_peaks[fit_pairs] == peak_id].copy()
        rng.shuffle(available)
        count = max(1, int(round(validation_fraction * available.size)))
        validation.extend(available[:count].tolist())
        train.extend(available[count:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def train_select_epochs(
    initial_state: dict[str, torch.Tensor] | None,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    validation_peaks: np.ndarray,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    selection: str,
) -> tuple[dict[str, torch.Tensor], int, dict[str, Any], list[dict[str, Any]]]:
    set_seed(seed)
    model = CompactWaveformCNN(2, width=WIDTH).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_values, train_labels, train_weights, batch_size, True, seed)
    validation_loader = make_loader(
        validation_values, validation_labels, validation_weights, batch_size, False, seed
    )
    best_key = (-np.inf, -np.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        labels, scores = predict(model, validation_loader, device)
        summary = metrics(labels, scores, validation_peaks)
        key = (
            (summary["macro_auroc"], summary["pooled_auroc"])
            if selection == "macro"
            else (summary["worst_peak_auroc"], summary["macro_auroc"])
        )
        history.append({"epoch": epoch, "train_loss": loss, "validation": summary})
        print(
            f"epoch={epoch} loss={loss:.6f} worst={summary['worst_peak_auroc']:.6f} "
            f"macro={summary['macro_auroc']:.6f}",
            flush=True,
        )
        if key > (best_key[0] + 1.0e-4, best_key[1]):
            best_key = key
            best_epoch = epoch
            best_metrics = summary
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("No epoch selected")
    return best_state, best_epoch, best_metrics, history


def train_fixed_epochs(
    initial_state: dict[str, torch.Tensor] | None,
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    set_seed(seed)
    model = CompactWaveformCNN(2, width=WIDTH).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = make_loader(values, labels, weights, batch_size, True, seed)
    history = [train_epoch(model, loader, optimizer, device) for _ in range(epochs)]
    return model, history


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--label-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_weight_scan_20260819",
    )
    result.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    result.add_argument(
        "--representation",
        choices=tuple(REPRESENTATIONS),
        default="shared_t10_relaxed",
    )
    result.add_argument("--stage1-validation-fraction", type=float, default=0.15)
    result.add_argument("--pretrain-epochs", type=int, default=8)
    result.add_argument("--finetune-epochs", type=int, default=10)
    result.add_argument("--patience", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--seed", type=int, default=20260820)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    representation = REPRESENTATIONS[args.representation]
    default_output = (
        PROJECT_ROOT / "outputs/models/shared_t10_raw_zscore_compact_20260820"
        if args.representation == "shared_t10_raw_zscore"
        else (
            PROJECT_ROOT / "outputs/models/shared_t10_relaxed_compact_20260820"
            if args.representation == "shared_t10_relaxed"
            else PROJECT_ROOT / "outputs/models/dual_anchor_compact_20260820"
        )
    )
    output_dir = (args.output_dir or default_output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = args.label_dir.resolve()
    train_csv = label_dir / "label_pairs_train.csv"
    heldout_csv = label_dir / "label_pairs_validation.csv"
    split_path = label_dir / "train_internal_split_indices.npz"
    split = np.load(split_path)
    fit_pairs = split["fit_pair_indices"]
    internal_pairs = split["internal_pair_indices"]
    all_pair_peaks = pair_peaks(train_csv)
    stage1_train_pairs, stage1_validation_pairs = stratified_stage1_split(
        fit_pairs, all_pair_peaks, args.stage1_validation_fraction, args.seed + 1
    )
    high_fit_pairs = fit_pairs[np.isin(all_pair_peaks[fit_pairs], HIGH_ENERGY_PEAK_IDS)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} representation={representation.name}", flush=True)

    raw_train = load_raw_partition(train_csv, args.event_store_dir.resolve())
    values_train, train_qc = build_representation(raw_train, representation)
    if values_train.shape[1:] != (2, representation.window_length):
        raise ValueError(f"Unexpected input shape: {values_train.shape}")
    labels_train = raw_train.labels
    peaks_train = raw_train.peak_ids
    del raw_train
    feature_statistics = {"means": [0.0, 0.0], "standard_deviations": [1.0, 1.0]}
    if representation.standardization == "fixed_current_peak_scale":
        fit_events_for_scale = event_indices(fit_pairs)
        positive_current_peaks = np.max(values_train[fit_events_for_scale, 1], axis=1)
        valid_peaks = positive_current_peaks[
            np.isfinite(positive_current_peaks) & (positive_current_peaks > 1.0e-12)
        ]
        if valid_peaks.size != positive_current_peaks.size:
            raise ValueError("Invalid fit-only current peaks for fixed scaling")
        current_scale = float(np.median(valid_peaks))
        values_train[:, 1] /= current_scale
        feature_statistics["standard_deviations"][1] = current_scale
    elif representation.standardization == "train_zscore":
        fit_events_for_scale = event_indices(fit_pairs)
        feature_statistics = fit_channel_statistics(values_train[fit_events_for_scale])
        apply_channel_statistics(values_train, feature_statistics)
    elif representation.standardization != "none":
        raise ValueError(f"Unsupported standardization: {representation.standardization}")

    def partition(pair_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        events = event_indices(pair_indices)
        selected_pair_peaks = all_pair_peaks[pair_indices]
        return (
            values_train[events],
            labels_train[events],
            equal_peak_event_weights(selected_pair_peaks),
            peaks_train[events],
        )

    s1_train = partition(stage1_train_pairs)
    s1_validation = partition(stage1_validation_pairs)
    print("stage=high_energy_epoch_selection", flush=True)
    _s1_state, pretrain_epochs, pretrain_metrics, pretrain_history = train_select_epochs(
        None,
        *s1_train[:3],
        *s1_validation[:3],
        s1_validation[3],
        args.pretrain_epochs,
        args.patience,
        args.batch_size,
        args.seed + 10,
        device,
        "macro",
    )

    high_fit = partition(high_fit_pairs)
    fit = partition(fit_pairs)
    internal = partition(internal_pairs)
    print(f"stage=pretrain_refit epochs={pretrain_epochs}", flush=True)
    pretrained_model, pretrain_refit_loss = train_fixed_epochs(
        None, *high_fit[:3], pretrain_epochs, args.batch_size, args.seed + 20, device
    )
    pretrained_state = copy.deepcopy(pretrained_model.state_dict())
    del pretrained_model

    candidates: dict[str, dict[str, Any]] = {}
    for index, (name, initial_state) in enumerate(
        (("high_energy_pretrained", pretrained_state), ("scratch", None))
    ):
        print(f"stage=balanced_finetune candidate={name}", flush=True)
        state, epochs, summary, history = train_select_epochs(
            initial_state,
            *fit[:3],
            *internal[:3],
            internal[3],
            args.finetune_epochs,
            args.patience,
            args.batch_size,
            args.seed + 30 + index,
            device,
            "worst",
        )
        candidates[name] = {
            "state": state,
            "selected_finetune_epochs": epochs,
            "internal_metrics": summary,
            "history": history,
        }
    selected_name = max(
        candidates,
        key=lambda name: (
            candidates[name]["internal_metrics"]["worst_peak_auroc"],
            candidates[name]["internal_metrics"]["macro_auroc"],
        ),
    )
    selected_epochs = int(candidates[selected_name]["selected_finetune_epochs"])
    print(f"selected={selected_name} finetune_epochs={selected_epochs}", flush=True)

    all_pairs = np.arange(all_pair_peaks.size, dtype=np.int64)
    all_train = partition(all_pairs)
    high_all_pairs = all_pairs[np.isin(all_pair_peaks, HIGH_ENERGY_PEAK_IDS)]
    final_initial: dict[str, torch.Tensor] | None = None
    final_pretrain_loss: list[float] = []
    if selected_name == "high_energy_pretrained":
        high_all = partition(high_all_pairs)
        final_pretrained, final_pretrain_loss = train_fixed_epochs(
            None, *high_all[:3], pretrain_epochs, args.batch_size, args.seed + 100, device
        )
        final_initial = copy.deepcopy(final_pretrained.state_dict())
        del final_pretrained
    final_model, final_finetune_loss = train_fixed_epochs(
        final_initial,
        *all_train[:3],
        selected_epochs,
        args.batch_size,
        args.seed + 110,
        device,
    )

    # The held-out file partition is intentionally first opened after strategy selection and refit.
    raw_heldout = load_raw_partition(heldout_csv, args.event_store_dir.resolve())
    values_heldout, heldout_qc = build_representation(raw_heldout, representation)
    if representation.standardization == "fixed_current_peak_scale":
        values_heldout[:, 1] /= feature_statistics["standard_deviations"][1]
    elif representation.standardization == "train_zscore":
        apply_channel_statistics(values_heldout, feature_statistics)
    heldout_pair_peaks = raw_heldout.peak_ids[::2]
    heldout_weights = equal_peak_event_weights(heldout_pair_peaks)
    heldout_loader = make_loader(
        values_heldout,
        raw_heldout.labels,
        heldout_weights,
        args.batch_size,
        False,
        args.seed,
    )
    heldout_labels, heldout_scores = predict(final_model, heldout_loader, device)
    heldout_metrics = metrics(heldout_labels, heldout_scores, raw_heldout.peak_ids)

    checkpoint_path = output_dir / "compact_cnn_best.pt"
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in final_model.state_dict().items()
        },
        "model_kind": "compact_cnn",
        "model_width": WIDTH,
        "representation_config": representation.as_dict(),
        "input_shape": [2, representation.window_length],
        "feature_statistics": feature_statistics,
        "training_strategy": selected_name,
        "pretrain_peak_ids": list(HIGH_ENERGY_PEAK_IDS),
        "pretrain_epochs": pretrain_epochs if selected_name == "high_energy_pretrained" else 0,
        "finetune_epochs": selected_epochs,
        "selection_metric": "internal_worst_peak_auroc_then_macro_auroc",
        "selection_internal_metrics": candidates[selected_name]["internal_metrics"],
        "label_train_sha256": sha256_file(train_csv),
        "label_heldout_sha256": sha256_file(heldout_csv),
        "split_sha256": sha256_file(split_path),
        "seed": args.seed,
        "test_partition_used": False,
    }
    torch.save(checkpoint, checkpoint_path)
    scores_path = output_dir / "held_out_scores.npz"
    np.savez_compressed(
        scores_path,
        labels=heldout_labels,
        scores=heldout_scores,
        peak_ids=raw_heldout.peak_ids,
    )
    report_candidates = {
        name: {key: value for key, value in candidate.items() if key != "state"}
        for name, candidate in candidates.items()
    }
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "device": str(device),
        "representation_config": representation.as_dict(),
        "input_shape": [2, representation.window_length],
        "feature_statistics": feature_statistics,
        "data_boundary": {
            "train_pair_count": int(all_pair_peaks.size),
            "fit_pair_count": int(fit_pairs.size),
            "internal_pair_count": int(internal_pairs.size),
            "heldout_pair_count": int(heldout_pair_peaks.size),
            "test_partition_used": False,
        },
        "stage1": {
            "peak_ids": list(HIGH_ENERGY_PEAK_IDS),
            "train_pair_count": int(stage1_train_pairs.size),
            "validation_pair_count": int(stage1_validation_pairs.size),
            "selected_epochs": pretrain_epochs,
            "selection_metric": "equal_high_energy_macro_auroc",
            "validation_metrics": pretrain_metrics,
            "history": pretrain_history,
            "refit_loss": pretrain_refit_loss,
        },
        "stage2_candidates": report_candidates,
        "selected_strategy": selected_name,
        "final_training": {
            "pretrain_loss": final_pretrain_loss,
            "finetune_loss": final_finetune_loss,
        },
        "held_out_metrics": heldout_metrics,
        "qc": {"train": train_qc, "heldout": heldout_qc},
        "inputs": {
            "train_labels": {"path": relative(train_csv), "sha256": sha256_file(train_csv)},
            "heldout_labels": {"path": relative(heldout_csv), "sha256": sha256_file(heldout_csv)},
            "split": {"path": relative(split_path), "sha256": sha256_file(split_path)},
        },
        "artifacts": {
            "checkpoint": {"path": relative(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "held_out_scores": {"path": relative(scores_path), "sha256": sha256_file(scores_path)},
        },
        "test_partition_used": False,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "held_out": heldout_metrics}, indent=2), flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
