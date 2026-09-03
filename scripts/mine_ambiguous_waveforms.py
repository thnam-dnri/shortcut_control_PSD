#!/usr/bin/env python3
"""Mine fit/internal events routed to the fixed Stage-1 ambiguous region.

Only the development train manifest is opened.  A fit-only Stage-1 checkpoint
defines the score bins, and the resulting fit/internal event-level subsets are
saved with raw waveforms and provenance for Stage-2 training and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    build_representation,
    load_raw_partition,
    read_event_references,
    representation_config_from_checkpoint,
    t10_anchor,
)
from src.cascade_refinement import (  # noqa: E402
    BASELINE_STOP,
    CHUNK_SIZE,
    SEARCH_START,
    SEARCH_STOP,
    SELECTED_PEAK_WEIGHTS,
    TAU_HIGH,
    TAU_LOW,
    event_indices,
    make_event_weights,
    moving_average,
    save_json,
    sha256_file,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


EXPERIMENT_ID = "cascaded_ambiguous_refinement_ds_cnn_20260821"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_STAGE1_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/experiments"
    / EXPERIMENT_ID
    / "stage1_dev"
    / "stage1_dev_best.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID / "mining"
SCORE_BATCH_SIZE = 512
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def load_stage1(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn" or checkpoint.get("model_role") != "stage1_development_only":
        raise ValueError("Checkpoint is not the fit-only Stage-1 development model")
    if checkpoint.get("test_partition_used", checkpoint.get("partition", {}).get("test_partition_used")) is not False:
        raise ValueError("Stage-1 checkpoint has an invalid test boundary")
    if checkpoint.get("held_out_partition_loaded", checkpoint.get("partition", {}).get("held_out_partition_loaded")) is not False:
        raise ValueError("Stage-1 checkpoint loaded held-out data")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    model = DSCNN(input_channels=config.channel_count, width=int(checkpoint["model_width"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, config


def score_model(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=SCORE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False)


def score_bin(score: float) -> str:
    if score < TAU_LOW:
        return "low_tail"
    if score <= TAU_HIGH:
        return "ambiguous"
    return "high_tail"


def make_counts(
    partition_name: str,
    indices: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    peak_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_name in ("low_tail", "ambiguous", "high_tail"):
        bin_mask = np.asarray([score_bin(float(value)) == bin_name for value in scores])
        for label in (0, 1):
            for peak_id in sorted(set(peak_ids.tolist())):
                mask = bin_mask & (labels == label) & (peak_ids == peak_id)
                rows.append(
                    {
                        "partition": partition_name,
                        "score_bin": bin_name,
                        "label": label,
                        "peak_id": peak_id,
                        "event_count": int(np.count_nonzero(mask)),
                        "event_fraction_of_partition": float(
                            np.count_nonzero(mask) / max(indices.size, 1)
                        ),
                    }
                )
    return rows


def write_counts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) if rows else ["partition", "score_bin", "label", "peak_id", "event_count"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_subset(
    path: Path,
    partition: str,
    raw: RawPartition,
    event_weights: np.ndarray,
    records: list[tuple[str, int, int, float, str]],
    indices: np.ndarray,
    scores: np.ndarray,
    source_event_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": "1",
                "experiment_id": EXPERIMENT_ID,
                "partition": partition,
                "score_bin": "0.4 <= score <= 0.6",
                "score_low": TAU_LOW,
                "score_high": TAU_HIGH,
                "event_level_selection": True,
                "raw_waveform_definition": "negative acquired polarity, 4500 samples",
            }
        )
        handle.create_dataset("waveform", data=raw.waveforms[selected], compression="lzf")
        handle.create_dataset("shaped_energy", data=raw.shaped_energy[selected])
        handle.create_dataset("labels", data=raw.labels[selected])
        handle.create_dataset("weights", data=event_weights[selected])
        handle.create_dataset("stage1_score", data=scores.astype(np.float32, copy=False))
        handle.create_dataset("source_event_index", data=source_event_indices[selected])
        handle.create_dataset("pair_index", data=source_event_indices[selected] // 2)
        handle.create_dataset("peak_ids", data=np.asarray(raw.peak_ids[selected], dtype=object), dtype=STRING_DTYPE)
        handle.create_dataset(
            "source_hdf5",
            data=np.asarray([records[int(index)][0] for index in selected], dtype=object),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "source_row",
            data=np.asarray([records[int(index)][1] for index in selected], dtype=np.int64),
        )


def diagnostic_windows(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = -np.asarray(waveforms, dtype=np.float32)
    baseline = np.median(positive[:, :BASELINE_STOP], axis=1).astype(np.float32)
    charge = moving_average(positive - baseline[:, None], 10)
    current = np.gradient(charge, 4.0, axis=1).astype(np.float32)
    anchors, _fallback = t10_anchor(charge)
    offsets = np.arange(-250, 500, dtype=np.int64)
    indices = anchors[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < charge.shape[1])
    clipped = np.clip(indices, 0, charge.shape[1] - 1)
    charge_window = charge[np.arange(charge.shape[0])[:, None], clipped]
    current_window = current[np.arange(current.shape[0])[:, None], clipped]
    charge_window[~valid] = 0.0
    current_window[~valid] = 0.0
    charge_peak = np.max(charge_window, axis=1)
    current_peak = np.max(np.abs(current_window), axis=1)
    charge_peak[~np.isfinite(charge_peak) | (charge_peak <= 1.0e-12)] = 1.0
    current_peak[~np.isfinite(current_peak) | (current_peak <= 1.0e-12)] = 1.0
    charge_window /= charge_peak[:, None]
    current_window /= current_peak[:, None]
    return charge_window.astype(np.float32), current_window.astype(np.float32)


def plot_diagnostics(
    output_dir: Path,
    raw: RawPartition,
    fit_events: np.ndarray,
    scores: np.ndarray,
) -> None:
    rng = np.random.default_rng(20260821)
    bin_masks = {
        "low_tail": scores < TAU_LOW,
        "ambiguous": (scores >= TAU_LOW) & (scores <= TAU_HIGH),
        "high_tail": scores > TAU_HIGH,
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    fit_mask = np.zeros(scores.size, dtype=bool)
    fit_mask[fit_events] = True
    x = np.arange(750, dtype=np.float32) * 4.0
    for name, mask in bin_masks.items():
        selected = np.flatnonzero(mask & fit_mask)
        if selected.size > 2000:
            selected = rng.choice(selected, size=2000, replace=False)
        if selected.size == 0:
            continue
        charge, current = diagnostic_windows(raw.waveforms[selected])
        axes[0].plot(x, np.mean(charge, axis=0), label=f"{name} (n={selected.size})")
        axes[1].plot(x, np.mean(current, axis=0), label=name)
    axes[0].set_ylabel("Normalized charge")
    axes[1].set_ylabel("Normalized |current| shape")
    axes[1].set_xlabel("Time from t10 (ns)")
    axes[0].set_title("Stage-1 score-bin average charge profiles (fit events)")
    axes[0].legend()
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "stage1_score_bin_average_profiles.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, color in ((0, "tab:blue"), (1, "tab:orange")):
        selected = np.flatnonzero(fit_mask & (raw.labels == label))
        if selected.size:
            axes[0].hist(scores[selected], bins=80, range=(0, 1), histtype="step", color=color, label=f"label {label}")
    axes[0].axvspan(TAU_LOW, TAU_HIGH, color="grey", alpha=0.2, label="ambiguous")
    axes[0].set_xlabel("Stage-1 score")
    axes[0].set_ylabel("Fit-event count")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    ambiguous = fit_mask & bin_masks["ambiguous"]
    peak_names = sorted(set(raw.peak_ids.tolist()))
    matrix = np.zeros((len(peak_names), 2), dtype=np.int64)
    for row, peak_id in enumerate(peak_names):
        for col, label in enumerate((0, 1)):
            matrix[row, col] = int(np.count_nonzero(ambiguous & (raw.peak_ids == peak_id) & (raw.labels == label)))
    axes[1].bar(np.arange(len(peak_names)) - 0.18, matrix[:, 0], width=0.36, label="label 0")
    axes[1].bar(np.arange(len(peak_names)) + 0.18, matrix[:, 1], width=0.36, label="label 1")
    axes[1].set_xticks(np.arange(len(peak_names)), peak_names, rotation=25, ha="right")
    axes[1].set_ylabel("Fit ambiguous events")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "stage1_ambiguous_counts_and_distribution.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    stage1_checkpoint = args.stage1_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    for path in (labels_dir, event_store_dir, stage1_checkpoint, train_csv, split_path):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, checkpoint, config = load_stage1(stage1_checkpoint, device)
    records = read_event_references(train_csv)
    raw = load_raw_partition(train_csv, event_store_dir)
    if len(records) != raw.labels.size:
        raise ValueError("Reference and event-store record counts differ")
    split = np.load(split_path)
    fit_pairs = np.asarray(split["fit_pair_indices"], dtype=np.int64)
    internal_pairs = np.asarray(split["internal_pair_indices"], dtype=np.int64)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)
    weights = make_event_weights(raw.peak_ids)

    print("building Stage-1 mining representation", flush=True)
    values, representation_qc = build_representation(raw, config)
    from src.ba133_cnn import apply_channel_statistics  # local to keep import grouping clear

    apply_channel_statistics(values, checkpoint["feature_statistics"])
    scores = score_model(model, values, device)
    del values, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.all(np.isfinite(scores)):
        raise ValueError("Stage-1 scores contain nonfinite values")
    ambiguous_mask = (scores >= TAU_LOW) & (scores <= TAU_HIGH)
    fit_ambiguous = np.intersect1d(fit_events, np.flatnonzero(ambiguous_mask))
    internal_ambiguous = np.intersect1d(internal_events, np.flatnonzero(ambiguous_mask))
    print(
        f"fit_events={fit_events.size} internal_events={internal_events.size} "
        f"fit_ambiguous={fit_ambiguous.size} internal_ambiguous={internal_ambiguous.size}",
        flush=True,
    )

    np.savez_compressed(
        output_dir / "stage1_train_scores.npz",
        scores=scores,
        labels=raw.labels,
        peak_ids=raw.peak_ids,
        weights=weights,
        event_indices=np.arange(raw.labels.size, dtype=np.int64),
        fit_event_mask=np.isin(np.arange(raw.labels.size), fit_events),
        internal_event_mask=np.isin(np.arange(raw.labels.size), internal_events),
        ambiguous_mask=ambiguous_mask,
    )
    write_subset(
        output_dir / "ambiguous_fit_events.h5",
        "fit",
        raw,
        weights,
        records,
        fit_ambiguous,
        scores[fit_ambiguous],
        np.arange(raw.labels.size, dtype=np.int64),
    )
    write_subset(
        output_dir / "ambiguous_internal_events.h5",
        "internal",
        raw,
        weights,
        records,
        internal_ambiguous,
        scores[internal_ambiguous],
        np.arange(raw.labels.size, dtype=np.int64),
    )

    rows = make_counts("fit", fit_events, scores[fit_events], raw.labels[fit_events], raw.peak_ids[fit_events])
    rows.extend(
        make_counts(
            "internal",
            internal_events,
            scores[internal_events],
            raw.labels[internal_events],
            raw.peak_ids[internal_events],
        )
    )
    write_counts_csv(output_dir / "ambiguous_counts.csv", rows)
    plot_diagnostics(output_dir, raw, fit_events, scores)
    summary = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "selection_rule": "event-level 0.4 <= Stage-1 score <= 0.6, inclusive",
        "score_thresholds": {"low": TAU_LOW, "high": TAU_HIGH},
        "stage1_checkpoint": stage1_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "stage1_checkpoint_sha256": sha256_file(stage1_checkpoint),
        "stage1_representation_config": config.as_dict(),
        "stage1_feature_statistics": checkpoint["feature_statistics"],
        "stage1_representation_qc": representation_qc,
        "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
        "train_csv_sha256": sha256_file(train_csv),
        "split_path": split_path.relative_to(PROJECT_ROOT).as_posix(),
        "split_sha256": sha256_file(split_path),
        "selected_peak_weights": SELECTED_PEAK_WEIGHTS,
        "partition_counts": {
            "train_pairs": int(raw.labels.size // 2),
            "fit_pairs": int(fit_pairs.size),
            "internal_pairs": int(internal_pairs.size),
            "fit_events": int(fit_events.size),
            "internal_events": int(internal_events.size),
            "fit_ambiguous_events": int(fit_ambiguous.size),
            "internal_ambiguous_events": int(internal_ambiguous.size),
            "fit_ambiguous_fraction": float(fit_ambiguous.size / fit_events.size),
            "internal_ambiguous_fraction": float(internal_ambiguous.size / internal_events.size),
        },
        "artifacts": {
            "stage1_train_scores": "stage1_train_scores.npz",
            "ambiguous_fit_events": "ambiguous_fit_events.h5",
            "ambiguous_internal_events": "ambiguous_internal_events.h5",
            "counts_csv": "ambiguous_counts.csv",
            "average_profiles_png": "stage1_score_bin_average_profiles.png",
            "distribution_png": "stage1_ambiguous_counts_and_distribution.png",
        },
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "target_data_used_for_selection": False,
    }
    save_json(output_dir / "mining_summary.json", summary)
    print(json.dumps(summary["partition_counts"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
