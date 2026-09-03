#!/usr/bin/env python3
"""Plot exact normalized Compact CNN inputs for three positive training peaks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    relative,
    sha256_file,
    utc_now,
)
from scripts.plot_o2_3p_normalized_training_waveforms import (  # noqa: E402
    PEAKS,
    resolve_store_indices,
    select_positive_rows,
)
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    RepresentationConfig,
    SAMPLE_PERIOD_NS,
    apply_channel_statistics,
    build_representation,
    representation_config_from_checkpoint,
)


def load_compact_features(selected, lookup, store_path: Path, config, statistics):
    records = []
    for peak_id, rows in selected.items():
        for row in rows:
            key = (row["positive_hdf5"], int(row["positive_row"]))
            records.append((peak_id, row, lookup[key]))
    store_indices = np.asarray([record[2] for record in records], dtype=np.int64)
    order = np.argsort(store_indices)
    with h5py.File(store_path, "r") as store:
        if str(store.attrs["partition"]) != "train":
            raise ValueError("Expected the train event store")
        waveforms_sorted = np.asarray(store["waveform"][store_indices[order]], dtype=np.float32)
        shaped_sorted = np.asarray(store["shaped_energy_unit"][store_indices[order]], dtype=np.float32)
    inverse = np.argsort(order)
    raw = RawPartition(
        waveforms=waveforms_sorted[inverse],
        shaped_energy=shaped_sorted[inverse],
        labels=np.ones(store_indices.size, dtype=np.float32),
        weights=np.ones(store_indices.size, dtype=np.float32),
        peak_ids=np.asarray([record[0] for record in records], dtype="U64"),
    )
    values, qc = build_representation(raw, config)
    if config.standardization in {"train_zscore", "fixed_current_peak_scale"}:
        apply_channel_statistics(values, statistics)
    elif config.standardization != "none":
        raise ValueError(f"Unknown standardization: {config.standardization}")
    features = {}
    start = 0
    for peak_id, rows in selected.items():
        stop = start + len(rows)
        features[peak_id] = {"charge": values[start:stop, 0], "current": values[start:stop, 1]}
        for row, store_index in zip(rows, store_indices[start:stop]):
            row["store_index"] = int(store_index)
        start = stop
    return features, qc


def plot_overlays(output_path: Path, features, config):
    time = (np.arange(config.window_length) - config.pre_samples) * SAMPLE_PERIOD_NS
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey="col")
    for row_index, (peak_id, label, color) in enumerate(PEAKS):
        for column, branch in enumerate(("charge", "current")):
            axis = axes[row_index, column]
            values = features[peak_id][branch]
            axis.plot(time, values.T, color=color, alpha=0.10, linewidth=0.55)
            axis.plot(time, np.mean(values, axis=0), color="black", linewidth=1.8)
            axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
            axis.set_title(f"{label}: normalized {branch} (100 events)")
            axis.set_ylabel("Model input")
            axis.grid(alpha=0.18)
            anchor_label = (
                "current-peak anchor"
                if branch == "current" and config.anchor == "dual_t10_current_peak"
                else "t10 anchor"
            )
            axis.text(0.99, 0.04, anchor_label, transform=axis.transAxes, ha="right", fontsize=9)
    axes[-1, 0].set_xlabel("Time relative to charge t10 (ns)")
    axes[-1, 1].set_xlabel(
        "Time relative to current peak (ns)"
        if config.anchor == "dual_t10_current_peak"
        else "Time relative to charge t10 (ns)"
    )
    figure.suptitle("Compact CNN normalized positive training inputs", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_heatmaps(output_path: Path, features, config):
    time = (np.arange(config.window_length) - config.pre_samples) * SAMPLE_PERIOD_NS
    limits = {}
    for branch in ("charge", "current"):
        combined = np.concatenate([features[peak_id][branch].ravel() for peak_id, _label, _color in PEAKS])
        limits[branch] = tuple(np.percentile(combined, (1.0, 99.0)))
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey="row")
    for row_index, (peak_id, label, _color) in enumerate(PEAKS):
        for column, branch in enumerate(("charge", "current")):
            axis = axes[row_index, column]
            image = axis.imshow(
                features[peak_id][branch],
                aspect="auto",
                origin="lower",
                extent=(time[0], time[-1], 0.5, features[peak_id][branch].shape[0] + 0.5),
                cmap="coolwarm",
                vmin=limits[branch][0],
                vmax=limits[branch][1],
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
            axis.set_title(f"{label}: normalized {branch}")
            axis.set_ylabel("Selected training event")
            figure.colorbar(image, ax=axis, pad=0.01, label="Model input")
    axes[-1, 0].set_xlabel("Time relative to charge t10 (ns)")
    axes[-1, 1].set_xlabel(
        "Time relative to current peak (ns)"
        if config.anchor == "dual_t10_current_peak"
        else "Time relative to charge t10 (ns)"
    )
    figure.suptitle("All 100 Compact CNN training inputs per positive peak", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_weight_scan_20260819/label_pairs_train.csv",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/three_peak_weight_scan_20260819/compact_cnn_best.pt",
    )
    parser.add_argument("--samples-per-peak", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.samples_per_peak <= 0:
        raise ValueError("samples-per-peak must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_csv = args.label_csv.resolve()
    checkpoint_path = args.checkpoint.resolve()
    store_dir = args.event_store_dir.resolve()
    store_path = store_dir / "train_events.h5"
    lookup_path = store_dir / "event_lookup_train.csv"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "compact_cnn" or checkpoint.get("test_partition_used") is not False:
        raise ValueError("Unexpected or test-contaminated checkpoint")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.input_mode != "both" or config.channel_count != 2:
        raise ValueError("Expected the frozen two-channel compact representation")
    statistics = checkpoint["feature_statistics"]
    selected = select_positive_rows(label_csv, args.samples_per_peak, args.seed)
    lookup = resolve_store_indices(selected, lookup_path)
    features, qc = load_compact_features(selected, lookup, store_path, config, statistics)

    overlay_path = output_dir / "normalized_training_inputs_overlay.png"
    heatmap_path = output_dir / "normalized_training_inputs_heatmap.png"
    npz_path = output_dir / "normalized_training_inputs.npz"
    selection_path = output_dir / "selected_training_events.csv"
    plot_overlays(overlay_path, features, config)
    plot_heatmaps(heatmap_path, features, config)
    np.savez_compressed(
        npz_path,
        **{
            f"{peak_id}_{branch}": features[peak_id][branch]
            for peak_id, _label, _color in PEAKS
            for branch in ("charge", "current")
        },
    )
    selection_rows = []
    for peak_id, rows in selected.items():
        for sample_index, row in enumerate(rows):
            selection_rows.append(
                {
                    "peak_id": peak_id,
                    "sample_index": sample_index,
                    "pair_id": row["pair_id"],
                    "positive_source": row["positive_source"],
                    "positive_event_id": row["positive_event_id"],
                    "positive_energy_kev": row["positive_energy_kev"],
                    "positive_hdf5": row["positive_hdf5"],
                    "positive_row": row["positive_row"],
                    "store_index": row["store_index"],
                    "positive_qc_status": row["positive_qc_status"],
                }
            )
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)

    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "model_name": "Three-peak Compact CNN",
        "partition": "train",
        "selection": {
            "positive_label_only": True,
            "samples_per_peak": args.samples_per_peak,
            "seed": args.seed,
            "without_replacement": True,
            "peak_ids": [peak_id for peak_id, _label, _color in PEAKS],
        },
        "representation_config": checkpoint["representation_config"],
        "feature_statistics": statistics,
        "representation_steps": (
            "negative-to-positive polarity; baseline subtraction; causal MA10; "
            "charge divided by positive peak, clipped to [0,1], and anchored at t10; "
            "current=d(smoothed charge)/dt, divided by its positive peak, and independently peak-anchored; "
            "inclusive +/-1 microsecond windows; no channel z-score"
            if config.anchor == "dual_t10_current_peak"
            else (
                "negative-to-positive polarity; baseline subtraction; causal MA10; "
                "charge divided by its positive peak without clipping; "
                "current=d(normalized charge)/dt and divided by a fixed fit-only current scale; "
                "both channels share an inclusive t10 +/-1 microsecond window"
                if config.normalization == "charge_peak_shared"
                else (
                    "negative-to-positive polarity; baseline subtraction; raw charge (MA1); "
                    "current=d(raw charge)/dt; shared inclusive t10 +/-1 microsecond windows; "
                    "charge-window RMS normalization followed by fit-only channel z-scores"
                    if config.moving_average == 1 and config.standardization == "train_zscore"
                    else "checkpoint-defined compact CNN preprocessing"
                )
            )
        ),
        "qc": qc,
        "inputs": {
            "label_csv": {"path": relative(label_csv), "sha256": sha256_file(label_csv)},
            "checkpoint": {"path": relative(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "event_store": {"path": relative(store_path), "sha256": sha256_file(store_path)},
            "event_lookup": {"path": relative(lookup_path), "sha256": sha256_file(lookup_path)},
        },
        "feature_ranges": {
            peak_id: {
                branch: {
                    "minimum": float(np.min(features[peak_id][branch])),
                    "maximum": float(np.max(features[peak_id][branch])),
                    "mean": float(np.mean(features[peak_id][branch])),
                    "standard_deviation": float(np.std(features[peak_id][branch])),
                }
                for branch in ("charge", "current")
            }
            for peak_id, _label, _color in PEAKS
        },
        "artifacts": {},
        "test_partition_used": False,
    }
    for path in (overlay_path, heatmap_path, npz_path, selection_path):
        report["artifacts"][path.name] = {"path": relative(path), "sha256": sha256_file(path)}
    report_path = output_dir / "normalized_training_inputs_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
