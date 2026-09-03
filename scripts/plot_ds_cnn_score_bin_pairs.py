#!/usr/bin/env python3
"""Plot one raw matched waveform pair per figure for a DS-CNN score bin.

Each figure contains the original acquired waveforms, the intermediate
positive-polarity/baseline-subtracted/causal-MA10 pulse, and the exact two
channels supplied to the frozen joint 3-peak DS-CNN after t10 alignment,
global normalization, and train-fitted z-score standardization.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    apply_channel_statistics,
    build_representation,
    moving_average,
    representation_config_from_checkpoint,
)
from src.cascade_refinement import sha256_file  # noqa: E402


EXPERIMENT_ID = "joint_3peak_ds_cnn_score_bins_20260821"
DEFAULT_HDF5 = (
    PROJECT_ROOT
    / "processed_data/visual_inspection"
    / EXPERIMENT_ID
    / "pairs_by_score_bin.h5"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/visual_inspection" / EXPERIMENT_ID
SAMPLE_PERIOD_NS = 4.0
BASELINE_STOP = 1000
MOVING_AVERAGE_WIDTH = 10
WAVEFORM_LENGTH = 4500
BIN_COUNT = 6


def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    margin = 0.05 * span if span > 0.0 else max(1.0, abs(low) * 0.05)
    return low - margin, high + margin


def plot_pair(
    output_path: Path,
    pair_index: int,
    pair_id: str,
    peak_id: str,
    scores: np.ndarray,
    energies: np.ndarray,
    raw_waveforms: np.ndarray,
    processed_waveforms: np.ndarray,
    model_values: np.ndarray,
    bin_low: float,
    bin_high: float,
    pre_samples: int,
) -> None:
    raw_time_us = np.arange(WAVEFORM_LENGTH, dtype=np.float32) * SAMPLE_PERIOD_NS / 1000.0
    model_time_us = (
        np.arange(model_values.shape[-1], dtype=np.float32) - float(pre_samples)
    ) * SAMPLE_PERIOD_NS / 1000.0
    positive_color = "#1b9e77"
    negative_color = "#d95f02"
    labels = ("positive label 1", "negative label 0")
    colors = (positive_color, negative_color)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    panels = (
        (axes[0], raw_time_us, raw_waveforms, "Raw acquired waveform", "ADC"),
        (
            axes[1],
            raw_time_us,
            processed_waveforms,
            "Positive polarity, baseline-subtracted, causal MA10",
            "ADC after preprocessing",
        ),
        (
            axes[2],
            model_time_us,
            model_values[:, 0],
            "Exact DS-CNN normalized charge channel",
            "z-score input",
        ),
        (
            axes[3],
            model_time_us,
            model_values[:, 1],
            "Exact DS-CNN normalized current channel",
            "z-score input",
        ),
    )
    for axis, time_axis, values, title, ylabel in panels:
        for member_index, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(
                time_axis,
                values[member_index],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                label=label,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("time (microseconds)" if axis in axes[:2] else "time relative to t10 (microseconds)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.set_ylim(*axis_limits(np.asarray(values)))
    axes[0].legend(loc="best", fontsize=8)
    axes[2].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Pair {pair_index:04d} | {pair_id} | {peak_id}\n"
        f"score bin [{bin_low:.1f}, {bin_high:.1f}] | "
        f"positive score={scores[0]:.4f}, negative score={scores[1]:.4f} | "
        f"energy={energies[0]:.2f}/{energies[1]:.2f} keV",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=120, facecolor="white")
    plt.close(figure)


def plot_average(
    output_path: Path,
    raw_waveforms: np.ndarray,
    processed_waveforms: np.ndarray,
    model_values: np.ndarray,
    bin_low: float,
    bin_high: float,
    pre_samples: int,
) -> None:
    raw_time_us = np.arange(WAVEFORM_LENGTH, dtype=np.float32) * SAMPLE_PERIOD_NS / 1000.0
    model_time_us = (
        np.arange(model_values.shape[-1], dtype=np.float32) - float(pre_samples)
    ) * SAMPLE_PERIOD_NS / 1000.0
    positive_color = "#1b9e77"
    negative_color = "#d95f02"
    labels = ("average positive label 1", "average negative label 0")
    colors = (positive_color, negative_color)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    panels = (
        (
            axes[0],
            raw_time_us,
            raw_waveforms,
            "Average raw acquired waveform (unaligned)",
            "ADC",
        ),
        (
            axes[1],
            raw_time_us,
            processed_waveforms,
            "Average positive polarity, baseline-subtracted, causal MA10",
            "ADC after preprocessing",
        ),
        (
            axes[2],
            model_time_us,
            model_values[:, 0],
            "Average exact DS-CNN normalized charge channel",
            "z-score input",
        ),
        (
            axes[3],
            model_time_us,
            model_values[:, 1],
            "Average exact DS-CNN normalized current channel",
            "z-score input",
        ),
    )
    for axis, time_axis, values, title, ylabel in panels:
        for member_index, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(
                time_axis,
                values[member_index],
                color=color,
                linewidth=1.2,
                alpha=0.95,
                label=label,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(
            "time (microseconds)"
            if axis in axes[:2]
            else "time relative to t10 (microseconds)"
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.set_ylim(*axis_limits(np.asarray(values)))
    axes[0].legend(loc="best", fontsize=8)
    axes[2].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Average of 1,000 matched pairs | score bin [{bin_low:.1f}, {bin_high:.1f}]\n"
        "The raw average is unaligned; the normalized channels are t10-aligned.",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=140, facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bin-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.bin_index < BIN_COUNT:
        raise ValueError(f"bin-index must be in [0, {BIN_COUNT - 1}]")
    input_hdf5 = args.input_hdf5.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / f"bin_{args.bin_index:02d}"
    manifest_path = output_dir / "plot_manifest.csv"
    report_path = output_dir / "plot_report.json"
    for path in (input_hdf5, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_hdf5, "r") as handle:
        required = {
            "waveform",
            "score",
            "energy_kev",
            "pair_id",
            "peak_id",
            "score_bin_index",
            "score_bin_low",
            "score_bin_high",
            "label",
        }
        missing = required - set(handle)
        if missing:
            raise ValueError(f"Input HDF5 lacks datasets: {sorted(missing)}")
        bin_indices = np.asarray(handle["score_bin_index"][:], dtype=np.int8)
        selected = np.flatnonzero(bin_indices == args.bin_index)
        if selected.size != 1000:
            raise ValueError(
                f"Expected 1,000 pairs in bin {args.bin_index}, found {selected.size}"
            )
        raw_waveforms = np.asarray(handle["waveform"][selected], dtype=np.float32)
        scores = np.asarray(handle["score"][selected], dtype=np.float32)
        energies = np.asarray(handle["energy_kev"][selected], dtype=np.float32)
        labels = np.asarray(handle["label"][selected], dtype=np.int8)
        pair_ids = [decode_string(value) for value in handle["pair_id"][selected]]
        peak_ids = [decode_string(value) for value in handle["peak_id"][selected]]
        bin_lows = np.asarray(handle["score_bin_low"][selected], dtype=np.float32)
        bin_highs = np.asarray(handle["score_bin_high"][selected], dtype=np.float32)

    if raw_waveforms.shape != (1000, 2, WAVEFORM_LENGTH):
        raise ValueError(f"Unexpected raw waveform shape: {raw_waveforms.shape}")
    if not np.all(labels == np.asarray([1, 0], dtype=np.int8)):
        raise ValueError("Pair labels are not ordered as [positive, negative]")
    if not np.allclose(bin_lows, bin_lows[0]) or not np.allclose(bin_highs, bin_highs[0]):
        raise ValueError("Selected rows do not share one score bin")
    bin_low = float(bin_lows[0])
    bin_high = float(bin_highs[0])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("The checkpoint is not a DS-CNN checkpoint")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.input_mode != "both" or config.anchor != "t10":
        raise ValueError(f"Unexpected DS-CNN input contract: {config.as_dict()}")
    flattened_waveforms = raw_waveforms.reshape(-1, WAVEFORM_LENGTH)
    flattened_labels = labels.reshape(-1).astype(np.float32)
    flattened_peaks = np.repeat(np.asarray(peak_ids, dtype="U64"), 2)
    raw = RawPartition(
        waveforms=flattened_waveforms,
        shaped_energy=np.ones(flattened_waveforms.shape[0], dtype=np.float32),
        labels=flattened_labels,
        weights=np.ones(flattened_waveforms.shape[0], dtype=np.float32),
        peak_ids=flattened_peaks,
    )
    model_values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(model_values, checkpoint["feature_statistics"])
    if not np.all(np.isfinite(model_values)):
        raise ValueError("Normalized DS-CNN representation contains nonfinite values")

    positive_polarity = -flattened_waveforms
    baseline = np.median(positive_polarity[:, :BASELINE_STOP], axis=1).astype(np.float32)
    processed = moving_average(positive_polarity - baseline[:, None], MOVING_AVERAGE_WIDTH)
    processed_pairs = processed.reshape(1000, 2, WAVEFORM_LENGTH)
    model_pairs = model_values.reshape(1000, 2, config.channel_count, config.window_length)

    plot_rows: list[dict[str, Any]] = []
    print(f"plotting {selected.size} pairs to {output_dir}", flush=True)
    for local_index in range(selected.size):
        filename = f"pair_{local_index:04d}_{safe_name(pair_ids[local_index])}.png"
        plot_path = output_dir / filename
        plot_pair(
            plot_path,
            local_index,
            pair_ids[local_index],
            peak_ids[local_index],
            scores[local_index],
            energies[local_index],
            raw_waveforms[local_index],
            processed_pairs[local_index],
            model_pairs[local_index],
            bin_low,
            bin_high,
            config.pre_samples,
        )
        plot_rows.append(
            {
                "plot_index": local_index,
                "filename": filename,
                "source_hdf5_pair_index": int(selected[local_index]),
                "pair_id": pair_ids[local_index],
                "peak_id": peak_ids[local_index],
                "positive_score": f"{scores[local_index, 0]:.9g}",
                "negative_score": f"{scores[local_index, 1]:.9g}",
                "positive_energy_kev": f"{energies[local_index, 0]:.9g}",
                "negative_energy_kev": f"{energies[local_index, 1]:.9g}",
            }
        )

    average_plot_path = output_dir / "average_of_1000_pairs.png"
    plot_average(
        average_plot_path,
        np.mean(raw_waveforms, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(processed_pairs, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(model_pairs, axis=0, dtype=np.float64).astype(np.float32),
        bin_low,
        bin_high,
        config.pre_samples,
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    report = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PLOTTED_RAW_AND_EXACT_DS_CNN_NORMALIZED_PAIRS",
        "score_bin_index": args.bin_index,
        "score_bin": [bin_low, bin_high],
        "pair_count": int(selected.size),
        "input_hdf5": input_hdf5.relative_to(PROJECT_ROOT).as_posix(),
        "input_hdf5_sha256": sha256_file(input_hdf5),
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "representation_config": config.as_dict(),
        "feature_statistics": checkpoint["feature_statistics"],
        "representation_qc": representation_qc,
        "raw_plot": {
            "polarity": "as acquired; negative pulse polarity retained",
            "sample_period_ns": SAMPLE_PERIOD_NS,
            "waveform_length": WAVEFORM_LENGTH,
        },
        "normalized_plot": (
            "Exact frozen joint DS-CNN input: negative acquired polarity converted to positive, "
            "median baseline subtraction, causal MA10, t10 alignment, global normalization, and train z-score."
        ),
        "artifacts": {
            "plot_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "plot_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "pair_plot_count": len(plot_rows),
            "aggregate_plot": average_plot_path.relative_to(PROJECT_ROOT).as_posix(),
            "aggregate_plot_sha256": sha256_file(average_plot_path),
            "plot_count": len(list(output_dir.glob("*.png"))),
            "plot_manifest_sha256": sha256_file(manifest_path),
        },
        "scientific_boundary": (
            "These are visual-inspection figures from the development-training sample; they are not a new "
            "model or threshold selection result."
        ),
    }
    save_json(report_path, report)
    print(f"wrote_manifest={manifest_path}", flush=True)
    print(f"wrote_report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
