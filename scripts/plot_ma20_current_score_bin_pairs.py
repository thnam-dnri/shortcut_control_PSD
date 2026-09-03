#!/usr/bin/env python3
"""Plot a small smoothed-current diagnostic for one DS-CNN score bin.

This is a visualization-only reprocessing. It keeps the frozen DS-CNN scores
and pair selection from the score-bin HDF5, but derives a new charge/current
view with a configurable causal moving average. Charge and current use the
same per-waveform global-RMS normalization so pulse-shape differences can be
inspected without energy scale.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (  # noqa: E402
    BASELINE_STOP,
    SAMPLE_PERIOD_NS,
    gather_window,
    moving_average,
    t10_anchor,
)
from src.cascade_refinement import sha256_file  # noqa: E402


EXPERIMENT_ID = "joint_3peak_ds_cnn_score_bins_20260821"
DEFAULT_INPUT_HDF5 = (
    PROJECT_ROOT
    / "processed_data/visual_inspection"
    / EXPERIMENT_ID
    / "pairs_by_score_bin.h5"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/visual_inspection"
    / EXPERIMENT_ID
    / "bin_00_ma20_100_pairs"
)
WAVEFORM_LENGTH = 4500
DEFAULT_MA_WIDTH = 20
PRE_SAMPLES = 250
POST_SAMPLES = 500


def decode_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    margin = 0.05 * span if span > 0.0 else max(1.0, abs(low) * 0.05)
    return low - margin, high + margin


def plot_views(
    output_path: Path,
    title: str,
    scores: np.ndarray,
    raw_waveforms: np.ndarray,
    smoothed_charge: np.ndarray,
    normalized_charge: np.ndarray,
    normalized_current: np.ndarray,
    ma_width: int,
    show_scores: bool = True,
) -> None:
    raw_time_us = np.arange(WAVEFORM_LENGTH, dtype=np.float32) * SAMPLE_PERIOD_NS / 1000.0
    aligned_time_us = (
        np.arange(normalized_charge.shape[-1], dtype=np.float32) - PRE_SAMPLES
    ) * SAMPLE_PERIOD_NS / 1000.0
    colors = ("#1b9e77", "#d95f02")
    labels = ("positive label 1", "negative label 0")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    panels = (
        (axes[0], raw_time_us, raw_waveforms, "Raw acquired waveform", "ADC", False),
        (
            axes[1],
            raw_time_us,
            smoothed_charge,
            f"Positive polarity, baseline-subtracted, causal MA{ma_width}",
            f"ADC after MA{ma_width}",
            False,
        ),
        (
            axes[2],
            aligned_time_us,
            normalized_charge,
            f"MA{ma_width} charge, t10-aligned, global-RMS normalized",
            "normalized charge",
            True,
        ),
        (
            axes[3],
            aligned_time_us,
            normalized_current,
            f"MA{ma_width}-derived current, t10-aligned, global-RMS normalized",
            "normalized current",
            True,
        ),
    )
    for axis, time_axis, values, panel_title, ylabel, aligned in panels:
        for member_index, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(
                time_axis,
                values[member_index],
                color=color,
                linewidth=0.85,
                alpha=0.9,
                label=label,
            )
        axis.set_title(panel_title, fontsize=10)
        axis.set_xlabel(
            "time relative to t10 (microseconds)" if aligned else "time (microseconds)"
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.set_ylim(*axis_limits(np.asarray(values)))
    axes[0].legend(loc="best", fontsize=8)
    axes[2].legend(loc="best", fontsize=8)
    title_text = title
    if show_scores:
        title_text += f"\npositive score={scores[0]:.4f}, negative score={scores[1]:.4f}"
    figure.suptitle(title_text, fontsize=12)
    figure.savefig(output_path, dpi=120, facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf5", type=Path, default=DEFAULT_INPUT_HDF5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-index", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument(
        "--ma-width",
        type=int,
        default=DEFAULT_MA_WIDTH,
        help="Causal moving-average width in samples; default preserves the MA20 diagnostic.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bin_index < 0:
        raise ValueError("bin-index must be non-negative")
    if args.pair_count < 1:
        raise ValueError("pair-count must be positive")
    if not 1 <= args.ma_width <= WAVEFORM_LENGTH:
        raise ValueError(f"ma-width must be between 1 and {WAVEFORM_LENGTH}")
    input_hdf5 = args.input_hdf5.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not input_hdf5.is_file():
        raise FileNotFoundError(input_hdf5)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_hdf5, "r") as handle:
        bin_mask = np.asarray(handle["score_bin_index"][:]) == args.bin_index
        available = np.flatnonzero(bin_mask)
        if available.size < args.pair_count:
            raise ValueError(
                f"Only {available.size} pairs are available in score bin index {args.bin_index}"
            )
        selected = available[: args.pair_count]
        raw_waveforms = np.asarray(handle["waveform"][selected], dtype=np.float32)
        scores = np.asarray(handle["score"][selected], dtype=np.float32)
        energies = np.asarray(handle["energy_kev"][selected], dtype=np.float32)
        labels = np.asarray(handle["label"][selected], dtype=np.int8)
        pair_ids = [decode_string(value) for value in handle["pair_id"][selected]]
        peak_ids = [decode_string(value) for value in handle["peak_id"][selected]]
        bin_lows = np.asarray(handle["score_bin_low"][selected], dtype=np.float32)
        bin_highs = np.asarray(handle["score_bin_high"][selected], dtype=np.float32)

    if raw_waveforms.shape != (args.pair_count, 2, WAVEFORM_LENGTH):
        raise ValueError(f"Unexpected raw waveform shape: {raw_waveforms.shape}")
    if not np.all(labels == np.asarray([1, 0], dtype=np.int8)):
        raise ValueError("Pair labels are not ordered as [positive, negative]")
    if not np.allclose(bin_lows, bin_lows[0]) or not np.allclose(bin_highs, bin_highs[0]):
        raise ValueError("Selected rows do not share one score bin")
    bin_low = float(bin_lows[0])
    bin_high = float(bin_highs[0])

    flattened = -raw_waveforms.reshape(-1, WAVEFORM_LENGTH)
    baseline = np.median(flattened[:, :BASELINE_STOP], axis=1).astype(np.float32)
    smoothed_charge = moving_average(
        flattened - baseline[:, None], args.ma_width
    )
    current = np.gradient(smoothed_charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
    anchors, fallback_count = t10_anchor(smoothed_charge)
    charge_window = gather_window(smoothed_charge, anchors, PRE_SAMPLES, POST_SAMPLES)
    current_window = gather_window(current, anchors, PRE_SAMPLES, POST_SAMPLES)
    scale = np.sqrt(np.mean(np.square(charge_window), axis=1, dtype=np.float64)).astype(np.float32)
    invalid_scale = ~np.isfinite(scale) | (scale <= 1.0e-12)
    scale[invalid_scale] = 1.0
    charge_window /= scale[:, None]
    current_window /= scale[:, None]

    pair_count = args.pair_count
    smoothed_charge_pairs = smoothed_charge.reshape(pair_count, 2, WAVEFORM_LENGTH)
    normalized_charge_pairs = charge_window.reshape(pair_count, 2, PRE_SAMPLES + POST_SAMPLES)
    normalized_current_pairs = current_window.reshape(pair_count, 2, PRE_SAMPLES + POST_SAMPLES)
    manifest_rows: list[dict[str, Any]] = []
    for local_index in range(pair_count):
        filename = f"pair_{local_index:04d}_{safe_name(pair_ids[local_index])}.png"
        output_path = output_dir / filename
        plot_views(
            output_path,
            f"Pair {local_index:04d} | {pair_ids[local_index]} | {peak_ids[local_index]} | "
            f"MA{args.ma_width} diagnostic, score bin [{bin_low:.1f},{bin_high:.1f}]",
            scores[local_index],
            raw_waveforms[local_index],
            smoothed_charge_pairs[local_index],
            normalized_charge_pairs[local_index],
            normalized_current_pairs[local_index],
            args.ma_width,
        )
        manifest_rows.append(
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

    average_path = output_dir / f"average_of_{pair_count}_pairs.png"
    plot_views(
        average_path,
        f"Average of {pair_count} pairs | MA{args.ma_width} diagnostic | score bin [{bin_low:.1f},{bin_high:.1f}]",
        np.mean(scores, axis=0),
        np.mean(raw_waveforms, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(smoothed_charge_pairs, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(normalized_charge_pairs, axis=0, dtype=np.float64).astype(np.float32),
        np.mean(normalized_current_pairs, axis=0, dtype=np.float64).astype(np.float32),
        args.ma_width,
        show_scores=False,
    )

    manifest_path = output_dir / "plot_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report_path = output_dir / "plot_report.json"
    report = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": f"PLOTTED_MA{args.ma_width}_CURRENT_DIAGNOSTIC",
        "score_bin_index": args.bin_index,
        "score_bin": [bin_low, bin_high],
        "pair_count": pair_count,
        "input_hdf5": input_hdf5.relative_to(PROJECT_ROOT).as_posix(),
        "input_hdf5_sha256": sha256_file(input_hdf5),
        "frozen_score_checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "frozen_score_checkpoint_sha256": sha256_file(checkpoint_path),
        "moving_average_width": args.ma_width,
        "current_definition": (
            "dQ/dt after baseline subtraction, positive-polarity conversion, "
            f"and causal MA{args.ma_width}"
        ),
        "normalization": "shared per-waveform global RMS of the t10-aligned charge window; no MA10 train z-score applied",
        "t10_fallback_count": int(fallback_count),
        "invalid_scale_count": int(np.count_nonzero(invalid_scale)),
        "raw_waveform": "as acquired; negative polarity retained",
        "artifacts": {
            "plot_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "pair_plot_count": pair_count,
            "aggregate_plot": average_path.relative_to(PROJECT_ROOT).as_posix(),
            "aggregate_plot_sha256": sha256_file(average_path),
            "plot_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "plot_manifest_sha256": sha256_file(manifest_path),
            "plot_count": len(list(output_dir.glob("*.png"))),
        },
        "scientific_boundary": (
            f"Visualization-only MA{args.ma_width} reprocessing; frozen DS-CNN scores and score-bin membership were not changed."
        ),
    }
    save_json(report_path, report)
    print(f"wrote_pair_plots={pair_count}", flush=True)
    print(f"wrote_average={average_path}", flush=True)
    print(f"wrote_report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
