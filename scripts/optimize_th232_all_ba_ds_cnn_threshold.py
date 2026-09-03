#!/usr/bin/env python3
"""Optimize one all-Ba DS-CNN score threshold directly on Th-232 P/B.

Th-232 is explicitly treated as an optimization dataset in this experiment.
The approximately 1592-keV double-escape region is reported as a diagnostic but
is excluded from the cross-energy optimization objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    interval_counts,
    peak_background_metrics,
    th232_admission_mask,
)
from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    apply_channel_statistics,
    build_representation,
    representation_config_from_checkpoint,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/experiments/relaxed_continuum_all_ba_t10_20260823/seed_20260823.pt"
)
DEFAULT_CACHE_MANIFEST = (
    PROJECT_ROOT
    / "processed_data/relaxed_continuum_all_ba_t10_20260823/cache_manifest.json"
)
DEFAULT_TH232_DIR = (
    PROJECT_ROOT / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823"
)
PRIMARY_REFERENCE_PEAKS_KEV = (238.632, 338.320, 583.187, 911.204, 968.971, 2614.511)
EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV = 1588.190
RETENTION_FLOORS = (0.90, 0.80, 0.70, 0.50, 0.30, 0.10, 0.00)
SCORE_BIN_COUNT = 2000
MINIMUM_BACKGROUND_COUNTS = 100.0
MINIMUM_NET_PEAK_COUNTS = 1000.0
EXPECTED_TH232_FILE_COUNT = 30
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_contract(
    checkpoint_path: Path,
    manifest_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, list[float]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("Checkpoint is not a DS-CNN")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Checkpoint is marked as test-contaminated")
    if int(checkpoint.get("seed", -1)) != 20260823:
        raise ValueError("Expected the selected all-Ba seed 20260823 checkpoint")
    if manifest.get("test_partition_used") is not False:
        raise ValueError("Cache manifest has an invalid test boundary")
    config = representation_config_from_checkpoint(manifest["representation_config"])
    if config.name != "both_ma10_global_t10_w750_positive_polarity":
        raise ValueError(f"Unexpected representation: {config.name}")
    if config.channel_count != 2 or config.window_length != 750:
        raise ValueError("Expected an MA10/t10 [2,750] representation")
    model = DSCNN(input_channels=2, width=24).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": int(checkpoint["seed"]),
        "epochs": int(checkpoint["epochs"]),
        "selection": (
            "seed 20260823 had the highest strict common-three macro AUROC and "
            "highest strict Cs-662 AUROC among the three all-Ba seeds"
        ),
        "representation_config": config.as_dict(),
        "feature_statistics": manifest["feature_statistics"],
        "cache_manifest": relative(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
    }
    return model, config, manifest["feature_statistics"], metadata


def create_score_cache(handle: h5py.File, file_count: int, chunk_size: int) -> h5py.Group:
    for name, dtype in (
        ("corrected_energy_kev", np.float32),
        ("score", np.float32),
        ("qc_rejection_bits", np.uint16),
        ("source_file_index", np.uint16),
        ("source_row", np.int64),
        ("event_id", np.uint32),
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


def resize_score_cache(handle: h5py.File, new_size: int) -> None:
    for name in (
        "corrected_energy_kev",
        "score",
        "qc_rejection_bits",
        "source_file_index",
        "source_row",
        "event_id",
    ):
        handle[name].resize((new_size,))


def score_th232(
    files: list[Path],
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    model, config, statistics, model_metadata = load_model_contract(
        checkpoint_path, manifest_path, device
    )
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    counts = {
        "input_events": 0,
        "admitted_events": 0,
        "rejected_energy_or_qc_bits_0_to_2": 0,
        "rejected_nonpositive_shaped_energy": 0,
        "anchor_fallback_count": 0,
        "invalid_scale_count": 0,
    }
    file_records: list[dict[str, Any]] = []
    next_index = 0
    try:
        with h5py.File(partial, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "model_kind": "all_ba_t10_ds_cnn",
                    "checkpoint": relative(checkpoint_path),
                    "checkpoint_sha256": model_metadata["checkpoint_sha256"],
                    "score_definition": "sigmoid of seed-20260823 DS-CNN logit",
                    "energy_dataset": "corrected_energy_kev",
                    "created_utc": utc_now(),
                    "th232_role": "direct_threshold_optimization",
                    "test_partition_used": False,
                }
            )
            source_table = create_score_cache(output, len(files), batch_size)
            for file_index, path in enumerate(files):
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
                        selected_waveforms = np.asarray(
                            source["waveform"][start:stop][admitted], dtype=np.float32
                        )
                        selected_shaped = shaped[admitted]
                        count = int(np.count_nonzero(admitted))
                        raw = RawPartition(
                            waveforms=selected_waveforms,
                            shaped_energy=selected_shaped,
                            labels=np.zeros(count, dtype=np.float32),
                            weights=np.ones(count, dtype=np.float32),
                            peak_ids=np.full(count, "th232", dtype="U8"),
                        )
                        values, qc = build_representation(raw, config)
                        apply_channel_statistics(values, statistics)
                        if not np.all(np.isfinite(values)):
                            raise ValueError("Nonfinite Th-232 model representation")
                        with torch.inference_mode():
                            scores = torch.sigmoid(
                                model(torch.from_numpy(values).to(device, non_blocking=True))
                            ).cpu().numpy().astype(np.float32, copy=False)
                        selected_rows = start + np.flatnonzero(admitted)
                        end_index = next_index + count
                        resize_score_cache(output, end_index)
                        output["corrected_energy_kev"][next_index:end_index] = energy[admitted]
                        output["score"][next_index:end_index] = scores
                        output["qc_rejection_bits"][next_index:end_index] = bits[admitted]
                        output["source_file_index"][next_index:end_index] = file_index
                        output["source_row"][next_index:end_index] = selected_rows
                        output["event_id"][next_index:end_index] = np.asarray(
                            source["event_id"][selected_rows], dtype=np.uint32
                        )
                        next_index = end_index
                        admitted_for_file += count
                        counts["admitted_events"] += count
                        counts["anchor_fallback_count"] += int(qc["anchor_fallback_count"])
                        counts["invalid_scale_count"] += int(qc["invalid_scale_count"])
                digest = sha256_file(path)
                source_table["path"][file_index] = relative(path)
                source_table["sha256"][file_index] = digest
                source_table["input_event_count"][file_index] = event_count
                source_table["admitted_event_count"][file_index] = admitted_for_file
                file_records.append(
                    {
                        "path": relative(path),
                        "sha256": digest,
                        "input_events": event_count,
                        "admitted_events": admitted_for_file,
                    }
                )
                print(
                    f"Th232 file {file_index + 1}/{len(files)} "
                    f"admitted={admitted_for_file} cumulative={next_index}",
                    flush=True,
                )
            output.attrs["event_count"] = next_index
            output.attrs["input_event_count"] = counts["input_events"]
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "score_cache": relative(output_path),
        "score_cache_sha256": sha256_file(output_path),
        "model": model_metadata,
        "counts": counts,
        "files": file_records,
    }


def reliable_metrics(histogram: np.ndarray, window: PeakWindow) -> dict[str, float]:
    try:
        metrics = peak_background_metrics(histogram, window)
    except ZeroDivisionError:
        roi_counts, _ = interval_counts(
            histogram, window.roi_low_kev, window.roi_high_kev
        )
        metrics = {
            "roi_counts": roi_counts,
            "estimated_background_counts": 0.0,
            "net_peak_counts": roi_counts,
            "peak_to_background": float("nan"),
        }
    metrics["statistically_reliable"] = bool(
        metrics["estimated_background_counts"] >= MINIMUM_BACKGROUND_COUNTS
        and metrics["net_peak_counts"] >= MINIMUM_NET_PEAK_COUNTS
        and np.isfinite(metrics["peak_to_background"])
        and metrics["peak_to_background"] > 0.0
    )
    return metrics


def evaluate_threshold_grid(
    energy: np.ndarray,
    scores: np.ndarray,
    windows: list[PeakWindow],
) -> tuple[list[dict[str, Any]], dict[float, np.ndarray], dict[float, dict[str, float]]]:
    score_edges = np.linspace(0.0, 1.0, SCORE_BIN_COUNT + 1, dtype=np.float64)
    histogram_2d, _, _ = np.histogram2d(energy, scores, bins=(ENERGY_EDGES, score_edges))
    cumulative = np.cumsum(histogram_2d[:, ::-1], axis=1)[:, ::-1]
    baseline_histogram = np.histogram(energy, ENERGY_EDGES)[0].astype(np.float64)
    baseline = {
        window.reference_kev: reliable_metrics(baseline_histogram, window)
        for window in windows
    }
    rows: list[dict[str, Any]] = []
    selected_histograms: dict[float, np.ndarray] = {0.0: baseline_histogram}
    for index, threshold in enumerate(score_edges[:-1]):
        histogram = cumulative[:, index]
        per_peak: dict[str, Any] = {}
        gains = []
        retentions = []
        reliable = True
        for window in windows:
            metrics = reliable_metrics(histogram, window)
            reference = window.reference_kev
            base = baseline[reference]
            gain = metrics["peak_to_background"] / base["peak_to_background"]
            retention = metrics["net_peak_counts"] / base["net_peak_counts"]
            per_peak[f"{reference:.3f}"] = {
                **metrics,
                "pb_improvement_factor_vs_no_cut": float(gain),
                "net_peak_retention_vs_no_cut": float(retention),
            }
            if reference in PRIMARY_REFERENCE_PEAKS_KEV:
                reliable = reliable and bool(metrics["statistically_reliable"])
                gains.append(float(gain))
                retentions.append(float(retention))
        valid_gain = bool(np.all(np.isfinite(gains)) and np.all(np.asarray(gains) > 0.0))
        objective = (
            float(np.exp(np.mean(np.log(gains)))) if valid_gain and reliable else float("nan")
        )
        rows.append(
            {
                "threshold": float(threshold),
                "selected_events": int(np.sum(histogram)),
                "selected_fraction": float(np.sum(histogram) / energy.size),
                "geometric_mean_pb_improvement": objective,
                "minimum_primary_peak_retention": float(np.min(retentions)),
                "mean_primary_peak_retention": float(np.mean(retentions)),
                "all_peak_statistics_reliable": reliable,
                "per_peak": per_peak,
            }
        )
    return rows, selected_histograms, baseline


def select_operating_points(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for floor in RETENTION_FLOORS:
        eligible = [
            row
            for row in rows
            if row["all_peak_statistics_reliable"]
            and np.isfinite(row["geometric_mean_pb_improvement"])
            and row["minimum_primary_peak_retention"] >= floor
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda row: row["geometric_mean_pb_improvement"])
        selected[f"minimum_retention_{int(round(100 * floor)):02d}pct"] = best
    return selected


def histogram_for_threshold(
    energy: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    return np.histogram(energy[scores >= threshold], ENERGY_EDGES)[0]


def write_scan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "threshold",
        "selected_events",
        "selected_fraction",
        "geometric_mean_pb_improvement",
        "minimum_primary_peak_retention",
        "mean_primary_peak_retention",
        "all_peak_statistics_reliable",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_operating_point_csv(path: Path, selected: dict[str, dict[str, Any]]) -> None:
    fields = [
        "operating_point",
        "threshold",
        "selected_fraction",
        "geometric_mean_pb_improvement",
        "minimum_primary_peak_retention",
        "mean_primary_peak_retention",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in selected.items():
            writer.writerow({"operating_point": name, **{field: row[field] for field in fields[1:]}})


def plot_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    energy: np.ndarray,
    scores: np.ndarray,
    windows: list[PeakWindow],
) -> None:
    threshold = np.asarray([row["threshold"] for row in rows])
    objective = np.asarray([row["geometric_mean_pb_improvement"] for row in rows])
    minimum_retention = np.asarray([row["minimum_primary_peak_retention"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(threshold, objective, linewidth=1.5)
    axes[0].set_xlabel("DS-CNN score threshold")
    axes[0].set_ylabel("Geometric mean P/B improvement")
    axes[0].grid(alpha=0.25)
    axes[1].plot(threshold, minimum_retention, linewidth=1.5)
    axes[1].set_xlabel("DS-CNN score threshold")
    axes[1].set_ylabel("Minimum net-peak retention")
    axes[1].grid(alpha=0.25)
    for name, row in selected.items():
        if name in {"minimum_retention_80pct", "minimum_retention_00pct"}:
            for axis in axes:
                axis.axvline(row["threshold"], linestyle="--", linewidth=1.0, label=name)
    axes[0].legend(fontsize=8)
    figure.suptitle("Direct Th-232 threshold optimization; 1592-keV region excluded")
    figure.savefig(output_dir / "threshold_optimization.png", dpi=180)
    plt.close(figure)

    recommended = selected["minimum_retention_80pct"]
    best_reliable = selected["minimum_retention_00pct"]
    histograms = {
        "no_cut": np.histogram(energy, ENERGY_EDGES)[0],
        "recommended_80pct_floor": histogram_for_threshold(
            energy, scores, recommended["threshold"]
        ),
        "maximum_reliable_pb": histogram_for_threshold(
            energy, scores, best_reliable["threshold"]
        ),
    }
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    colors = {"no_cut": "black", "recommended_80pct_floor": "#0072B2", "maximum_reliable_pb": "#D55E00"}
    labels = {
        "no_cut": "No cut",
        "recommended_80pct_floor": f"Recommended threshold {recommended['threshold']:.4f}",
        "maximum_reliable_pb": f"Maximum reliable P/B threshold {best_reliable['threshold']:.4f}",
    }
    for name, histogram in histograms.items():
        for axis in axes:
            axis.step(ENERGY_CENTERS, histogram, where="mid", linewidth=0.8, color=colors[name], label=labels[name])
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[0].set_ylabel("Counts / 1 keV")
    axes[1].set_ylabel("Counts / 1 keV")
    axes[0].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axvspan(1575.0, 1605.0, color="#CC79A7", alpha=0.12)
    figure.suptitle("Th-232 spectrum after directly optimized all-Ba DS-CNN cuts")
    figure.savefig(output_dir / "th232_spectrum_optimized_thresholds.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for axis, window in zip(axes.flat, windows):
        half_width = max(18.0, 6.0 * window.sigma_kev)
        mask = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (ENERGY_CENTERS <= window.centroid_kev + half_width)
        for name, histogram in histograms.items():
            axis.step(ENERGY_CENTERS[mask], histogram[mask], where="mid", linewidth=0.8, color=colors[name], label=labels[name])
        excluded = window.reference_kev == EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV
        axis.set_title(f"{window.reference_kev:g} keV" + (" (excluded)" if excluded else ""))
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(windows):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7)
    figure.suptitle("Th-232 peak windows; double-escape region is diagnostic only")
    figure.savefig(output_dir / "th232_peak_zooms_optimized_thresholds.png", dpi=180)
    plt.close(figure)


def make_report_markdown(
    selected: dict[str, dict[str, Any]],
    windows: list[PeakWindow],
) -> str:
    recommended = selected["minimum_retention_80pct"]
    maximum = selected["minimum_retention_00pct"]
    lines = [
        "# Direct Th-232 threshold optimization",
        "",
        "## Simple result",
        "",
        "- Model: all Ba-133 peaks + Na-511 + Cs-662 MA10/t10 DS-CNN, seed 20260823.",
        "- Th-232 is used directly for threshold optimization, so this is not external validation.",
        "- The approximately 1592-keV double-escape region is excluded from the objective.",
        f"- Recommended balanced threshold: `{recommended['threshold']:.4f}`.",
        f"- At that threshold, the cross-energy geometric-mean P/B improvement is `{recommended['geometric_mean_pb_improvement']:.4f}x`.",
        f"- The lowest retained net photopeak fraction is `{recommended['minimum_primary_peak_retention']:.2%}`.",
        f"- Maximum statistically reliable P/B threshold without a retention floor: `{maximum['threshold']:.4f}`.",
        "",
        "## Recommended threshold by peak",
        "",
        "| Energy (keV) | P/B improvement | Net peak retention |",
        "|---:|---:|---:|",
    ]
    for window in windows:
        if window.reference_kev not in PRIMARY_REFERENCE_PEAKS_KEV:
            continue
        row = recommended["per_peak"][f"{window.reference_kev:.3f}"]
        lines.append(
            f"| {window.reference_kev:.3f} | {row['pb_improvement_factor_vs_no_cut']:.4f}x | {row['net_peak_retention_vs_no_cut']:.2%} |"
        )
    excluded = recommended["per_peak"][f"{EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV:.3f}"]
    lines.extend(
        [
            "",
            "## Future observation outside this project",
            "",
            "The feature in the approximately 1592-keV double-escape region is strongly suppressed as the score threshold rises. It was excluded from threshold selection. No mechanism study or physical claim is made here.",
            "",
            f"At the recommended threshold its fitted-window net-count retention is `{excluded['net_peak_retention_vs_no_cut']:.2%}`. This number is diagnostic only because the historical 1588.190-keV reference window overlaps the double-escape region.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--th232-dir", type=Path, default=DEFAULT_TH232_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--expected-files", type=int, default=EXPECTED_TH232_FILE_COUNT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.cache_manifest.resolve()
    th232_dir = args.th232_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(th232_dir.glob("*.h5"))
    if len(files) != args.expected_files:
        raise ValueError(f"Expected {args.expected_files} Th-232 files, found {len(files)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    score_cache = output_dir / "th232_all_ba_ds_cnn_scores.h5"
    existing_outputs = list(output_dir.iterdir())
    if existing_outputs and not args.overwrite and existing_outputs != [score_cache]:
        raise FileExistsError(output_dir)
    print(f"device={device} files={len(files)}", flush=True)
    if score_cache.exists() and not args.overwrite:
        with h5py.File(score_cache, "r") as cache:
            expected_checkpoint_sha = sha256_file(checkpoint_path)
            if str(cache.attrs.get("checkpoint_sha256")) != expected_checkpoint_sha:
                raise ValueError("Existing score cache uses a different checkpoint")
            admitted_events = int(cache.attrs["event_count"])
            input_events = int(cache.attrs["input_event_count"])
        _, _, _, model_metadata = load_model_contract(
            checkpoint_path, manifest_path, device
        )
        scoring = {
            "score_cache": relative(score_cache),
            "score_cache_sha256": sha256_file(score_cache),
            "model": model_metadata,
            "counts": {
                "input_events": input_events,
                "admitted_events": admitted_events,
            },
            "files": [],
            "reused_completed_score_cache": True,
        }
        print(f"reusing_score_cache admitted={admitted_events}", flush=True)
    else:
        if args.overwrite and score_cache.exists():
            score_cache.unlink()
        scoring = score_th232(
            files,
            checkpoint_path,
            manifest_path,
            score_cache,
            args.batch_size,
            device,
        )
    with h5py.File(score_cache, "r") as cache:
        energy = np.asarray(cache["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(cache["score"], dtype=np.float32)
    no_cut_histogram = np.histogram(energy, ENERGY_EDGES)[0]
    windows = fit_peak_windows(no_cut_histogram)
    if tuple(window.reference_kev for window in windows) != REFERENCE_PEAKS_KEV:
        raise ValueError("Unexpected Th-232 peak-window order")
    rows, _, baseline = evaluate_threshold_grid(energy, scores, windows)
    selected = select_operating_points(rows)
    required = {"minimum_retention_80pct", "minimum_retention_00pct"}
    if not required.issubset(selected):
        raise ValueError("No statistically reliable threshold satisfied required operating points")
    scan_csv = output_dir / "threshold_scan.csv"
    operating_csv = output_dir / "operating_points.csv"
    write_scan_csv(scan_csv, rows)
    write_operating_point_csv(operating_csv, selected)
    plot_results(output_dir, rows, selected, energy, scores, windows)
    report_markdown = output_dir / "report.md"
    report_markdown.write_text(make_report_markdown(selected, windows), encoding="utf-8")
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "TH232_DIRECT_THRESHOLD_OPTIMIZATION_COMPLETE",
        "model": scoring["model"],
        "th232_role": "threshold optimization; not external validation",
        "scoring": scoring,
        "admission": {
            "energy_range_kev": [ENERGY_MIN_KEV, ENERGY_MAX_KEV],
            "qc_rule": "reject qc bits 0-2; retain bits 3-4",
            "admitted_events": int(energy.size),
        },
        "optimization": {
            "threshold_kind": "one global DS-CNN score threshold",
            "objective": "geometric mean P/B improvement versus no cut across primary peaks",
            "primary_reference_peaks_kev": list(PRIMARY_REFERENCE_PEAKS_KEV),
            "excluded_reference_region_kev": EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV,
            "excluded_region_reason": "approximately 1592-keV double-escape behavior is outside project scope",
            "score_bin_count": SCORE_BIN_COUNT,
            "statistical_reliability": {
                "minimum_background_counts_per_peak": MINIMUM_BACKGROUND_COUNTS,
                "minimum_net_peak_counts_per_peak": MINIMUM_NET_PEAK_COUNTS,
            },
            "selected_operating_points": selected,
        },
        "peak_windows": [asdict(window) for window in windows],
        "baseline_peak_metrics": {f"{key:.3f}": value for key, value in baseline.items()},
        "future_observation": {
            "region": "approximately 1592-keV Tl-208 double-escape region",
            "status": "record_only_outside_current_project",
            "physical_interpretation_attempted": False,
        },
        "claim_boundary": (
            "The all-Ba model is fixed, but Th-232 is used directly to choose the "
            "threshold. Results are deployment optimization, not external validation."
        ),
        "test_partition_used": False,
        "artifacts": {},
    }
    artifact_paths = [
        score_cache,
        scan_csv,
        operating_csv,
        report_markdown,
        output_dir / "threshold_optimization.png",
        output_dir / "th232_spectrum_optimized_thresholds.png",
        output_dir / "th232_peak_zooms_optimized_thresholds.png",
    ]
    for path in artifact_paths:
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "admitted_events": int(energy.size),
                "recommended": selected["minimum_retention_80pct"],
                "maximum_reliable_pb": selected["minimum_retention_00pct"],
                "report": relative(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
