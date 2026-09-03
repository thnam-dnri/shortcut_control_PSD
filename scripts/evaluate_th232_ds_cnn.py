#!/usr/bin/env python3
"""Apply the frozen corrected-input DS-CNN to the external Th-232 cache.

The requested percentages are positive-retention operating points. Their score
cuts are calibrated from the already-frozen development-positive score
artifact, before Th-232 scores are inspected. Th-232 is used only for the
transfer spectrum and peak-to-background report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit

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
    apply_channel_statistics,
    build_representation,
    representation_config_from_checkpoint,
)

MODEL_DIR = PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn"
DEFAULT_CHECKPOINT = MODEL_DIR / "ds_cnn_best.pt"
DEFAULT_HELD_OUT_SCORES = (
    PROJECT_ROOT
    / "outputs/models/compact_ds_cnn_performance_20260820/held_out_evaluation/held_out_scores.npz"
)
DEFAULT_HELD_OUT_REPORT = (
    PROJECT_ROOT
    / "outputs/models/compact_ds_cnn_performance_20260820/held_out_evaluation/held_out_evaluation.json"
)
DEFAULT_TH232_DIR = (
    PROJECT_ROOT / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/th232_ds_cnn_evaluation_20260820"

ACCEPTANCES = (0.99, 0.95, 0.90, 0.80, 0.50, 0.30, 0.10)
EXPECTED_TH232_FILE_COUNT = 30
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True)
class ThresholdInfo:
    name: str
    requested_weighted_acceptance: float
    score_threshold: float
    actual_weighted_acceptance: float
    actual_unweighted_acceptance: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def threshold_name(acceptance: float) -> str:
    return f"{int(round(100.0 * acceptance))}pct"


def weighted_acceptance_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    acceptance: float,
) -> ThresholdInfo:
    if scores.ndim != 1 or weights.ndim != 1 or scores.size != weights.size:
        raise ValueError("Scores and weights must be matching one-dimensional arrays")
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Positive calibration scores must be finite and nonempty")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Positive calibration weights must be finite and positive")
    if not 0.0 < acceptance <= 1.0:
        raise ValueError(f"Invalid acceptance: {acceptance}")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    target_weight = acceptance * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), target_weight, side="left"))
    threshold = float(sorted_scores[min(index, sorted_scores.size - 1)])
    accepted = scores >= threshold
    return ThresholdInfo(
        name=threshold_name(acceptance),
        requested_weighted_acceptance=float(acceptance),
        score_threshold=threshold,
        actual_weighted_acceptance=float(np.sum(weights[accepted]) / np.sum(weights)),
        actual_unweighted_acceptance=float(np.mean(accepted)),
    )


def load_thresholds(
    held_out_scores_path: Path,
    held_out_report_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    report = json.loads(held_out_report_path.read_text(encoding="utf-8"))
    model_report = report.get("models", {}).get("ds_cnn", {})
    if model_report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Held-out score report does not match the frozen DS-CNN checkpoint")
    if report.get("test_partition_used") is not False:
        raise ValueError("Held-out score report has an invalid test boundary")

    with np.load(held_out_scores_path, allow_pickle=False) as values:
        labels = np.asarray(values["labels"], dtype=np.float32)
        peak_ids = np.asarray(values["peak_ids"])
        weights = np.asarray(values["weights"], dtype=np.float64)
        scores = np.asarray(values["ds_cnn_scores"], dtype=np.float64)
    if not (labels.shape == peak_ids.shape == weights.shape == scores.shape):
        raise ValueError("Held-out score arrays have inconsistent shapes")
    positive = labels == 1.0
    if not np.all(np.isin(labels, [0.0, 1.0])) or not np.any(positive):
        raise ValueError("Invalid held-out labels")
    threshold_values = {
        threshold_name(acceptance): asdict(
            weighted_acceptance_threshold(
                scores[positive],
                weights[positive],
                acceptance,
            )
        )
        for acceptance in ACCEPTANCES
    }
    positive_peak_counts = {
        str(peak): int(np.count_nonzero(positive & (peak_ids == peak)))
        for peak in sorted({str(value) for value in peak_ids[positive]})
    }
    return {
        "source_scores_path": relative(held_out_scores_path),
        "source_scores_sha256": sha256_file(held_out_scores_path),
        "source_report_path": relative(held_out_report_path),
        "source_report_sha256": sha256_file(held_out_report_path),
        "source_partition": report.get("partition"),
        "held_out_partition_already_consumed_for_model_comparison": True,
        "positive_event_count": int(np.count_nonzero(positive)),
        "positive_peak_counts": positive_peak_counts,
        "thresholds": threshold_values,
        "selection_rule": (
            "weighted positive-event score retention; fixed before Th-232 scoring; "
            "no Th-232 P/B or spectrum information used"
        ),
    }


def load_frozen_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Any, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("The requested checkpoint is not a DS-CNN checkpoint")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("DS-CNN checkpoint is marked as test-contaminated")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError("DS-CNN checkpoint was trained with held-out data")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.name != "both_ma10_global_t10_w750_positive_polarity":
        raise ValueError(f"Unexpected DS-CNN representation: {config.name}")
    if config.channel_count != 2 or config.window_length != 750:
        raise ValueError("Frozen DS-CNN input contract is not [2, 750]")
    model_width = int(checkpoint.get("model_width", 24))
    model = DSCNN(input_channels=config.channel_count, width=model_width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    digest = sha256_file(checkpoint_path)
    metadata = {
        "model_kind": checkpoint["model_kind"],
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": digest,
        "parameter_count": int(checkpoint["parameter_count"]),
        "model_width": model_width,
        "representation_config": config.as_dict(),
        "feature_statistics": checkpoint["feature_statistics"],
        "selected_peak_weights": checkpoint["selected_peak_weights"],
        "scan_best_epoch": int(checkpoint["scan_best_epoch"]),
        "refit_epochs": int(checkpoint["refit_epochs"]),
        "internal_selection_metrics": checkpoint["scan_best_internal_metrics"],
        "test_partition_used": checkpoint["test_partition_used"],
        "held_out_partition_loaded": checkpoint["held_out_partition_loaded"],
    }
    return model, metadata, config, digest


def score_representation_batch(
    waveforms: np.ndarray,
    shaped_energy: np.ndarray,
    config: Any,
    feature_statistics: dict[str, list[float]],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, int]]:
    count = waveforms.shape[0]
    raw = RawPartition(
        waveforms=waveforms,
        shaped_energy=shaped_energy,
        labels=np.zeros(count, dtype=np.float32),
        weights=np.ones(count, dtype=np.float32),
        peak_ids=np.full(count, "th232", dtype="U16"),
    )
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, feature_statistics)
    if not np.all(np.isfinite(values)):
        raise ValueError("Th-232 representation contains nonfinite values")
    with torch.inference_mode():
        logits = model(torch.from_numpy(values).to(device, non_blocking=True))
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("DS-CNN produced nonfinite Th-232 scores")
    return scores, representation_qc


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


def score_th232(
    files: list[Path],
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    model, metadata, config, _ = load_frozen_model(checkpoint_path, device)
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    counts: dict[str, int] = {
        "input_events": 0,
        "admitted_events": 0,
        "rejected_energy_or_qc_bits_0_to_2": 0,
        "rejected_nonpositive_shaped_energy": 0,
        "admitted_with_noise_bit3": 0,
        "admitted_with_pulse_bit4": 0,
        "anchor_fallback_count": 0,
        "invalid_scale_count": 0,
        "charge_clipped_sample_count": 0,
        "charge_clipped_event_count": 0,
    }
    file_records: list[dict[str, Any]] = []
    next_index = 0
    try:
        with h5py.File(partial, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "model_kind": "ds_cnn",
                    "checkpoint": relative(checkpoint_path),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "energy_dataset": "corrected_energy_kev",
                    "score_definition": "sigmoid of DS-CNN logit",
                    "representation_config": json.dumps(
                        config.as_dict(), sort_keys=True
                    ),
                    "created_utc": utc_now(),
                    "test_partition_used": False,
                    "retention_policy": "all source events remain in input HDF5; admitted score sidecar preserves QC bits and source row references",
                }
            )
            source_table = create_score_cache(output, len(files), batch_size)
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
                        waveforms = np.asarray(
                            source["waveform"][start:stop][admitted], dtype=np.float32
                        )
                        scores, representation_qc = score_representation_batch(
                            waveforms,
                            shaped[admitted],
                            config,
                            metadata["feature_statistics"],
                            model,
                            device,
                        )
                        selected_rows = start + np.flatnonzero(admitted)
                        selected_bits = bits[admitted]
                        selected_event_ids = np.asarray(
                            source["event_id"][selected_rows], dtype=np.uint32
                        )
                        count = int(selected_rows.size)
                        end_index = next_index + count
                        resize_score_cache(output, end_index)
                        output["corrected_energy_kev"][next_index:end_index] = energy[
                            admitted
                        ]
                        output["score"][next_index:end_index] = scores
                        output["qc_rejection_bits"][next_index:end_index] = selected_bits
                        output["source_file_index"][next_index:end_index] = file_index
                        output["source_row"][next_index:end_index] = selected_rows
                        output["event_id"][next_index:end_index] = selected_event_ids
                        next_index = end_index
                        admitted_for_file += count
                        counts["admitted_events"] += count
                        counts["admitted_with_noise_bit3"] += int(
                            np.count_nonzero(selected_bits & np.uint16(1 << 3))
                        )
                        counts["admitted_with_pulse_bit4"] += int(
                            np.count_nonzero(selected_bits & np.uint16(1 << 4))
                        )
                        for key in (
                            "anchor_fallback_count",
                            "invalid_scale_count",
                            "charge_clipped_sample_count",
                            "charge_clipped_event_count",
                        ):
                            counts[key] += int(representation_qc[key])
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
            output.attrs["anchor_fallback_count"] = counts["anchor_fallback_count"]
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "score_cache": relative(output_path),
        "score_cache_sha256": sha256_file(output_path),
        "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "counts": counts,
        "files": file_records,
    }


def make_histograms(
    energy: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks: dict[str, np.ndarray] = {"no_cut": np.ones(energy.size, dtype=bool)}
    histograms: dict[str, np.ndarray] = {
        "no_cut": np.histogram(energy, ENERGY_EDGES)[0]
    }
    for acceptance in ACCEPTANCES:
        name = threshold_name(acceptance)
        mask = scores >= float(thresholds["thresholds"][name]["score_threshold"])
        masks[name] = mask
        histograms[name] = np.histogram(energy[mask], ENERGY_EDGES)[0]
    return histograms, masks


def write_spectrum_csv(path: Path, histograms: dict[str, np.ndarray]) -> None:
    names = ["no_cut"] + [threshold_name(value) for value in ACCEPTANCES]
    columns = ["energy_kev_bin_center"] + [f"ds_cnn_{name}" for name in names]
    arrays = [ENERGY_CENTERS] + [histograms[name] for name in names]
    formats = ["%.1f"] + ["%d"] * len(names)
    np.savetxt(
        path,
        np.column_stack(arrays),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt=formats,
    )


def make_peak_rows(
    histograms: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    thresholds: dict[str, Any],
    windows: list[PeakWindow],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        baseline = {
            key: float(value)
            for key, value in peak_background_metrics(histograms["no_cut"], window).items()
        }
        rows.append(
            {
                "reference_energy_kev": float(window.reference_kev),
                "observed_centroid_kev": float(window.centroid_kev),
                "model": "ds_cnn",
                "acceptance": "none",
                "score_threshold": "",
                "events": int(np.count_nonzero(masks["no_cut"])),
                **baseline,
                "pb_improvement_factor_vs_no_cut": 1.0,
                "net_peak_retention_vs_no_cut": 1.0,
            }
        )
        for acceptance in ACCEPTANCES:
            name = threshold_name(acceptance)
            metrics = {
                key: float(value)
                for key, value in peak_background_metrics(histograms[name], window).items()
            }
            rows.append(
                {
                    "reference_energy_kev": float(window.reference_kev),
                    "observed_centroid_kev": float(window.centroid_kev),
                    "model": "ds_cnn",
                    "acceptance": name,
                    "score_threshold": float(
                        thresholds["thresholds"][name]["score_threshold"]
                    ),
                    "requested_weighted_acceptance": float(
                        thresholds["thresholds"][name][
                            "requested_weighted_acceptance"
                        ]
                    ),
                    "actual_weighted_acceptance": float(
                        thresholds["thresholds"][name]["actual_weighted_acceptance"]
                    ),
                    "events": int(np.count_nonzero(masks[name])),
                    **metrics,
                    "pb_improvement_factor_vs_no_cut": float(
                        metrics["peak_to_background"]
                        / baseline["peak_to_background"]
                    ),
                    "net_peak_retention_vs_no_cut": float(
                        metrics["net_peak_counts"] / baseline["net_peak_counts"]
                    ),
                }
            )
    return rows


def write_peak_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_spectra(
    output_dir: Path,
    histograms: dict[str, np.ndarray],
    windows: list[PeakWindow],
) -> None:
    names = ["no_cut"] + [threshold_name(value) for value in ACCEPTANCES]
    colors = {"no_cut": "black"}
    palette = plt.cm.viridis(np.linspace(0.08, 0.94, len(ACCEPTANCES)))
    colors.update({name: color for name, color in zip(names[1:], palette)})
    labels = {"no_cut": "No cut"}
    labels.update({name: f"{name[:-3]}% positive retention" for name in names[1:]})

    figure, axes = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True, constrained_layout=True
    )
    for name in names:
        for index, axis in enumerate(axes):
            axis.step(
                ENERGY_CENTERS,
                histograms[name],
                where="mid",
                linewidth=0.85,
                color=colors[name],
                label=labels[name] if index == 0 else "_nolegend_",
            )
    axes[0].set_ylabel("Counts / 1 keV")
    axes[1].set_ylabel("Counts / 1 keV")
    axes[1].set_xlabel("Corrected energy (keV; preliminary calibration)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[0].set_title("Th-232 spectrum after frozen DS-CNN score cuts")
    axes[0].legend(ncol=4, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
        for window in windows:
            axis.axvline(window.centroid_kev, color="0.6", linewidth=0.45, alpha=0.5)
    figure.savefig(output_dir / "th232_ds_cnn_energy_spectra.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    flat_axes = list(axes.flat)
    for axis, window in zip(flat_axes, windows):
        half_width = max(18.0, 6.0 * window.sigma_kev)
        selected = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (
            ENERGY_CENTERS <= window.centroid_kev + half_width
        )
        for name in names:
            axis.step(
                ENERGY_CENTERS[selected],
                histograms[name][selected],
                where="mid",
                linewidth=0.8,
                color=colors[name],
                label=labels[name],
            )
        axis.axvspan(window.roi_low_kev, window.roi_high_kev, color="#984ea3", alpha=0.10)
        axis.axvspan(window.left_low_kev, window.left_high_kev, color="0.5", alpha=0.08)
        axis.axvspan(window.right_low_kev, window.right_high_kev, color="0.5", alpha=0.08)
        axis.set_title(
            f"{window.reference_kev:g} keV ref.\nobserved {window.centroid_kev:.2f} keV"
        )
        axis.grid(alpha=0.2)
    for axis in flat_axes[len(windows) :]:
        axis.axis("off")
    flat_axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle("Th-232 peak windows after frozen DS-CNN score cuts")
    figure.supxlabel("Corrected energy (keV)")
    figure.supylabel("Counts / 1 keV")
    figure.savefig(output_dir / "th232_ds_cnn_peak_zooms.png", dpi=180)
    plt.close(figure)


def plot_peak_to_background(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    references = [float(value) for value in REFERENCE_PEAKS_KEV]
    by_acceptance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_acceptance.setdefault(str(row["acceptance"]), []).append(row)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    names = ["none"] + [threshold_name(value) for value in ACCEPTANCES]
    colors = ["black", *plt.cm.viridis(np.linspace(0.08, 0.94, len(ACCEPTANCES)))]
    for name, color in zip(names, colors):
        values = by_acceptance[name]
        axis.plot(
            references,
            [float(row["peak_to_background"]) for row in values],
            marker="o",
            linewidth=1.2,
            color=color,
            label="No cut" if name == "none" else f"{name[:-3]}% positive retention",
        )
    axis.set_xlabel("Reference peak energy (keV)")
    axis.set_ylabel("Peak / background")
    axis.set_title("Th-232 peak-to-background ratio after frozen DS-CNN cuts")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    figure.savefig(output_dir / "th232_ds_cnn_peak_to_background.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--held-out-scores", type=Path, default=DEFAULT_HELD_OUT_SCORES)
    parser.add_argument("--held-out-report", type=Path, default=DEFAULT_HELD_OUT_REPORT)
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
    held_out_scores_path = args.held_out_scores.resolve()
    held_out_report_path = args.held_out_report.resolve()
    th232_dir = args.th232_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not held_out_scores_path.is_file() or not held_out_report_path.is_file():
        raise FileNotFoundError("Missing held-out DS-CNN score/report artifact")
    files = sorted(th232_dir.glob("*.h5"))
    if len(files) != args.expected_files:
        raise ValueError(
            f"Expected {args.expected_files} corrected Th-232 files, found {len(files)}"
        )

    _, frozen_metadata, _, checkpoint_sha = load_frozen_model(checkpoint_path, device)
    thresholds = load_thresholds(
        held_out_scores_path,
        held_out_report_path,
        checkpoint_sha,
    )
    print(f"device={device}", flush=True)
    print(f"frozen_checkpoint={relative(checkpoint_path)}", flush=True)
    print(f"th232_files={len(files)}", flush=True)
    scoring = score_th232(
        files,
        checkpoint_path,
        output_dir / "th232_ds_cnn_scores.h5",
        args.batch_size,
        device,
    )
    with h5py.File(output_dir / "th232_ds_cnn_scores.h5", "r") as cache:
        energy = np.asarray(cache["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(cache["score"], dtype=np.float32)
    histograms, masks = make_histograms(energy, scores, thresholds)
    windows = fit_peak_windows(histograms["no_cut"])
    peak_rows = make_peak_rows(histograms, masks, thresholds, windows)

    spectrum_csv = output_dir / "th232_ds_cnn_energy_spectra_1kev.csv"
    peak_csv = output_dir / "th232_ds_cnn_peak_to_background.csv"
    write_spectrum_csv(spectrum_csv, histograms)
    write_peak_csv(peak_csv, peak_rows)
    plot_spectra(output_dir, histograms, windows)
    plot_peak_to_background(output_dir, peak_rows)

    threshold_names = [threshold_name(value) for value in ACCEPTANCES]
    global_retention = {
        name: {
            "score_threshold": float(thresholds["thresholds"][name]["score_threshold"]),
            "events": int(np.count_nonzero(masks[name])),
            "fraction_of_admitted_th232": float(np.mean(masks[name])),
            "requested_weighted_acceptance": float(
                thresholds["thresholds"][name]["requested_weighted_acceptance"]
            ),
            "actual_weighted_acceptance": float(
                thresholds["thresholds"][name]["actual_weighted_acceptance"]
            ),
            "actual_unweighted_acceptance": float(
                thresholds["thresholds"][name]["actual_unweighted_acceptance"]
            ),
        }
        for name in threshold_names
    }
    report: dict[str, Any] = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "status": "EXTERNAL_EVALUATION",
        "model_frozen_before_th232": True,
        "frozen_model": frozen_metadata,
        "threshold_calibration": thresholds,
        "th232_scoring": scoring,
        "admission": {
            "energy_dataset": "corrected_energy_kev",
            "energy_range_kev": [ENERGY_MIN_KEV, ENERGY_MAX_KEV],
            "qc_rule": "reject qc_rejection_bits 0-2; retain noise bit 3 and pulse bit 4",
            "shaped_energy_rule": "finite and positive",
            "raw_events_retained_in_input_cache": True,
        },
        "representation_qc": {
            "anchor_fallback_count": scoring["counts"]["anchor_fallback_count"],
            "invalid_scale_count": scoring["counts"]["invalid_scale_count"],
            "charge_clipped_sample_count": scoring["counts"][
                "charge_clipped_sample_count"
            ],
            "charge_clipped_event_count": scoring["counts"][
                "charge_clipped_event_count"
            ],
        },
        "global_retention": global_retention,
        "energy_calibration": {
            "dataset": "corrected_energy_kev",
            "status": "PRELIMINARY",
            "source": "corrected Th-232 HDF5 attributes",
        },
        "peak_windows": [asdict(window) for window in windows],
        "peak_background_definition": (
            "(counts in observed centroid +/-2 sigma ROI - linearly interpolated "
            "3--5 sigma sideband background) / estimated background"
        ),
        "peak_background_rows": peak_rows,
        "scientific_boundary": (
            "Historical corrected Th-232 cache and same-domain development-positive "
            "threshold calibration; no independent isotope/session claim. Th-232 "
            "was not used to select the model or score cuts, and the locked test "
            "partition was not used."
        ),
        "artifacts": {},
    }
    artifact_paths = [
        output_dir / "th232_ds_cnn_scores.h5",
        spectrum_csv,
        peak_csv,
        output_dir / "th232_ds_cnn_energy_spectra.png",
        output_dir / "th232_ds_cnn_peak_zooms.png",
        output_dir / "th232_ds_cnn_peak_to_background.png",
    ]
    for path in artifact_paths:
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "th232_ds_cnn_evaluation.json"
    report["artifacts"][report_path.name] = {"path": relative(report_path)}
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"admitted_th232_events={energy.size}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
