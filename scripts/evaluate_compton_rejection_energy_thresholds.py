#!/usr/bin/env python3
"""Calibrate energy-dependent Compton rejection and apply it to Th-232."""

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

from scripts.evaluate_o2_3p_co60_threshold_curve import (  # noqa: E402
    closest_constant_pass_threshold,
)
from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    REFERENCE_PEAKS_KEV,
    fit_peak_windows,
    peak_background_metrics,
)
from scripts.optimize_th232_all_ba_ds_cnn_threshold import (  # noqa: E402
    EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV,
    PRIMARY_REFERENCE_PEAKS_KEV,
    load_model_contract,
)
from src.ba133_cnn import RawPartition, apply_channel_statistics, build_representation  # noqa: E402

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/experiments/relaxed_continuum_all_ba_t10_20260823/seed_20260823.pt"
)
DEFAULT_CACHE_MANIFEST = (
    PROJECT_ROOT
    / "processed_data/relaxed_continuum_all_ba_t10_20260823/cache_manifest.json"
)
DEFAULT_CO60_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/co60_continuum_100_1000kev_20260819"
)
DEFAULT_CS137_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/cs137_continuum_100_400kev_20260823"
)
DEFAULT_TH232_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/compton_rejection_energy_thresholds_20260823"
)
SOURCE_CONFIG = {
    "co60": {
        "code": 0,
        "edges": np.arange(100.0, 1050.0, 50.0),
        "minimum_energy_kev": 100.0,
        "maximum_energy_kev": 1000.0,
    },
    "cs137": {
        "code": 1,
        "edges": np.arange(100.0, 450.0, 50.0),
        "minimum_energy_kev": 100.0,
        "maximum_energy_kev": 400.0,
    },
}
PARTITION_CODE = {"train": 0, "validation": 1}
REJECTION_TARGETS_PERCENT = (10, 20, 30, 50)
GLOBAL_BASELINE_THRESHOLD = 0.437
POLYNOMIAL_CENTER_KEV = 550.0
POLYNOMIAL_SCALE_KEV = 450.0
FIT_EVALUATION_MINIMUM_KEV = 125.0
FIT_EVALUATION_MAXIMUM_KEV = 975.0
MAXIMUM_POLYNOMIAL_DEGREE = 6
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


def bin_mask(values: np.ndarray, low: float, high: float, last: bool) -> np.ndarray:
    if last:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def validate_store(path: Path, source_name: str, partition: str) -> int:
    config = SOURCE_CONFIG[source_name]
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("source", "")).lower() != source_name:
            raise ValueError(f"Source mismatch: {path}")
        if str(handle.attrs.get("partition", "")) != partition:
            raise ValueError(f"Partition mismatch: {path}")
        if bool(handle.attrs.get("test_partition_used", True)):
            raise ValueError(f"Locked-test contamination: {path}")
        count = int(handle["waveform"].shape[0])
        if handle["waveform"].shape != (count, 4500):
            raise ValueError(f"Unexpected waveform shape: {path}")
        energy = handle["corrected_energy_kev"]
        shaped = handle["shaped_energy_unit"]
        if energy.shape != (count,) or shaped.shape != (count,):
            raise ValueError(f"Unexpected scalar shape: {path}")
        if float(handle.attrs["minimum_energy_kev_inclusive"]) != config["minimum_energy_kev"]:
            raise ValueError(f"Unexpected minimum energy: {path}")
        if float(handle.attrs["maximum_energy_kev_inclusive"]) != config["maximum_energy_kev"]:
            raise ValueError(f"Unexpected maximum energy: {path}")
    return count


def create_score_cache(handle: h5py.File, chunk_size: int) -> None:
    for name, dtype in (
        ("corrected_energy_kev", np.float32),
        ("score", np.float32),
        ("source_code", np.uint8),
        ("partition_code", np.uint8),
        ("source_row", np.int64),
    ):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=dtype,
        )
    table = handle.create_group("input_stores")
    table.create_dataset("path", shape=(4,), dtype=STRING_DTYPE)
    table.create_dataset("sha256", shape=(4,), dtype=STRING_DTYPE)
    table.create_dataset("source", shape=(4,), dtype=STRING_DTYPE)
    table.create_dataset("partition", shape=(4,), dtype=STRING_DTYPE)
    table.create_dataset("event_count", shape=(4,), dtype=np.int64)


def resize_score_cache(handle: h5py.File, size: int) -> None:
    for name in (
        "corrected_energy_kev",
        "score",
        "source_code",
        "partition_code",
        "source_row",
    ):
        handle[name].resize((size,))


def score_continuum_stores(
    stores: list[tuple[str, str, Path]],
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
    next_index = 0
    records = []
    qc = {"anchor_fallback_count": 0, "invalid_scale_count": 0}
    try:
        with h5py.File(partial, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "model_kind": "all_ba_t10_ds_cnn",
                    "checkpoint": relative(checkpoint_path),
                    "checkpoint_sha256": model_metadata["checkpoint_sha256"],
                    "score_definition": "sigmoid of seed-20260823 DS-CNN logit",
                    "selection": "all events in approved source continuum stores",
                    "test_partition_used": False,
                    "created_utc": utc_now(),
                }
            )
            create_score_cache(output, batch_size)
            for store_index, (source_name, partition, path) in enumerate(stores):
                event_count = validate_store(path, source_name, partition)
                with h5py.File(path, "r") as source:
                    for start in range(0, event_count, batch_size):
                        stop = min(start + batch_size, event_count)
                        waveforms = np.asarray(source["waveform"][start:stop], dtype=np.float32)
                        shaped = np.asarray(source["shaped_energy_unit"][start:stop], dtype=np.float32)
                        energy = np.asarray(source["corrected_energy_kev"][start:stop], dtype=np.float32)
                        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(shaped)):
                            raise ValueError(f"Nonfinite continuum scalar in {path}")
                        count = stop - start
                        raw = RawPartition(
                            waveforms=waveforms,
                            shaped_energy=shaped,
                            labels=np.zeros(count, dtype=np.float32),
                            weights=np.ones(count, dtype=np.float32),
                            peak_ids=np.full(count, source_name, dtype="U8"),
                        )
                        values, batch_qc = build_representation(raw, config)
                        apply_channel_statistics(values, statistics)
                        with torch.inference_mode():
                            score = torch.sigmoid(
                                model(torch.from_numpy(values).to(device, non_blocking=True))
                            ).cpu().numpy().astype(np.float32, copy=False)
                        end_index = next_index + count
                        resize_score_cache(output, end_index)
                        output["corrected_energy_kev"][next_index:end_index] = energy
                        output["score"][next_index:end_index] = score
                        output["source_code"][next_index:end_index] = SOURCE_CONFIG[source_name]["code"]
                        output["partition_code"][next_index:end_index] = PARTITION_CODE[partition]
                        output["source_row"][next_index:end_index] = np.arange(start, stop, dtype=np.int64)
                        next_index = end_index
                        for key in qc:
                            qc[key] += int(batch_qc[key])
                digest = sha256_file(path)
                table = output["input_stores"]
                table["path"][store_index] = relative(path)
                table["sha256"][store_index] = digest
                table["source"][store_index] = source_name
                table["partition"][store_index] = partition
                table["event_count"][store_index] = event_count
                record = {
                    "path": relative(path),
                    "sha256": digest,
                    "source": source_name,
                    "partition": partition,
                    "event_count": event_count,
                }
                records.append(record)
                print(
                    f"continuum store {store_index + 1}/4 {source_name}/{partition} "
                    f"events={event_count} cumulative={next_index}",
                    flush=True,
                )
            output.attrs["event_count"] = next_index
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "path": relative(output_path),
        "sha256": sha256_file(output_path),
        "event_count": next_index,
        "input_stores": records,
        "representation_qc": qc,
        "model": model_metadata,
    }


def derive_empirical_threshold_rows(
    energy: np.ndarray,
    scores: np.ndarray,
    source_codes: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for source_name, config in SOURCE_CONFIG.items():
        source_mask = source_codes == config["code"]
        source_energy = energy[source_mask]
        source_scores = scores[source_mask]
        edges = config["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        for index, (low, high, center) in enumerate(zip(edges[:-1], edges[1:], centers)):
            selected = bin_mask(source_energy, low, high, index == centers.size - 1)
            if np.count_nonzero(selected) < 100:
                raise ValueError(f"Too few {source_name} events in {low:g}-{high:g} keV")
            selected_scores = source_scores[selected]
            for rejection in REJECTION_TARGETS_PERCENT:
                threshold, passed, pass_fraction = closest_constant_pass_threshold(
                    selected_scores, 1.0 - rejection / 100.0
                )
                rows.append(
                    {
                        "source": source_name,
                        "rejection_target_percent": rejection,
                        "energy_low_kev": float(low),
                        "energy_high_kev": float(high),
                        "energy_center_kev": float(center),
                        "event_count": int(selected_scores.size),
                        "empirical_threshold": float(threshold),
                        "empirical_rejected_count": int(selected_scores.size - passed),
                        "empirical_rejection_fraction": float(1.0 - pass_fraction),
                    }
                )
    return rows


def normalized_energy(energy_kev: np.ndarray) -> np.ndarray:
    return (np.asarray(energy_kev, dtype=np.float64) - POLYNOMIAL_CENTER_KEV) / POLYNOMIAL_SCALE_KEV


def fit_polynomial(points: list[dict[str, Any]], rejection: int) -> dict[str, Any]:
    selected = [row for row in points if row["rejection_target_percent"] == rejection]
    energy = np.asarray([row["energy_center_kev"] for row in selected], dtype=np.float64)
    threshold = np.asarray([row["empirical_threshold"] for row in selected], dtype=np.float64)
    unique_centers = np.unique(energy)
    candidates = []
    for degree in range(1, MAXIMUM_POLYNOMIAL_DEGREE + 1):
        coefficients = np.polynomial.polynomial.polyfit(normalized_energy(energy), threshold, degree)
        prediction = np.polynomial.polynomial.polyval(normalized_energy(energy), coefficients)
        loo_errors = []
        for center in unique_centers:
            retained = energy != center
            held_out = energy == center
            fitted = np.polynomial.polynomial.polyfit(
                normalized_energy(energy[retained]), threshold[retained], degree
            )
            predicted = np.polynomial.polynomial.polyval(
                normalized_energy(energy[held_out]), fitted
            )
            loo_errors.extend((threshold[held_out] - predicted).tolist())
        candidates.append(
            {
                "degree": degree,
                "coefficients_ascending": coefficients.tolist(),
                "fit_rmse": float(np.sqrt(np.mean(np.square(threshold - prediction)))),
                "grouped_leave_one_energy_out_rmse": float(
                    np.sqrt(np.mean(np.square(loo_errors)))
                ),
            }
        )
    best = min(candidates, key=lambda row: row["grouped_leave_one_energy_out_rmse"])
    overlap = {}
    for center in sorted(set(energy[energy <= 400.0])):
        values = threshold[energy == center]
        overlap[f"{center:.1f}"] = {
            "co60_minus_cs137": float(values[0] - values[1]),
            "absolute_difference": float(abs(values[0] - values[1])),
        }
    return {
        "family": "polynomial_in_normalized_energy",
        "equation": "T(E)=sum_k c[k]*x^k; x=(clip(E,125,975)-550)/450; clip T to [0,1]",
        "rejection_target_percent": rejection,
        "energy_domain_kev": [100.0, 1000.0],
        "outside_domain_policy": "hold the fitted threshold constant below the first 125-keV bin center and above the last 975-keV bin center",
        "source_weighting": "one equal-weight point per source and 50-keV bin",
        "selected_degree": int(best["degree"]),
        "coefficients_ascending": best["coefficients_ascending"],
        "fit_rmse": best["fit_rmse"],
        "grouped_leave_one_energy_out_rmse": best["grouped_leave_one_energy_out_rmse"],
        "candidate_degrees": candidates,
        "overlap_source_differences": overlap,
    }


def evaluate_fit(energy_kev: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    bounded = np.clip(
        np.asarray(energy_kev, dtype=np.float64),
        FIT_EVALUATION_MINIMUM_KEV,
        FIT_EVALUATION_MAXIMUM_KEV,
    )
    values = np.polynomial.polynomial.polyval(
        normalized_energy(bounded),
        np.asarray(fit["coefficients_ascending"], dtype=np.float64),
    )
    return np.clip(values, 0.0, 1.0)


def audit_fitted_rejection(
    rows: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
    energy: np.ndarray,
    scores: np.ndarray,
    source_codes: np.ndarray,
) -> None:
    for row in rows:
        source_name = row["source"]
        config = SOURCE_CONFIG[source_name]
        selected = (
            (source_codes == config["code"])
            & bin_mask(
                energy,
                row["energy_low_kev"],
                row["energy_high_kev"],
                row["energy_high_kev"] == config["maximum_energy_kev"],
            )
        )
        fit = fits[f"rejection_{row['rejection_target_percent']}pct"]
        rejected = scores[selected] < evaluate_fit(energy[selected], fit)
        row["fitted_rejected_count"] = int(np.count_nonzero(rejected))
        row["fitted_rejection_fraction"] = float(np.mean(rejected))


def safe_peak_metrics(histogram: np.ndarray, window: Any) -> dict[str, float]:
    try:
        return peak_background_metrics(histogram, window)
    except ZeroDivisionError:
        return {
            "roi_counts": float("nan"),
            "estimated_background_counts": 0.0,
            "net_peak_counts": float("nan"),
            "peak_to_background": float("nan"),
        }


def evaluate_th232(
    th232_path: Path,
    fits: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[Any]]:
    with h5py.File(th232_path, "r") as source:
        if str(source.attrs.get("checkpoint_sha256")) == "":
            raise ValueError("Th-232 score cache lacks checkpoint provenance")
        energy = np.asarray(source["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(source["score"], dtype=np.float32)
    masks = {
        "no_cut": np.ones(energy.size, dtype=bool),
        "global_0p437": scores >= GLOBAL_BASELINE_THRESHOLD,
    }
    for condition, fit in fits.items():
        masks[condition] = scores >= evaluate_fit(energy, fit)
    histograms = {
        condition: np.histogram(energy[mask], ENERGY_EDGES)[0]
        for condition, mask in masks.items()
    }
    windows = fit_peak_windows(histograms["no_cut"])
    baselines = {
        "no_cut": {
            window.reference_kev: safe_peak_metrics(histograms["no_cut"], window)
            for window in windows
        },
        "global_0p437": {
            window.reference_kev: safe_peak_metrics(histograms["global_0p437"], window)
            for window in windows
        },
    }
    rows = []
    for condition, histogram in histograms.items():
        for window in windows:
            metrics = safe_peak_metrics(histogram, window)
            no_cut = baselines["no_cut"][window.reference_kev]
            global_baseline = baselines["global_0p437"][window.reference_kev]
            rows.append(
                {
                    "condition": condition,
                    "reference_energy_kev": window.reference_kev,
                    "observed_centroid_kev": window.centroid_kev,
                    **metrics,
                    "pb_improvement_factor_vs_no_cut": metrics["peak_to_background"] / no_cut["peak_to_background"],
                    "pb_factor_vs_global_0p437": metrics["peak_to_background"] / global_baseline["peak_to_background"],
                    "net_peak_retention_vs_no_cut": metrics["net_peak_counts"] / no_cut["net_peak_counts"],
                    "net_peak_retention_vs_global_0p437": metrics["net_peak_counts"] / global_baseline["net_peak_counts"],
                    "excluded_double_escape_region": window.reference_kev == EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV,
                }
            )
    summaries = {}
    for condition, mask in masks.items():
        primary = [
            row
            for row in rows
            if row["condition"] == condition
            and row["reference_energy_kev"] in PRIMARY_REFERENCE_PEAKS_KEV
        ]
        summaries[condition] = {
            "selected_events": int(np.count_nonzero(mask)),
            "selected_fraction": float(np.mean(mask)),
            "geometric_mean_pb_improvement_vs_no_cut": float(
                np.exp(np.mean(np.log([row["pb_improvement_factor_vs_no_cut"] for row in primary])))
            ),
            "geometric_mean_pb_factor_vs_global_0p437": float(
                np.exp(np.mean(np.log([row["pb_factor_vs_global_0p437"] for row in primary])))
            ),
            "minimum_net_peak_retention_vs_no_cut": float(
                min(row["net_peak_retention_vs_no_cut"] for row in primary)
            ),
            "mean_net_peak_retention_vs_no_cut": float(
                np.mean([row["net_peak_retention_vs_no_cut"] for row in primary])
            ),
        }
    return summaries, rows, histograms, windows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_thresholds(
    output_dir: Path,
    rows: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    dense = np.linspace(100.0, 1000.0, 1000)
    colors = {"co60": "#0072B2", "cs137": "#D55E00"}
    for axis, rejection in zip(axes.flat, REJECTION_TARGETS_PERCENT):
        for source_name in SOURCE_CONFIG:
            selected = [row for row in rows if row["source"] == source_name and row["rejection_target_percent"] == rejection]
            axis.scatter(
                [row["energy_center_kev"] for row in selected],
                [row["empirical_threshold"] for row in selected],
                s=24,
                color=colors[source_name],
                label=source_name,
            )
        fit = fits[f"rejection_{rejection}pct"]
        axis.plot(dense, evaluate_fit(dense, fit), color="black", linewidth=1.5, label=f"degree {fit['selected_degree']} fit")
        axis.axhline(GLOBAL_BASELINE_THRESHOLD, color="0.5", linestyle="--", linewidth=1.0, label="global 0.437")
        axis.set_title(f"{rejection}% Compton rejection")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.supxlabel("Corrected energy (keV)")
    figure.supylabel("DS-CNN score threshold")
    figure.suptitle("Co-60 and Cs-137 50-keV thresholds with joint polynomial fits")
    figure.savefig(output_dir / "compton_threshold_energy_dependence.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    for axis, rejection in zip(axes.flat, REJECTION_TARGETS_PERCENT):
        for source_name in SOURCE_CONFIG:
            selected = [row for row in rows if row["source"] == source_name and row["rejection_target_percent"] == rejection]
            axis.plot(
                [row["energy_center_kev"] for row in selected],
                [100.0 * row["fitted_rejection_fraction"] for row in selected],
                marker="o",
                color=colors[source_name],
                label=source_name,
            )
        axis.axhline(rejection, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"Target {rejection}%")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.supxlabel("Corrected energy (keV)")
    figure.supylabel("Achieved continuum rejection (%)")
    figure.suptitle("Rejection achieved by fitted threshold functions")
    figure.savefig(output_dir / "compton_fitted_rejection_by_energy.png", dpi=180)
    plt.close(figure)


def plot_th232(
    output_dir: Path,
    summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    histograms: dict[str, np.ndarray],
) -> None:
    conditions = ["global_0p437", *[f"rejection_{value}pct" for value in REJECTION_TARGETS_PERCENT]]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for condition in conditions:
        selected = [
            row for row in rows
            if row["condition"] == condition and row["reference_energy_kev"] in PRIMARY_REFERENCE_PEAKS_KEV
        ]
        label = "Global 0.437" if condition == "global_0p437" else condition.replace("rejection_", "").replace("pct", "% rejection")
        axes[0].plot(
            [row["reference_energy_kev"] for row in selected],
            [row["pb_improvement_factor_vs_no_cut"] for row in selected],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["reference_energy_kev"] for row in selected],
            [row["net_peak_retention_vs_no_cut"] for row in selected],
            marker="o",
            label=label,
        )
    axes[0].axhline(1.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("P/B improvement vs no cut")
    axes[1].set_ylabel("Net peak retention vs no cut")
    for axis in axes:
        axis.set_xlabel("Th-232 peak energy (keV)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Th-232 energy-dependent thresholds versus global threshold 0.437")
    figure.savefig(output_dir / "th232_pb_retention_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for condition in ["no_cut", *conditions]:
        label = "No cut" if condition == "no_cut" else ("Global 0.437" if condition == "global_0p437" else condition.replace("rejection_", "").replace("pct", "% rejection"))
        for axis in axes:
            axis.step(ENERGY_CENTERS, histograms[condition], where="mid", linewidth=0.75, label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8)
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[0].set_ylabel("Counts / 1 keV")
    axes[1].set_ylabel("Counts / 1 keV")
    axes[0].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axvspan(1575.0, 1605.0, color="#CC79A7", alpha=0.10)
    figure.suptitle("Th-232 after fitted Compton-rejection threshold functions")
    figure.savefig(output_dir / "th232_compton_rejection_spectra.png", dpi=180)
    plt.close(figure)


def make_report_markdown(
    fits: dict[str, dict[str, Any]],
    continuum_summary: dict[str, Any],
    th232_summary: dict[str, Any],
) -> str:
    lines = [
        "# Energy-dependent Compton rejection thresholds",
        "",
        "## Simple result",
        "",
        "- Model: selected all-Ba MA10/t10 DS-CNN, seed 20260823.",
        "- Calibration data: Co-60 100-1000 keV and Cs-137 100-400 keV, in 50-keV bins.",
        "- Locked-test files were excluded; all approved train+validation continuum events were retained.",
        "- Fits use equal weight for every source/bin threshold point.",
        "- Th-232 is used as deployment optimization data, not external validation.",
        "- Global threshold 0.437 is the baseline.",
        "",
        "| Target continuum rejection | Polynomial degree | Co-60 achieved | Cs-137 achieved | Th-232 retained | Th-232 P/B vs no cut | P/B vs global 0.437 | Min peak retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rejection in REJECTION_TARGETS_PERCENT:
        condition = f"rejection_{rejection}pct"
        fit = fits[condition]
        continuum = continuum_summary[condition]
        th = th232_summary[condition]
        lines.append(
            f"| {rejection}% | {fit['selected_degree']} | {continuum['co60_overall_rejection']:.2%} | {continuum['cs137_overall_rejection']:.2%} | {th['selected_fraction']:.2%} | {th['geometric_mean_pb_improvement_vs_no_cut']:.4f}x | {th['geometric_mean_pb_factor_vs_global_0p437']:.4f}x | {th['minimum_net_peak_retention_vs_no_cut']:.2%} |"
        )
    baseline = th232_summary["global_0p437"]
    lines.extend(
        [
            "",
            f"Global 0.437 baseline: `{baseline['geometric_mean_pb_improvement_vs_no_cut']:.4f}x` geometric-mean P/B improvement, `{baseline['minimum_net_peak_retention_vs_no_cut']:.2%}` minimum peak retention, and `{baseline['selected_fraction']:.2%}` overall Th-232 event retention.",
            "",
            "The approximately 1592-keV double-escape region remains diagnostic only and is excluded from all cross-energy summary metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--co60-store-dir", type=Path, default=DEFAULT_CO60_STORE_DIR)
    parser.add_argument("--cs137-store-dir", type=Path, default=DEFAULT_CS137_STORE_DIR)
    parser.add_argument("--th232-score-cache", type=Path, default=DEFAULT_TH232_SCORE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_cache = output_dir / "continuum_all_ba_ds_cnn_scores.h5"
    existing = list(output_dir.iterdir())
    if existing and not args.overwrite and existing != [score_cache]:
        raise FileExistsError(output_dir)
    stores = []
    for source_name, directory in (
        ("co60", args.co60_store_dir.resolve()),
        ("cs137", args.cs137_store_dir.resolve()),
    ):
        for partition in PARTITION_CODE:
            stores.append((source_name, partition, directory / f"{partition}_events.h5"))
    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.cache_manifest.resolve()
    if score_cache.exists():
        with h5py.File(score_cache, "r") as cache:
            if str(cache.attrs.get("checkpoint_sha256")) != sha256_file(checkpoint_path):
                raise ValueError("Existing continuum cache uses another checkpoint")
            event_count = int(cache.attrs["event_count"])
        _, _, _, model_metadata = load_model_contract(checkpoint_path, manifest_path, device)
        scoring = {
            "path": relative(score_cache),
            "sha256": sha256_file(score_cache),
            "event_count": event_count,
            "reused_completed_score_cache": True,
            "model": model_metadata,
        }
        print(f"reusing continuum score cache events={event_count}", flush=True)
    else:
        scoring = score_continuum_stores(
            stores,
            checkpoint_path,
            manifest_path,
            score_cache,
            args.batch_size,
            device,
        )
    with h5py.File(score_cache, "r") as cache:
        energy = np.asarray(cache["corrected_energy_kev"], dtype=np.float32)
        scores = np.asarray(cache["score"], dtype=np.float32)
        source_codes = np.asarray(cache["source_code"], dtype=np.uint8)
    threshold_rows = derive_empirical_threshold_rows(energy, scores, source_codes)
    fits = {
        f"rejection_{rejection}pct": fit_polynomial(threshold_rows, rejection)
        for rejection in REJECTION_TARGETS_PERCENT
    }
    audit_fitted_rejection(threshold_rows, fits, energy, scores, source_codes)
    continuum_summary = {}
    for rejection in REJECTION_TARGETS_PERCENT:
        condition = f"rejection_{rejection}pct"
        fit = fits[condition]
        applied = scores < evaluate_fit(energy, fit)
        summary = {}
        for source_name, config in SOURCE_CONFIG.items():
            selected = source_codes == config["code"]
            summary[f"{source_name}_overall_rejection"] = float(np.mean(applied[selected]))
            bin_rows = [row for row in threshold_rows if row["source"] == source_name and row["rejection_target_percent"] == rejection]
            summary[f"{source_name}_bin_rejection_range"] = [
                float(min(row["fitted_rejection_fraction"] for row in bin_rows)),
                float(max(row["fitted_rejection_fraction"] for row in bin_rows)),
            ]
        continuum_summary[condition] = summary
    th232_path = args.th232_score_cache.resolve()
    th232_summary, pb_rows, histograms, windows = evaluate_th232(th232_path, fits)
    threshold_csv = output_dir / "continuum_thresholds_50kev.csv"
    pb_csv = output_dir / "th232_peak_background.csv"
    write_csv(threshold_csv, threshold_rows)
    write_csv(pb_csv, pb_rows)
    spectrum_csv = output_dir / "th232_spectra_1kev.csv"
    names = list(histograms)
    np.savetxt(
        spectrum_csv,
        np.column_stack((ENERGY_CENTERS, *[histograms[name] for name in names])),
        delimiter=",",
        header=",".join(["energy_kev_bin_center", *names]),
        comments="",
        fmt=["%.1f", *("%d" for _ in names)],
    )
    plot_thresholds(output_dir, threshold_rows, fits)
    plot_th232(output_dir, th232_summary, pb_rows, histograms)
    report_markdown = output_dir / "report.md"
    report_markdown.write_text(
        make_report_markdown(fits, continuum_summary, th232_summary), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "COMPTON_ENERGY_THRESHOLD_FUNCTIONS_COMPLETE",
        "model": scoring["model"],
        "continuum_scoring": scoring,
        "source_domains": {
            "co60": [100.0, 1000.0],
            "cs137": [100.0, 400.0],
            "bin_width_kev": 50.0,
            "partitions": ["train", "validation"],
            "locked_test_used": False,
        },
        "rejection_targets_percent": list(REJECTION_TARGETS_PERCENT),
        "threshold_rule": "score >= threshold(E) passes; lower scores are rejected",
        "empirical_threshold_rows": threshold_rows,
        "fits": fits,
        "continuum_rejection_summary": continuum_summary,
        "th232": {
            "score_cache": relative(th232_path),
            "score_cache_sha256": sha256_file(th232_path),
            "role": "direct deployment optimization; not external validation",
            "global_baseline_threshold": GLOBAL_BASELINE_THRESHOLD,
            "conditions": th232_summary,
            "peak_windows": [asdict(window) for window in windows],
            "peak_background_rows": pb_rows,
            "excluded_double_escape_region_reference_kev": EXCLUDED_DOUBLE_ESCAPE_REGION_REFERENCE_KEV,
        },
        "claim_boundary": (
            "Threshold functions use all approved development Co-60/Cs-137 continuum "
            "events and are assessed directly on historical Th-232 optimization data. "
            "No locked test or Eu-152 data were used."
        ),
        "test_partition_used": False,
        "artifacts": {},
    }
    artifact_paths = [
        score_cache,
        threshold_csv,
        pb_csv,
        spectrum_csv,
        report_markdown,
        output_dir / "compton_threshold_energy_dependence.png",
        output_dir / "compton_fitted_rejection_by_energy.png",
        output_dir / "th232_pb_retention_comparison.png",
        output_dir / "th232_compton_rejection_spectra.png",
    ]
    for path in artifact_paths:
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "continuum": continuum_summary, "th232": th232_summary, "report": relative(report_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
