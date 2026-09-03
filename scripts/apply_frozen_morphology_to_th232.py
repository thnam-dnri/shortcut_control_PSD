#!/usr/bin/env python3
"""Apply the frozen six-group morphology catalogue to corrected Th-232 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_relaxed_six_group_datasets import apply_catalogue
from scripts.evaluate_th232_o2_3p_energy_threshold import (
    ENERGY_CENTERS,
    ENERGY_EDGES,
    REFERENCE_PEAKS_KEV,
    fit_peak_windows,
    peak_background_metrics,
    th232_admission_mask,
)
from src.waveform_morphology import MorphologyConfig, extract_morphology_features

MINIMUM_INDEX_LOW = 1000
MINIMUM_INDEX_HIGH = 1500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT)
        else str(resolved)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_group_spectra(
    output_dir: Path,
    histograms: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    colors = plt.cm.tab10(np.arange(6))
    for group, (axis, color) in enumerate(zip(axes.flat, colors), 1):
        values = histograms[f"group_{group}"]
        axis.step(ENERGY_CENTERS, values, where="mid", color=color, linewidth=0.8)
        axis.set_yscale("log")
        axis.set_ylim(bottom=0.8)
        axis.set_title(
            f"Group {group}: {int(values.sum()):,} events "
            f"({values.sum() / histograms['all'].sum():.1%})"
        )
        axis.set_ylabel("Counts / 1 keV")
        axis.grid(alpha=0.2)
        for peak in REFERENCE_PEAKS_KEV:
            axis.axvline(peak, color="black", linestyle=":", linewidth=0.5, alpha=0.45)
    for axis in axes[-1]:
        axis.set_xlabel("Corrected energy (keV)")
    figure.suptitle("Th-232 energy spectrum for each frozen morphology group")
    figure.tight_layout()
    figure.savefig(output_dir / "th232_energy_spectrum_by_group.png", dpi=180)
    plt.close(figure)

    rebin = 5
    stop = (histograms["all"].size // rebin) * rebin
    rebinned_center = ENERGY_CENTERS[:stop].reshape(-1, rebin).mean(axis=1)
    total = histograms["all"][:stop].reshape(-1, rebin).sum(axis=1).astype(np.float64)
    figure, axis = plt.subplots(figsize=(13, 6))
    for group, color in enumerate(colors, 1):
        group_counts = histograms[f"group_{group}"][:stop].reshape(-1, rebin).sum(axis=1)
        fraction = np.divide(
            group_counts,
            total,
            out=np.full_like(total, np.nan),
            where=total >= 20,
        )
        axis.plot(rebinned_center, fraction, color=color, linewidth=0.9, label=f"Group {group}")
    for peak in REFERENCE_PEAKS_KEV:
        axis.axvline(peak, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    axis.set(
        xlim=(0, 2700),
        ylim=(0, 1),
        xlabel="Corrected energy (keV)",
        ylabel="Fraction of events in morphology group",
        title="Frozen morphology-group fraction across the Th-232 spectrum (5-keV bins)",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output_dir / "th232_group_fraction_vs_energy.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--th232-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813",
    )
    parser.add_argument(
        "--catalogue-model",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/morphology_catalogue_minimum_1000_1500_20260821/catalogue/catalogue_model.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/th232_frozen_morphology_groups_20260822",
    )
    parser.add_argument("--expected-files", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    th232_dir = args.th232_dir.resolve()
    catalogue_path = args.catalogue_model.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(th232_dir.glob("*.h5"))
    if len(files) != args.expected_files:
        raise ValueError(f"Expected {args.expected_files} Th-232 files, found {len(files)}")
    catalogue = joblib.load(catalogue_path)
    if len(catalogue["component_order"]) != 8:
        raise ValueError("Expected frozen eight-component source catalogue")
    config = MorphologyConfig()

    energy_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    confidence_blocks: list[np.ndarray] = []
    file_index_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    counts = {
        "input_events": 0,
        "admitted_energy_shaped_qc_events": 0,
        "excluded_outside_minimum_index": 0,
        "excluded_morphology_invalid": 0,
        "selected_grouped_events": 0,
        "anchor_fallback_count": 0,
        "invalid_scale_count": 0,
    }
    file_rows: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        file_selected = 0
        with h5py.File(path, "r") as source:
            if str(source.attrs.get("processing_status")) != "OK":
                raise ValueError(f"Non-OK processing status: {path}")
            if str(source.attrs.get("source_label", "")).lower() != "th232":
                raise ValueError(f"Unexpected source label: {path}")
            event_count = int(source["waveform"].shape[0])
            if source["waveform"].shape != (event_count, 4500):
                raise ValueError(f"Unexpected waveform shape: {path}")
            counts["input_events"] += event_count
            for start in range(0, event_count, args.batch_size):
                stop = min(start + args.batch_size, event_count)
                energy = np.asarray(source["corrected_energy_kev"][start:stop], dtype=np.float32)
                shaped = np.asarray(source["shaped_energy_unit"][start:stop], dtype=np.float32)
                bits = np.asarray(source["qc_rejection_bits"][start:stop], dtype=np.uint16)
                admitted, _energy_valid, _shaped_valid = th232_admission_mask(
                    energy, shaped, bits
                )
                if not np.any(admitted):
                    continue
                admitted_rows = start + np.flatnonzero(admitted)
                waveforms = np.asarray(source["waveform"][start:stop][admitted], dtype=np.float32)
                selected_energy = energy[admitted]
                counts["admitted_energy_shaped_qc_events"] += int(admitted_rows.size)
                raw_minimum_index = np.argmin(waveforms, axis=1)
                features, qc = extract_morphology_features(waveforms, config)
                morphology_valid = qc["valid"]
                quality_valid = (
                    (raw_minimum_index >= MINIMUM_INDEX_LOW)
                    & (raw_minimum_index <= MINIMUM_INDEX_HIGH)
                )
                selected = morphology_valid & quality_valid
                counts["excluded_outside_minimum_index"] += int(np.count_nonzero(~quality_valid))
                counts["excluded_morphology_invalid"] += int(
                    np.count_nonzero(quality_valid & ~morphology_valid)
                )
                counts["anchor_fallback_count"] += int(qc["anchor_fallback_count"])
                counts["invalid_scale_count"] += int(qc["invalid_scale_count"])
                if not np.any(selected):
                    continue
                probability, assignment = apply_catalogue(features, selected, catalogue)
                chosen = np.flatnonzero(selected)
                energy_blocks.append(selected_energy[chosen])
                group_blocks.append((assignment[chosen] + 1).astype(np.uint8))
                confidence_blocks.append(np.max(probability[chosen], axis=1).astype(np.float32))
                file_index_blocks.append(np.full(chosen.size, file_index, dtype=np.uint8))
                row_blocks.append(admitted_rows[chosen].astype(np.int32))
                file_selected += int(chosen.size)
                counts["selected_grouped_events"] += int(chosen.size)
        file_rows.append(
            {
                "file_index": file_index,
                "path": relative(path),
                "input_events": event_count,
                "selected_grouped_events": file_selected,
            }
        )
        print(
            f"Th232 morphology {file_index + 1}/{len(files)} "
            f"selected={file_selected:,} cumulative={counts['selected_grouped_events']:,}",
            flush=True,
        )

    energies = np.concatenate(energy_blocks)
    groups = np.concatenate(group_blocks)
    confidence = np.concatenate(confidence_blocks)
    source_file_index = np.concatenate(file_index_blocks)
    source_row = np.concatenate(row_blocks)
    assignments_output = output_dir / "th232_frozen_morphology_assignments.npz"
    np.savez_compressed(
        assignments_output,
        corrected_energy_kev=energies,
        group=groups,
        maximum_group_probability=confidence,
        source_file_index=source_file_index,
        source_row=source_row,
    )

    histograms = {"all": np.histogram(energies, bins=ENERGY_EDGES)[0]}
    for group in range(1, 7):
        histograms[f"group_{group}"] = np.histogram(
            energies[groups == group], bins=ENERGY_EDGES
        )[0]
    spectrum_path = output_dir / "th232_energy_spectrum_by_group_1kev.csv"
    spectrum_values = np.column_stack(
        (ENERGY_CENTERS, histograms["all"], *(histograms[f"group_{group}"] for group in range(1, 7)))
    )
    np.savetxt(
        spectrum_path,
        spectrum_values,
        delimiter=",",
        header="energy_kev_bin_center,all,group_1,group_2,group_3,group_4,group_5,group_6",
        comments="",
        fmt=("%.1f", "%d", "%d", "%d", "%d", "%d", "%d", "%d"),
    )
    windows = fit_peak_windows(histograms["all"])
    peak_rows: list[dict[str, Any]] = []
    for window in windows:
        baseline = peak_background_metrics(histograms["all"], window)
        for group in range(1, 7):
            result = peak_background_metrics(histograms[f"group_{group}"], window)
            peak_retention = result["net_peak_counts"] / baseline["net_peak_counts"]
            background_retention = (
                result["estimated_background_counts"]
                / baseline["estimated_background_counts"]
            )
            peak_rows.append(
                {
                    "reference_energy_kev": window.reference_kev,
                    "observed_centroid_kev": window.centroid_kev,
                    "fwhm_kev": 2.354820045 * window.sigma_kev,
                    "group": group,
                    **result,
                    "net_peak_retention_vs_all_groups": peak_retention,
                    "background_retention_vs_all_groups": background_retention,
                    "peak_to_background_improvement_vs_all_groups": (
                        result["peak_to_background"] / baseline["peak_to_background"]
                    ),
                    "peak_retention_divided_by_background_retention": (
                        peak_retention / background_retention
                    ),
                }
            )
    peak_path = output_dir / "th232_group_peak_background.csv"
    write_csv(peak_path, peak_rows)
    plot_group_spectra(output_dir, histograms)

    group_counts = {
        f"group_{group}": int(np.count_nonzero(groups == group))
        for group in range(1, 7)
    }
    group_2_rows = [row for row in peak_rows if row["group"] == 2]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_MORPHOLOGY_APPLIED_TO_TH232",
        "authorization": "Explicit user request in current session",
        "catalogue": {
            "kind": "frozen merged six-group GMM",
            "model": relative(catalogue_path),
            "model_sha256": sha256_file(catalogue_path),
            "refitted_on_th232": False,
        },
        "selection": {
            "corrected_energy_range_kev": [0.0, 3200.0],
            "established_th232_admission": True,
            "raw_negative_minimum_index_inclusive": [MINIMUM_INDEX_LOW, MINIMUM_INDEX_HIGH],
        },
        "counts": counts,
        "group_counts": group_counts,
        "group_fractions": {
            key: value / counts["selected_grouped_events"]
            for key, value in group_counts.items()
        },
        "group_2_peak_results": group_2_rows,
        "files": file_rows,
        "artifacts": {
            "assignments": relative(assignments_output),
            "assignments_sha256": sha256_file(assignments_output),
            "spectrum_csv": relative(spectrum_path),
            "spectrum_csv_sha256": sha256_file(spectrum_path),
            "peak_background_csv": relative(peak_path),
            "peak_background_csv_sha256": sha256_file(peak_path),
            "spectrum_plot": relative(output_dir / "th232_energy_spectrum_by_group.png"),
            "fraction_plot": relative(output_dir / "th232_group_fraction_vs_energy.png"),
        },
        "claim_boundary": (
            "Historical corrected Th-232 external diagnostic. Morphology was frozen "
            "before access; no model, group mapping, energy window, or threshold was "
            "selected from Th-232. Peak/background results test transfer only."
        ),
        "test_partition_used": False,
        "th232_used": True,
        "eu152_used": False,
    }
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": counts,
                "group_counts": group_counts,
                "group_2_peak_results": group_2_rows,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
