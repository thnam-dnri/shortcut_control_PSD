#!/usr/bin/env python3
"""Plot exact normalized O2-3P charge/current inputs for three training peaks."""

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

from scripts.train_o2_late_fusion import (  # noqa: E402
    CHARGE_PRE_SAMPLES,
    CURRENT_HALF_WIDTH,
    SAMPLE_PERIOD_NS,
    extract_o2_features,
)
from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    relative,
    sha256_file,
    utc_now,
)

PEAKS = (
    ("ba133_356kev", "Ba-133 356 keV", "#377eb8"),
    ("na22_511kev", "Na-22 511 keV", "#4daf4a"),
    ("cs137_662kev", "Cs-137 662 keV", "#e41a1c"),
)


def select_positive_rows(label_csv: Path, samples_per_peak: int, seed: int):
    grouped = {peak_id: [] for peak_id, _label, _color in PEAKS}
    with label_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("partition") == "train"
                and int(row["positive_label"]) == 1
                and row["peak_id"] in grouped
            ):
                grouped[row["peak_id"]].append(dict(row))
    rng = np.random.default_rng(seed)
    selected = {}
    for peak_id, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                row["positive_hdf5"],
                int(row["positive_row"]),
                row["pair_id"],
            )
        )
        if len(rows) < samples_per_peak:
            raise ValueError(f"Only {len(rows)} positive rows available for {peak_id}")
        indices = np.sort(rng.choice(len(rows), size=samples_per_peak, replace=False))
        selected[peak_id] = [rows[int(index)] for index in indices]
    return selected


def resolve_store_indices(selected, lookup_csv: Path):
    required = {
        (row["positive_hdf5"], int(row["positive_row"]))
        for rows in selected.values()
        for row in rows
    }
    lookup = {}
    with lookup_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            if key in required:
                lookup[key] = int(row["store_index"])
    missing = required - set(lookup)
    if missing:
        raise KeyError(f"Event store lookup is missing {len(missing)} selected rows")
    if len(lookup) != len(required):
        raise ValueError("Selected positive training events are not unique")
    return lookup


def load_normalized_features(selected, lookup, store_path: Path, statistics):
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
        waveforms = np.asarray(store["waveform"][store_indices[order]], dtype=np.float32)
        shaped = np.asarray(store["shaped_energy_unit"][store_indices[order]], dtype=np.float32)
    charge_sorted, current_sorted, fallback_count = extract_o2_features(waveforms, shaped)
    inverse = np.argsort(order)
    charge = charge_sorted[inverse]
    current = current_sorted[inverse]
    charge = (charge - np.float32(statistics["charge_mean"])) / np.float32(
        statistics["charge_std"]
    )
    current = (current - np.float32(statistics["current_mean"])) / np.float32(
        statistics["current_std"]
    )
    features = {}
    start = 0
    for peak_id, rows in selected.items():
        stop = start + len(rows)
        features[peak_id] = {"charge": charge[start:stop], "current": current[start:stop]}
        for row, store_index in zip(rows, store_indices[start:stop]):
            row["store_index"] = int(store_index)
        start = stop
    return features, fallback_count


def plot_overlays(output_path: Path, features):
    charge_time = (np.arange(features[PEAKS[0][0]]["charge"].shape[1]) - CHARGE_PRE_SAMPLES) * SAMPLE_PERIOD_NS
    current_time = (np.arange(features[PEAKS[0][0]]["current"].shape[1]) - CURRENT_HALF_WIDTH) * SAMPLE_PERIOD_NS
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col", sharey="col")
    for row_index, (peak_id, label, color) in enumerate(PEAKS):
        for column, (branch, time, alignment) in enumerate(
            (("charge", charge_time, "t10"), ("current", current_time, "current peak"))
        ):
            axis = axes[row_index, column]
            values = features[peak_id][branch]
            axis.plot(time, values.T, color=color, alpha=0.10, linewidth=0.55)
            axis.plot(time, np.mean(values, axis=0), color="black", linewidth=1.8, label="mean")
            axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
            axis.set_title(f"{label}: normalized {branch} (100 events)")
            axis.set_ylabel("Standardized model input")
            axis.grid(alpha=0.18)
            axis.text(0.99, 0.04, f"aligned to {alignment}", transform=axis.transAxes, ha="right", fontsize=9)
    axes[-1, 0].set_xlabel("Time relative to t10 (ns)")
    axes[-1, 1].set_xlabel("Time relative to current peak (ns)")
    figure.suptitle("O2-3P Late Fusion normalized positive training inputs", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_heatmaps(output_path: Path, features):
    charge_time = (np.arange(features[PEAKS[0][0]]["charge"].shape[1]) - CHARGE_PRE_SAMPLES) * SAMPLE_PERIOD_NS
    current_time = (np.arange(features[PEAKS[0][0]]["current"].shape[1]) - CURRENT_HALF_WIDTH) * SAMPLE_PERIOD_NS
    limits = {}
    for branch in ("charge", "current"):
        combined = np.concatenate([features[peak_id][branch].ravel() for peak_id, _label, _color in PEAKS])
        limits[branch] = tuple(np.percentile(combined, (1.0, 99.0)))
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col", sharey="row")
    for row_index, (peak_id, label, _color) in enumerate(PEAKS):
        for column, (branch, time) in enumerate((("charge", charge_time), ("current", current_time))):
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
            figure.colorbar(image, ax=axis, pad=0.01, label="Standardized input")
    axes[-1, 0].set_xlabel("Time relative to t10 (ns)")
    axes[-1, 1].set_xlabel("Time relative to current peak (ns)")
    figure.suptitle("All 100 O2-3P training inputs per positive peak", fontsize=15)
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
        default=PROJECT_ROOT / "outputs/models/three_peak_weight_scan_20260819/late_fusion_best.pt",
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
    if checkpoint.get("model_kind") != "late_fusion" or checkpoint.get("test_partition_used") is not False:
        raise ValueError("Unexpected or test-contaminated checkpoint")
    statistics = checkpoint["feature_statistics"]
    selected = select_positive_rows(label_csv, args.samples_per_peak, args.seed)
    lookup = resolve_store_indices(selected, lookup_path)
    features, fallback_count = load_normalized_features(selected, lookup, store_path, statistics)

    overlay_path = output_dir / "normalized_training_inputs_overlay.png"
    heatmap_path = output_dir / "normalized_training_inputs_heatmap.png"
    npz_path = output_dir / "normalized_training_inputs.npz"
    selection_path = output_dir / "selected_training_events.csv"
    plot_overlays(overlay_path, features)
    plot_heatmaps(heatmap_path, features)
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
        "model_name": "O2-3P Late Fusion",
        "partition": "train",
        "selection": {
            "positive_label_only": True,
            "samples_per_peak": args.samples_per_peak,
            "seed": args.seed,
            "without_replacement": True,
            "peak_ids": [peak_id for peak_id, _label, _color in PEAKS],
        },
        "representation": {
            "charge": "baseline-subtracted, causal MA10, t10-aligned [-250,+499], divided by shaped_energy_unit",
            "current": "d(baseline-subtracted charge)/dt, peak-aligned [-125,+125], divided by shaped_energy_unit",
            "sample_period_ns": SAMPLE_PERIOD_NS,
            "feature_statistics": statistics,
            "standardization": "(energy-normalized branch - train mean) / train standard deviation",
            "t10_fallback_count": fallback_count,
        },
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
