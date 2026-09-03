#!/usr/bin/env python3
"""Build unpaired DS-CNN caches with a relaxed continuum ROI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_energy_matched_labels import TRAINING_PEAKS, sha256_file
from build_source_ablation_labels import NEGATIVE_SOURCES, FileRecord, load_files
from src.ba133_cnn import (
    RawPartition,
    apply_channel_statistics,
    build_representation,
    representation_config_from_checkpoint,
)
from src.data_access_guards import assert_no_forbidden_path

SEED = 20260822
PEAK_SETS = {
    "three_peak": ("ba133_356kev", "na22_511kev", "cs137_662kev"),
    "all_ba": (
        "ba133_276kev",
        "ba133_303kev",
        "ba133_356kev",
        "ba133_384kev",
        "na22_511kev",
        "cs137_662kev",
    ),
}


def source_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "little")


def strict_internal_references(
    train_csv: Path, split_path: Path
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    with train_csv.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    split = np.load(split_path)
    indices = np.sort(split["internal_pair_indices"].astype(np.int64))
    references: set[tuple[str, int]] = set()
    events: list[dict[str, Any]] = []
    for pair_index in indices:
        row = pairs[int(pair_index)]
        for side, label in (("positive", 1), ("negative", 0)):
            event = {
                "hdf5": row[f"{side}_hdf5"],
                "row": int(row[f"{side}_row"]),
                "label": label,
                "peak_id": row["peak_id"],
                "source": row[f"{side}_source"],
                "energy_kev": float(row[f"{side}_energy_kev"]),
            }
            references.add((event["hdf5"], event["row"]))
            events.append(event)
    return references, events


def candidate_events(
    files: list[FileRecord],
    partition: str,
    peak: Any,
    label: int,
    excluded: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    sources = (peak.source,) if label == 1 else NEGATIVE_SOURCES[peak.source]
    half_width = (0.5 if label == 1 else 1.5) * peak.fwhm_kev
    low = peak.fitted_center_kev - half_width
    high = peak.fitted_center_kev + half_width
    events: list[dict[str, Any]] = []
    for record in files:
        if record.partition != partition or record.source not in sources:
            continue
        rows = np.flatnonzero((record.energies >= low) & (record.energies <= high))
        for row in rows.tolist():
            key = (record.hdf5, int(row))
            if key in excluded:
                continue
            events.append(
                {
                    "hdf5": record.hdf5,
                    "row": int(row),
                    "label": label,
                    "peak_id": peak.peak_id,
                    "source": record.source,
                    "energy_kev": float(record.energies[row]),
                }
            )
    return events


def build_balanced_selection(
    files: list[FileRecord],
    partition: str,
    excluded: set[tuple[str, int]],
    seed: int,
    peak_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    peaks = [peak for peak in TRAINING_PEAKS if peak.peak_id in peak_ids]
    if {peak.peak_id for peak in peaks} != set(peak_ids):
        raise ValueError(f"Unknown or missing peak IDs: {peak_ids}")
    for peak in peaks:
        positive = candidate_events(files, partition, peak, 1, excluded)
        negative = candidate_events(files, partition, peak, 0, excluded)
        rng = np.random.default_rng(source_seed(seed, f"{partition}:{peak.peak_id}"))
        target = min(len(positive), len(negative))
        positive_indices = rng.choice(len(positive), size=target, replace=False)
        selected.extend(positive[int(index)] for index in positive_indices)
        selected.extend(negative if len(negative) == target else [negative[int(index)] for index in rng.choice(len(negative), size=target, replace=False)])
        counts.append(
            {
                "peak_id": peak.peak_id,
                "positive_candidates": len(positive),
                "negative_candidates": len(negative),
                "selected_per_class": target,
                "selected_events": 2 * target,
            }
        )
    selected.sort(key=lambda event: (event["hdf5"], event["row"], event["label"]))
    keys = [(event["hdf5"], event["row"]) for event in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Event reuse detected in {partition} selection")
    return selected, counts


def write_cache(
    name: str,
    events: list[dict[str, Any]],
    output_dir: Path,
    config: Any,
    statistics: dict[str, list[float]],
) -> dict[str, Any]:
    values_path = output_dir / f"{name}_values.npy"
    values = np.lib.format.open_memmap(
        values_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(events), config.channel_count, config.window_length),
    )
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for destination, event in enumerate(events):
        by_file[event["hdf5"]].append((int(event["row"]), destination))
    qc = {"anchor_fallback_count": 0, "invalid_scale_count": 0}
    for file_index, (relative, indexed_rows) in enumerate(sorted(by_file.items()), 1):
        source_path = PROJECT_ROOT / relative
        assert_no_forbidden_path(source_path)
        indexed_rows.sort()
        with h5py.File(source_path, "r") as handle:
            for start in range(0, len(indexed_rows), 512):
                block = indexed_rows[start : start + 512]
                rows = np.asarray([item[0] for item in block], dtype=np.int64)
                destinations = np.asarray([item[1] for item in block], dtype=np.int64)
                waveforms = np.asarray(handle["waveform"][rows], dtype=np.float32)
                shaped = np.asarray(handle["shaped_energy_unit"][rows], dtype=np.float32)
                labels = np.asarray(
                    [events[int(index)]["label"] for index in destinations],
                    dtype=np.float32,
                )
                peaks = np.asarray(
                    [events[int(index)]["peak_id"] for index in destinations],
                    dtype="U32",
                )
                raw = RawPartition(
                    waveforms=waveforms,
                    shaped_energy=shaped,
                    labels=labels,
                    weights=np.ones(labels.size, dtype=np.float32),
                    peak_ids=peaks,
                )
                represented, block_qc = build_representation(raw, config)
                apply_channel_statistics(represented, statistics)
                values[destinations] = represented.astype(np.float16)
                qc["anchor_fallback_count"] += int(block_qc["anchor_fallback_count"])
                qc["invalid_scale_count"] += int(block_qc["invalid_scale_count"])
        if file_index % 10 == 0 or file_index == len(by_file):
            print(f"{name}: cached {file_index}/{len(by_file)} files", flush=True)
    values.flush()
    del values
    metadata_path = output_dir / f"{name}_metadata.npz"
    np.savez_compressed(
        metadata_path,
        label=np.asarray([event["label"] for event in events], dtype=np.int8),
        peak_id=np.asarray([event["peak_id"] for event in events], dtype="U32"),
        source=np.asarray([event["source"] for event in events], dtype="U16"),
        energy_kev=np.asarray([event["energy_kev"] for event in events], dtype=np.float32),
        hdf5=np.asarray([event["hdf5"] for event in events], dtype="U256"),
        source_row=np.asarray([event["row"] for event in events], dtype=np.int64),
    )
    return {
        "event_count": len(events),
        "values": values_path.relative_to(PROJECT_ROOT).as_posix(),
        "values_sha256": sha256_file(values_path),
        "metadata": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "qc": qc,
    }


def fit_streaming_channel_statistics(
    events: list[dict[str, Any]],
    config: Any,
) -> dict[str, list[float]]:
    """Fit population channel mean/std on training representations only."""
    channel_sum = np.zeros(config.channel_count, dtype=np.float64)
    channel_square_sum = np.zeros(config.channel_count, dtype=np.float64)
    value_count = 0
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for destination, event in enumerate(events):
        by_file[event["hdf5"]].append((int(event["row"]), destination))
    for file_index, (relative, indexed_rows) in enumerate(sorted(by_file.items()), 1):
        source_path = PROJECT_ROOT / relative
        assert_no_forbidden_path(source_path)
        indexed_rows.sort()
        with h5py.File(source_path, "r") as handle:
            for start in range(0, len(indexed_rows), 512):
                block = indexed_rows[start : start + 512]
                rows = np.asarray([item[0] for item in block], dtype=np.int64)
                waveforms = np.asarray(handle["waveform"][rows], dtype=np.float32)
                shaped = np.asarray(handle["shaped_energy_unit"][rows], dtype=np.float32)
                raw = RawPartition(
                    waveforms=waveforms,
                    shaped_energy=shaped,
                    labels=np.zeros(rows.size, dtype=np.float32),
                    weights=np.ones(rows.size, dtype=np.float32),
                    peak_ids=np.full(rows.size, "statistics", dtype="U32"),
                )
                represented, _qc = build_representation(raw, config)
                channel_sum += np.sum(represented, axis=(0, 2), dtype=np.float64)
                channel_square_sum += np.sum(
                    np.square(represented, dtype=np.float64),
                    axis=(0, 2),
                    dtype=np.float64,
                )
                value_count += represented.shape[0] * represented.shape[2]
        if file_index % 10 == 0 or file_index == len(by_file):
            print(
                f"train_statistics: processed {file_index}/{len(by_file)} files",
                flush=True,
            )
    means = channel_sum / value_count
    variances = channel_square_sum / value_count - np.square(means)
    stds = np.sqrt(np.maximum(variances, 0.0))
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("Invalid streaming channel statistics")
    return {"means": means.tolist(), "standard_deviations": stds.tolist()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/labels/architecture_pass_warn_20260815/file_partition_manifest.json",
    )
    parser.add_argument(
        "--strict-labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/three_peak_positive_polarity_20260820/compact_cnn_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--moving-average",
        type=int,
        default=None,
        help=(
            "Override the reference representation moving-average width and "
            "refit channel statistics on the selected training pool."
        ),
    )
    parser.add_argument(
        "--anchor",
        choices=("t10", "t50", "t90"),
        default=None,
        help=(
            "Override the reference shared charge/current window anchor and "
            "refit channel statistics on the selected training pool."
        ),
    )
    parser.add_argument(
        "--peak-set",
        choices=tuple(PEAK_SETS),
        default="three_peak",
        help="Training and relaxed-validation photopeak set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    for path in (
        args.partition_manifest,
        args.strict_labels_dir,
        args.reference_checkpoint,
        output_dir,
    ):
        assert_no_forbidden_path(Path(path).resolve())
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        args.reference_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    reference_moving_average = config.moving_average
    reference_anchor = config.anchor
    if args.moving_average is not None:
        if args.moving_average < 1:
            raise ValueError("moving-average must be positive")
        config = replace(
            config,
            moving_average=args.moving_average,
            name=config.name.replace(
                f"ma{reference_moving_average}", f"ma{args.moving_average}", 1
            ),
        )
    if args.anchor is not None:
        config = replace(
            config,
            anchor=args.anchor,
            name=config.name.replace(
                f"_{reference_anchor}_", f"_{args.anchor}_", 1
            ),
        )
    strict_dir = args.strict_labels_dir.resolve()
    excluded, strict_internal = strict_internal_references(
        strict_dir / "label_pairs_train.csv",
        strict_dir / "train_internal_split_indices.npz",
    )
    print("Loading train/validation energy arrays; test files remain unopened", flush=True)
    files = load_files(args.partition_manifest.resolve())
    peak_ids = PEAK_SETS[args.peak_set]
    train, train_counts = build_balanced_selection(
        files, "train", excluded, args.seed, peak_ids
    )
    validation, validation_counts = build_balanced_selection(
        files, "validation", set(), args.seed, peak_ids
    )
    train_keys = {(event["hdf5"], event["row"]) for event in train}
    strict_keys = {(event["hdf5"], event["row"]) for event in strict_internal}
    if train_keys & strict_keys:
        raise ValueError("Strict internal events leaked into relaxed training")
    representation_changed = (
        config.moving_average != reference_moving_average
        or config.anchor != reference_anchor
    )
    training_pool_changed = args.peak_set != "three_peak"
    if not representation_changed and not training_pool_changed:
        statistics = checkpoint["feature_statistics"]
        statistics_source = "reference_checkpoint"
    else:
        statistics = fit_streaming_channel_statistics(train, config)
        statistics_source = "selected_training_pool_only"
    cache = {
        "train": write_cache("train", train, output_dir, config, statistics),
        "relaxed_file_validation": write_cache(
            "relaxed_file_validation", validation, output_dir, config, statistics
        ),
        "strict_internal": write_cache(
            "strict_internal", strict_internal, output_dir, config, statistics
        ),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEVELOPMENT_ONLY_RELAXED_CONTINUUM_SHORTCUT_TEST_CACHE",
        "selection": {
            "peak_set": args.peak_set,
            "peak_ids": list(peak_ids),
            "positive_roi": "fitted centroid +/- 0.5 FWHM",
            "continuum_roi": "fitted centroid +/- 1.5 FWHM",
            "energy_bin_matching": False,
            "one_to_one_pairing": False,
            "class_balance": "independent random positive sampling to retain every unique continuum candidate within each peak",
            "strict_internal_excluded_from_training": True,
        },
        "train_counts": train_counts,
        "relaxed_file_validation_counts": validation_counts,
        "representation_config": config.as_dict(),
        "feature_statistics": statistics,
        "feature_statistics_source": statistics_source,
        "cache": cache,
        "input": {
            "partition_manifest_sha256": sha256_file(args.partition_manifest.resolve()),
            "strict_train_csv_sha256": sha256_file(strict_dir / "label_pairs_train.csv"),
            "strict_split_sha256": sha256_file(strict_dir / "train_internal_split_indices.npz"),
            "reference_checkpoint_sha256": sha256_file(args.reference_checkpoint.resolve()),
        },
        "held_out_file_partition_used_for_training": False,
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"train_counts": train_counts, "validation_counts": validation_counts, "cache": cache}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
