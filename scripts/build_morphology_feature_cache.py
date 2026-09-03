#!/usr/bin/env python3
"""Build fit/internal MA20 morphology descriptors from development data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import load_raw_partition, read_event_references
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path
from src.waveform_morphology import (
    FEATURE_NAMES,
    MorphologyConfig,
    extract_morphology_features,
)

SEED = 20260821


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pairs, 2 * pairs + 1)).reshape(-1)


def event_metadata(rows: pd.DataFrame, registry: dict[str, str]) -> dict[str, np.ndarray]:
    values: dict[str, list[object]] = {
        key: []
        for key in ("source", "hdf5", "source_row", "energy_kev", "qc_status", "session")
    }
    for row in rows.to_dict(orient="records"):
        for side in ("positive", "negative"):
            path = str(row[f"{side}_hdf5"])
            values["source"].append(str(row[f"{side}_source"]))
            values["hdf5"].append(path)
            values["source_row"].append(int(row[f"{side}_row"]))
            values["energy_kev"].append(float(row[f"{side}_energy_kev"]))
            values["qc_status"].append(str(row[f"{side}_qc_status"]))
            values["session"].append(registry.get(path, "UNKNOWN"))
    return {
        "source": np.asarray(values["source"], dtype="U16"),
        "hdf5": np.asarray(values["hdf5"], dtype="U256"),
        "source_row": np.asarray(values["source_row"], dtype=np.int64),
        "energy_kev": np.asarray(values["energy_kev"], dtype=np.float32),
        "qc_status": np.asarray(values["qc_status"], dtype="U16"),
        "session": np.asarray(values["session"], dtype="U96"),
    }


def store_metadata(
    csv_path: Path,
    event_store_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    records = read_event_references(csv_path)
    required = {(record[0], record[1]) for record in records}
    lookup: dict[tuple[str, int], int] = {}
    with (event_store_dir / "event_lookup_train.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            if key in required:
                lookup[key] = int(row["store_index"])
    rows = np.asarray([lookup[(record[0], record[1])] for record in records])
    order = np.argsort(rows)
    trigger = np.empty(rows.size, dtype=np.float64)
    baseline_noise = np.empty(rows.size, dtype=np.float32)
    with h5py.File(event_store_dir / "train_events.h5", "r") as handle:
        for start in range(0, rows.size, 1024):
            stop = min(start + 1024, rows.size)
            selected = rows[order[start:stop]]
            destination = order[start:stop]
            trigger[destination] = handle["trigger_time_s"][selected]
            noise = handle["noise_rms_adc"][selected]
            baseline_noise[destination] = np.mean(noise, axis=1)
    return trigger, baseline_noise


def select_discovery_pairs(
    rows: pd.DataFrame,
    fit_pairs: np.ndarray,
    seed: int,
    eligible_events: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    fit_rows = rows.iloc[fit_pairs]
    for peak in sorted(fit_rows["peak_id"].unique()):
        candidates = fit_pairs[fit_rows["peak_id"].to_numpy() == peak]
        if eligible_events is not None:
            candidates = candidates[
                eligible_events[2 * candidates]
                & eligible_events[2 * candidates + 1]
            ]
        if candidates.size < 5000:
            raise ValueError(
                f"Peak {peak} has only {candidates.size} eligible fit pairs"
            )
        selected.extend(rng.choice(candidates, size=5000, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def save_partition(
    path: Path,
    indices: np.ndarray,
    discovery_events: set[int],
    features: np.ndarray,
    valid: np.ndarray,
    peak_count: np.ndarray,
    raw_labels: np.ndarray,
    raw_peak_ids: np.ndarray,
    metadata: dict[str, np.ndarray],
    trigger_time: np.ndarray,
    baseline_noise: np.ndarray,
    raw_minimum_index: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        features=features[indices],
        valid=valid[indices],
        detected_peak_count=peak_count[indices],
        original_event_index=indices,
        pair_index=indices // 2,
        pair_member=indices % 2,
        is_discovery=np.asarray([int(i) in discovery_events for i in indices]),
        label=raw_labels[indices].astype(np.int8),
        peak_id=raw_peak_ids[indices],
        trigger_time_s=trigger_time[indices],
        baseline_noise_rms_adc=baseline_noise[indices],
        raw_minimum_index=raw_minimum_index[indices],
        **{key: value[indices] for key, value in metadata.items()},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/morphology_catalogue_20260821",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-minimum-index-low", type=int)
    parser.add_argument("--raw-minimum-index-high", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    registry_path = PROJECT_ROOT / "outputs/protocol/domain_registry.csv"
    for path in (train_csv, split_path, event_store_dir, registry_path):
        if not path.exists():
            raise FileNotFoundError(path)
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(train_csv)
    split = np.load(split_path)
    fit_pairs = np.sort(split["fit_pair_indices"].astype(np.int64))
    internal_pairs = np.sort(split["internal_pair_indices"].astype(np.int64))
    registry_table = pd.read_csv(registry_path)
    registry = dict(
        zip(registry_table["hdf5"], registry_table["canonical_session_id"])
    )
    metadata = event_metadata(rows, registry)

    print("Loading 97,474 train-manifest waveforms ...", flush=True)
    raw = load_raw_partition(train_csv, event_store_dir)
    raw_minimum_index = np.argmin(raw.waveforms, axis=1).astype(np.int16)
    if (args.raw_minimum_index_low is None) != (
        args.raw_minimum_index_high is None
    ):
        raise ValueError("Both raw-minimum-index bounds must be provided together")
    if args.raw_minimum_index_low is None:
        quality_valid = np.ones(raw_minimum_index.size, dtype=bool)
        minimum_index_range = None
    else:
        if not 0 <= args.raw_minimum_index_low <= args.raw_minimum_index_high < raw.waveforms.shape[1]:
            raise ValueError("Invalid raw-minimum-index range")
        quality_valid = (
            (raw_minimum_index >= args.raw_minimum_index_low)
            & (raw_minimum_index <= args.raw_minimum_index_high)
        )
        minimum_index_range = [
            args.raw_minimum_index_low,
            args.raw_minimum_index_high,
        ]
    discovery_pairs = select_discovery_pairs(
        rows, fit_pairs, args.seed, quality_valid
    )
    discovery_event_set = set(event_indices(discovery_pairs).tolist())
    print("Loading nuisance metadata ...", flush=True)
    trigger_time, baseline_noise = store_metadata(train_csv, event_store_dir)
    config = MorphologyConfig()
    print("Extracting MA20 morphology descriptors ...", flush=True)
    features, qc = extract_morphology_features(raw.waveforms, config)
    valid = np.asarray(qc["valid"], dtype=bool)
    peak_count = np.asarray(qc["detected_peak_count"], dtype=np.int16)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)
    retained_fit_events = fit_events[quality_valid[fit_events]]
    retained_internal_events = internal_events[quality_valid[internal_events]]
    save_partition(
        output_dir / "fit_features.npz",
        retained_fit_events,
        discovery_event_set,
        features,
        valid,
        peak_count,
        raw.labels,
        raw.peak_ids,
        metadata,
        trigger_time,
        baseline_noise,
        raw_minimum_index,
    )
    save_partition(
        output_dir / "internal_features.npz",
        retained_internal_events,
        set(),
        features,
        valid,
        peak_count,
        raw.labels,
        raw.peak_ids,
        metadata,
        trigger_time,
        baseline_noise,
        raw_minimum_index,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_ENGINEERING_MORPHOLOGY_FEATURE_CACHE",
        "config": config.as_dict(),
        "feature_names": list(FEATURE_NAMES),
        "seed": args.seed,
        "waveform_quality_selection": {
            "raw_negative_polarity_minimum_index_inclusive": minimum_index_range,
        },
        "input": {
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "split_path": split_path.relative_to(PROJECT_ROOT).as_posix(),
            "split_sha256": sha256_file(split_path),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
            "domain_registry_sha256": sha256_file(registry_path),
        },
        "counts": {
            "fit_events": int(fit_events.size),
            "internal_events": int(internal_events.size),
            "retained_fit_events": int(retained_fit_events.size),
            "retained_internal_events": int(retained_internal_events.size),
            "excluded_fit_events": int(fit_events.size - retained_fit_events.size),
            "excluded_internal_events": int(
                internal_events.size - retained_internal_events.size
            ),
            "discovery_events": int(len(discovery_event_set)),
            "discovery_pairs_per_peak": 5000,
            "valid_fit_events": int(np.count_nonzero(valid[retained_fit_events])),
            "valid_internal_events": int(
                np.count_nonzero(valid[retained_internal_events])
            ),
            "valid_discovery_events": int(
                np.count_nonzero(valid[np.asarray(sorted(discovery_event_set))])
            ),
            "anchor_fallback_count_all_train": int(qc["anchor_fallback_count"]),
            "invalid_scale_count_all_train": int(qc["invalid_scale_count"]),
        },
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
