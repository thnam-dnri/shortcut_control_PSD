#!/usr/bin/env python3
"""Assign relaxed three-peak events to the frozen merged six-group catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_morphology_catalogue_groups import (
    DEFAULT_GROUPS,
    merge_assignments,
)
from src.data_access_guards import assert_no_forbidden_path
from src.waveform_morphology import MorphologyConfig, extract_morphology_features

POPULATIONS = ("train", "relaxed_file_validation", "strict_internal")
MINIMUM_INDEX_LOW = 1000
MINIMUM_INDEX_HIGH = 1500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_features(
    metadata: Any,
    population: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    count = metadata["label"].size
    features = np.full((count, 8), np.nan, dtype=np.float32)
    raw_minimum_index = np.full(count, -1, dtype=np.int16)
    morphology_valid = np.zeros(count, dtype=bool)
    fallback_count = 0
    invalid_scale_count = 0
    hdf5_values = metadata["hdf5"].astype(str)
    source_rows = metadata["source_row"].astype(np.int64)
    unique_files = sorted(np.unique(hdf5_values))
    config = MorphologyConfig()
    for file_index, relative in enumerate(unique_files, 1):
        destinations = np.flatnonzero(hdf5_values == relative)
        order = np.argsort(source_rows[destinations])
        destinations = destinations[order]
        rows = source_rows[destinations]
        source_path = PROJECT_ROOT / relative
        assert_no_forbidden_path(source_path)
        with h5py.File(source_path, "r") as handle:
            for start in range(0, rows.size, 512):
                stop = min(start + 512, rows.size)
                block_rows = rows[start:stop]
                block_destinations = destinations[start:stop]
                waveforms = np.asarray(
                    handle["waveform"][block_rows], dtype=np.float32
                )
                raw_minimum_index[block_destinations] = np.argmin(
                    waveforms, axis=1
                ).astype(np.int16)
                block_features, qc = extract_morphology_features(
                    waveforms, config
                )
                features[block_destinations] = block_features
                morphology_valid[block_destinations] = qc["valid"]
                fallback_count += int(qc["anchor_fallback_count"])
                invalid_scale_count += int(qc["invalid_scale_count"])
        if file_index % 10 == 0 or file_index == len(unique_files):
            print(
                f"{population}: processed {file_index}/{len(unique_files)} files",
                flush=True,
            )
    return features, raw_minimum_index, morphology_valid, {
        "anchor_fallback_count": fallback_count,
        "invalid_scale_count": invalid_scale_count,
    }


def apply_catalogue(
    features: np.ndarray,
    selected: np.ndarray,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base_probability = np.full((features.shape[0], 8), np.nan, dtype=np.float32)
    standardized = (
        features[selected].astype(np.float64) - model["median"]
    ) / model["scale"]
    projected = model["pca"].transform(standardized)
    base_probability[selected] = model["gmm"].predict_proba(projected)[
        :, model["component_order"]
    ].astype(np.float32)
    base_assignment = np.full(features.shape[0], -1, dtype=np.int16)
    base_assignment[selected] = np.argmax(
        base_probability[selected], axis=1
    ).astype(np.int16)
    merged_probability = np.full((features.shape[0], 6), np.nan, dtype=np.float32)
    merged_assignment = np.full(features.shape[0], -1, dtype=np.int16)
    probability, assignment = merge_assignments(
        base_probability[selected],
        base_assignment[selected],
        np.ones(np.count_nonzero(selected), dtype=bool),
        DEFAULT_GROUPS,
    )
    merged_probability[selected] = probability
    merged_assignment[selected] = assignment
    return merged_probability, merged_assignment


def sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = ["group", "peak_id", "label"]
    counts = frame.groupby(keys)["cache_index"].transform("count").to_numpy()
    weights = 1.0 / counts.astype(np.float64)
    return (weights / np.mean(weights)).astype(np.float32)


def write_population(
    population: str,
    cache_dir: Path,
    output_dir: Path,
    model: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = cache_dir / f"{population}_metadata.npz"
    metadata = np.load(metadata_path)
    features, raw_minimum_index, morphology_valid, qc = extract_features(
        metadata, population
    )
    quality_valid = (
        (raw_minimum_index >= MINIMUM_INDEX_LOW)
        & (raw_minimum_index <= MINIMUM_INDEX_HIGH)
    )
    selected = morphology_valid & quality_valid
    probability, assignment = apply_catalogue(features, selected, model)
    np.savez_compressed(
        output_dir / f"{population}_assignments.npz",
        features=features,
        probability=probability,
        assignment=assignment,
        selected=selected,
        morphology_valid=morphology_valid,
        quality_valid=quality_valid,
        raw_minimum_index=raw_minimum_index,
    )
    indices = np.flatnonzero(selected)
    frame = pd.DataFrame(
        {
            "cache_index": indices,
            "group": assignment[indices] + 1,
            "label": metadata["label"][indices].astype(np.int8),
            "class_name": np.where(
                metadata["label"][indices] == 1, "positive", "negative"
            ),
            "peak_id": metadata["peak_id"][indices].astype(str),
            "source": metadata["source"][indices].astype(str),
            "energy_kev": metadata["energy_kev"][indices],
            "hdf5": metadata["hdf5"][indices].astype(str),
            "source_row": metadata["source_row"][indices],
            "raw_minimum_index": raw_minimum_index[indices],
        }
    )
    if population == "train":
        frame["sampling_weight"] = sampling_weights(frame)
    frame.to_csv(
        output_dir / f"{population}_group_manifest.csv.gz",
        index=False,
        compression="gzip",
    )
    for group in range(1, 7):
        group_frame = frame[frame["group"] == group]
        payload: dict[str, Any] = {
            "cache_index": group_frame["cache_index"].to_numpy(np.int64),
            "label": group_frame["label"].to_numpy(np.int8),
            "peak_id": group_frame["peak_id"].to_numpy(dtype="U32"),
            "source": group_frame["source"].to_numpy(dtype="U16"),
        }
        if population == "train":
            payload["sampling_weight"] = group_frame["sampling_weight"].to_numpy(
                np.float32
            )
        np.savez_compressed(
            output_dir / f"{population}_group_{group}_indices.npz", **payload
        )
    return frame, {
        "input_events": int(metadata["label"].size),
        "selected_events": int(selected.sum()),
        "excluded_outside_minimum_index": int((~quality_valid).sum()),
        "excluded_morphology_invalid": int(
            np.count_nonzero(quality_valid & ~morphology_valid)
        ),
        "qc": qc,
        "metadata_sha256": sha256_file(metadata_path),
        "assignments_sha256": sha256_file(
            output_dir / f"{population}_assignments.npz"
        ),
        "manifest_sha256": sha256_file(
            output_dir / f"{population}_group_manifest.csv.gz"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
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
        / "processed_data/relaxed_continuum_six_group_20260822",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    catalogue_path = args.catalogue_model.resolve()
    output_dir = args.output_dir.resolve()
    for path in (cache_dir, catalogue_path, output_dir):
        assert_no_forbidden_path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(catalogue_path)
    if len(model["component_order"]) != 8:
        raise ValueError("Expected the frozen eight-component source catalogue")

    frames: dict[str, pd.DataFrame] = {}
    population_reports: dict[str, Any] = {}
    for population in POPULATIONS:
        frames[population], population_reports[population] = write_population(
            population, cache_dir, output_dir, model
        )
    count_rows: list[dict[str, Any]] = []
    for population, frame in frames.items():
        counts = (
            frame.groupby(["group", "label", "class_name", "peak_id", "source"])
            .size()
            .rename("events")
            .reset_index()
        )
        counts.insert(0, "population", population)
        count_rows.extend(counts.to_dict(orient="records"))
    pd.DataFrame(count_rows).to_csv(
        output_dir / "counts_by_group_class_peak_source.csv", index=False
    )
    feasibility_rows: list[dict[str, Any]] = []
    for group in range(1, 7):
        train = frames["train"][frames["train"]["group"] == group]
        validation = frames["relaxed_file_validation"][
            frames["relaxed_file_validation"]["group"] == group
        ]
        train_positive = int(np.count_nonzero(train["label"] == 1))
        train_negative = int(np.count_nonzero(train["label"] == 0))
        validation_positive = int(np.count_nonzero(validation["label"] == 1))
        validation_negative = int(np.count_nonzero(validation["label"] == 0))
        feasibility_rows.append(
            {
                "group": group,
                "train_positive": train_positive,
                "train_negative": train_negative,
                "train_pair_capacity": min(train_positive, train_negative),
                "train_target_10000_pairs_met": min(
                    train_positive, train_negative
                )
                >= 10000,
                "file_validation_positive": validation_positive,
                "file_validation_negative": validation_negative,
                "file_validation_pair_capacity": min(
                    validation_positive, validation_negative
                ),
                "file_validation_target_5000_pairs_met": min(
                    validation_positive, validation_negative
                )
                >= 5000,
            }
        )
    feasibility = pd.DataFrame(feasibility_rows)
    feasibility.to_csv(output_dir / "group_feasibility.csv", index=False)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SIX_GROUP_DATASET_PREPARED_COUNTS_ONLY_NO_MODEL_TRAINING",
        "group_mapping_new_to_source": {
            str(index + 1): list(groups)
            for index, groups in enumerate(DEFAULT_GROUPS)
        },
        "selection": {
            "raw_negative_minimum_index_inclusive": [
                MINIMUM_INDEX_LOW,
                MINIMUM_INDEX_HIGH,
            ],
            "frozen_catalogue": True,
            "all_unique_passing_events_retained": True,
            "training_balance": (
                "Inverse-frequency weights for every group/peak/class stratum; "
                "no event downsampling or duplicated stored rows."
            ),
        },
        "populations": population_reports,
        "feasibility": feasibility_rows,
        "input": {
            "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
            "catalogue_model_sha256": sha256_file(catalogue_path),
        },
        "model_training_performed": False,
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(feasibility.to_string(index=False), flush=True)
    print(json.dumps(population_reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
