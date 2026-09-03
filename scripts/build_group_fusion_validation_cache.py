#!/usr/bin/env python3
"""Build an all-candidate MA20 validation cache for group-score fusion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_relaxed_continuum_roi_ds_cnn_cache import (
    PEAK_IDS,
    candidate_events,
    sha256_file,
    write_cache,
)
from scripts.build_relaxed_six_group_datasets import apply_catalogue, extract_features
from scripts.build_source_ablation_labels import load_files
from scripts.build_energy_matched_labels import TRAINING_PEAKS
from src.ba133_cnn import representation_config_from_checkpoint
from src.data_access_guards import assert_no_forbidden_path

MINIMUM_INDEX_LOW = 1000
MINIMUM_INDEX_HIGH = 1500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/labels/architecture_pass_warn_20260815/file_partition_manifest.json",
    )
    parser.add_argument(
        "--ma20-cache-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_ma20_20260822/cache_manifest.json",
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
        / "processed_data/group_fusion_natural_validation_ma20_20260822",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    partition_manifest = args.partition_manifest.resolve()
    ma20_manifest_path = args.ma20_cache_manifest.resolve()
    catalogue_path = args.catalogue_model.resolve()
    output_dir = args.output_dir.resolve()
    for path in (
        partition_manifest,
        ma20_manifest_path,
        catalogue_path,
        output_dir,
    ):
        assert_no_forbidden_path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ma20_manifest = json.loads(ma20_manifest_path.read_text(encoding="utf-8"))
    config = representation_config_from_checkpoint(
        ma20_manifest["representation_config"]
    )
    if config.moving_average != 20 or config.anchor != "t10":
        raise ValueError("Expected frozen MA20/t10 representation")
    statistics = ma20_manifest["feature_statistics"]
    files = load_files(partition_manifest)
    events: list[dict[str, Any]] = []
    candidate_counts: list[dict[str, Any]] = []
    for peak in TRAINING_PEAKS:
        if peak.peak_id not in PEAK_IDS:
            continue
        for label in (0, 1):
            selected = candidate_events(
                files=files,
                partition="validation",
                peak=peak,
                label=label,
                excluded=set(),
            )
            events.extend(selected)
            candidate_counts.append(
                {
                    "peak_id": peak.peak_id,
                    "label": label,
                    "class_name": "positive" if label == 1 else "negative",
                    "events": len(selected),
                }
            )
    events.sort(key=lambda event: (event["hdf5"], event["row"], event["label"]))
    keys = [(event["hdf5"], event["row"]) for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("Event reuse detected in all-candidate validation pool")

    cache_report = write_cache(
        "natural_file_validation", events, output_dir, config, statistics
    )
    metadata_path = output_dir / "natural_file_validation_metadata.npz"
    metadata = np.load(metadata_path)
    features, raw_minimum_index, morphology_valid, morphology_qc = extract_features(
        metadata, "natural_file_validation"
    )
    quality_valid = (
        (raw_minimum_index >= MINIMUM_INDEX_LOW)
        & (raw_minimum_index <= MINIMUM_INDEX_HIGH)
    )
    selected = morphology_valid & quality_valid
    catalogue = joblib.load(catalogue_path)
    probability, assignment = apply_catalogue(
        features, selected, catalogue
    )
    assignments_path = output_dir / "natural_file_validation_assignments.npz"
    np.savez_compressed(
        assignments_path,
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
            "label": metadata["label"][indices],
            "peak_id": metadata["peak_id"][indices].astype(str),
            "source": metadata["source"][indices].astype(str),
            "energy_kev": metadata["energy_kev"][indices],
            "hdf5": metadata["hdf5"][indices].astype(str),
            "source_row": metadata["source_row"][indices],
            "raw_minimum_index": raw_minimum_index[indices],
        }
    )
    manifest_path = output_dir / "natural_file_validation_group_manifest.csv.gz"
    frame.to_csv(manifest_path, index=False, compression="gzip")
    group_counts = (
        frame.groupby(["group", "label"])
        .size()
        .rename("events")
        .reset_index()
        .to_dict(orient="records")
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ALL_CANDIDATE_FILE_VALIDATION_CACHE_PREPARED",
        "selection": {
            "partition": "validation only",
            "positive_roi": "fitted centroid +/- 0.5 FWHM",
            "continuum_roi": "fitted centroid +/- 1.5 FWHM",
            "class_downsampling": False,
            "one_to_one_pairing": False,
            "all_unique_candidates_retained": True,
            "raw_negative_minimum_index_inclusive": [
                MINIMUM_INDEX_LOW,
                MINIMUM_INDEX_HIGH,
            ],
        },
        "candidate_counts_before_timing_cut": candidate_counts,
        "input_events": len(events),
        "selected_events": int(selected.sum()),
        "excluded_outside_minimum_index": int((~quality_valid).sum()),
        "excluded_morphology_invalid": int(
            np.count_nonzero(quality_valid & ~morphology_valid)
        ),
        "group_class_counts": group_counts,
        "cache": cache_report,
        "morphology_qc": morphology_qc,
        "input": {
            "partition_manifest_sha256": sha256_file(partition_manifest),
            "ma20_cache_manifest_sha256": sha256_file(ma20_manifest_path),
            "catalogue_model_sha256": sha256_file(catalogue_path),
        },
        "artifacts": {
            "assignments_sha256": sha256_file(assignments_path),
            "group_manifest_sha256": sha256_file(manifest_path),
        },
        "claim_boundary": (
            "Unbalanced source/ROI candidate distribution from development "
            "validation files; not measured deployment interaction truth."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "input_events": len(events),
                "selected_events": int(selected.sum()),
                "candidate_counts": candidate_counts,
                "group_class_counts": group_counts,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
