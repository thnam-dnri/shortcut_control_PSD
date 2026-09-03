#!/usr/bin/env python3
"""Create an explicitly merged morphology catalogue from frozen assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path


DEFAULT_GROUPS = ((1,), (2,), (3,), (4, 6), (5,), (7, 8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def merge_assignments(
    probability: np.ndarray,
    assignment: np.ndarray,
    valid: np.ndarray,
    groups: tuple[tuple[int, ...], ...] = DEFAULT_GROUPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum soft probabilities and exactly remap frozen hard assignments."""

    components = probability.shape[1]
    zero_based = tuple(tuple(group - 1 for group in merged) for merged in groups)
    flattened = sorted(group for merged in zero_based for group in merged)
    if flattened != list(range(components)):
        raise ValueError("Merged groups must cover each source group exactly once")
    merged_probability = np.column_stack(
        [np.sum(probability[:, merged], axis=1) for merged in zero_based]
    ).astype(np.float32)
    source_to_merged = np.empty(components, dtype=np.int16)
    for merged_index, source_indices in enumerate(zero_based):
        source_to_merged[list(source_indices)] = merged_index
    merged_assignment = np.full(assignment.shape, -1, dtype=np.int16)
    selected = np.asarray(valid, dtype=bool)
    merged_assignment[selected] = source_to_merged[assignment[selected]]
    return merged_probability, merged_assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-experiment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/morphology_catalogue_minimum_1000_1500_20260821",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/morphology_catalogue_minimum_1000_1500_20260821",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/morphology_catalogue_minimum_1000_1500_merged_20260822",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_dir = args.source_experiment_dir.resolve()
    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (source_dir, feature_dir, output_dir):
        assert_no_forbidden_path(path)
    catalogue_dir = output_dir / "catalogue"
    if catalogue_dir.exists() and any(catalogue_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(catalogue_dir)
    catalogue_dir.mkdir(parents=True, exist_ok=True)

    source_catalogue = source_dir / "catalogue"
    source_report_path = source_catalogue / "catalogue_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if int(source_report["selected_components"]) != 8:
        raise ValueError("The requested merge requires an eight-group source catalogue")

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for partition in ("fit", "internal"):
        source_path = source_catalogue / f"{partition}_assignments.npz"
        source = np.load(source_path)
        probability, assignment = merge_assignments(
            source["probability"], source["assignment"], source["valid"]
        )
        destination = catalogue_dir / f"{partition}_assignments.npz"
        np.savez_compressed(
            destination,
            probability=probability,
            assignment=assignment,
            valid=source["valid"],
            original_event_index=source["original_event_index"],
        )
        outputs[partition] = {
            "assignment": assignment,
            "valid": source["valid"].astype(bool),
        }

    fit_features = np.load(feature_dir / "fit_features.npz")
    discovery = fit_features["is_discovery"].astype(bool) & outputs["fit"]["valid"]
    internal_valid = outputs["internal"]["valid"]
    discovery_counts = np.bincount(
        outputs["fit"]["assignment"][discovery], minlength=len(DEFAULT_GROUPS)
    )
    internal_counts = np.bincount(
        outputs["internal"]["assignment"][internal_valid],
        minlength=len(DEFAULT_GROUPS),
    )
    discovery_fraction = discovery_counts / discovery_counts.sum()
    internal_fraction = internal_counts / internal_counts.sum()
    divergence = float(
        jensenshannon(discovery_fraction, internal_fraction, base=2.0) ** 2
    )
    mapping = {
        str(index + 1): list(source_groups)
        for index, source_groups in enumerate(DEFAULT_GROUPS)
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "USER_DIRECTED_GROUP_MERGE",
        "selection_is_label_blind": False,
        "selected_components": len(DEFAULT_GROUPS),
        "group_mapping_new_to_source": mapping,
        "assignment_rule": (
            "Hard assignments are mapped exactly; merged posterior columns are "
            "the sums of their source posterior columns."
        ),
        "component_fraction": {
            "discovery": discovery_fraction.tolist(),
            "internal": internal_fraction.tolist(),
            "jensen_shannon_divergence": divergence,
        },
        "input": {
            "source_catalogue_report": source_report_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "source_catalogue_report_sha256": sha256_file(source_report_path),
        },
        "claim_boundary": (
            "This is a user-directed aggregation of the frozen eight-group "
            "catalogue, not a new unsupervised model-selection result."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (catalogue_dir / "catalogue_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
