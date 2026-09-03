#!/usr/bin/env python3
"""Build frozen-development labels for controlled positive-peak combinations.

Rows are filtered from the existing source-ablation manifests and concatenated
without rematching or changing file partitions. Source weights are recomputed so
each retained positive peak has equal total weight. Locked-test and external
manifests are not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402
from train_o2_late_fusion import sha256_file  # noqa: E402


PARTITIONS = ("train", "validation")
SEED = 20260816
BA_PEAKS = (
    "ba133_276kev",
    "ba133_303kev",
    "ba133_356kev",
    "ba133_384kev",
)
NA_PEAKS = ("na22_511kev",)
COMBINATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "ba_low": {"ba133": ("ba133_276kev", "ba133_303kev")},
    "ba_high": {"ba133": ("ba133_356kev", "ba133_384kev")},
    "ba_low_na511": {
        "ba133": ("ba133_276kev", "ba133_303kev"),
        "na22": NA_PEAKS,
    },
    "ba_high_na511": {
        "ba133": ("ba133_356kev", "ba133_384kev"),
        "na22": NA_PEAKS,
    },
    "ba_all": {"ba133": BA_PEAKS},
    "ba_all_na511": {"ba133": BA_PEAKS, "na22": NA_PEAKS},
}


def display_path(path: Path) -> str:
    """Return a concise project-relative path, or an absolute test path."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combination",
        choices=tuple(COMBINATIONS) + ("all",),
        default="all",
    )
    parser.add_argument(
        "--source-label-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/peak_combinations",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    assert_development_csv(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader), reader.fieldnames


def build_combination(
    name: str,
    specification: dict[str, tuple[str, ...]],
    source_label_root: Path,
    output_root: Path,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_root / name
    assert_no_forbidden_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_hashes: dict[str, dict[str, str]] = {}
    partitions: dict[str, Any] = {}
    input_peaks: dict[str, list[str]] = {}
    fieldnames: list[str] | None = None
    for partition_index, partition in enumerate(PARTITIONS):
        rows: list[dict[str, str]] = []
        input_hashes[partition] = {}
        input_peaks[partition] = []
        for source, peaks in specification.items():
            input_csv = source_label_root / f"{source}_positive/label_pairs_{partition}.csv"
            assert_no_forbidden_path(input_csv)
            source_rows, source_fieldnames = read_rows(input_csv)
            if fieldnames is None:
                fieldnames = source_fieldnames
            if source_fieldnames != fieldnames:
                raise ValueError(f"Mismatched label fields in {input_csv}")
            selected = [row for row in source_rows if row["peak_id"] in peaks]
            if not selected:
                raise ValueError(f"No selected rows for {source} in {partition}")
            rows.extend(selected)
            input_peaks[partition].extend(peaks)
            input_hashes[partition][source] = sha256_file(input_csv)

        if fieldnames is None or not rows:
            raise ValueError(f"No rows selected for {name}/{partition}")
        peak_counts = Counter(row["peak_id"] for row in rows)
        total_pairs = len(rows)
        for row in rows:
            row["source_weight"] = str(
                total_pairs / (len(peak_counts) * peak_counts[row["peak_id"]])
            )
        rng = np.random.default_rng(seed + partition_index)
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["pair_id"] = f"{partition}_{index:08d}"

        output_csv = output_dir / f"label_pairs_{partition}.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        event_keys = [
            key
            for row in rows
            for key in (
                (row["positive_hdf5"], int(row["positive_row"])),
                (row["negative_hdf5"], int(row["negative_row"])),
            )
        ]
        if len(set(event_keys)) != len(event_keys):
            raise ValueError(f"Event reuse in {name}/{partition}")
        energy_differences = [
            abs(float(row["positive_energy_kev"]) - float(row["negative_energy_kev"]))
            for row in rows
        ]
        partitions[partition] = {
            "pair_count": total_pairs,
            "event_count": 2 * total_pairs,
            "peak_counts": dict(peak_counts),
            "negative_source_counts": dict(
                Counter(row["negative_source"] for row in rows)
            ),
            "maximum_absolute_energy_difference_kev": max(energy_differences),
            "event_unique": True,
            "output_csv_sha256": sha256_file(output_csv),
        }

    source_partition_manifest = (
        source_label_root / "ba133_positive/file_partition_manifest.json"
    )
    output_partition_manifest = output_dir / "file_partition_manifest.json"
    shutil.copyfile(source_partition_manifest, output_partition_manifest)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "combination": name,
        "positive_domains": specification,
        "negative_policy": "Inherited from frozen source-ablation rows; no rematching.",
        "construction": "Filtered and concatenated frozen source-ablation pairs without replacement.",
        "matching": "One-to-one without replacement in 0.5-keV bins inherited from source-ablation manifests.",
        "sample_weighting": "Inverse peak frequency; every retained peak has equal total weight per partition.",
        "seed": seed,
        "input_csv_hashes": input_hashes,
        "partitions": partitions,
        "file_partition_manifest_sha256": sha256_file(output_partition_manifest),
        "test_partition_used": False,
        "external_data_used": False,
    }
    (output_dir / "label_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": display_path(output_dir),
        "positive_domains": specification,
        "partitions": partitions,
        "test_partition_used": False,
    }


def main() -> int:
    args = build_parser().parse_args()
    source_label_root = args.source_label_root.resolve()
    output_root = args.output_root.resolve()
    assert_no_forbidden_path(source_label_root)
    assert_no_forbidden_path(output_root)
    names = tuple(COMBINATIONS) if args.combination == "all" else (args.combination,)
    results = [
        build_combination(
            name,
            COMBINATIONS[name],
            source_label_root,
            output_root,
            args.seed,
            args.overwrite,
        )
        for name in names
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
