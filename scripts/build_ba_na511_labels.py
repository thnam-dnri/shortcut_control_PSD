#!/usr/bin/env python3
"""Combine frozen Ba-all-peak and Na-511 energy-matched pair manifests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_o2_late_fusion import sha256_file  # noqa: E402

PARTITIONS = ("train", "validation")
SEED = 20260813


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/ba_all_na511_positive",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    label_root = args.label_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_details: dict[str, object] = {}
    input_hashes: dict[str, dict[str, str]] = {}

    for partition_index, partition in enumerate(PARTITIONS):
        rows: list[dict[str, str]] = []
        fieldnames: list[str] | None = None
        input_hashes[partition] = {}
        for source in ("ba133", "na22"):
            path = label_root / f"{source}_positive/label_pairs_{partition}.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    raise ValueError(f"Missing CSV header: {path}")
                fieldnames = reader.fieldnames
                rows.extend(reader)
            input_hashes[partition][source] = sha256_file(path)
        if fieldnames is None:
            raise RuntimeError("No input rows")
        peak_counts = Counter(row["peak_id"] for row in rows)
        total_pairs = len(rows)
        for row in rows:
            row["source_weight"] = str(
                total_pairs / (len(peak_counts) * peak_counts[row["peak_id"]])
            )
        rng = np.random.default_rng(args.seed + partition_index)
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
            raise ValueError(f"Event reuse in combined {partition} pairs")
        energy_differences = [
            abs(float(row["positive_energy_kev"]) - float(row["negative_energy_kev"]))
            for row in rows
        ]
        partition_details[partition] = {
            "pair_count": total_pairs,
            "event_count": 2 * total_pairs,
            "peak_counts": dict(peak_counts),
            "negative_source_counts": dict(Counter(row["negative_source"] for row in rows)),
            "maximum_absolute_energy_difference_kev": max(energy_differences),
            "event_unique": True,
            "output_csv_sha256": sha256_file(output_csv),
        }

    source_partition_manifest = label_root / "ba133_positive/file_partition_manifest.json"
    output_partition_manifest = output_dir / "file_partition_manifest.json"
    shutil.copyfile(source_partition_manifest, output_partition_manifest)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_domains": {
            "ba133": ["ba133_276kev", "ba133_303kev", "ba133_356kev", "ba133_384kev"],
            "na22": ["na22_511kev"],
        },
        "negative_policy": {
            "ba133": ["co60", "cs137", "na22"],
            "na22": ["co60"],
        },
        "construction": "Concatenated frozen source-ablation pairs without rematching.",
        "matching": "One-to-one without replacement in 0.5-keV bins.",
        "sample_weighting": "Inverse peak frequency; each of five peaks has equal total weight per partition.",
        "seed": args.seed,
        "input_csv_hashes": input_hashes,
        "partitions": partition_details,
        "file_partition_manifest_sha256": sha256_file(output_partition_manifest),
        "test_partition_used": False,
    }
    (output_dir / "label_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(partition_details, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
