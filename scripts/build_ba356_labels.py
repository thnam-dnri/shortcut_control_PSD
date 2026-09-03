#!/usr/bin/env python3
"""Build Ba-133 356-keV-only labels from frozen source-ablation pairs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_o2_late_fusion import sha256_file  # noqa: E402

PARTITIONS = ("train", "validation")
PEAK_ID = "ba133_356kev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/ba356_positive",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions: dict[str, object] = {}
    for partition in PARTITIONS:
        input_csv = input_dir / f"label_pairs_{partition}.csv"
        output_csv = output_dir / f"label_pairs_{partition}.csv"
        with input_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Missing CSV header: {input_csv}")
            rows = [row for row in reader if row["peak_id"] == PEAK_ID]
        if not rows:
            raise ValueError(f"No {PEAK_ID} rows in {input_csv}")
        for index, row in enumerate(rows):
            row["pair_id"] = f"{partition}_{index:08d}"
            row["source_weight"] = "1.0"
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        energy_differences = [
            abs(float(row["positive_energy_kev"]) - float(row["negative_energy_kev"]))
            for row in rows
        ]
        partitions[partition] = {
            "pair_count": len(rows),
            "event_count": 2 * len(rows),
            "negative_source_counts": dict(Counter(row["negative_source"] for row in rows)),
            "maximum_absolute_energy_difference_kev": max(energy_differences),
            "input_csv": input_csv.relative_to(PROJECT_ROOT).as_posix(),
            "input_csv_sha256": sha256_file(input_csv),
            "output_csv_sha256": sha256_file(output_csv),
        }

    source_partition_manifest = input_dir / "file_partition_manifest.json"
    output_partition_manifest = output_dir / "file_partition_manifest.json"
    shutil.copyfile(source_partition_manifest, output_partition_manifest)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_source": "ba133",
        "positive_peak_id": PEAK_ID,
        "positive_peak_energy_kev": 356.0,
        "negative_sources": ["co60", "cs137", "na22"],
        "construction": "Filtered without rematching from frozen Ba-positive source-ablation pairs.",
        "matching": "One-to-one without replacement in 0.5-keV bins.",
        "same_source_negative_excluded": True,
        "partitions": partitions,
        "file_partition_manifest_sha256": sha256_file(output_partition_manifest),
        "test_partition_used": False,
    }
    (output_dir / "label_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(partitions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
