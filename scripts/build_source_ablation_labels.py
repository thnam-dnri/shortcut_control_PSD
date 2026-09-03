#!/usr/bin/env python3
"""Build single-positive-source labels with physically valid mixed continua.

Three independent internal ablations are produced: Ba-133 peaks only, Na-22
511-keV only, and Cs-137 662-keV only.  Negative events are energy matched in
0.5-keV bins and drawn without replacement from continuum-capable Co-60, Cs-137,
and Na-22 sources.  The positive source itself is never used as a negative source
because events in its photopeak ROI are not truth-labeled Compton interactions.

Only the existing train and validation file partitions are read.  Test files remain
locked and no test pair CSV is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_energy_matched_labels import (  # noqa: E402
    BIN_WIDTH_KEV,
    CSV_FIELDS,
    TRAINING_PEAKS,
    PeakDefinition,
    sha256_file,
)

PARTITIONS = ("train", "validation")
SEED = 20260811
EXPERIMENT_SOURCES = ("ba133", "na22", "cs137")
NEGATIVE_SOURCES = {
    "ba133": ("co60", "cs137", "na22"),
    "na22": ("co60",),
    "cs137": ("co60", "na22"),
}
CONTINUUM_JUSTIFICATION = {
    "ba133": (
        "Co-60, Cs-137, and Na-22 have physically supported Compton continua "
        "across 276-384 keV."
    ),
    "na22": (
        "Only Co-60 is used at 511 keV: Cs-137's 477.3-keV Compton edge is below "
        "the ROI, and same-source Na-22 events in the ROI are photopeak-contaminated."
    ),
    "cs137": (
        "Co-60 and the 1274.5-keV Na-22 gamma have continua at 662 keV; same-source "
        "Cs-137 ROI events are photopeak-contaminated."
    ),
}


@dataclass
class FileRecord:
    source: str
    hdf5: str
    partition: str
    qc_status: str
    hdf5_sha256: str
    energies: np.ndarray
    event_ids: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "little")


def load_files(partition_manifest: Path) -> list[FileRecord]:
    manifest = json.loads(partition_manifest.read_text(encoding="utf-8"))
    records: list[FileRecord] = []
    for entry in manifest["files"]:
        if (
            entry["partition"] not in PARTITIONS
            or entry["processing_status"] != "OK"
            or not entry["complete_input"]
        ):
            continue
        path = PROJECT_ROOT / entry["hdf5"]
        with h5py.File(path, "r") as handle:
            dset_name = "corrected_energy_kev" if "corrected_energy_kev" in handle else "reconstructed_energy_kev"
            energies = np.asarray(handle[dset_name], dtype=np.float64)
            event_ids = np.asarray(handle["event_id"])
        if energies.size != int(entry["accepted_entries"]):
            raise ValueError(f"Accepted-entry mismatch: {path}")
        records.append(
            FileRecord(
                source=entry["source"],
                hdf5=entry["hdf5"],
                partition=entry["partition"],
                qc_status=entry["qc_status"],
                hdf5_sha256=entry["hdf5_sha256"],
                energies=energies,
                event_ids=event_ids,
            )
        )
    return records


def candidates(
    files: list[FileRecord],
    partition: str,
    source: str,
    peak: PeakDefinition,
) -> dict[int, list[tuple[FileRecord, int]]]:
    grouped: dict[int, list[tuple[FileRecord, int]]] = defaultdict(list)
    for record in files:
        if record.partition != partition or record.source != source:
            continue
        rows = np.flatnonzero(
            (record.energies >= peak.roi_low_kev)
            & (record.energies <= peak.roi_high_kev)
        )
        bins = np.floor(record.energies[rows] / BIN_WIDTH_KEV).astype(np.int64)
        for row, bin_index in zip(rows.tolist(), bins.tolist()):
            grouped[int(bin_index)].append((record, int(row)))
    return grouped


def match_peak(
    files: list[FileRecord],
    partition: str,
    peak: PeakDefinition,
    negative_sources: tuple[str, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(source_seed(seed, f"{partition}:{peak.peak_id}"))
    positive = candidates(files, partition, peak.source, peak)
    negative = {
        source: candidates(files, partition, source, peak)
        for source in negative_sources
    }
    rows: list[dict[str, Any]] = []
    available_counts = {
        source: sum(len(items) for items in grouped.values())
        for source, grouped in negative.items()
    }
    for bin_index in sorted(positive):
        pos_items = positive[bin_index].copy()
        rng.shuffle(pos_items)
        source_items: dict[str, list[tuple[FileRecord, int]]] = {}
        for source in negative_sources:
            items = negative[source].get(bin_index, []).copy()
            rng.shuffle(items)
            if items:
                source_items[source] = items
        if not source_items:
            continue
        source_order = list(source_items)
        rng.shuffle(source_order)
        selected_negatives: list[tuple[str, FileRecord, int]] = []
        offsets = {source: 0 for source in source_order}
        while len(selected_negatives) < len(pos_items):
            progress = False
            for source in source_order:
                offset = offsets[source]
                if offset >= len(source_items[source]):
                    continue
                record, row = source_items[source][offset]
                offsets[source] += 1
                selected_negatives.append((source, record, row))
                progress = True
                if len(selected_negatives) >= len(pos_items):
                    break
            if not progress:
                break
        for (pos_record, pos_row), (neg_source, neg_record, neg_row) in zip(
            pos_items, selected_negatives
        ):
            rows.append(
                {
                    "partition": partition,
                    "peak_id": peak.peak_id,
                    "match_bin_index": bin_index,
                    "match_bin_low_kev": bin_index * BIN_WIDTH_KEV,
                    "positive_label": 1,
                    "positive_source": peak.source,
                    "positive_hdf5": pos_record.hdf5,
                    "positive_row": pos_row,
                    "positive_event_id": str(pos_record.event_ids[pos_row]),
                    "positive_energy_kev": float(pos_record.energies[pos_row]),
                    "positive_qc_status": pos_record.qc_status,
                    "negative_label": 0,
                    "negative_source": neg_source,
                    "negative_hdf5": neg_record.hdf5,
                    "negative_row": neg_row,
                    "negative_event_id": str(neg_record.event_ids[neg_row]),
                    "negative_energy_kev": float(neg_record.energies[neg_row]),
                    "negative_qc_status": neg_record.qc_status,
                    "source_weight": 1.0,
                }
            )
    energy_delta = np.asarray(
        [row["positive_energy_kev"] - row["negative_energy_kev"] for row in rows]
    )
    return rows, {
        "peak_id": peak.peak_id,
        "positive_candidates": sum(len(items) for items in positive.values()),
        "negative_candidates_by_source": available_counts,
        "matched_pairs": len(rows),
        "matched_negative_sources": dict(Counter(row["negative_source"] for row in rows)),
        "median_abs_energy_difference_kev": float(np.median(np.abs(energy_delta)))
        if energy_delta.size
        else None,
        "max_abs_energy_difference_kev": float(np.max(np.abs(energy_delta)))
        if energy_delta.size
        else None,
    }


def assign_ids_and_peak_weights(rows: list[dict[str, Any]], partition: str) -> None:
    peak_counts = Counter(row["peak_id"] for row in rows)
    total = len(rows)
    for index, row in enumerate(rows):
        row["pair_id"] = f"{partition}_{index:08d}"
        row["source_weight"] = total / (len(peak_counts) * peak_counts[row["peak_id"]])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/file_partition_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    partition_manifest = args.partition_manifest.resolve()
    files = load_files(partition_manifest)
    source_peaks = {
        source: tuple(peak for peak in TRAINING_PEAKS if peak.source == source)
        for source in EXPERIMENT_SOURCES
    }
    summary: dict[str, Any] = {
        "created_utc": utc_now(),
        "seed": args.seed,
        "bin_width_kev": BIN_WIDTH_KEV,
        "input_partition_manifest": partition_manifest.relative_to(PROJECT_ROOT).as_posix(),
        "input_partition_manifest_sha256": sha256_file(partition_manifest),
        "test_partition_used": False,
        "experiments": {},
    }
    for positive_source in EXPERIMENT_SOURCES:
        experiment_dir = output_root / f"{positive_source}_positive"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_details: dict[str, Any] = {
            "positive_source": positive_source,
            "positive_peaks": [peak.as_dict() for peak in source_peaks[positive_source]],
            "negative_sources": list(NEGATIVE_SOURCES[positive_source]),
            "continuum_justification": CONTINUUM_JUSTIFICATION[positive_source],
            "same_source_negative_excluded": True,
            "partitions": {},
        }
        for partition in PARTITIONS:
            all_rows: list[dict[str, Any]] = []
            peak_details: list[dict[str, Any]] = []
            for peak in source_peaks[positive_source]:
                matched, details = match_peak(
                    files,
                    partition,
                    peak,
                    NEGATIVE_SOURCES[positive_source],
                    args.seed,
                )
                all_rows.extend(matched)
                peak_details.append(details)
            rng = np.random.default_rng(source_seed(args.seed, f"rows:{positive_source}:{partition}"))
            rng.shuffle(all_rows)
            assign_ids_and_peak_weights(all_rows, partition)
            csv_path = experiment_dir / f"label_pairs_{partition}.csv"
            write_csv(csv_path, all_rows)
            experiment_details["partitions"][partition] = {
                "matched_pairs": len(all_rows),
                "event_count": 2 * len(all_rows),
                "negative_source_counts": dict(
                    Counter(row["negative_source"] for row in all_rows)
                ),
                "peak_details": peak_details,
                "csv_sha256": sha256_file(csv_path),
            }
        label_manifest = {
            "created_utc": utc_now(),
            "experiment": experiment_details,
            "matching": {
                "energy_bin_width_kev": BIN_WIDTH_KEV,
                "one_to_one_without_replacement": True,
                "negative_source_balancing": "round-robin within each energy bin, subject to availability",
                "sample_weighting": "inverse peak frequency within each partition",
                "continuum_truth_warning": "Label-0 events are continuum candidates, not interaction-truth-labeled Compton events.",
            },
            "test_partition_used": False,
        }
        (experiment_dir / "label_dataset_manifest.json").write_text(
            json.dumps(label_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Trainers require a provenance file with this conventional name.
        (experiment_dir / "file_partition_manifest.json").write_text(
            json.dumps(
                {
                    "source_manifest": partition_manifest.relative_to(PROJECT_ROOT).as_posix(),
                    "source_manifest_sha256": sha256_file(partition_manifest),
                    "eligible_hdf5_files": [
                        {
                            "source": record.source,
                            "hdf5": record.hdf5,
                            "partition": record.partition,
                            "qc_status": record.qc_status,
                            "hdf5_sha256": record.hdf5_sha256,
                        }
                        for record in files
                    ],
                    "test_partition_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["experiments"][positive_source] = experiment_details
        print(
            positive_source,
            {
                partition: experiment_details["partitions"][partition]["matched_pairs"]
                for partition in PARTITIONS
            },
            flush=True,
        )
    (output_root / "source_ablation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
