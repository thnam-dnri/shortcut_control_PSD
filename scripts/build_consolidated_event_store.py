#!/usr/bin/env python3
"""Build or append partitioned, model-flexible waveform event stores.

The store contains unique accepted 4,500-sample waveforms and stable scalar
metadata, not fixed charge/current windows or normalization.  Pair CSVs remain the
label authority.  Repeated references to the same source HDF5 row are stored once.

Future acquisitions are supported through ``--append``: first preprocess and assign
new complete files to a partition, build label-pair CSVs, then append their referenced
events.  Existing store rows are immutable; each append has consistency checks and a
recorded transaction manifest.  Test is deliberately not an accepted partition name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PARTITIONS = ("train", "validation", "external_validation")
WAVEFORM_LENGTH = 4500
NOISE_RMS_SECTIONS = 5
CHUNK_EVENTS = 128
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
LOOKUP_FIELDS = (
    "store_index",
    "partition",
    "source_hdf5",
    "source_row",
    "event_id",
    "source",
    "reconstructed_energy_kev",
    "qc_status",
    "source_hdf5_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_default_csvs() -> list[Path]:
    paths: list[Path] = []
    for partition in ("train", "validation"):
        paths.extend(
            sorted(
                (PROJECT_ROOT / "outputs/labels").glob(
                    f"**/label_pairs_{partition}.csv"
                )
            )
        )
    return sorted(set(path.resolve() for path in paths))


def collect_references(
    csv_paths: Iterable[Path],
) -> tuple[dict[str, set[tuple[str, int]]], list[dict[str, Any]]]:
    references: dict[str, set[tuple[str, int]]] = defaultdict(set)
    inputs: list[dict[str, Any]] = []
    for csv_path in sorted(set(path.resolve() for path in csv_paths)):
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        row_count = 0
        partitions: set[str] = set()
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                partition = row["partition"]
                if partition not in ALLOWED_PARTITIONS:
                    raise ValueError(
                        f"Unsupported or locked partition {partition!r} in {csv_path}"
                    )
                partitions.add(partition)
                references[partition].add(
                    (row["positive_hdf5"], int(row["positive_row"]))
                )
                references[partition].add(
                    (row["negative_hdf5"], int(row["negative_row"]))
                )
                row_count += 1
        if len(partitions) != 1:
            raise ValueError(f"Pair CSV must contain exactly one partition: {csv_path}")
        inputs.append(
            {
                "path": csv_path.relative_to(PROJECT_ROOT).as_posix()
                if csv_path.is_relative_to(PROJECT_ROOT)
                else str(csv_path),
                "sha256": sha256_file(csv_path),
                "pair_count": row_count,
                "partition": next(iter(partitions)),
            }
        )
    partitions = sorted(references)
    for index, first in enumerate(partitions):
        first_files = {path for path, _ in references[first]}
        for second in partitions[index + 1 :]:
            duplicate_events = references[first] & references[second]
            if duplicate_events:
                raise ValueError(
                    f"Events occur in both {first} and {second}: "
                    f"example={next(iter(duplicate_events))}"
                )
            duplicate_files = first_files & {
                path for path, _ in references[second]
            }
            if duplicate_files:
                raise ValueError(
                    f"Source files occur in both {first} and {second}: "
                    f"example={next(iter(duplicate_files))}"
                )
    return references, inputs


def load_qc_lookup() -> dict[str, str]:
    manifest_path = PROJECT_ROOT / "outputs/labels/file_partition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        entry["hdf5"]: entry["qc_status"]
        for entry in manifest["files"]
        if entry.get("hdf5")
    }


def create_datasets(handle: h5py.File, partition: str) -> None:
    handle.attrs.update(
        {
            "schema_version": "1",
            "partition": partition,
            "waveform_length": WAVEFORM_LENGTH,
            "sample_period_ns": 4.0,
            "noise_rms_sections": NOISE_RMS_SECTIONS,
            "noise_rms_window": "five 200-sample sections covering samples 0:1000",
            "representation": "accepted raw waveform; no fixed window, alignment, smoothing, or normalization",
            "append_only": True,
            "created_utc": utc_now(),
        }
    )
    handle.create_dataset(
        "waveform",
        shape=(0, WAVEFORM_LENGTH),
        maxshape=(None, WAVEFORM_LENGTH),
        chunks=(CHUNK_EVENTS, WAVEFORM_LENGTH),
        dtype=np.float32,
    )
    scalar_specs = {
        "event_id": np.uint64,
        "source_row": np.int64,
        "reconstructed_energy_kev": np.float32,
        "shaped_energy_unit": np.float32,
        "pulse_extremum_adc": np.float32,
        "pulse_extremum_index": np.int32,
        "trigger_time_s": np.float64,
    }
    for name, dtype in scalar_specs.items():
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(CHUNK_EVENTS,),
            dtype=dtype,
        )
    handle.create_dataset(
        "noise_rms_adc",
        shape=(0, NOISE_RMS_SECTIONS),
        maxshape=(None, NOISE_RMS_SECTIONS),
        chunks=(CHUNK_EVENTS, NOISE_RMS_SECTIONS),
        dtype=np.float32,
    )
    for name in ("source", "source_hdf5", "source_hdf5_sha256", "qc_status"):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(CHUNK_EVENTS,),
            dtype=STRING_DTYPE,
        )


def resize_all(handle: h5py.File, new_size: int) -> None:
    for dataset in handle.values():
        if dataset.ndim == 2:
            dataset.resize((new_size, dataset.shape[1]))
        else:
            dataset.resize((new_size,))


def load_existing_lookup(path: Path) -> dict[tuple[str, int], int]:
    if not path.is_file():
        return {}
    lookup: dict[tuple[str, int], int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            lookup[key] = int(row["store_index"])
    return lookup


def source_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def append_partition(
    partition: str,
    references: set[tuple[str, int]],
    output_dir: Path,
    qc_lookup: dict[str, str],
    append: bool,
) -> dict[str, Any]:
    store_path = output_dir / f"{partition}_events.h5"
    lookup_path = output_dir / f"event_lookup_{partition}.csv"
    existing_lookup = load_existing_lookup(lookup_path) if append else {}
    new_references = sorted(references - set(existing_lookup))
    if append and not store_path.is_file():
        raise FileNotFoundError(f"Cannot append without existing store: {store_path}")
    if append and not new_references:
        with h5py.File(store_path, "r") as output:
            if output.attrs.get("partition") != partition:
                raise ValueError(f"Partition attribute mismatch in {store_path}")
            event_count = int(output["waveform"].shape[0])
        if len(existing_lookup) != event_count:
            raise ValueError(f"Lookup/store size mismatch for {partition}")
        return {
            "partition": partition,
            "store_file": store_path.relative_to(PROJECT_ROOT).as_posix(),
            "lookup_file": lookup_path.relative_to(PROJECT_ROOT).as_posix(),
            "events_before": event_count,
            "events_appended": 0,
            "events_after": event_count,
            "requested_unique_references": len(references),
            "store_sha256": sha256_file(store_path),
            "lookup_sha256": sha256_file(lookup_path),
        }
    mode = "r+" if append else "w"
    source_hash_cache: dict[Path, str] = {}
    lookup_rows: list[dict[str, Any]] = []
    with h5py.File(store_path, mode) as output:
        if not append:
            create_datasets(output, partition)
        elif output.attrs.get("partition") != partition:
            raise ValueError(f"Partition attribute mismatch in {store_path}")
        original_size = int(output["waveform"].shape[0])
        if existing_lookup and max(existing_lookup.values()) != original_size - 1:
            raise ValueError(f"Lookup/store size mismatch for {partition}")
        try:
            grouped: dict[str, list[int]] = defaultdict(list)
            for source_hdf5, source_row in new_references:
                grouped[source_hdf5].append(source_row)
            next_index = original_size
            for relative_path, rows in sorted(grouped.items()):
                source_path = PROJECT_ROOT / relative_path
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                source_digest = source_sha256(source_path, source_hash_cache)
                with h5py.File(source_path, "r") as source_file:
                    source_label = str(source_file.attrs["source_label"])
                    noise_rms = source_file["noise_rms_adc"]
                    if noise_rms.ndim != 2 or noise_rms.shape[1] < NOISE_RMS_SECTIONS:
                        raise ValueError(
                            f"Expected at least {NOISE_RMS_SECTIONS} noise RMS sections "
                            f"in {source_path}; found shape={noise_rms.shape}"
                        )
                    sorted_rows = np.asarray(sorted(rows), dtype=np.int64)
                    for start in range(0, sorted_rows.size, CHUNK_EVENTS):
                        selected = sorted_rows[start : start + CHUNK_EVENTS]
                        count = int(selected.size)
                        stop_index = next_index + count
                        resize_all(output, stop_index)
                        output["waveform"][next_index:stop_index] = source_file["waveform"][selected]
                        for name in (
                            "event_id",
                            "reconstructed_energy_kev",
                            "shaped_energy_unit",
                            "pulse_extremum_adc",
                            "pulse_extremum_index",
                            "trigger_time_s",
                        ):
                            output[name][next_index:stop_index] = source_file[name][selected]
                        output["noise_rms_adc"][next_index:stop_index] = noise_rms[
                            selected, :NOISE_RMS_SECTIONS
                        ]
                        output["source_row"][next_index:stop_index] = selected
                        output["source"][next_index:stop_index] = [source_label] * count
                        output["source_hdf5"][next_index:stop_index] = [relative_path] * count
                        output["source_hdf5_sha256"][next_index:stop_index] = [source_digest] * count
                        qc_status = qc_lookup.get(relative_path, "UNKNOWN")
                        output["qc_status"][next_index:stop_index] = [qc_status] * count
                        event_ids = np.asarray(source_file["event_id"][selected], dtype=np.uint64)
                        energies = np.asarray(
                            source_file["reconstructed_energy_kev"][selected],
                            dtype=np.float32,
                        )
                        for offset, (source_row, event_id, energy) in enumerate(
                            zip(selected.tolist(), event_ids.tolist(), energies.tolist())
                        ):
                            store_index = next_index + offset
                            existing_lookup[(relative_path, int(source_row))] = store_index
                            lookup_rows.append(
                                {
                                    "store_index": store_index,
                                    "partition": partition,
                                    "source_hdf5": relative_path,
                                    "source_row": int(source_row),
                                    "event_id": int(event_id),
                                    "source": source_label,
                                    "reconstructed_energy_kev": float(energy),
                                    "qc_status": qc_status,
                                    "source_hdf5_sha256": source_digest,
                                }
                            )
                        next_index = stop_index
            output.attrs["last_append_utc"] = utc_now()
            output.attrs["event_count"] = next_index
            output.flush()
        except Exception:
            resize_all(output, original_size)
            output.attrs["event_count"] = original_size
            output.flush()
            raise
    lookup_mode = "a" if append and lookup_path.exists() else "w"
    with lookup_path.open(lookup_mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOOKUP_FIELDS)
        if lookup_mode == "w":
            writer.writeheader()
        writer.writerows(lookup_rows)
    final_count = len(existing_lookup)
    return {
        "partition": partition,
        "store_file": store_path.relative_to(PROJECT_ROOT).as_posix(),
        "lookup_file": lookup_path.relative_to(PROJECT_ROOT).as_posix(),
        "events_before": final_count - len(lookup_rows),
        "events_appended": len(lookup_rows),
        "events_after": final_count,
        "requested_unique_references": len(references),
        "store_sha256": sha256_file(store_path),
        "lookup_sha256": sha256_file(lookup_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-csv",
        type=Path,
        action="append",
        default=None,
        help="Pair CSV to include; repeat as needed. Defaults to all active train/validation pair CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/event_store",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.append or args.overwrite):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --append or --overwrite"
        )
    if args.overwrite and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "--overwrite does not delete an existing populated store; choose a new output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = args.pair_csv if args.pair_csv else discover_default_csvs()
    references, input_records = collect_references(csv_paths)
    qc_lookup = load_qc_lookup()
    transaction = {
        "created_utc": utc_now(),
        "mode": "append" if args.append else "create",
        "input_pair_csvs": input_records,
        "test_partition_used": False,
        "partitions": {},
    }
    for partition in sorted(references):
        print(
            f"{partition}: requested_unique_events={len(references[partition])}",
            flush=True,
        )
        transaction["partitions"][partition] = append_partition(
            partition,
            references[partition],
            output_dir,
            qc_lookup,
            args.append,
        )
        print(transaction["partitions"][partition], flush=True)
    transaction_text = json.dumps(transaction, indent=2, sort_keys=True) + "\n"
    transaction_hash = hashlib.sha256(transaction_text.encode("utf-8")).hexdigest()[:16]
    transaction_path = manifest_dir / f"transaction_{transaction_hash}.json"
    transaction_path.write_text(transaction_text, encoding="utf-8")
    manifest = {
        "schema_version": "1",
        "updated_utc": utc_now(),
        "store_policy": {
            "event_identity": "source_hdf5 path plus source row",
            "storage": "one append-only HDF5 event store per assigned partition",
            "waveforms": "accepted raw 4500-sample float32; no fixed model representation",
            "labels": "remain in independently versioned pair CSV manifests",
            "partition_rule": "assign complete source files before appending; never move existing store rows",
            "future_data": "preprocess, QC, partition by complete file, build labels, then append new unique references",
            "external_validation": "use a separate external_validation store; never append it to train or internal validation",
            "test_partition_used": False,
        },
        "latest_transaction": transaction_path.relative_to(PROJECT_ROOT).as_posix(),
        "latest_transaction_sha256": sha256_file(transaction_path),
        "partitions": transaction["partitions"],
    }
    (manifest_dir / "event_store_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
