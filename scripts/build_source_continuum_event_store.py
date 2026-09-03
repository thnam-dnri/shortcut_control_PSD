#!/usr/bin/env python3
"""Build full-retention source continuum stores from a frozen development manifest.

Every finite event within an inclusive ``corrected_energy_kev`` interval is
copied with its raw 4,500-sample waveform and stable metadata. The output keeps
train and validation separate and deliberately rejects test/external partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PARTITIONS = ("train", "validation")
ENERGY_DATASET = "corrected_energy_kev"
WAVEFORM_LENGTH = 4500
NOISE_RMS_SECTIONS = 5
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
COPIED_SCALARS = (
    "event_id",
    "reconstructed_energy_kev",
    "corrected_energy_kev",
    "shaped_energy_unit",
    "pulse_extremum_adc",
    "pulse_extremum_index",
    "trigger_time_s",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def select_source_files(
    manifest: dict[str, Any], source: str, partitions: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    requested = tuple(partitions)
    unsupported = sorted(set(requested) - set(ALLOWED_PARTITIONS))
    if unsupported:
        raise ValueError(f"Unsupported or locked partitions: {unsupported}")
    manifest_partitions = {entry.get("partition") for entry in manifest["files"]}
    if manifest_partitions - set(ALLOWED_PARTITIONS):
        raise ValueError(
            "Input manifest is not development-only; use the frozen PASS+WARN "
            "train/validation manifest"
        )
    selected = {partition: [] for partition in requested}
    for entry in manifest["files"]:
        partition = entry.get("partition")
        if entry.get("source", "").lower() != source.lower() or partition not in selected:
            continue
        if not entry.get("complete_input", False):
            raise ValueError(f"Incomplete input: {entry.get('hdf5')}")
        if entry.get("processing_status") != "OK":
            raise ValueError(f"Non-OK processing status: {entry.get('hdf5')}")
        if entry.get("qc_status") not in {"PASS", "WARN"}:
            raise ValueError(f"Disallowed QC status: {entry.get('hdf5')}")
        selected[partition].append(dict(entry))
    for partition, entries in selected.items():
        entries.sort(key=lambda item: item["hdf5"])
        if not entries:
            raise ValueError(f"No approved {source} files found for {partition}")
    return selected


def validate_source_file(handle: h5py.File, path: Path, source: str) -> int:
    source_label = str(handle.attrs.get("source_label", "")).lower()
    if source_label != source.lower():
        raise ValueError(
            f"Source mismatch for {path}: expected {source}, found {source_label}"
        )
    required = {"waveform", "noise_rms_adc", *COPIED_SCALARS}
    missing = sorted(required - set(handle.keys()))
    if missing:
        raise ValueError(f"Missing datasets in {path}: {missing}")
    event_count = int(handle[ENERGY_DATASET].shape[0])
    for name in required:
        if int(handle[name].shape[0]) != event_count:
            raise ValueError(f"Dataset length mismatch for {name} in {path}")
    if handle["waveform"].shape != (event_count, WAVEFORM_LENGTH):
        raise ValueError(f"Unexpected waveform shape in {path}")
    noise_shape = handle["noise_rms_adc"].shape
    if len(noise_shape) != 2 or noise_shape[1] < NOISE_RMS_SECTIONS:
        raise ValueError(f"Unexpected noise RMS shape in {path}: {noise_shape}")
    return event_count


def selected_rows(
    handle: h5py.File, minimum_energy_kev: float, maximum_energy_kev: float
) -> np.ndarray:
    energies = np.asarray(handle[ENERGY_DATASET], dtype=np.float32)
    mask = (
        np.isfinite(energies)
        & (energies >= minimum_energy_kev)
        & (energies <= maximum_energy_kev)
    )
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def scan_inputs(
    selected: dict[str, list[dict[str, Any]]],
    source: str,
    minimum_energy_kev: float,
    maximum_energy_kev: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"partitions": {}, "total_selected_events": 0}
    for partition, entries in selected.items():
        file_records = []
        partition_count = 0
        for entry in entries:
            path = resolve_project_path(entry["hdf5"])
            with h5py.File(path, "r") as handle:
                source_events = validate_source_file(handle, path, source)
                count = int(
                    selected_rows(handle, minimum_energy_kev, maximum_energy_kev).size
                )
            partition_count += count
            file_records.append(
                {
                    "path": entry["hdf5"],
                    "source_events": source_events,
                    "selected_events": count,
                    "qc_status": entry["qc_status"],
                }
            )
        result["partitions"][partition] = {
            "file_count": len(entries),
            "selected_events": partition_count,
            "files": file_records,
        }
        result["total_selected_events"] += partition_count
    return result


def create_event_datasets(handle: h5py.File, chunk_events: int) -> None:
    compression = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }
    handle.create_dataset(
        "waveform",
        shape=(0, WAVEFORM_LENGTH),
        maxshape=(None, WAVEFORM_LENGTH),
        chunks=(chunk_events, WAVEFORM_LENGTH),
        dtype=np.float32,
        **compression,
    )
    scalar_types = {
        "event_id": np.uint64,
        "source_row": np.int64,
        "source_file_index": np.uint16,
        "reconstructed_energy_kev": np.float32,
        "corrected_energy_kev": np.float32,
        "shaped_energy_unit": np.float32,
        "pulse_extremum_adc": np.float32,
        "pulse_extremum_index": np.int32,
        "trigger_time_s": np.float64,
    }
    for name, dtype in scalar_types.items():
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_events,),
            dtype=dtype,
            **compression,
        )
    handle.create_dataset(
        "noise_rms_adc",
        shape=(0, NOISE_RMS_SECTIONS),
        maxshape=(None, NOISE_RMS_SECTIONS),
        chunks=(chunk_events, NOISE_RMS_SECTIONS),
        dtype=np.float32,
        **compression,
    )


def resize_event_datasets(handle: h5py.File, new_size: int) -> None:
    for dataset in handle.values():
        if isinstance(dataset, h5py.Group):
            continue
        shape = (new_size, dataset.shape[1]) if dataset.ndim == 2 else (new_size,)
        dataset.resize(shape)


def create_source_file_table(handle: h5py.File, file_count: int) -> h5py.Group:
    group = handle.create_group("source_files")
    group.create_dataset("path", shape=(file_count,), dtype=STRING_DTYPE)
    group.create_dataset("sha256", shape=(file_count,), dtype=STRING_DTYPE)
    group.create_dataset("qc_status", shape=(file_count,), dtype=STRING_DTYPE)
    group.create_dataset("source_event_count", shape=(file_count,), dtype=np.int64)
    group.create_dataset("selected_event_count", shape=(file_count,), dtype=np.int64)
    return group


def build_partition_store(
    partition: str,
    entries: list[dict[str, Any]],
    output_path: Path,
    source: str,
    minimum_energy_kev: float,
    maximum_energy_kev: float,
    chunk_events: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    if partial_path.exists():
        raise FileExistsError(partial_path)
    file_records: list[dict[str, Any]] = []
    try:
        with h5py.File(partial_path, "w") as output:
            output.attrs.update(
                {
                    "schema_version": "1",
                    "source": source,
                    "partition": partition,
                    "energy_dataset": ENERGY_DATASET,
                    "minimum_energy_kev_inclusive": minimum_energy_kev,
                    "maximum_energy_kev_inclusive": maximum_energy_kev,
                    "waveform_length": WAVEFORM_LENGTH,
                    "sample_period_ns": 4.0,
                    "noise_rms_sections": NOISE_RMS_SECTIONS,
                    "selection": "all finite corrected-energy rows in inclusive bounds",
                    "representation": "accepted raw waveform; no fixed model representation",
                    "created_utc": utc_now(),
                    "test_partition_used": False,
                    "external_data_used": False,
                }
            )
            create_event_datasets(output, chunk_events)
            source_table = create_source_file_table(output, len(entries))
            next_index = 0
            for file_index, entry in enumerate(entries):
                path = resolve_project_path(entry["hdf5"])
                current_digest = sha256_file(path)
                expected_digest = entry.get("hdf5_sha256")
                if expected_digest and current_digest != expected_digest:
                    raise ValueError(f"Source HDF5 hash mismatch: {path}")
                with h5py.File(path, "r") as source_file:
                    source_count = validate_source_file(source_file, path, source)
                    rows = selected_rows(
                        source_file, minimum_energy_kev, maximum_energy_kev
                    )
                    source_table["path"][file_index] = entry["hdf5"]
                    source_table["sha256"][file_index] = current_digest
                    source_table["qc_status"][file_index] = entry["qc_status"]
                    source_table["source_event_count"][file_index] = source_count
                    source_table["selected_event_count"][file_index] = rows.size
                    for start in range(0, rows.size, chunk_events):
                        chosen = rows[start : start + chunk_events]
                        stop_index = next_index + chosen.size
                        resize_event_datasets(output, stop_index)
                        output["waveform"][next_index:stop_index] = source_file[
                            "waveform"
                        ][chosen]
                        output["noise_rms_adc"][next_index:stop_index] = source_file[
                            "noise_rms_adc"
                        ][chosen, :NOISE_RMS_SECTIONS]
                        for name in COPIED_SCALARS:
                            output[name][next_index:stop_index] = source_file[name][chosen]
                        output["source_row"][next_index:stop_index] = chosen
                        output["source_file_index"][next_index:stop_index] = file_index
                        next_index = int(stop_index)
                file_records.append(
                    {
                        "path": entry["hdf5"],
                        "sha256": current_digest,
                        "qc_status": entry["qc_status"],
                        "source_events": source_count,
                        "selected_events": int(rows.size),
                    }
                )
                print(
                    f"{partition}: file {file_index + 1}/{len(entries)} "
                    f"selected={rows.size} cumulative={next_index} {path.name}",
                    flush=True,
                )
            output.attrs["event_count"] = next_index
            output.attrs["source_file_count"] = len(entries)
            output.flush()
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "partition": partition,
        "store_file": display_path(output_path),
        "store_sha256": sha256_file(output_path),
        "source_file_count": len(entries),
        "event_count": sum(record["selected_events"] for record in file_records),
        "files": file_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="co60")
    parser.add_argument("--minimum-energy-kev", type=float, default=100.0)
    parser.add_argument("--maximum-energy-kev", type=float, default=1000.0)
    parser.add_argument(
        "--file-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/labels/architecture_pass_warn_20260815/file_partition_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--chunk-events", type=int, default=256)
    parser.add_argument("--scan-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not np.isfinite(args.minimum_energy_kev) or not np.isfinite(
        args.maximum_energy_kev
    ):
        raise ValueError("Energy bounds must be finite")
    if args.minimum_energy_kev > args.maximum_energy_kev:
        raise ValueError("Minimum energy must not exceed maximum energy")
    if args.chunk_events <= 0:
        raise ValueError("--chunk-events must be positive")
    file_manifest_path = args.file_manifest.resolve()
    manifest_data = json.loads(file_manifest_path.read_text(encoding="utf-8"))
    selected = select_source_files(manifest_data, args.source, ALLOWED_PARTITIONS)
    scan = scan_inputs(
        selected,
        args.source,
        args.minimum_energy_kev,
        args.maximum_energy_kev,
    )
    print(json.dumps(scan, indent=2, sort_keys=True), flush=True)
    if args.scan_only:
        return 0
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    stores = {}
    for partition in ALLOWED_PARTITIONS:
        stores[partition] = build_partition_store(
            partition,
            selected[partition],
            output_dir / f"{partition}_events.h5",
            args.source,
            args.minimum_energy_kev,
            args.maximum_energy_kev,
            args.chunk_events,
        )
    result = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "builder": display_path(Path(__file__)),
        "builder_sha256": sha256_file(Path(__file__)),
        "source": args.source,
        "selection": {
            "energy_dataset": ENERGY_DATASET,
            "minimum_energy_kev_inclusive": args.minimum_energy_kev,
            "maximum_energy_kev_inclusive": args.maximum_energy_kev,
            "finite_energy_required": True,
            "sampling": "none; retain every admitted row in range",
            "qc_policy": "frozen PASS+WARN development manifest",
            "partitions": list(ALLOWED_PARTITIONS),
            "test_partition_used": False,
            "external_data_used": False,
        },
        "input_manifest": display_path(file_manifest_path),
        "input_manifest_sha256": sha256_file(file_manifest_path),
        "scan": scan,
        "stores": stores,
        "total_event_count": sum(item["event_count"] for item in stores.values()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(f"manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
