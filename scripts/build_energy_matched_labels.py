#!/usr/bin/env python3
"""Build file-partitioned, energy-matched label manifests from waveform HDF5 files.

The output is an index-only dataset: waveform arrays remain in their source HDF5
files and the CSV manifests store the source file plus row index for each matched
label-1/label-0 pair.  No waveform data is copied.

The default exploratory policy follows the current user decision: retain every
complete HDF5 product with processing_status=OK, while retaining all individual QC
observations in the manifest rather than filtering on them.  A complete source file
has 100,000 input entries; incomplete and invalid products cannot participate in the
required file-level split and are recorded as excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5_DIR = PROJECT_ROOT / "processed_data" / "waveform_hdf5_corrected"
DEFAULT_PREPROCESSING_MANIFESTS = (
    PROJECT_ROOT / "outputs" / "waveform_hdf5" / "preprocessing_manifest.json",
    PROJECT_ROOT / "outputs" / "waveform_hdf5" / "preprocessing_manifest_20260814_remaining.json",
)
DEFAULT_QC_DIR = PROJECT_ROOT / "outputs" / "data_quality"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "labels"
DEFAULT_PARTITION_MANIFEST = DEFAULT_OUTPUT_DIR / "file_partition_manifest.json"

BIN_WIDTH_KEV = 0.5
COMPLETE_INPUT_ENTRIES = 100_000
PARTITIONS = ("train", "validation", "test")
SOURCE_ORDER = ("ba133", "na22", "cs137", "co60")


@dataclass(frozen=True)
class PeakDefinition:
    peak_id: str
    source: str
    nominal_energy_kev: float
    fitted_center_kev: float
    fwhm_kev: float
    training_label: bool
    note: str

    @property
    def sigma_kev(self) -> float:
        return self.fwhm_kev / 2.354820045

    @property
    def roi_half_width_kev(self) -> float:
        return 0.5 * self.fwhm_kev

    @property
    def roi_low_kev(self) -> float:
        return self.fitted_center_kev - self.roi_half_width_kev

    @property
    def roi_high_kev(self) -> float:
        return self.fitted_center_kev + self.roi_half_width_kev

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_id": self.peak_id,
            "source": self.source,
            "nominal_energy_kev": self.nominal_energy_kev,
            "fitted_center_kev": self.fitted_center_kev,
            "fwhm_kev": self.fwhm_kev,
            "sigma_kev": self.sigma_kev,
            "roi_definition": "fitted_center +/- 0.5 FWHM",
            "roi_low_kev": self.roi_low_kev,
            "roi_high_kev": self.roi_high_kev,
            "training_label": self.training_label,
            "note": self.note,
        }


# These centroids and widths are fitted from the gain-corrected HDF5 dataset
# (processed_data/waveform_hdf5_corrected/). They must not be refit after file partitioning.
TRAINING_PEAKS = (
    PeakDefinition(
        "ba133_276kev",
        "ba133",
        276.3989,
        276.146,
        3.986,
        True,
        "Ba-133 full-energy candidate",
    ),
    PeakDefinition(
        "ba133_303kev",
        "ba133",
        302.8508,
        303.139,
        3.971,
        True,
        "Ba-133 full-energy candidate",
    ),
    PeakDefinition(
        "ba133_356kev",
        "ba133",
        356.0129,
        355.709,
        3.941,
        True,
        "Ba-133 full-energy candidate",
    ),
    PeakDefinition(
        "ba133_384kev",
        "ba133",
        383.8485,
        383.978,
        4.120,
        True,
        "Ba-133 full-energy candidate",
    ),
    PeakDefinition(
        "na22_511kev",
        "na22",
        510.99895,
        510.926,
        4.447,
        True,
        "Na-22 annihilation full-energy candidate",
    ),
    PeakDefinition(
        "cs137_662kev",
        "cs137",
        661.657,
        661.668,
        3.749,
        True,
        "Cs-137 full-energy candidate",
    ),
)

DEVELOPMENT_PEAKS = (
    PeakDefinition(
        "na22_1275kev",
        "na22",
        1274.537,
        1274.532,
        3.930,
        False,
        "Reserved development/transfer check; not a training label",
    ),
)


@dataclass
class FileInfo:
    source: str
    hdf5_path: Path
    input_root: str
    input_entries: int
    accepted_entries: int
    processing_status: str
    qc_status: str
    qc_reports: list[str]
    qc_observations: list[dict[str, Any]]
    hdf5_sha256: str
    complete_input: bool
    partition: str | None = None
    excluded_reason: str | None = None
    energies: np.ndarray | None = field(default=None, repr=False)
    event_ids: np.ndarray | None = field(default=None, repr=False)

    @property
    def hdf5_relative(self) -> str:
        return self.hdf5_path.relative_to(PROJECT_ROOT).as_posix()

    @property
    def input_root_name(self) -> str:
        return Path(self.input_root).name

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "hdf5": self.hdf5_relative,
            "input_root": self.input_root,
            "input_root_name": self.input_root_name,
            "input_entries": self.input_entries,
            "accepted_entries": self.accepted_entries,
            "processing_status": self.processing_status,
            "qc_status": self.qc_status,
            "qc_reports": self.qc_reports,
            "qc_observations": self.qc_observations,
            "hdf5_sha256": self.hdf5_sha256,
            "complete_input": self.complete_input,
            "partition": self.partition,
            "excluded_reason": self.excluded_reason,
        }


CSV_FIELDS = (
    "pair_id",
    "partition",
    "peak_id",
    "match_bin_index",
    "match_bin_low_kev",
    "positive_label",
    "positive_source",
    "positive_hdf5",
    "positive_row",
    "positive_event_id",
    "positive_energy_kev",
    "positive_qc_status",
    "negative_label",
    "negative_source",
    "negative_hdf5",
    "negative_row",
    "negative_event_id",
    "negative_energy_kev",
    "negative_qc_status",
    "source_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build index-only energy-matched label manifests from HDF5 files."
    )
    parser.add_argument("--hdf5-dir", type=Path, default=DEFAULT_HDF5_DIR)
    parser.add_argument(
        "--preprocessing-manifests",
        type=Path,
        nargs="*",
        default=list(DEFAULT_PREPROCESSING_MANIFESTS),
        help="One or more preprocessing manifest paths.",
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        default=None,
        help="Single preprocessing manifest path (legacy compatibility).",
    )
    parser.add_argument(
        "--energy-dataset",
        type=str,
        default="corrected_energy_kev",
        help="Energy dataset to load from HDF5 (default: corrected_energy_kev).",
    )
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--approved-allowlist",
        type=Path,
        default=None,
        help="Frozen QC allowlist produced after WP0 approval; changes file admission without changing partitions.",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Build train/validation labels only; requires --approved-allowlist and never opens test HDF5 event data.",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=DEFAULT_PARTITION_MANIFEST,
        help="Existing immutable file-partition manifest to inherit when an allowlist is supplied.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing label-manifest outputs.",
    )
    return parser.parse_args()


def text_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(json_value(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessing_records(
    manifest_paths: Path | Iterable[Path],
) -> dict[str, dict[str, Any]]:
    if isinstance(manifest_paths, Path):
        paths = [manifest_paths]
    else:
        paths = list(manifest_paths)
    by_hdf5: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("files")
        if not isinstance(records, list):
            raise ValueError(f"Expected a files list in {path}")
        for record in records:
            output = record.get("output_hdf5")
            if not output:
                raise ValueError(f"Manifest record has no output_hdf5: {record}")
            name = Path(output).name
            if name in by_hdf5:
                continue
            by_hdf5[name] = record
    return by_hdf5


def load_qc_statuses(qc_dir: Path) -> dict[str, dict[str, Any]]:
    """Return every QC observation plus a derived worst status per input file."""
    rank = {"UNKNOWN": 0, "PASS": 1, "WARN": 2, "FAIL": 3}
    statuses: dict[str, dict[str, Any]] = {}
    reports = sorted(qc_dir.glob("**/session_qc_report.json"))
    for report in reports:
        report_bytes = report.read_bytes()
        data = json.loads(report_bytes.decode("utf-8"))
        report_rel = report.relative_to(PROJECT_ROOT).as_posix()
        report_hash = hashlib.sha256(report_bytes).hexdigest()
        for file_report in data.get("files", []):
            raw_name = file_report.get("file_name") or file_report.get("file")
            if not raw_name:
                continue
            name = Path(str(raw_name)).name
            status = text_value(file_report.get("status", "UNKNOWN")).upper()
            observation = {
                "report": report_rel,
                "report_sha256": report_hash,
                "session_id": data.get("session_id"),
                "created_utc": data.get("created_utc"),
                "sample_mode": data.get("sample_mode"),
                "status": status,
                "failures": list(file_report.get("failures", [])),
                "warnings": list(file_report.get("warnings", [])),
            }
            current = statuses.setdefault(
                name,
                {"worst_status": "UNKNOWN", "observations": []},
            )
            current["observations"].append(observation)
            if rank.get(status, 0) > rank.get(current["worst_status"], 0):
                current["worst_status"] = status
    return statuses


def load_file_infos(
    hdf5_dir: Path,
    preprocessing_records: dict[str, dict[str, Any]],
    qc_statuses: dict[str, dict[str, Any]],
    energy_dataset: str = "corrected_energy_kev",
    selected_hdf5_names: set[str] | None = None,
) -> tuple[list[FileInfo], list[FileInfo]]:
    eligible: list[FileInfo] = []
    excluded: list[FileInfo] = []
    for name, record in sorted(preprocessing_records.items()):
        if selected_hdf5_names is not None and name not in selected_hdf5_names:
            continue
        source = text_value(record.get("source", "")).lower()
        if source not in SOURCE_ORDER:
            continue
        hdf5_path = hdf5_dir / name
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Manifest HDF5 does not exist: {hdf5_path}")
        input_entries = int(record.get("input_entries", 0))
        accepted_entries = int(record.get("accepted_entries", 0))
        processing_status = text_value(record.get("status", "UNKNOWN")).upper()
        qc_name = Path(text_value(record.get("input_root", name))).name
        qc = qc_statuses.get(
            qc_name,
            {"worst_status": "UNKNOWN", "observations": []},
        )
        complete = input_entries == COMPLETE_INPUT_ENTRIES
        excluded_reason: str | None = None
        if processing_status != "OK":
            excluded_reason = f"processing_status={processing_status}"
        elif not complete:
            excluded_reason = (
                f"incomplete_input_entries={input_entries}; "
                f"required={COMPLETE_INPUT_ENTRIES}"
            )

        with h5py.File(hdf5_path, "r") as h5_file:
            h5_status = text_value(h5_file.attrs.get("processing_status", "UNKNOWN")).upper()
            dset_name = energy_dataset if energy_dataset in h5_file else "reconstructed_energy_kev"
            h5_count = len(h5_file[dset_name])
        if h5_status != processing_status:
            raise ValueError(
                f"Status mismatch for {name}: manifest={processing_status}, HDF5={h5_status}"
            )
        if h5_count != accepted_entries:
            raise ValueError(
                f"Accepted-count mismatch for {name}: manifest={accepted_entries}, HDF5={h5_count}"
            )

        qc_observations = list(qc.get("observations", []))
        info = FileInfo(
            source=source,
            hdf5_path=hdf5_path,
            input_root=text_value(record.get("input_root", "")),
            input_entries=input_entries,
            accepted_entries=accepted_entries,
            processing_status=processing_status,
            qc_status=text_value(qc.get("worst_status", "UNKNOWN")).upper(),
            qc_reports=[observation["report"] for observation in qc_observations],
            qc_observations=qc_observations,
            hdf5_sha256=sha256_file(hdf5_path),
            complete_input=complete,
            excluded_reason=excluded_reason,
        )
        if excluded_reason is None:
            eligible.append(info)
        else:
            excluded.append(info)
    return eligible, excluded


def load_approved_allowlist(path: Path) -> tuple[dict[str, Any], set[str]]:
    """Load a user-approved QC allowlist without accepting draft artifacts."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "FROZEN":
        raise ValueError(
            f"QC allowlist is not frozen: {path}; approval is required before rebuilding"
        )
    approved_policy = data.get("approved_policy")
    admitted = data.get("approved_hdf5")
    if not isinstance(admitted, list):
        candidates = data.get("policy_candidates", {})
        candidate = candidates.get(approved_policy, {}) if approved_policy else {}
        admitted = candidate.get("admitted_hdf5")
    if not approved_policy or not isinstance(admitted, list) or not admitted:
        raise ValueError(
            f"Frozen QC allowlist must contain approved_policy and approved_hdf5: {path}"
        )
    return data, {str(item) for item in admitted}


def load_frozen_partition_map(path: Path) -> dict[str, str]:
    """Read immutable train/validation/test assignments from a prior manifest."""

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("files")
    if not isinstance(records, list):
        raise ValueError(f"Expected files list in partition manifest: {path}")
    mapping: dict[str, str] = {}
    for record in records:
        hdf5 = str(record.get("hdf5", ""))
        partition = record.get("partition")
        if not hdf5 or partition not in PARTITIONS:
            continue
        previous = mapping.get(hdf5)
        if previous is not None and previous != partition:
            raise ValueError(f"Conflicting immutable partitions for {hdf5}")
        mapping[hdf5] = str(partition)
    if not mapping:
        raise ValueError(f"No complete partition assignments found in {path}")
    return mapping


def apply_approved_allowlist(
    eligible: list[FileInfo],
    excluded: list[FileInfo],
    admitted_hdf5: set[str],
    partition_map: dict[str, str],
) -> tuple[list[FileInfo], list[FileInfo]]:
    """Filter files by an approved allowlist and inherit, never reassign, partitions."""

    current_hdf5 = {info.hdf5_relative for info in eligible}
    unknown = admitted_hdf5 - current_hdf5
    if unknown:
        raise ValueError(
            "Approved allowlist admits files that are not complete processing-valid inputs: "
            + ", ".join(sorted(unknown)[:10])
        )
    selected: list[FileInfo] = []
    rejected = list(excluded)
    for info in eligible:
        if info.hdf5_relative not in admitted_hdf5:
            info.excluded_reason = "approved_allowlist_excluded"
            rejected.append(info)
            continue
        partition = partition_map.get(info.hdf5_relative)
        if partition not in PARTITIONS:
            raise ValueError(
                f"Approved file has no immutable train/validation/test assignment: {info.hdf5_relative}"
            )
        info.partition = partition
        selected.append(info)
    validate_partition_disjointness(selected)
    return selected, rejected


def source_seed(seed: int, source: str) -> int:
    digest = hashlib.sha256(source.encode("ascii")).digest()
    return seed + int.from_bytes(digest[:4], "little")


def partition_counts(count: int) -> dict[str, int]:
    if count < 3:
        raise ValueError(f"Need at least three complete files for a source; got {count}")
    train = max(1, int(round(count * 0.60)))
    validation = max(1, int(round(count * 0.20)))
    test = count - train - validation
    while test < 1:
        if train >= validation and train > 1:
            train -= 1
        else:
            validation -= 1
        test = count - train - validation
    return {"train": train, "validation": validation, "test": test}


def assign_partitions(infos: list[FileInfo], seed: int) -> None:
    by_source: dict[str, list[FileInfo]] = defaultdict(list)
    for info in infos:
        by_source[info.source].append(info)
    for source in SOURCE_ORDER:
        source_infos = sorted(by_source.get(source, []), key=lambda item: item.input_root_name)
        if not source_infos:
            continue
        counts = partition_counts(len(source_infos))
        order = np.arange(len(source_infos))
        rng = np.random.default_rng(source_seed(seed, source))
        rng.shuffle(order)
        offset = 0
        for partition in PARTITIONS:
            size = counts[partition]
            for index in order[offset : offset + size]:
                source_infos[int(index)].partition = partition
            offset += size
        if offset != len(source_infos):
            raise AssertionError(f"Partition assignment lost files for {source}")


def validate_partition_disjointness(infos: Iterable[FileInfo]) -> None:
    seen: dict[str, str] = {}
    for info in infos:
        if info.partition is None:
            raise AssertionError(f"Unassigned eligible file: {info.hdf5_relative}")
        key = info.hdf5_relative
        previous = seen.get(key)
        if previous is not None and previous != info.partition:
            raise AssertionError(
                f"HDF5 file assigned to multiple partitions: {key} ({previous}, {info.partition})"
            )
        seen[key] = info.partition


def load_arrays(infos: Iterable[FileInfo], energy_dataset: str = "corrected_energy_kev") -> None:
    for info in infos:
        with h5py.File(info.hdf5_path, "r") as h5_file:
            dset_name = energy_dataset if energy_dataset in h5_file else "reconstructed_energy_kev"
            info.energies = np.asarray(
                h5_file[dset_name], dtype=np.float64
            )
            info.event_ids = np.asarray(h5_file["event_id"])
        if len(info.energies) != info.accepted_entries:
            raise ValueError(f"Array length mismatch after loading {info.hdf5_path}")
        if len(info.event_ids) != len(info.energies):
            raise ValueError(f"event_id length mismatch in {info.hdf5_path}")


def grouped_candidates(
    infos: Iterable[FileInfo],
    partition: str,
    source: str,
    peak: PeakDefinition,
) -> tuple[dict[int, list[tuple[FileInfo, int]]], int]:
    grouped: dict[int, list[tuple[FileInfo, int]]] = defaultdict(list)
    candidate_count = 0
    for info in infos:
        if info.partition != partition or info.source != source:
            continue
        if info.energies is None:
            raise AssertionError("File arrays must be loaded before matching")
        mask = (info.energies >= peak.roi_low_kev) & (
            info.energies <= peak.roi_high_kev
        )
        rows = np.flatnonzero(mask)
        candidate_count += int(rows.size)
        if rows.size == 0:
            continue
        bins = np.floor(info.energies[rows] / BIN_WIDTH_KEV).astype(np.int64)
        for row, bin_index in zip(rows.tolist(), bins.tolist()):
            grouped[int(bin_index)].append((info, int(row)))
    return grouped, candidate_count


def two_sample_ks_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size == 0 or second.size == 0:
        return None
    first_sorted = np.sort(first)
    second_sorted = np.sort(second)
    values = np.sort(np.concatenate((first_sorted, second_sorted)))
    first_cdf = np.searchsorted(first_sorted, values, side="right") / first_sorted.size
    second_cdf = np.searchsorted(second_sorted, values, side="right") / second_sorted.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def match_peak(
    positive_infos: Iterable[FileInfo],
    negative_infos: Iterable[FileInfo],
    partition: str,
    peak: PeakDefinition,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive, positive_count = grouped_candidates(
        positive_infos, partition, peak.source, peak
    )
    negative, negative_count = grouped_candidates(
        negative_infos, partition, "co60", peak
    )
    rng = np.random.default_rng(seed + sum(ord(char) for char in peak.peak_id))
    rows: list[dict[str, Any]] = []
    matched_by_bin: dict[int, int] = {}
    for bin_index in sorted(set(positive) & set(negative)):
        pos_candidates = positive[bin_index].copy()
        neg_candidates = negative[bin_index].copy()
        rng.shuffle(pos_candidates)
        rng.shuffle(neg_candidates)
        count = min(len(pos_candidates), len(neg_candidates))
        matched_by_bin[bin_index] = count
        for pos, neg in zip(pos_candidates[:count], neg_candidates[:count]):
            pos_info, pos_row = pos
            neg_info, neg_row = neg
            if pos_info.energies is None or pos_info.event_ids is None:
                raise AssertionError("Positive arrays were not loaded")
            if neg_info.energies is None or neg_info.event_ids is None:
                raise AssertionError("Negative arrays were not loaded")
            rows.append(
                {
                    "partition": partition,
                    "peak_id": peak.peak_id,
                    "match_bin_index": bin_index,
                    "match_bin_low_kev": bin_index * BIN_WIDTH_KEV,
                    "positive_label": 1,
                    "positive_source": peak.source,
                    "positive_hdf5": pos_info.hdf5_relative,
                    "positive_row": pos_row,
                    "positive_event_id": text_value(pos_info.event_ids[pos_row]),
                    "positive_energy_kev": float(pos_info.energies[pos_row]),
                    "positive_qc_status": pos_info.qc_status,
                    "negative_label": 0,
                    "negative_source": "co60",
                    "negative_hdf5": neg_info.hdf5_relative,
                    "negative_row": neg_row,
                    "negative_event_id": text_value(neg_info.event_ids[neg_row]),
                    "negative_energy_kev": float(neg_info.energies[neg_row]),
                    "negative_qc_status": neg_info.qc_status,
                    "source_weight": 1.0,
                }
            )
    positive_selected = np.asarray(
        [row["positive_energy_kev"] for row in rows], dtype=np.float64
    )
    negative_selected = np.asarray(
        [row["negative_energy_kev"] for row in rows], dtype=np.float64
    )
    energy_delta = positive_selected - negative_selected
    energy_diagnostics = {
        "mean_positive_minus_negative_kev": float(np.mean(energy_delta))
        if energy_delta.size
        else None,
        "median_abs_energy_difference_kev": float(np.median(np.abs(energy_delta)))
        if energy_delta.size
        else None,
        "p95_abs_energy_difference_kev": float(np.percentile(np.abs(energy_delta), 95))
        if energy_delta.size
        else None,
        "max_abs_energy_difference_kev": float(np.max(np.abs(energy_delta)))
        if energy_delta.size
        else None,
        "unpaired_energy_ks_distance": two_sample_ks_distance(
            positive_selected, negative_selected
        ),
    }
    return rows, {
        "peak_id": peak.peak_id,
        "source": peak.source,
        "roi_low_kev": peak.roi_low_kev,
        "roi_high_kev": peak.roi_high_kev,
        "positive_roi_candidates": positive_count,
        "negative_roi_candidates": negative_count,
        "common_energy_bins": len(set(positive) & set(negative)),
        "matchable_pairs_before_random_selection": sum(matched_by_bin.values()),
        "matched_pairs": len(rows),
        "matched_bins": matched_by_bin,
        "matched_energy_diagnostics": energy_diagnostics,
    }


def assign_pair_ids_and_weights(
    rows: list[dict[str, Any]], partition: str
) -> dict[str, float]:
    source_counts = Counter(row["positive_source"] for row in rows)
    active_sources = sorted(source_counts)
    if not active_sources:
        return {}
    total = len(rows)
    weights = {
        source: total / (len(active_sources) * count)
        for source, count in source_counts.items()
    }
    for index, row in enumerate(rows):
        row["pair_id"] = f"{partition}_{index:08d}"
        row["source_weight"] = weights[row["positive_source"]]
    return weights


def write_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def count_roi(energies: np.ndarray, low: float, high: float) -> int:
    return int(np.count_nonzero((energies >= low) & (energies <= high)))


def continuum_audit(
    co60_infos: Iterable[FileInfo],
    peaks: Iterable[PeakDefinition],
    partition: str | None = None,
) -> list[dict[str, Any]]:
    selected_infos = [
        info
        for info in co60_infos
        if info.energies is not None
        and (partition is None or info.partition == partition)
    ]
    arrays = [info.energies for info in selected_infos if info.energies is not None]
    if not arrays:
        return []
    energies = np.concatenate(arrays)
    audit: list[dict[str, Any]] = []
    for peak in peaks:
        half_w = peak.roi_half_width_kev
        window_low = peak.fitted_center_kev - 3 * peak.fwhm_kev
        window_high = peak.fitted_center_kev + 3 * peak.fwhm_kev
        bins = np.arange(window_low, window_high + BIN_WIDTH_KEV, BIN_WIDTH_KEV)
        counts, edges = np.histogram(energies, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        roi_mask = (centers >= peak.roi_low_kev) & (centers <= peak.roi_high_kev)
        sideband_mask = (np.abs(centers - peak.fitted_center_kev) >= 0.75 * peak.fwhm_kev) & (
            np.abs(centers - peak.fitted_center_kev) <= 2.0 * peak.fwhm_kev
        )
        roi_count = int(counts[roi_mask].sum())
        sideband_count = int(counts[sideband_mask].sum())
        sideband_width = 2.5 * peak.fwhm_kev
        roi_width = 2.0 * half_w
        estimated_background = sideband_count * (roi_width) / sideband_width if sideband_width > 0 else 0.0
        median_sideband_bin = float(np.median(counts[sideband_mask])) if np.any(sideband_mask) else 0.0
        max_roi_bin = int(counts[roi_mask].max()) if np.any(roi_mask) else 0
        ratio = (
            max_roi_bin / median_sideband_bin
            if median_sideband_bin > 0
            else float("inf") if max_roi_bin > 0 else 0.0
        )
        linear_background = None
        if np.count_nonzero(sideband_mask) >= 2:
            coefficients = np.polyfit(
                centers[sideband_mask], counts[sideband_mask], deg=1
            )
            linear_background = float(
                np.maximum(0.0, np.polyval(coefficients, centers[roi_mask])).sum()
            )
        audit.append(
            {
                "partition": partition or "all",
                "peak_id": peak.peak_id,
                "roi_count": roi_count,
                "sideband_count": sideband_count,
                "sideband_median_bin_count": median_sideband_bin,
                "max_roi_bin_count": max_roi_bin,
                "max_roi_to_sideband_median_ratio": ratio,
                "estimated_background_from_sidebands": estimated_background,
                "estimated_background_from_linear_sidebands": linear_background,
                "contamination_flag": bool(ratio > 3.0),
                "interpretation": "continuum candidate count; not individually proven Compton",
            }
        )
    return audit


def ensure_output_directory(
    output_dir: Path,
    overwrite: bool,
    include_test: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    expected = [
        output_dir / "label_dataset_manifest.json",
        output_dir / "file_partition_manifest.json",
        output_dir / "label_pairs_train.csv",
        output_dir / "label_pairs_validation.csv",
    ]
    if include_test:
        expected.append(output_dir / "label_pairs_test.csv")
    existing = [path for path in expected if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Outputs already exist: {names}; use --overwrite to replace them"
        )


def main() -> int:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.resolve()
    manifest_paths = (
        [args.preprocessing_manifest.resolve()]
        if args.preprocessing_manifest is not None
        else [p.resolve() for p in args.preprocessing_manifests]
    )
    qc_dir = args.qc_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.development_only and args.approved_allowlist is None:
        raise ValueError("--development-only requires --approved-allowlist to protect the locked test partition")

    records = load_preprocessing_records(manifest_paths)
    qc_statuses = load_qc_statuses(qc_dir)
    allowlist_data: dict[str, Any] | None = None
    partition_map: dict[str, str] | None = None
    admitted_hdf5: set[str] | None = None
    selected_hdf5_names: set[str] | None = None
    build_partitions = ("train", "validation") if args.development_only else PARTITIONS
    if args.approved_allowlist is not None:
        allowlist_path = args.approved_allowlist.resolve()
        allowlist_data, admitted_hdf5 = load_approved_allowlist(allowlist_path)
        partition_map = load_frozen_partition_map(args.partition_manifest.resolve())
        if args.development_only:
            admitted_hdf5 = {
                path for path in admitted_hdf5
                if partition_map.get(path) in build_partitions
            }
        selected_hdf5_names = {Path(path).name for path in admitted_hdf5}

    ensure_output_directory(
        output_dir,
        args.overwrite,
        include_test=not args.development_only,
    )
    eligible, excluded = load_file_infos(
        hdf5_dir,
        records,
        qc_statuses,
        args.energy_dataset,
        selected_hdf5_names=selected_hdf5_names,
    )
    if allowlist_data is not None and admitted_hdf5 is not None and partition_map is not None:
        eligible, excluded = apply_approved_allowlist(
            eligible,
            excluded,
            admitted_hdf5,
            partition_map,
        )
    else:
        assign_partitions(eligible, args.seed)
        validate_partition_disjointness(eligible)

    by_source = defaultdict(list)
    for info in eligible:
        by_source[info.source].append(info)
    co60_infos = by_source["co60"]
    if not co60_infos:
        raise RuntimeError("No complete processing_status=OK Co-60 files available")

    # In development-only mode this list contains no locked-test files. The
    # full waveform/energy arrays are loaded only for partitions being built.
    load_arrays(eligible, args.energy_dataset)

    all_peak_definitions = (*TRAINING_PEAKS, *DEVELOPMENT_PEAKS)
    audit = continuum_audit(co60_infos, all_peak_definitions)
    audit_by_partition = {
        partition: continuum_audit(co60_infos, all_peak_definitions, partition)
        for partition in build_partitions
    }

    pairs_by_partition: dict[str, list[dict[str, Any]]] = {}
    match_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_weights: dict[str, dict[str, float]] = {}
    for partition in build_partitions:
        partition_rows: list[dict[str, Any]] = []
        for peak in TRAINING_PEAKS:
            rows, detail = match_peak(
                eligible,
                co60_infos,
                partition,
                peak,
                args.seed,
            )
            partition_rows.extend(rows)
            match_details[partition].append(detail)
        partition_rows.sort(
            key=lambda row: (
                row["peak_id"],
                int(row["match_bin_index"]),
                row["positive_hdf5"],
                int(row["positive_row"]),
            )
        )
        source_weights[partition] = assign_pair_ids_and_weights(
            partition_rows, partition
        )
        pairs_by_partition[partition] = partition_rows
        write_pairs(output_dir / f"label_pairs_{partition}.csv", partition_rows)

    output_sha256 = {
        partition: sha256_file(output_dir / f"label_pairs_{partition}.csv")
        for partition in build_partitions
    }
    preprocessing_manifest_sha256 = {
        p.name: sha256_file(p) for p in manifest_paths if p.exists()
    }
    hdf5_hash_groups: dict[str, list[str]] = defaultdict(list)
    for info in eligible:
        hdf5_hash_groups[info.hdf5_sha256].append(info.hdf5_relative)
    duplicate_hdf5_hashes = {
        digest: paths for digest, paths in hdf5_hash_groups.items() if len(paths) > 1
    }
    qc_status_counts = dict(
        sorted(Counter(info.qc_status for info in eligible).items())
    )
    qc_status_counts_by_source = {
        source: dict(
            sorted(
                Counter(
                    info.qc_status
                    for info in eligible
                    if info.source == source
                ).items()
            )
        )
        for source in SOURCE_ORDER
    }

    file_records = [info.as_dict() for info in [*eligible, *excluded]]
    file_records.sort(key=lambda record: (record["source"], record["input_root_name"]))
    partition_counts = {
        source: {
            partition: sum(
                info.partition == partition
                for info in eligible
                if info.source == source
            )
            for partition in build_partitions
        }
        for source in SOURCE_ORDER
    }

    if allowlist_data is None:
        qc_policy = {
            "user_selected_mode": "all valid HDF5",
            "effective_file_policy": (
                "all complete processing-valid HDF5 files; input_entries must equal "
                f"{COMPLETE_INPUT_ENTRIES}"
            ),
            "processing_filter": "processing_status=OK",
            "file_completeness_filter": f"input_entries={COMPLETE_INPUT_ENTRIES}",
            "qc_status_filter": "none; every PASS/WARN/FAIL/UNKNOWN observation is retained",
            "warning": "Exploratory manifest; not a QC-locked training dataset.",
        }
        file_split_policy = {
            "seed": args.seed,
            "partitions": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "unit": "complete source HDF5 files, never individual events",
            "assignment": "new deterministic source-level assignment",
            "source_order": list(SOURCE_ORDER),
        }
    else:
        approved_policy = str(allowlist_data["approved_policy"])
        qc_policy = {
            "user_selected_mode": "approved_allowlist",
            "approved_policy": approved_policy,
            "effective_file_policy": "exactly the HDF5 paths listed by the frozen allowlist",
            "processing_filter": "processing_status=OK",
            "file_completeness_filter": f"input_entries={COMPLETE_INPUT_ENTRIES}",
            "qc_status_filter": "frozen allowlist decision; no status recomputation during rebuild",
            "warning": "Partition assignments are inherited from the prior immutable file manifest.",
        }
        file_split_policy = {
            "seed": None,
            "partitions": {"train": None, "validation": None, "test": None},
            "unit": "complete source HDF5 files, never individual events",
            "assignment": "inherited immutable assignments from --partition-manifest",
            "partition_manifest": args.partition_manifest.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "source_order": list(SOURCE_ORDER),
        }

    dataset_manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_mode": "index_only; source HDF5 waveform data is not copied",
        "development_only": args.development_only,
        "locked_test_event_rows_loaded": not args.development_only,
        "energy_dataset": args.energy_dataset,
        "qc_policy": qc_policy,
        "input_identity": {
            "preprocessing_manifests": [p.relative_to(PROJECT_ROOT).as_posix() for p in manifest_paths if p.exists()],
            "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
            "hdf5_sha256_algorithm": "sha256",
            "eligible_hdf5_hash_count": len(hdf5_hash_groups),
            "duplicate_hdf5_hashes": duplicate_hdf5_hashes,
            "output_csv_sha256": output_sha256,
            "approved_allowlist": (
                args.approved_allowlist.resolve().relative_to(PROJECT_ROOT).as_posix()
                if args.approved_allowlist is not None else None
            ),
            "approved_allowlist_sha256": (
                sha256_file(args.approved_allowlist.resolve())
                if args.approved_allowlist is not None else None
            ),
            "locked_test_event_rows_loaded": not args.development_only,
        },
        "file_split_policy": file_split_policy,
        "matching_policy": {
            "negative_source": "co60 continuum candidates",
            "energy_bin_width_kev": BIN_WIDTH_KEV,
            "one_to_one_without_replacement": True,
            "matching_scope": "independently within each partition and energy bin",
            "source_domain_weighting": "inverse frequency weight in each partition; all matched pairs retained",
            "continuum_warning": "Co-60 ROI events are continuum candidates, not individually proven Compton interactions.",
        },
        "training_peaks": [peak.as_dict() for peak in TRAINING_PEAKS],
        "development_peaks_not_matched": [peak.as_dict() for peak in DEVELOPMENT_PEAKS],
        "eligible_file_count": len(eligible),
        "excluded_file_count": len(excluded),
        "eligible_files_with_nonpass_qc_status": sum(
            info.qc_status not in {"PASS", "UNKNOWN"} for info in eligible
        ),
        "qc_status_counts": qc_status_counts,
        "qc_status_counts_by_source": qc_status_counts_by_source,
        "file_partition_disjoint": True,
        "accepted_events_by_source": {
            source: sum(info.accepted_entries for info in eligible if info.source == source)
            for source in SOURCE_ORDER
        },
        "complete_files_by_source_and_partition": partition_counts,
        "match_details_by_partition": match_details,
        "matched_pairs_by_partition": {
            partition: len(pairs_by_partition[partition]) for partition in build_partitions
        },
        "matched_pairs_total": sum(len(rows) for rows in pairs_by_partition.values()),
        "source_weights_by_partition": source_weights,
        "co60_continuum_audit_before_matching": audit,
        "co60_continuum_audit_by_partition": audit_by_partition,
        "output_files": {
            partition: f"outputs/labels/label_pairs_{partition}.csv"
            for partition in build_partitions
        },
    }
    partition_manifest = {
        "manifest_version": 1,
        "created_utc": dataset_manifest["created_utc"],
        "selection_policy": dataset_manifest["qc_policy"],
        "files": file_records,
    }
    write_json(output_dir / "label_dataset_manifest.json", dataset_manifest)
    write_json(output_dir / "file_partition_manifest.json", partition_manifest)

    print(json.dumps({
        "eligible_files": len(eligible),
        "excluded_files": len(excluded),
        "eligible_events_by_source": dataset_manifest["accepted_events_by_source"],
        "eligible_files_with_nonpass_qc_status": dataset_manifest[
            "eligible_files_with_nonpass_qc_status"
        ],
        "complete_files_by_source_and_partition": partition_counts,
        "matched_pairs_by_partition": dataset_manifest["matched_pairs_by_partition"],
        "matched_pairs_total": dataset_manifest["matched_pairs_total"],
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
