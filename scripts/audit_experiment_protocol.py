#!/usr/bin/env python3
"""Build an auditable draft protocol for the architecture experiment.

This is a non-destructive WP0 audit. It reads the existing file-partition
manifest and only the train/validation pair manifests; it deliberately does not
read locked-test event rows, Th-232 files, or waveform arrays. The output is a
machine-readable decision bundle under ``outputs/protocol``.

The QC policy is not silently frozen. Historical QC observations are preserved,
classified by reason, and mapped to candidate acquisition-session domains. A
user-approved allowlist is required before labels or event stores are rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE_MANIFEST = PROJECT_ROOT / "outputs" / "labels" / "file_partition_manifest.json"
DEFAULT_LABEL_DIR = PROJECT_ROOT / "outputs" / "labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "protocol"

POLICIES: dict[str, dict[str, Any]] = {
    "current_complete": {
        "admit_qc_statuses": ["PASS", "WARN", "FAIL", "UNKNOWN"],
        "rationale": "Current exploratory eligibility: retain every complete processing_status=OK file.",
    },
    "pass_warn": {
        "admit_qc_statuses": ["PASS", "WARN"],
        "rationale": (
            "Candidate development policy: retain valid complete files with PASS or WARN; "
            "exclude hard FAIL and UNKNOWN files pending reason-level adjudication."
        ),
    },
    "pass_only": {
        "admit_qc_statuses": ["PASS"],
        "rationale": "Candidate strict policy: retain only complete processing_status=OK files with PASS status.",
    },
}

HARD_QC_REASON_CATEGORIES = {
    "io_or_missing_root",
    "structural_integrity",
    "nonfinite_waveform",
    "metadata_mismatch",
    "timestamp_integrity",
}
SOFT_QC_REASON_CATEGORIES = {"baseline_noise", "rail_or_clipping", "baseline_other"}
PAIR_COLUMNS = {
    "pair_id",
    "partition",
    "peak_id",
    "positive_source",
    "positive_hdf5",
    "positive_row",
    "positive_event_id",
    "positive_energy_kev",
    "positive_qc_status",
    "negative_source",
    "negative_hdf5",
    "negative_row",
    "negative_event_id",
    "negative_energy_kev",
    "negative_qc_status",
}
TIMESTAMP_RE = re.compile(r"_(?P<date>20\d{6})_(?P<time>\d{6})_thr", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty files list in {path}")
    by_hdf5: dict[str, dict[str, Any]] = {}
    for record in records:
        hdf5 = record.get("hdf5")
        if not hdf5:
            raise ValueError(f"File record has no hdf5 path in {path}: {record}")
        if hdf5 in by_hdf5:
            raise ValueError(f"Duplicate HDF5 record in {path}: {hdf5}")
        by_hdf5[str(hdf5)] = record
    return data, by_hdf5


def parse_acquisition_block(input_root_name: str, source: str) -> dict[str, str]:
    match = TIMESTAMP_RE.search(input_root_name)
    if not match:
        return {
            "acquisition_date": "",
            "acquisition_time": "",
            "acquisition_block_id": Path(input_root_name).stem,
        }
    date = match.group("date")
    time = match.group("time")
    return {
        "acquisition_date": date,
        "acquisition_time": time,
        "acquisition_block_id": f"{source}_{date}_{time}",
    }


def report_role(session_id: str | None) -> str:
    """Classify report provenance without treating every report as authoritative."""

    value = str(session_id or "").lower()
    if "all_sources" in value or "raw_data_all" in value:
        return "aggregate_audit"
    if "progress" in value:
        return "progress_audit"
    if "postprocess" in value:
        return "postprocess_audit"
    if value.endswith("_corrected") or value.endswith("_tighter"):
        return "alternate_threshold_audit"
    return "canonical_session"


def normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "")
    value = re.sub(r"_(corrected|tighter)$", "", value)
    return value


def classify_qc_message(message: str) -> str:
    value = message.lower()
    if "cannot read root" in value or "failed to open file" in value or "cannot open root" in value:
        return "io_or_missing_root"
    if "non-finite" in value or "nonfinite" in value:
        return "nonfinite_waveform"
    if "metadata mismatch" in value:
        return "metadata_mismatch"
    if "timestamp" in value or "trigger" in value:
        return "timestamp_integrity"
    if "entry count" in value or "waveform shape" in value or "sample could not be read" in value:
        return "structural_integrity"
    if "baseline noise" in value:
        return "baseline_noise"
    if "rail" in value or "clipping" in value:
        return "rail_or_clipping"
    if "baseline" in value:
        return "baseline_other"
    return "other"


def extract_qc_metrics(report_data: dict[str, Any], report_file: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable QC metrics and thresholds from one file observation."""

    reference = report_data.get("reference_metrics") or {}
    reference_baseline = reference.get("baseline") or {}
    reference_noise = reference_baseline.get("baseline_noise_rms_adc") or {}
    baseline = report_file.get("baseline") or {}
    noise = baseline.get("baseline_noise_rms_adc") or {}
    integrity = report_file.get("integrity") or {}
    waveform_integrity = report_file.get("waveform_integrity") or {}
    timing = report_file.get("timing") or {}
    thresholds = report_data.get("thresholds") or {}
    warn_factor = thresholds.get("noise_warn_factor")
    fail_factor = thresholds.get("noise_fail_factor")

    def scaled(value: Any, factor: Any) -> float | None:
        if value is None or factor is None:
            return None
        return float(value) * float(factor)

    p95_reference = reference_noise.get("p95")
    p99_reference = reference_noise.get("p99")
    p95_value = noise.get("p95")
    p99_value = noise.get("p99")
    return {
        "reference_root": report_data.get("reference_root"),
        "report_sample_mode": report_data.get("sample_mode"),
        "file_sample_mode": integrity.get("sample_mode"),
        "baseline_noise_numpy_slice": baseline.get("noise_numpy_slice"),
        "baseline_noise_samples_inclusive": baseline.get("noise_samples_inclusive"),
        "baseline_noise_p95_adc": p95_value,
        "baseline_noise_p99_adc": p99_value,
        "reference_noise_p95_adc": p95_reference,
        "reference_noise_p99_adc": p99_reference,
        "noise_warn_factor": warn_factor,
        "noise_fail_factor": fail_factor,
        "noise_p95_warn_limit_adc": scaled(p95_reference, warn_factor),
        "noise_p95_fail_limit_adc": scaled(p95_reference, fail_factor),
        "noise_p99_warn_limit_adc": scaled(p99_reference, warn_factor),
        "noise_p99_fail_limit_adc": scaled(p99_reference, fail_factor),
        "noise_p95_ratio_to_reference": (
            float(p95_value) / float(p95_reference)
            if p95_value is not None and p95_reference not in (None, 0) else None
        ),
        "noise_p99_ratio_to_reference": (
            float(p99_value) / float(p99_reference)
            if p99_value is not None and p99_reference not in (None, 0) else None
        ),
        "lower_rail_event_fraction": waveform_integrity.get("lower_rail_event_fraction"),
        "upper_rail_event_fraction": waveform_integrity.get("upper_rail_event_fraction"),
        "max_rail_event_fraction": max(
            float(waveform_integrity.get("lower_rail_event_fraction") or 0.0),
            float(waveform_integrity.get("upper_rail_event_fraction") or 0.0),
        ),
        "flatline_event_fraction": waveform_integrity.get("flatline_event_fraction"),
        "duplicate_consecutive_event_fraction": waveform_integrity.get("duplicate_consecutive_event_fraction"),
        "entries": integrity.get("entries"),
        "sampled_entries": integrity.get("sampled_entries"),
        "waveform_shape": integrity.get("waveform_shape"),
        "timestamp_monotonic": timing.get("timestamp_monotonic"),
        "failure_limit_rules": {
            "pedestal_warn_adc": thresholds.get("pedestal_warn_adc"),
            "pedestal_fail_adc": thresholds.get("pedestal_fail_adc"),
            "drift_warn_adc": thresholds.get("drift_warn_adc"),
            "drift_fail_adc": thresholds.get("drift_fail_adc"),
        },
    }


def observation_values(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    observations = record.get("qc_observations") or []
    statuses = sorted({str(item.get("status", "UNKNOWN")).upper() for item in observations})
    report_paths = sorted({str(item.get("report", "")) for item in observations if item.get("report")})
    session_ids = sorted(
        {
            normalize_session_id(item.get("session_id"))
            for item in observations
            if item.get("session_id") and report_role(item.get("session_id")) == "canonical_session"
        }
    )
    return statuses, report_paths, session_ids


def policy_admits(record: dict[str, Any], policy_id: str) -> bool:
    policy = POLICIES[policy_id]
    return (
        record.get("processing_status") == "OK"
        and bool(record.get("complete_input"))
        and str(record.get("qc_status", "UNKNOWN")).upper() in policy["admit_qc_statuses"]
    )


def count_nested(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def source_partition_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{record.get('source')}|{record.get('partition')}"
                for record in records
            ).items()
        )
    )


def summarize_policy(records: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    admitted = [record for record in records if policy_admits(record, policy_id)]
    excluded = [record for record in records if not policy_admits(record, policy_id)]
    reasons = Counter()
    for record in excluded:
        if record.get("processing_status") != "OK":
            reasons[f"processing_status={record.get('processing_status')}"] += 1
        elif not record.get("complete_input"):
            reasons["incomplete_input"] += 1
        else:
            reasons[f"qc_status={str(record.get('qc_status', 'UNKNOWN')).upper()}"] += 1
    return {
        "policy_id": policy_id,
        "admit_processing_status": ["OK"],
        "require_complete_input": True,
        "admit_qc_statuses": POLICIES[policy_id]["admit_qc_statuses"],
        "rationale": POLICIES[policy_id]["rationale"],
        "admitted_file_count": len(admitted),
        "excluded_file_count": len(excluded),
        "admitted_by_source": count_nested(admitted, "source"),
        "admitted_by_partition": count_nested(admitted, "partition"),
        "admitted_by_source_partition": source_partition_counts(admitted),
        "excluded_reasons": dict(sorted(reasons.items())),
        "admitted_hdf5": sorted(str(record["hdf5"]) for record in admitted),
        "excluded_hdf5": sorted(str(record["hdf5"]) for record in excluded),
    }


def load_qc_adjudication(
    file_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load all QC reports and retain competing observations per HDF5 file."""

    report_cache: dict[str, tuple[dict[str, Any], str | None, bool]] = {}
    details_by_hdf5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_reports: set[str] = set()
    report_hash_mismatches: list[dict[str, str]] = []

    for record in file_records.values():
        hdf5 = str(record["hdf5"])
        for observation in record.get("qc_observations") or []:
            report_rel = str(observation.get("report", ""))
            if not report_rel:
                continue
            if report_rel not in report_cache:
                report_path = PROJECT_ROOT / report_rel
                if not report_path.exists():
                    report_cache[report_rel] = ({}, None, False)
                    missing_reports.add(report_rel)
                else:
                    report_bytes = report_path.read_bytes()
                    report_hash = hashlib.sha256(report_bytes).hexdigest()
                    report_data = json.loads(report_bytes.decode("utf-8"))
                    report_cache[report_rel] = (report_data, report_hash, True)
            report_data, current_hash, exists = report_cache[report_rel]
            recorded_hash = observation.get("report_sha256")
            if exists and recorded_hash and recorded_hash != current_hash:
                report_hash_mismatches.append(
                    {
                        "report": report_rel,
                        "recorded_sha256": str(recorded_hash),
                        "current_sha256": str(current_hash),
                    }
                )
            file_name = Path(str(record.get("input_root_name", ""))).name
            report_file = next(
                (
                    item
                    for item in report_data.get("files", [])
                    if Path(str(item.get("file_name") or item.get("file") or "")).name == file_name
                ),
                None,
            )
            failures = list((report_file or {}).get("failures", observation.get("failures", [])) or [])
            warnings = list((report_file or {}).get("warnings", observation.get("warnings", [])) or [])
            status = str((report_file or {}).get("status", observation.get("status", "UNKNOWN"))).upper()
            failure_categories = sorted({classify_qc_message(str(message)) for message in failures})
            warning_categories = sorted({classify_qc_message(str(message)) for message in warnings})
            session_id = str(observation.get("session_id") or report_data.get("session_id") or "")
            details_by_hdf5[hdf5].append(
                {
                    "report": report_rel,
                    "report_sha256_recorded": recorded_hash,
                    "report_sha256_current": current_hash,
                    "report_exists": exists,
                    "report_hash_match": bool(exists and (not recorded_hash or recorded_hash == current_hash)),
                    "session_id": session_id,
                    "normalized_session_id": normalize_session_id(session_id),
                    "report_role": report_role(session_id),
                    "created_utc": observation.get("created_utc") or report_data.get("created_utc"),
                    "sample_mode": observation.get("sample_mode") or report_data.get("sample_mode"),
                    "status": status,
                    "failures": failures,
                    "warnings": warnings,
                    "failure_categories": failure_categories,
                    "warning_categories": warning_categories,
                    **extract_qc_metrics(report_data, report_file or {}),
                }
            )

    adjudication: dict[str, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for hdf5, record in file_records.items():
        observations = details_by_hdf5.get(hdf5, [])
        canonical_sessions = sorted(
            {
                item["normalized_session_id"]
                for item in observations
                if item["report_role"] == "canonical_session" and item["normalized_session_id"]
            }
        )
        all_categories = sorted(
            {
                category
                for item in observations
                for category in (*item["failure_categories"], *item["warning_categories"])
            }
        )
        for category in all_categories:
            category_counts[category] += 1
        if any(category in HARD_QC_REASON_CATEGORIES for category in all_categories):
            decision = "hard_exclude_candidate"
        elif any(category in SOFT_QC_REASON_CATEGORIES for category in all_categories):
            decision = "soft_issue_review_required"
        elif not observations:
            decision = "unknown_review_required"
        elif "other" in all_categories:
            decision = "unclassified_review_required"
        else:
            decision = "no_reason_issue_observed"
        decision_counts[decision] += 1
        if len(canonical_sessions) == 1:
            canonical_session_id = canonical_sessions[0]
            session_mapping_status = "mapped_from_canonical_qc_report"
        elif len(canonical_sessions) > 1:
            canonical_session_id = "AMBIGUOUS:" + ";".join(canonical_sessions)
            session_mapping_status = "multiple_canonical_qc_reports"
        else:
            input_root_name = str(record.get("input_root_name", Path(str(record.get("input_root", ""))).name))
            block = parse_acquisition_block(input_root_name, str(record.get("source", "")))
            canonical_session_id = block["acquisition_block_id"]
            session_mapping_status = "filename_block_fallback"
        adjudication[hdf5] = {
            "observations": observations,
            "observation_count": len(observations),
            "historical_worst_status": str(record.get("qc_status", "UNKNOWN")).upper(),
            "canonical_session_candidates": canonical_sessions,
            "canonical_session_id": canonical_session_id,
            "session_mapping_status": session_mapping_status,
            "reason_categories": all_categories,
            "reason_level_decision": decision,
        }

    summary = {
        "report_count": len(report_cache),
        "missing_reports": sorted(missing_reports),
        "report_hash_mismatches": report_hash_mismatches,
        "reason_category_file_counts": dict(sorted(category_counts.items())),
        "reason_level_decision_counts": dict(sorted(decision_counts.items())),
        "hard_reason_categories": sorted(HARD_QC_REASON_CATEGORIES),
        "soft_reason_categories": sorted(SOFT_QC_REASON_CATEGORIES),
        "authoritative_status": None,
        "authoritative_precedence": [
            "No single status is selected in this draft.",
            "Canonical session reports are separated from aggregate, progress, postprocess, and alternate-threshold audits.",
            "Reason-level adjudication and user approval are required before an allowlist is frozen.",
        ],
    }
    return adjudication, summary


def session_domain_audit(
    file_records: dict[str, dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sessions: dict[str, dict[str, Any]] = defaultdict(lambda: {"partitions": set(), "sources": set(), "hdf5": []})
    ambiguous = 0
    fallback = 0
    for hdf5, record in file_records.items():
        if record.get("processing_status") != "OK" or not record.get("complete_input"):
            continue
        decision = adjudication[hdf5]
        mapping_status = decision["session_mapping_status"]
        if mapping_status == "multiple_canonical_qc_reports":
            ambiguous += 1
        if mapping_status == "filename_block_fallback":
            fallback += 1
        session_id = decision["canonical_session_id"]
        if session_id.startswith("AMBIGUOUS:"):
            continue
        sessions[session_id]["partitions"].add(str(record.get("partition")))
        sessions[session_id]["sources"].add(str(record.get("source")))
        sessions[session_id]["hdf5"].append(hdf5)
    spanning = []
    for session_id, value in sorted(sessions.items()):
        partitions = sorted(value["partitions"])
        if len(partitions) > 1:
            spanning.append(
                {
                    "session_id": session_id,
                    "partitions": partitions,
                    "sources": sorted(value["sources"]),
                    "file_count": len(value["hdf5"]),
                    "hdf5": sorted(value["hdf5"]),
                }
            )
    return {
        "canonical_session_count": len(sessions),
        "ambiguous_file_count": ambiguous,
        "filename_fallback_file_count": fallback,
        "sessions_spanning_multiple_partitions": spanning,
        "session_holdout_is_file_partition_disjoint": not spanning and ambiguous == 0,
        "interpretation": (
            "A session appearing in multiple train/validation/test partitions is not a clean session holdout."
        ),
    }


def verify_input_integrity(
    file_records: dict[str, dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
    verify_hdf5_hashes: bool,
) -> dict[str, Any]:
    missing_hdf5: list[str] = []
    hdf5_mismatches: list[dict[str, str]] = []
    hdf5_checked = 0
    for hdf5, record in file_records.items():
        path = PROJECT_ROOT / hdf5
        if not path.exists():
            missing_hdf5.append(hdf5)
            continue
        if verify_hdf5_hashes:
            hdf5_checked += 1
            current_hash = sha256_file(path)
            recorded_hash = str(record.get("hdf5_sha256", ""))
            if recorded_hash and current_hash != recorded_hash:
                hdf5_mismatches.append(
                    {
                        "hdf5": hdf5,
                        "recorded_sha256": recorded_hash,
                        "current_sha256": current_hash,
                    }
                )
    report_hashes = [
        observation
        for value in adjudication.values()
        for observation in value["observations"]
    ]
    report_mismatches = [
        {
            "report": str(item["report"]),
            "recorded_sha256": str(item.get("report_sha256_recorded")),
            "current_sha256": str(item.get("report_sha256_current")),
        }
        for item in report_hashes
        if item.get("report_exists") and not item.get("report_hash_match")
    ]
    return {
        "hdf5_hashes_requested": verify_hdf5_hashes,
        "hdf5_hashes_checked": hdf5_checked,
        "missing_hdf5": sorted(set(missing_hdf5)),
        "hdf5_hash_mismatches": hdf5_mismatches,
        "qc_report_hash_observations": len(report_hashes),
        "qc_report_hash_mismatches": report_mismatches,
        "integrity_pass": not missing_hdf5 and not hdf5_mismatches and not report_mismatches,
        "waveform_arrays_loaded": False,
    }


def pair_reference(record: dict[str, Any], side: str) -> tuple[str, str]:
    return str(record[f"{side}_hdf5"]), str(record[f"{side}_row"])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    index = fraction * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_pairs(
    label_dir: Path,
    file_records: dict[str, dict[str, Any]],
    policy_ids: Iterable[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "locked_test_event_rows_read": False,
        "partitions_audited": ["train", "validation"],
        "partitions_reserved": ["test"],
        "by_partition": {},
    }
    for partition in ("train", "validation"):
        path = label_dir / f"label_pairs_{partition}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        total = 0
        invalid_rows = 0
        duplicate_positive_events = 0
        duplicate_negative_events = 0
        positive_events: set[tuple[str, str]] = set()
        negative_events: set[tuple[str, str]] = set()
        source_counts: Counter[str] = Counter()
        peak_counts: Counter[str] = Counter()
        qc_counts: Counter[str] = Counter()
        deltas: list[float] = []
        policy_kept = Counter()
        referenced_files: set[str] = set()
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = PAIR_COLUMNS - columns
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row in reader:
                total += 1
                try:
                    positive_hdf5, positive_row = pair_reference(row, "positive")
                    negative_hdf5, negative_row = pair_reference(row, "negative")
                    positive_energy = float(row["positive_energy_kev"])
                    negative_energy = float(row["negative_energy_kev"])
                    if row["partition"] != partition:
                        raise ValueError("partition column disagrees with filename")
                    if positive_hdf5 not in file_records or negative_hdf5 not in file_records:
                        raise ValueError("pair references an unknown HDF5 file")
                    if file_records[positive_hdf5].get("partition") != partition:
                        raise ValueError("positive file partition mismatch")
                    if file_records[negative_hdf5].get("partition") != partition:
                        raise ValueError("negative file partition mismatch")
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
                    continue
                referenced_files.update((positive_hdf5, negative_hdf5))
                source_counts[row["positive_source"]] += 1
                peak_counts[row["peak_id"]] += 1
                qc_counts[f"positive={row['positive_qc_status']}"] += 1
                qc_counts[f"negative={row['negative_qc_status']}"] += 1
                deltas.append(abs(positive_energy - negative_energy))
                positive_key = (positive_hdf5, positive_row)
                negative_key = (negative_hdf5, negative_row)
                if positive_key in positive_events:
                    duplicate_positive_events += 1
                positive_events.add(positive_key)
                if negative_key in negative_events:
                    duplicate_negative_events += 1
                negative_events.add(negative_key)
                for policy_id in policy_ids:
                    if policy_admits(file_records[positive_hdf5], policy_id) and policy_admits(
                        file_records[negative_hdf5], policy_id
                    ):
                        policy_kept[policy_id] += 1
        deltas.sort()
        output["by_partition"][partition] = {
            "csv": path.relative_to(PROJECT_ROOT).as_posix(),
            "csv_sha256": sha256_file(path),
            "pair_count": total,
            "invalid_rows": invalid_rows,
            "referenced_file_count": len(referenced_files),
            "positive_event_duplicate_rows": duplicate_positive_events,
            "negative_event_duplicate_rows": duplicate_negative_events,
            "positive_source_pair_counts": dict(sorted(source_counts.items())),
            "peak_pair_counts": dict(sorted(peak_counts.items())),
            "qc_status_counts": dict(sorted(qc_counts.items())),
            "energy_match_kev": {
                "max_abs_difference": max(deltas) if deltas else None,
                "p95_abs_difference": percentile(deltas, 0.95) if deltas else None,
                "median_abs_difference": percentile(deltas, 0.50) if deltas else None,
            },
            "estimated_pairs_retained_by_policy": {
                policy_id: policy_kept[policy_id] for policy_id in policy_ids
            },
            "retention_estimates_are_not_rebuilt_labels": True,
        }
    return output


def build_metric_adjudication_rows(
    records: Iterable[dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten every QC observation into a reviewable metric-level CSV row."""

    rows: list[dict[str, Any]] = []
    for record in records:
        hdf5 = str(record.get("hdf5", ""))
        source = str(record.get("source", ""))
        input_root_name = str(record.get("input_root_name", Path(str(record.get("input_root", ""))).name))
        file_adjudication = adjudication[hdf5]
        observations = file_adjudication["observations"]
        statuses = sorted({str(item.get("status", "UNKNOWN")) for item in observations})
        roles = sorted({str(item.get("report_role", "")) for item in observations})
        contradictory = len(statuses) > 1 or len(roles) > 1
        for index, observation in enumerate(observations):
            rows.append(
                {
                    "hdf5": hdf5,
                    "input_root_name": input_root_name,
                    "source": source,
                    "partition": record.get("partition") or "",
                    "canonical_session_id": file_adjudication["canonical_session_id"],
                    "session_mapping_status": file_adjudication["session_mapping_status"],
                    "observation_index": index,
                    "observation_count": len(observations),
                    "historical_worst_status": file_adjudication["historical_worst_status"],
                    "status_set": ";".join(statuses),
                    "report_role_set": ";".join(roles),
                    "contradictory_or_superseded_observations": contradictory,
                    "report": observation.get("report"),
                    "report_role": observation.get("report_role"),
                    "session_id": observation.get("session_id"),
                    "created_utc": observation.get("created_utc"),
                    "sample_mode": observation.get("sample_mode"),
                    "status": observation.get("status"),
                    "reference_root": observation.get("reference_root"),
                    "baseline_noise_numpy_slice": observation.get("baseline_noise_numpy_slice"),
                    "baseline_noise_samples_inclusive": json.dumps(observation.get("baseline_noise_samples_inclusive")),
                    "baseline_noise_p95_adc": observation.get("baseline_noise_p95_adc"),
                    "baseline_noise_p99_adc": observation.get("baseline_noise_p99_adc"),
                    "reference_noise_p95_adc": observation.get("reference_noise_p95_adc"),
                    "reference_noise_p99_adc": observation.get("reference_noise_p99_adc"),
                    "noise_warn_factor": observation.get("noise_warn_factor"),
                    "noise_fail_factor": observation.get("noise_fail_factor"),
                    "noise_p95_warn_limit_adc": observation.get("noise_p95_warn_limit_adc"),
                    "noise_p95_fail_limit_adc": observation.get("noise_p95_fail_limit_adc"),
                    "noise_p99_warn_limit_adc": observation.get("noise_p99_warn_limit_adc"),
                    "noise_p99_fail_limit_adc": observation.get("noise_p99_fail_limit_adc"),
                    "noise_p95_ratio_to_reference": observation.get("noise_p95_ratio_to_reference"),
                    "noise_p99_ratio_to_reference": observation.get("noise_p99_ratio_to_reference"),
                    "lower_rail_event_fraction": observation.get("lower_rail_event_fraction"),
                    "upper_rail_event_fraction": observation.get("upper_rail_event_fraction"),
                    "max_rail_event_fraction": observation.get("max_rail_event_fraction"),
                    "flatline_event_fraction": observation.get("flatline_event_fraction"),
                    "duplicate_consecutive_event_fraction": observation.get("duplicate_consecutive_event_fraction"),
                    "entries": observation.get("entries"),
                    "sampled_entries": observation.get("sampled_entries"),
                    "waveform_shape": json.dumps(observation.get("waveform_shape")),
                    "timestamp_monotonic": observation.get("timestamp_monotonic"),
                    "failure_categories": ";".join(observation.get("failure_categories", [])),
                    "warning_categories": ";".join(observation.get("warning_categories", [])),
                    "failures": " | ".join(str(value) for value in observation.get("failures", [])),
                    "warnings": " | ".join(str(value) for value in observation.get("warnings", [])),
                    "reason_level_decision": file_adjudication["reason_level_decision"],
                    "authoritative_status": "PENDING_ADJUDICATION",
                }
            )
    return rows


def build_domain_rows(
    records: Iterable[dict[str, Any]],
    adjudication: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        hdf5 = str(record.get("hdf5", ""))
        source = str(record.get("source", ""))
        input_root_name = str(record.get("input_root_name", Path(str(record.get("input_root", ""))).name))
        timestamp = parse_acquisition_block(input_root_name, source)
        statuses, reports, session_ids = observation_values(record)
        file_adjudication = adjudication[hdf5]
        rows.append(
            {
                "hdf5": hdf5,
                "input_root_name": input_root_name,
                "source": source,
                "partition": record.get("partition") or "",
                "acquisition_date": timestamp["acquisition_date"],
                "acquisition_time": timestamp["acquisition_time"],
                "acquisition_block_id": timestamp["acquisition_block_id"],
                "canonical_session_id": file_adjudication["canonical_session_id"],
                "session_mapping_status": file_adjudication["session_mapping_status"],
                "canonical_session_candidates": ";".join(file_adjudication["canonical_session_candidates"]),
                "processing_status": record.get("processing_status", ""),
                "complete_input": bool(record.get("complete_input")),
                "input_entries": record.get("input_entries", ""),
                "accepted_entries": record.get("accepted_entries", ""),
                "qc_status": record.get("qc_status", "UNKNOWN"),
                "qc_observed_statuses": ";".join(statuses),
                "qc_session_ids": ";".join(session_ids),
                "qc_reports": ";".join(reports),
                "qc_reason_categories": ";".join(file_adjudication["reason_categories"]),
                "qc_reason_level_decision": file_adjudication["reason_level_decision"],
                "hdf5_sha256": record.get("hdf5_sha256", ""),
                "hdf5_exists": (PROJECT_ROOT / hdf5).exists(),
                "currently_eligible": policy_admits(record, "current_complete"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_architecture_protocol(
    manifest_path: Path,
    train_csv_hash: str,
    validation_csv_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "DRAFT_PENDING_QC_ADJUDICATION",
        "created_utc": utc_now(),
        "purpose": "Controlled comparison of FPGA-oriented HPGe waveform architectures.",
        "selection_boundary": {
            "architecture_selection_may_use": [
                "train partition",
                "internal validation partition",
                "source/session/QC robustness probes",
                "parameter/MAC/memory estimates",
                "hardware synthesis results after target definition",
            ],
            "architecture_selection_must_not_use": [
                "locked test event rows",
                "Th-232 scores or spectra",
                "Eu-152 scores or spectra",
            ],
            "external_transfer_status": (
                "Existing Th-232 results are secondary exploratory/confirmatory evidence only; "
                "clean architecture-level confirmation requires untouched external data."
            ),
        },
        "input_data": {
            "file_partition_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "train_pairs": "outputs/labels/label_pairs_train.csv",
            "validation_pairs": "outputs/labels/label_pairs_validation.csv",
            "train_pairs_sha256": train_csv_hash,
            "validation_pairs_sha256": validation_csv_hash,
            "locked_test_manifest": "outputs/labels/label_pairs_test.csv",
            "locked_test_metadata_inspected_via_file_partition_manifest": True,
            "locked_test_event_rows_read_by_this_audit": False,
            "locked_test_procedure_caveat": (
                "Legacy O2 provenance hashed test CSV bytes; no test rows were parsed or scored."
            ),
        },
        "common_experiment_contract": {
            "waveform_samples_available": 4500,
            "common_window_samples": None,
            "input_channels": ["baseline_subtracted_charge", "current_or_derivative"],
            "normalization": "TBD and must be identical unless intrinsic to a declared architecture hypothesis",
            "loss_family": "TBD; retain current weighting policy unless explicitly frozen otherwise",
            "checkpoint_metric": "validation source-weighted AUROC",
            "stopping_policy": "TBD before screening",
            "seed_schedule": "TBD before screening",
            "tuning_budget": "TBD; bounded screening before finalist optimization",
            "provenance_inputs": ["pair CSV hashes", "event-store hashes", "code revision", "environment"],
        },
        "architectures": [
            {"id": "compact_cnn", "role": "simple baseline", "status": "existing"},
            {"id": "registered_residual_cnn", "role": "current performance baseline", "status": "existing"},
            {"id": "o2_late_fusion", "role": "charge/current fusion baseline", "status": "existing"},
            {"id": "ds_cnn", "role": "depthwise-separable hardware-efficiency hypothesis", "status": "to_implement"},
            {"id": "tcn", "role": "systematic long-receptive-field hypothesis", "status": "to_implement"},
            {"id": "multi_rate_hpge", "role": "fast/slow detector-timescale hypothesis", "status": "to_implement"},
            {"id": "cnn_gru", "role": "explicit temporal-memory comparison", "status": "to_implement"},
        ],
        "hardware_contract": {
            "status": "BLOCKED_UNTIL_DEFINED",
            "target_device": None,
            "tool_flow": None,
            "input_precision": None,
            "throughput_requirement": None,
            "latency_budget": None,
            "resource_limits": None,
            "note": "MAC/parameter estimates are not FPGA feasibility evidence without this contract.",
        },
        "gates": [
            {"id": "qc", "status": "pending_adjudication_and_user_approval", "artifact": "outputs/protocol/qc_allowlist.json"},
            {"id": "preflight", "status": "audit_generated", "artifact": "outputs/protocol/preflight_audit.json"},
            {"id": "shortcut", "status": "pending", "artifact": None},
            {"id": "architecture_screen", "status": "blocked_until_qc_and_protocol", "artifact": None},
            {"id": "hardware", "status": "blocked_until_target_contract", "artifact": None},
            {"id": "external", "status": "predeclared_only", "artifact": None},
        ],
    }


def build_preflight(
    manifest_data: dict[str, Any],
    manifest_path: Path,
    file_records: dict[str, dict[str, Any]],
    pair_summary: dict[str, Any],
    policy_summaries: dict[str, dict[str, Any]],
    qc_summary: dict[str, Any],
    integrity: dict[str, Any],
    session_audit: dict[str, Any],
) -> dict[str, Any]:
    complete_records = [
        record
        for record in file_records.values()
        if record.get("processing_status") == "OK" and record.get("complete_input")
    ]
    partition_files: dict[str, set[str]] = {}
    for record in complete_records:
        partition = record.get("partition")
        if partition:
            partition_files.setdefault(str(partition), set()).add(str(record["hdf5"]))
    overlap = {
        left: {
            right: sorted(partition_files.get(left, set()) & partition_files.get(right, set()))
            for right in partition_files
            if right > left
        }
        for left in partition_files
    }
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "AUDIT_COMPLETE_QC_POLICY_NOT_FROZEN",
        "source_manifest": {
            "path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_version": manifest_data.get("manifest_version"),
            "record_count": len(file_records),
        },
        "file_counts": {
            "all_records": len(file_records),
            "complete_processing_ok": len(complete_records),
            "by_source": count_nested(complete_records, "source"),
            "by_partition": count_nested(complete_records, "partition"),
            "by_qc_status": count_nested(complete_records, "qc_status"),
            "by_source_partition": source_partition_counts(complete_records),
        },
        "partition_audit": {
            "complete_file_partition_overlap": overlap,
            "partition_assignments_are_inherited": True,
            "file_partition_manifest_reassigned": False,
        },
        "session_domain_audit": session_audit,
        "qc_observation_adjudication": qc_summary,
        "input_integrity": integrity,
        "pair_audit": pair_summary,
        "qc_policy_candidates": policy_summaries,
        "external_access": {
            "locked_test_metadata_inspected_via_file_partition_manifest": True,
            "locked_test_event_rows_read_by_this_audit": False,
            "th232_read_by_this_audit": False,
            "eu152_read_by_this_audit": False,
        },
        "limitations": [
            "Candidate pair retention is an estimate from existing pairs; any admitted-file change requires label regeneration.",
            "Canonical sessions are inferred from non-aggregate QC report groups; unresolved or filename-fallback mappings cannot support a clean session holdout.",
            "Existing cross-source results show source/domain dependence; this audit does not claim the shortcut gate passes.",
            "No FPGA target/resource/latency contract is defined in this artifact.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-manifest", type=Path, default=DEFAULT_FILE_MANIFEST)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Recompute all recorded HDF5 hashes; otherwise only existence and QC-report hashes are checked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.file_manifest.resolve()
    label_dir = args.label_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_allowlist_path = output_dir / "qc_allowlist.json"
    if frozen_allowlist_path.exists():
        existing_allowlist = json.loads(frozen_allowlist_path.read_text(encoding="utf-8"))
        if existing_allowlist.get("status") == "FROZEN":
            raise RuntimeError(
                f"Refusing to overwrite frozen QC allowlist: {frozen_allowlist_path}; "
                "use a new output directory for a fresh audit"
            )
    expected = [
        output_dir / "qc_allowlist.json",
        output_dir / "domain_registry.csv",
        output_dir / "qc_metric_adjudication.csv",
        output_dir / "architecture_experiment.json",
        output_dir / "preflight_audit.json",
    ]
    if not args.overwrite:
        existing = [path for path in expected if path.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist; use --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    manifest_data, file_records = load_manifest(manifest_path)
    policy_summaries = {
        policy_id: summarize_policy(list(file_records.values()), policy_id)
        for policy_id in POLICIES
    }
    adjudication, qc_summary = load_qc_adjudication(file_records)
    integrity = verify_input_integrity(file_records, adjudication, args.verify_hashes)
    pair_summary = summarize_pairs(label_dir, file_records, POLICIES)
    session_audit = session_domain_audit(file_records, adjudication)
    domain_rows = build_domain_rows(file_records.values(), adjudication)
    metric_rows = build_metric_adjudication_rows(file_records.values(), adjudication)
    train_hash = pair_summary["by_partition"]["train"]["csv_sha256"]
    validation_hash = pair_summary["by_partition"]["validation"]["csv_sha256"]

    qc_allowlist = {
        "schema_version": 2,
        "status": "DRAFT_ADJUDICATION_REQUIRED",
        "created_utc": utc_now(),
        "approved_policy": None,
        "approval_required_before_rebuild": True,
        "source_manifest": {
            "path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "policy_candidates": policy_summaries,
        "qc_observation_adjudication": qc_summary,
        "session_domain_audit": session_audit,
        "decision_notes": [
            "This artifact does not alter labels, partitions, event stores, or model outputs.",
            "Historical worst status is retained for audit but is not treated as the final authoritative status.",
            "Aggregate, progress, postprocess, and alternate-threshold reports are separated from canonical session reports.",
            "Reason-level decisions remain review candidates until an authoritative QC precedence and allowlist are approved.",
        ],
    }
    architecture_protocol = build_architecture_protocol(manifest_path, train_hash, validation_hash)
    preflight = build_preflight(
        manifest_data,
        manifest_path,
        file_records,
        pair_summary,
        policy_summaries,
        qc_summary,
        integrity,
        session_audit,
    )

    write_json(output_dir / "qc_allowlist.json", qc_allowlist)
    write_csv(output_dir / "domain_registry.csv", domain_rows)
    write_csv(output_dir / "qc_metric_adjudication.csv", metric_rows)
    write_json(output_dir / "architecture_experiment.json", architecture_protocol)
    write_json(output_dir / "preflight_audit.json", preflight)

    print(json.dumps({
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "status": qc_allowlist["status"],
        "policy_candidates": {
            policy_id: {
                "files": summary["admitted_file_count"],
                "train_pair_estimate": pair_summary["by_partition"]["train"]["estimated_pairs_retained_by_policy"][policy_id],
                "validation_pair_estimate": pair_summary["by_partition"]["validation"]["estimated_pairs_retained_by_policy"][policy_id],
            }
            for policy_id, summary in policy_summaries.items()
        },
        "integrity_pass": integrity["integrity_pass"],
        "hdf5_hashes_checked": integrity["hdf5_hashes_checked"],
        "canonical_session_count": session_audit["canonical_session_count"],
        "sessions_spanning_multiple_partitions": len(session_audit["sessions_spanning_multiple_partitions"]),
        "locked_test_metadata_inspected": True,
        "locked_test_event_rows_read": False,
        "th232_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
