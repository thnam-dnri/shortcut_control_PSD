#!/usr/bin/env python3
"""Freeze an approved QC allowlist with immutable provenance.

This command is intentionally narrow: it freezes a policy candidate from the
non-destructive WP0 audit and records the exact source-manifest and partition
hashes. It does not rebuild labels, event stores, models, test rows, or external
scores. The frozen output can subsequently be passed to
``build_energy_matched_labels.py --approved-allowlist``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT = PROJECT_ROOT / "outputs" / "protocol" / "qc_allowlist.json"
DEFAULT_PARTITION_MANIFEST = PROJECT_ROOT / "outputs" / "labels" / "file_partition_manifest.json"
DEFAULT_PREFLIGHT = PROJECT_ROOT / "outputs" / "protocol" / "preflight_audit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "protocol" / "qc_allowlist.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--partition-manifest", type=Path, default=DEFAULT_PARTITION_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", choices=("current_complete", "pass_warn", "pass_only"), required=True)
    parser.add_argument(
        "--session-strategy",
        choices=("keep_split_narrow_claims",),
        default="keep_split_narrow_claims",
    )
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    draft_path = args.draft.resolve()
    partition_path = args.partition_manifest.resolve()
    preflight_path = args.preflight.resolve()
    output_path = args.output.resolve()
    if not draft_path.exists():
        raise FileNotFoundError(draft_path)
    if not partition_path.exists():
        raise FileNotFoundError(partition_path)
    if not preflight_path.exists():
        raise FileNotFoundError(preflight_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if not str(draft.get("status", "")).startswith("DRAFT"):
        raise ValueError(f"Expected a draft allowlist, got status={draft.get('status')!r}")
    candidates = draft.get("policy_candidates", {})
    candidate = candidates.get(args.policy)
    if not isinstance(candidate, dict):
        raise ValueError(f"Policy candidate is missing from draft: {args.policy}")
    admitted_hdf5 = sorted({str(path) for path in candidate.get("admitted_hdf5", [])})
    if not admitted_hdf5:
        raise ValueError(f"Policy candidate has no admitted HDF5 files: {args.policy}")

    partition_manifest = json.loads(partition_path.read_text(encoding="utf-8"))
    partition_records = {
        str(record.get("hdf5")): record
        for record in partition_manifest.get("files", [])
        if record.get("hdf5")
    }
    unknown = sorted(set(admitted_hdf5) - set(partition_records))
    if unknown:
        raise ValueError("Allowlist contains files absent from partition manifest: " + ", ".join(unknown[:10]))
    invalid = [
        path
        for path in admitted_hdf5
        if partition_records[path].get("processing_status") != "OK"
        or not partition_records[path].get("complete_input")
        or str(partition_records[path].get("qc_status", "UNKNOWN")).upper()
        not in set(candidate.get("admit_qc_statuses", []))
        or partition_records[path].get("partition") not in {"train", "validation", "test"}
    ]
    if invalid:
        raise ValueError("Candidate failed frozen admission validation: " + ", ".join(invalid[:10]))

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("input_integrity", {}).get("integrity_pass"):
        raise ValueError("Cannot freeze an allowlist while preflight input integrity is not PASS")
    if preflight.get("external_access", {}).get("locked_test_event_rows_read_by_this_audit"):
        raise ValueError("Preflight records locked-test event access")
    if preflight.get("external_access", {}).get("th232_read_by_this_audit"):
        raise ValueError("Preflight records Th-232 access")

    if output_path == draft_path:
        backup_path = output_path.with_name("qc_allowlist_draft.json")
        if backup_path.exists() and not args.overwrite:
            raise FileExistsError(f"Draft backup exists; use --overwrite: {backup_path}")
        write_json(backup_path, draft)

    by_source = Counter(str(partition_records[path].get("source")) for path in admitted_hdf5)
    by_partition = Counter(str(partition_records[path].get("partition")) for path in admitted_hdf5)
    frozen = {
        "schema_version": 3,
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "approved_policy": args.policy,
        "approval_note": args.approval_note,
        "session_strategy": args.session_strategy,
        "session_claim_boundary": (
            "Existing file partitions are retained; 17 sessions span multiple partitions, "
            "so results are not session-held-out. Use session-clustered uncertainty and narrow claims."
        ),
        "admitted_file_count": len(admitted_hdf5),
        "admitted_by_source": dict(sorted(by_source.items())),
        "admitted_by_partition": dict(sorted(by_partition.items())),
        "approved_hdf5": admitted_hdf5,
        "admit_processing_status": ["OK"],
        "require_complete_input": True,
        "admit_qc_statuses": candidate.get("admit_qc_statuses", []),
        "source_provenance": {
            "draft_allowlist": draft_path.relative_to(PROJECT_ROOT).as_posix(),
            "draft_allowlist_sha256": sha256_file(draft_path),
            "partition_manifest": partition_path.relative_to(PROJECT_ROOT).as_posix(),
            "partition_manifest_sha256": sha256_file(partition_path),
            "preflight": preflight_path.relative_to(PROJECT_ROOT).as_posix(),
            "preflight_sha256": sha256_file(preflight_path),
            "qc_metric_adjudication": "outputs/protocol/qc_metric_adjudication.csv",
            "qc_metric_adjudication_sha256": sha256_file(
                PROJECT_ROOT / "outputs" / "protocol" / "qc_metric_adjudication.csv"
            ),
            "domain_registry": "outputs/protocol/domain_registry.csv",
            "domain_registry_sha256": sha256_file(
                PROJECT_ROOT / "outputs" / "protocol" / "domain_registry.csv"
            ),
        },
        "policy_candidate_snapshot": candidate,
        "locked_data_boundary": {
            "locked_test_event_rows_read_by_freeze": False,
            "th232_read_by_freeze": False,
            "eu152_read_by_freeze": False,
            "legacy_test_csv_hash_caveat": True,
        },
        "rebuild_requirements": {
            "inherit_partition_manifest": True,
            "do_not_call_random_partition_assignment": True,
            "rebuild_labels_before_event_store": True,
            "retrain_baselines_if_counts_or_files_change": True,
        },
    }
    write_json(output_path, frozen)

    architecture_path = output_path.parent / "architecture_experiment.json"
    architecture_updated = False
    if architecture_path.exists():
        architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
        architecture["status"] = "QC_FROZEN_PENDING_REBUILD"
        architecture["qc_freeze"] = {
            "status": "FROZEN",
            "allowlist": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "allowlist_sha256": sha256_file(output_path),
            "approved_policy": args.policy,
            "session_strategy": args.session_strategy,
        }
        for gate in architecture.get("gates", []):
            if gate.get("id") == "qc":
                gate["status"] = "frozen"
            elif gate.get("id") == "architecture_screen":
                gate["status"] = "blocked_until_rebuild_and_shortcut_audit"
        write_json(architecture_path, architecture)
        architecture_updated = True

    print(json.dumps({
        "status": frozen["status"],
        "policy": args.policy,
        "admitted_files": len(admitted_hdf5),
        "by_source": frozen["admitted_by_source"],
        "by_partition": frozen["admitted_by_partition"],
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "architecture_protocol_updated": architecture_updated,
        "draft_backup": (
            output_path.with_name("qc_allowlist_draft.json").relative_to(PROJECT_ROOT).as_posix()
            if output_path == draft_path else None
        ),
        "test_event_rows_read": False,
        "th232_read": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
