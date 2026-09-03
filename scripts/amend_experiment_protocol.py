#!/usr/bin/env python3
"""Preserve the frozen architecture protocol and record a versioned amendment.

The status and input-path edits made after QC freeze are converted into a
one-shot, explicit provenance chain. This script does not read event rows or
external data; it only versions protocol JSON and records linked artifact hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "outputs" / "protocol" / "architecture_experiment.json"
AMENDMENT_ID = "20260816"


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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    output_dir = protocol_path.parent
    base_path = output_dir / "architecture_experiment_frozen_base.json"
    amended_snapshot_path = output_dir / f"architecture_experiment_amended_{AMENDMENT_ID}.json"
    amendment_path = output_dir / f"architecture_experiment_amendment_{AMENDMENT_ID}.json"
    targets = [base_path, amended_snapshot_path, amendment_path]
    if any(path.exists() for path in targets):
        raise FileExistsError("Amendment outputs already exist; this provenance conversion is one-shot")

    current = json.loads(protocol_path.read_text(encoding="utf-8"))
    if current.get("status") != "SCALAR_SHORTCUT_CONTROL_BLOCKS_SCIENTIFIC_RANKING":
        raise ValueError(
            "Expected the pre-conversion protocol to be the amended scalar-shortcut status; "
            "refusing to reconstruct from the canonical base snapshot"
        )
    amended_snapshot = copy.deepcopy(current)
    amended_snapshot_path.write_text(
        json.dumps(amended_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    amended_sha256 = sha256_file(amended_snapshot_path)

    # Reconstruct the pre-amendment state produced by freeze_experiment_protocol.py.
    # This is an internal reconstruction, not historical byte-level evidence.
    base = copy.deepcopy(current)
    base["status"] = "QC_FROZEN_PENDING_REBUILD"
    base["input_data"]["file_partition_manifest"] = "outputs/labels/file_partition_manifest.json"
    base["input_data"]["train_pairs"] = "outputs/labels/label_pairs_train.csv"
    base["input_data"]["train_pairs_sha256"] = "ab095a48a0914982b5bfe8925cdfaf2598d71bb1f57f2b9214929fc9bd7babd6"
    base["input_data"]["validation_pairs"] = "outputs/labels/label_pairs_validation.csv"
    base["input_data"]["validation_pairs_sha256"] = "9efba2b1ff10d1ea3a569d13fa266f1b4916b8cc40a098e3eb98328f854405e0"
    for gate in base.get("gates", []):
        if gate.get("id") == "shortcut":
            gate["artifact"] = None
            gate["status"] = "pending"
        elif gate.get("id") == "architecture_screen":
            gate["artifact"] = None
            gate["status"] = "blocked_until_rebuild_and_shortcut_audit"
    write_json(base_path, base)
    base_sha256 = sha256_file(base_path)

    linked = {
        "qc_allowlist": "outputs/protocol/qc_allowlist.json",
        "preflight_audit": "outputs/protocol/preflight_audit.json",
        "nuisance_shortcut_audit": "outputs/shortcut_audit/architecture_pass_warn_20260815_nuisance/nuisance_shortcut_audit.json",
        "waveform_shortcut_audit": "outputs/shortcut_audit/architecture_pass_warn_20260815/shortcut_audit.json",
        "data_protocol_redesign": "docs/data_protocol_redesign.md",
        "hardware_profile": "outputs/architecture_profile/architecture_candidates_20260816/architecture_profile.json",
    }
    linked_hashes = {
        key: {
            "path": path,
            "sha256": sha256_file(PROJECT_ROOT / path),
        }
        for key, path in linked.items()
    }
    amendment = {
        "schema_version": 1,
        "status": "VERSIONED_PROTOCOL_AMENDMENT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_id": AMENDMENT_ID,
        "original_frozen_snapshot": {
            "path": base_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": base_sha256,
        },
        "amended_snapshot": {
            "path": amended_snapshot_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": amended_sha256,
        },
        "canonical_protocol_restored_to": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": base_sha256,
        },
        "reason": (
            "The scalar nuisance audit blocks scientific architecture ranking; "
            "the protocol status and development-input paths were amended without changing the frozen allowlist or immutable partitions."
        ),
        "changes": [
            "Point the engineering development contract to the versioned PASS+WARN train/validation labels.",
            "Link the waveform and scalar shortcut audits.",
            "Mark scientific architecture ranking blocked until a corrected dataset exists.",
            "Record synthetic candidate profiles as hardware-only exercises.",
        ],
        "immutable_data_boundary": {
            "allowlist_path": "outputs/protocol/qc_allowlist.json",
            "allowlist_sha256": sha256_file(PROJECT_ROOT / "outputs/protocol/qc_allowlist.json"),
            "partition_manifest_sha256": json.loads(
                (PROJECT_ROOT / "outputs/protocol/qc_allowlist.json").read_text(encoding="utf-8")
            )["source_provenance"]["partition_manifest_sha256"],
            "test_event_rows_read": False,
            "th232_read": False,
            "eu152_read": False,
        },
        "linked_artifacts": linked_hashes,
    }
    write_json(amendment_path, amendment)
    # Restore canonical protocol to the immutable base snapshot after recording the amendment.
    write_json(protocol_path, base)
    print(json.dumps({
        "canonical_protocol": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
        "base_snapshot": base_path.relative_to(PROJECT_ROOT).as_posix(),
        "amended_snapshot": amended_snapshot_path.relative_to(PROJECT_ROOT).as_posix(),
        "amendment": amendment_path.relative_to(PROJECT_ROOT).as_posix(),
        "base_sha256": base_sha256,
        "amended_sha256": amended_sha256,
        "status": amendment["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
