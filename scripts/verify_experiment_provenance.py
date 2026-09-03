#!/usr/bin/env python3
"""Verify hashes and locked-data boundaries for the engineering experiment bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_access_guards import assert_no_forbidden_path  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def check_hash(path: Path, expected: str | None, checks: list[dict[str, Any]]) -> bool:
    actual = sha256_file(path)
    passed = expected is None or actual == expected
    checks.append({"path": rel(path), "expected_sha256": expected, "actual_sha256": actual, "pass": passed})
    return passed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_event_store(
    store_dir: Path,
    manifest_dir: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = manifest_dir / "event_store_manifest.json"
    manifest = load_json(manifest_path)
    transaction_path = PROJECT_ROOT / manifest["latest_transaction"]
    transaction_hash_pass = check_hash(
        transaction_path,
        manifest.get("latest_transaction_sha256"),
        checks,
    )
    transaction = load_json(transaction_path)
    test_guard = transaction.get("test_partition_used") is False and manifest.get("store_policy", {}).get("test_partition_used") is False
    partition_results = {}
    for partition, values in transaction.get("partitions", {}).items():
        store_path = PROJECT_ROOT / values["store_file"]
        lookup_path = PROJECT_ROOT / values["lookup_file"]
        store_pass = check_hash(store_path, values.get("store_sha256"), checks)
        lookup_pass = check_hash(lookup_path, values.get("lookup_sha256"), checks)
        partition_results[partition] = {
            "store_hash_pass": store_pass,
            "lookup_hash_pass": lookup_pass,
            "events_after": values.get("events_after"),
        }
    return {
        "manifest": rel(manifest_path),
        "transaction_hash_pass": transaction_hash_pass,
        "test_partition_used": transaction.get("test_partition_used"),
        "partitions": partition_results,
        "pass": transaction_hash_pass and test_guard and all(
            item["store_hash_pass"] and item["lookup_hash_pass"]
            for item in partition_results.values()
        ),
    }


def verify_development_labels(
    label_dir: Path,
    allowlist: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_manifest_path = label_dir / "label_dataset_manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    input_identity = dataset_manifest["input_identity"]
    allowlist_path = PROJECT_ROOT / input_identity["approved_allowlist"]
    allowlist_hash_pass = check_hash(allowlist_path, input_identity["approved_allowlist_sha256"], checks)
    csv_hash_pass = True
    for partition, expected in input_identity["output_csv_sha256"].items():
        csv_path = label_dir / f"label_pairs_{partition}.csv"
        csv_hash_pass &= check_hash(csv_path, expected, checks)
        if partition == "test" or "test" in csv_path.name.lower():
            raise ValueError(f"Development label output contains locked test CSV: {csv_path}")
    partition_manifest_path = label_dir / "file_partition_manifest.json"
    partition_manifest = load_json(partition_manifest_path)
    no_test_files = all(record.get("partition") in {"train", "validation"} for record in partition_manifest["files"])
    no_test_csv = not (label_dir / "label_pairs_test.csv").exists()
    approved_train_val = {
        path
        for path in allowlist["approved_hdf5"]
        if any(
            record.get("hdf5") == path and record.get("partition") in {"train", "validation"}
            for record in load_json(PROJECT_ROOT / "outputs/labels/file_partition_manifest.json")["files"]
        )
    }
    dev_records = {record["hdf5"]: record for record in partition_manifest["files"]}
    hdf5_hash_pass = True
    hdf5_checked = 0
    for path, record in sorted(dev_records.items()):
        if path not in approved_train_val:
            hdf5_hash_pass = False
            continue
        hdf5_checked += 1
        hdf5_hash_pass &= check_hash(PROJECT_ROOT / path, record.get("hdf5_sha256"), checks)
    return {
        "label_dir": rel(label_dir),
        "allowlist_hash_pass": allowlist_hash_pass,
        "csv_hash_pass": csv_hash_pass,
        "development_hdf5_hashes_checked": hdf5_checked,
        "development_hdf5_hash_pass": hdf5_hash_pass,
        "no_test_files": no_test_files,
        "no_test_csv": no_test_csv,
        "locked_test_hashes_not_recomputed": True,
        "pass": allowlist_hash_pass and csv_hash_pass and hdf5_hash_pass and no_test_files and no_test_csv,
    }


def verify_protocol_amendment(checks: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = PROJECT_ROOT / "outputs/protocol/architecture_experiment.json"
    base = PROJECT_ROOT / "outputs/protocol/architecture_experiment_frozen_base.json"
    amended = PROJECT_ROOT / "outputs/protocol/architecture_experiment_amended_20260816.json"
    amendment = PROJECT_ROOT / "outputs/protocol/architecture_experiment_amendment_20260816.json"
    data = load_json(amendment)
    base_pass = check_hash(base, data["original_frozen_snapshot"]["sha256"], checks)
    amended_pass = check_hash(amended, data["amended_snapshot"]["sha256"], checks)
    canonical_pass = check_hash(protocol, data["canonical_protocol_restored_to"]["sha256"], checks)
    linked_results = {}
    for key, entry in data["linked_artifacts"].items():
        linked_path = PROJECT_ROOT / entry["path"]
        linked_results[key] = {
            "path": entry["path"],
            "pass": check_hash(linked_path, entry["sha256"], checks),
        }
        if "summary_csv_sha256" in entry:
            summary_path = linked_path.with_name("nuisance_shortcut_summary.csv")
            linked_results[key]["summary_csv_pass"] = check_hash(
                summary_path, entry["summary_csv_sha256"], checks
            )
            linked_results[key]["pass"] &= linked_results[key]["summary_csv_pass"]
    return {
        "base_is_reconstructed_pre_amendment": True,
        "historical_byte_identity_proven": False,
        "base_pass": base_pass,
        "amended_snapshot_pass": amended_pass,
        "canonical_restored_pass": canonical_pass,
        "linked_artifacts": linked_results,
        "pass": base_pass and amended_pass and canonical_pass and all(
            item["pass"] for item in linked_results.values()
        ),
    }


def verify_model_checkpoints(checks: list[dict[str, Any]]) -> dict[str, Any]:
    models = {
        "o2_late_fusion": PROJECT_ROOT / "outputs/models/architecture_pass_warn_20260815/o2_late_fusion/o2_late_fusion_best.pt",
        "registered_residual_cnn": PROJECT_ROOT / "outputs/models/architecture_pass_warn_20260815/registered_residual_cnn/multiscale_registered_cnn_best.pt",
        "compact_cnn": PROJECT_ROOT / "outputs/models/architecture_pass_warn_20260815/compact_cnn/both_ma10_global_t10_w750.pt",
    }
    compact_results = load_json(PROJECT_ROOT / "outputs/models/architecture_pass_warn_20260815/compact_cnn/optimization_results.json")
    compact_expected = compact_results["trials"][0].get("checkpoint_sha256")
    baseline = load_json(PROJECT_ROOT / "outputs/protocol/checkpoint_hash_manifest_20260816.json")
    result: dict[str, Any] = {}
    for name, path in models.items():
        training_expected = compact_expected if name == "compact_cnn" else None
        baseline_entry = baseline["checkpoints"][name]
        baseline_path = PROJECT_ROOT / baseline_entry["path"]
        if baseline_path != path:
            raise ValueError(f"Checkpoint baseline path mismatch for {name}")
        baseline_pass = path.is_file() and check_hash(path, baseline_entry["sha256"], checks)
        training_pass = training_expected is None or check_hash(path, training_expected, checks)
        passed = baseline_pass and training_pass
        result[name] = {
            "path": rel(path),
            "training_manifest_sha256": training_expected,
            "post_run_baseline_sha256": baseline_entry["sha256"],
            "current_sha256": sha256_file(path) if path.is_file() else None,
            "recorded_in_training_manifest": training_expected is not None,
            "pass": passed,
        }
    return {
        "baseline_manifest": "outputs/protocol/checkpoint_hash_manifest_20260816.json",
        "baseline_is_post_run_not_historical_training_provenance": baseline["not_historical_training_manifest"],
        "models": result,
        "pass": all(item["pass"] for item in result.values()),
        "unrecorded_checkpoint_hashes": [
            name for name, item in result.items() if not item["recorded_in_training_manifest"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    checks: list[dict[str, Any]] = []
    allowlist_path = PROJECT_ROOT / "outputs/protocol/qc_allowlist.json"
    allowlist = load_json(allowlist_path)
    if allowlist.get("status") != "FROZEN":
        raise ValueError("QC allowlist is not frozen")
    # This validates path tokens in the approved development artifact names,
    # while deliberately not opening locked-test HDF5 files.
    for path in (
        "outputs/labels/architecture_pass_warn_20260815/label_pairs_train.csv",
        "outputs/labels/architecture_pass_warn_20260815/label_pairs_validation.csv",
        "processed_data/event_store/architecture_pass_warn_20260815/train_events.h5",
        "processed_data/event_store/architecture_pass_warn_20260815/validation_events.h5",
    ):
        assert_no_forbidden_path(path)

    results = {
        "schema_version": 1,
        "status": "DEVELOPMENT_ARTIFACT_INTEGRITY_VERIFIED_WITH_LIMITATIONS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_data_boundary": {
            "test_event_rows_read_by_verifier": False,
            "test_hdf5_hashes_recomputed": False,
            "th232_read_by_verifier": False,
            "eu152_read_by_verifier": False,
        },
        "protocol_amendment": verify_protocol_amendment(checks),
        "primary_labels": verify_development_labels(
            PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815",
            allowlist,
            checks,
        ),
        "primary_event_store": verify_event_store(
            PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815",
            PROJECT_ROOT / "outputs/event_store/architecture_pass_warn_20260815",
            checks,
        ),
        "source_ablation_event_store": verify_event_store(
            PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
            PROJECT_ROOT / "outputs/event_store/architecture_pass_warn_20260815_source_ablation",
            checks,
        ),
        "model_checkpoints": verify_model_checkpoints(checks),
        "checks": checks,
    }
    results["pass"] = all(
        results[key].get("pass", False)
        for key in ("protocol_amendment", "primary_labels", "primary_event_store", "source_ablation_event_store", "model_checkpoints")
    )
    if not results["pass"]:
        results["status"] = "DEVELOPMENT_ARTIFACT_INTEGRITY_VERIFICATION_FAILED"
    write_json(output, results)
    print(json.dumps({
        "output": rel(output),
        "status": results["status"],
        "pass": results["pass"],
        "hash_checks": len(checks),
        "test_event_rows_read": False,
        "test_hdf5_hashes_recomputed": False,
    }, indent=2, sort_keys=True))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
