#!/usr/bin/env python3
"""Summarize compact-CNN and late-fusion peak-combination screens.

This report is restricted to train/validation artifacts. It records the
validation metrics and per-peak strata for controlled Ba-low/Ba-high and
optional Na-22 511-keV combinations without opening target or external data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path  # noqa: E402


DEFAULT_COMBINATIONS = (
    "ba_low",
    "ba_high",
    "ba_low_na511",
    "ba_high_na511",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/peak_combinations",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/peak_combinations",
    )
    parser.add_argument(
        "--combinations",
        default=",".join(DEFAULT_COMBINATIONS),
        help="Comma-separated combination directory names to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/peak_combinations/peak_combination_screen_summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def selected_compact_trial(result: dict[str, Any]) -> dict[str, Any]:
    ranking = result.get("ranking")
    trials = result.get("trials")
    if not isinstance(ranking, list) or not ranking or not isinstance(trials, list):
        raise ValueError("Compact result lacks ranking/trials")
    selected_name = ranking[0]["config"]
    for trial in trials:
        if trial.get("config", {}).get("name") == selected_name:
            return trial
    raise ValueError(f"Compact selected trial not found: {selected_name}")


def summarize_combination(
    name: str,
    labels_root: Path,
    model_root: Path,
) -> dict[str, Any]:
    label_dir = labels_root / name
    compact_dir = model_root / name / "compact_cnn"
    late_dir = model_root / name / "late_fusion"
    manifest_path = label_dir / "label_dataset_manifest.json"
    compact_path = compact_dir / "optimization_results.json"
    late_path = late_dir / "internal_metrics.json"
    for path in (manifest_path, compact_path, late_path):
        assert_no_forbidden_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    compact_result = read_json(compact_path)
    late_result = read_json(late_path)
    compact_trial = selected_compact_trial(compact_result)
    compact_validation = compact_trial["validation"]
    late_validation = late_result["validation"]
    if manifest.get("test_partition_used") or manifest.get("external_data_used"):
        raise ValueError(f"Forbidden partition/external flag in {manifest_path}")

    return {
        "positive_domains": manifest["positive_domains"],
        "label_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "label_manifest_sha256": sha256_file(manifest_path),
        "partition_counts": manifest["partitions"],
        "test_partition_used": False,
        "external_data_used": False,
        "compact_cnn": {
            "config": compact_trial["config"],
            "best_epoch": compact_trial["best_epoch"],
            "validation": compact_validation,
            "output_dir": compact_dir.relative_to(PROJECT_ROOT).as_posix(),
        },
        "late_fusion": {
            "best_epoch": late_result["best_epoch"],
            "validation": late_validation,
            "output_dir": late_dir.relative_to(PROJECT_ROOT).as_posix(),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    labels_root = args.labels_root.resolve()
    model_root = args.model_root.resolve()
    output_path = args.output.resolve()
    assert_no_forbidden_path(labels_root)
    assert_no_forbidden_path(model_root)
    assert_no_forbidden_path(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    names = tuple(item.strip() for item in args.combinations.split(",") if item.strip())
    if not names:
        raise ValueError("At least one combination is required")
    combinations = {
        name: summarize_combination(name, labels_root, model_root) for name in names
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "development_internal_validation_only",
        "target_domains_opened": False,
        "test_partition_used": False,
        "external_data_used": False,
        "screened_combinations": list(names),
        "combinations": combinations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
