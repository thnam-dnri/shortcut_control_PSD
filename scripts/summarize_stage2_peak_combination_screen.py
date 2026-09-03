#!/usr/bin/env python3
"""Summarize the Stage 2 architecture-by-peak-combination screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path  # noqa: E402


ARCHITECTURES = ("ds_cnn", "tcn", "multi_rate_hpge", "cnn_gru")
COMBINATIONS = ("ba_low", "ba_high", "ba_low_na511", "ba_high_na511")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_peak_combinations_warning_20260817",
    )
    parser.add_argument(
        "--combinations",
        default=",".join(COMBINATIONS),
        help="Comma-separated combination directories to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_peak_combinations_warning_20260817/stage2_peak_combination_summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def summarize_combination(screen_root: Path, name: str) -> dict[str, Any]:
    combo_dir = screen_root / name
    comparison_path = combo_dir / "comparison.json"
    assert_no_forbidden_path(comparison_path)
    comparison = read_json(comparison_path)
    if comparison.get("warning_status") != "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED":
        raise ValueError(f"Unexpected warning status in {comparison_path}")
    if comparison.get("input", {}).get("test_partition_used") is not False:
        raise ValueError(f"Test partition flag is not false in {comparison_path}")

    candidates: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        candidate = comparison["candidates"][architecture]
        metrics_path = combo_dir / architecture / "internal_metrics.json"
        metrics = read_json(metrics_path)
        checkpoint_path = PROJECT_ROOT / candidate["checkpoint"]
        assert_no_forbidden_path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        candidates[architecture] = {
            "best_epoch": candidate["best_epoch"],
            "parameter_count": candidate["parameter_count"],
            "checkpoint": candidate["checkpoint"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation": candidate["validation"],
            "validation_per_peak": metrics["validation"]["per_peak"],
            "internal_metrics": relative(metrics_path),
        }

    return {
        "comparison": relative(comparison_path),
        "labels_dir": comparison["input"]["labels_dir"],
        "event_store_dir": comparison["input"]["event_store_dir"],
        "train_event_count": comparison["input"]["train_event_count"],
        "validation_event_count": comparison["input"]["validation_event_count"],
        "train_selection": comparison["input"]["train_selection"],
        "validation_selection": comparison["input"]["validation_selection"],
        "test_partition_used": False,
        "external_data_used": False,
        "candidates": candidates,
    }


def main() -> int:
    args = build_parser().parse_args()
    screen_root = args.screen_root.resolve()
    output_path = args.output.resolve()
    assert_no_forbidden_path(screen_root)
    assert_no_forbidden_path(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")

    names = tuple(item.strip() for item in args.combinations.split(",") if item.strip())
    if not names:
        raise ValueError("At least one combination is required")
    unknown = sorted(set(names) - set(COMBINATIONS))
    if unknown:
        raise ValueError(f"Unknown combinations: {unknown}")

    combinations = {
        name: summarize_combination(screen_root, name) for name in names
    }
    by_architecture: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        values = [
            combinations[name]["candidates"][architecture]["validation"][
                "weighted_auroc"
            ]
            for name in names
        ]
        by_architecture[architecture] = {
            "mean_weighted_auroc": sum(values) / len(values),
            "minimum_weighted_auroc": min(values),
            "maximum_weighted_auroc": max(values),
            "best_combination": names[values.index(max(values))],
        }
    by_combination = {
        name: {
            "best_architecture": max(
                ARCHITECTURES,
                key=lambda architecture: combinations[name]["candidates"][
                    architecture
                ]["validation"]["weighted_auroc"],
            ),
            "best_weighted_auroc": max(
                combinations[name]["candidates"][architecture]["validation"][
                    "weighted_auroc"
                ]
                for architecture in ARCHITECTURES
            ),
        }
        for name in names
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage_2_new_architecture_screening",
        "status": "PROVISIONAL_SHORTCUT_WARNING",
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
        "architectures": list(ARCHITECTURES),
        "combinations": list(names),
        "test_partition_used": False,
        "external_data_used": False,
        "by_architecture": by_architecture,
        "by_combination": by_combination,
        "screens": combinations,
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
