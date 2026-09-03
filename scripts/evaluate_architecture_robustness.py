#!/usr/bin/env python3
"""Evaluate warning-gated architecture candidates by development strata.

This is a Stage 3 development diagnostic. It rescored only the frozen
train/validation-derived candidate checkpoints on the validation partition and
reports peak, source, QC, and acquisition-session strata. It does not open the
locked test partition or external Th-232/Eu-152 data, and it does not select a
threshold or make a claim-grade architecture decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.architecture_candidates import build_candidate  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
)
from src.data_access_guards import (  # noqa: E402
    assert_development_csv,
    assert_no_forbidden_path,
)
from train_architecture_candidates import (  # noqa: E402
    CANDIDATES,
    REPRESENTATION,
    REPRESENTATION_NAME,
    WARNING_STATUS,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_session_registry(path: Path) -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    for row in read_rows(path):
        registry[row["hdf5"]] = row
    return registry


def session_name(
    hdf5_path: str,
    registry: dict[str, dict[str, str]],
) -> str:
    return registry.get(hdf5_path, {}).get("canonical_session_id", "UNKNOWN")


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    if labels.size == 0:
        raise ValueError("Cannot score an empty group")
    if np.unique(labels).size < 2:
        raise ValueError("A robustness group must contain both labels")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "event_count": int(labels.size),
        "pair_count": int(labels.size // 2),
    }


def pair_event_indices(pair_mask: np.ndarray) -> np.ndarray:
    pair_indices = np.flatnonzero(pair_mask)
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def grouped_metrics(
    group_values: Iterable[str],
    raw: RawPartition,
    scores: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(list(group_values), dtype="U128")
    pair_count = raw.labels.size // 2
    if values.size != pair_count:
        raise ValueError(
            f"Expected {pair_count} pair groups, received {values.size}"
        )
    result: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(values.tolist())):
        indices = pair_event_indices(values == group)
        result[group] = metric_summary(
            raw.labels[indices], scores[indices], raw.weights[indices]
        )
    return result


def worst_group(
    groups: dict[str, dict[str, float | int]],
    minimum_pairs: int,
) -> dict[str, float | int | str | None]:
    eligible = [
        (name, values)
        for name, values in groups.items()
        if int(values["pair_count"]) >= minimum_pairs
    ]
    if not eligible:
        return {
            "group": None,
            "auroc": None,
            "weighted_auroc": None,
            "pair_count": 0,
            "eligible_group_count": 0,
        }
    name, values = min(eligible, key=lambda item: float(item[1]["weighted_auroc"]))
    return {
        "group": name,
        "auroc": float(values["auroc"]),
        "weighted_auroc": float(values["weighted_auroc"]),
        "pair_count": int(values["pair_count"]),
        "eligible_group_count": len(eligible),
    }


def score_candidate(
    checkpoint_path: Path,
    candidate: str,
    values: np.ndarray,
    raw: RawPartition,
    width: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("warning_status") != WARNING_STATUS:
        raise ValueError(f"Unexpected warning status in {checkpoint_path}")
    model = build_candidate(candidate, input_channels=2, width=width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(scores)
    if result.size != raw.labels.size:
        raise ValueError("Candidate score count does not match validation events")
    return result


def build_pair_groups(
    pair_rows: list[dict[str, str]],
    registry: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    return {
        "peak": [row["peak_id"] for row in pair_rows],
        "positive_source": [row["positive_source"] for row in pair_rows],
        "negative_source": [row["negative_source"] for row in pair_rows],
        "qc_pair": [
            f"{row.get('positive_qc_status', 'UNKNOWN')}__"
            f"{row.get('negative_qc_status', 'UNKNOWN')}"
            for row in pair_rows
        ],
        "positive_session": [
            session_name(row["positive_hdf5"], registry) for row in pair_rows
        ],
        "negative_session": [
            session_name(row["negative_hdf5"], registry) for row in pair_rows
        ],
        "positive_acquisition_block": [
            registry.get(row["positive_hdf5"], {}).get(
                "acquisition_block_id", "UNKNOWN"
            )
            for row in pair_rows
        ],
        "negative_acquisition_block": [
            registry.get(row["negative_hdf5"], {}).get(
                "acquisition_block_id", "UNKNOWN"
            )
            for row in pair_rows
        ],
    }


def candidate_report(
    raw: RawPartition,
    scores: np.ndarray,
    pair_groups: dict[str, list[str]],
    minimum_group_pairs: int,
) -> dict[str, Any]:
    grouped = {
        name: grouped_metrics(values, raw, scores)
        for name, values in pair_groups.items()
    }
    return {
        "overall": metric_summary(raw.labels, scores, raw.weights),
        "by_group": grouped,
        "worst_eligible_group": {
            name: worst_group(values, minimum_group_pairs)
            for name, values in grouped.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "outputs/protocol/domain_registry.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_warning_balanced_20260816",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_warning_balanced_20260816/robustness_validation.json",
    )
    parser.add_argument("--candidate", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-validation-events", type=int, default=None)
    parser.add_argument("--minimum-group-pairs", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.minimum_group_pairs < 1:
        raise ValueError("batch-size and minimum-group-pairs must be positive")
    if args.max_validation_events is not None and args.max_validation_events < 2:
        raise ValueError("max-validation-events must be at least 2")

    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    registry_path = args.registry.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    validation_csv = labels_dir / "label_pairs_validation.csv"
    comparison_path = model_dir / "comparison.json"
    for path in (validation_csv, registry_path, comparison_path, event_store_dir, model_dir):
        assert_no_forbidden_path(path)
    assert_development_csv(validation_csv)
    if not args.overwrite and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    width = int(comparison["training"]["width"])
    if comparison.get("warning_status") != WARNING_STATUS:
        raise ValueError("Candidate comparison is not marked with the active warning")
    for candidate in args.candidate:
        checkpoint = model_dir / candidate / f"{candidate}_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        assert_no_forbidden_path(checkpoint)

    pair_rows = read_rows(validation_csv)
    registry = load_session_registry(registry_path)
    print("Loading validation event store ...", flush=True)
    raw = load_raw_partition(
        validation_csv,
        event_store_dir,
        args.max_validation_events,
    )
    print("Building validation representation ...", flush=True)
    values, representation_stats = build_representation(raw, REPRESENTATION)
    checkpoint_reference = model_dir / args.candidate[0] / f"{args.candidate[0]}_best.pt"
    reference_checkpoint = torch.load(
        checkpoint_reference, map_location="cpu", weights_only=False
    )
    apply_channel_statistics(values, reference_checkpoint["feature_statistics"])
    pair_count = raw.labels.size // 2
    pair_groups = build_pair_groups(pair_rows[:pair_count], registry)
    if len(pair_rows) != pair_count and args.max_validation_events is None:
        raise ValueError("Validation pair rows do not match loaded event count")
    if any(len(groups) != pair_count for groups in pair_groups.values()):
        raise ValueError("Validation group metadata does not match event pairs")

    device = resolve_device(args.device)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PROVISIONAL_STAGE_3_ROBUSTNESS_WITH_SHORTCUT_WARNING",
        "warning_status": WARNING_STATUS,
        "created_utc": utc_now(),
        "device": str(device),
        "representation": {
            "name": REPRESENTATION_NAME,
            **REPRESENTATION.as_dict(),
            "build_statistics": representation_stats,
            "normalization_statistics_source": "reference_candidate_train_only",
        },
        "input": {
            "labels_dir": labels_dir.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(validation_csv),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
            "domain_registry": registry_path.relative_to(PROJECT_ROOT).as_posix(),
            "model_dir": model_dir.relative_to(PROJECT_ROOT).as_posix(),
            "event_count": int(raw.labels.size),
            "pair_count": int(pair_count),
            "max_validation_events": args.max_validation_events,
            "test_partition_used": False,
            "th232_used_for_selection": False,
            "eu152_used_for_selection": False,
        },
        "configuration": {
            "candidates": list(args.candidate),
            "model_width": width,
            "batch_size": args.batch_size,
            "minimum_group_pairs": args.minimum_group_pairs,
            "group_types": sorted(pair_groups),
        },
        "caveats": [
            "This is a development stratification diagnostic, not a source/session-held-out claim.",
            "Internal validation remains shortcut-sensitive because train and validation can share acquisition domains.",
            "No threshold was selected and no external spectrum was opened.",
            "Label 0 denotes Co-60 Compton-continuum background without asserting a unique microscopic interaction history.",
        ],
        "candidates": {},
    }
    for candidate in args.candidate:
        checkpoint_path = model_dir / candidate / f"{candidate}_best.pt"
        print(f"Scoring {candidate} ...", flush=True)
        scores = score_candidate(
            checkpoint_path,
            candidate,
            values,
            raw,
            width,
            args.batch_size,
            device,
        )
        report = candidate_report(
            raw,
            scores,
            pair_groups,
            args.minimum_group_pairs,
        )
        result["candidates"][candidate] = {
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            **report,
        }
        print(
            candidate,
            {
                "overall_auroc": round(float(report["overall"]["auroc"]), 6),
                "overall_weighted_auroc": round(
                    float(report["overall"]["weighted_auroc"]), 6
                ),
            },
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
