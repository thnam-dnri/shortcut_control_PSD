#!/usr/bin/env python3
"""Establish the canonical equal-weight strict DS-CNN Paper 1 holdout scores.

The canonical checkpoint is the first predeclared seed from the completed
three-seed equal-weight reproducibility study.  This rule avoids choosing a
seed from held-out or Th-232 performance.  The resulting score artifact and
report are compatible with the downstream Th-232 evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_strict_ds_cnn_reproducibility import (  # noqa: E402
    make_event_weights,
    metrics,
    sha256_file,
)
from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


STUDY_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/strict_ds_cnn_reproducibility_20260826/three_peak_equal_weight"
)
DEFAULT_CHECKPOINT = STUDY_DIR / "seed_20260825/ds_cnn_best.pt"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT
    / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/paper1_equal_weight_ds_cnn_20260826/held_out"
)
EXPECTED_PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def canonical_contract(checkpoint_path: Path) -> dict[str, Any]:
    experiment = json.loads((STUDY_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in experiment["training"]["seeds"]]
    if not seeds:
        raise ValueError("Equal-weight experiment has no predeclared seeds")
    first_seed = seeds[0]
    expected_checkpoint = (STUDY_DIR / f"seed_{first_seed}/ds_cnn_best.pt").resolve()
    if checkpoint_path != expected_checkpoint:
        raise ValueError(
            "Paper 1 canonical checkpoint must be the first predeclared equal-weight seed: "
            f"{relative(expected_checkpoint)}"
        )
    expected_weights = {peak: 1.0 / 3.0 for peak in EXPECTED_PEAKS}
    if experiment["peak_weights"] != expected_weights:
        raise ValueError("Equal-weight experiment configuration has changed")
    return {
        "study_config": relative(STUDY_DIR / "experiment_config.json"),
        "study_config_sha256": sha256_file(STUDY_DIR / "experiment_config.json"),
        "predeclared_seeds": seeds,
        "canonical_seed": first_seed,
        "selection_rule": (
            "first predeclared seed; selected without held-out, Th-232, Eu-152, "
            "or locked-test comparison"
        ),
    }


def predict(
    checkpoint: dict[str, Any],
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = DSCNN(
        input_channels=int(values.shape[1]),
        width=int(checkpoint.get("model_width", 24)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            chunks.append(torch.sigmoid(model(batch.to(device, non_blocking=True))).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    checkpoint_path = args.checkpoint.resolve()
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    output_dir = args.output_dir.resolve()
    validation_csv = labels_dir / "label_pairs_validation.csv"
    for path in (checkpoint_path, labels_dir, event_store_dir, validation_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    assert_no_forbidden_path(validation_csv)
    assert_development_csv(validation_csv)
    lineage = canonical_contract(checkpoint_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("Canonical checkpoint is not a DS-CNN")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Canonical checkpoint is marked as test-contaminated")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError("Canonical checkpoint was trained with held-out data")
    if checkpoint.get("target_data_used_for_selection") is not False:
        raise ValueError("Canonical checkpoint was selected with target data")
    peak_weights = {str(key): float(value) for key, value in checkpoint["selected_peak_weights"].items()}
    expected_weights = {peak: 1.0 / 3.0 for peak in EXPECTED_PEAKS}
    if peak_weights != expected_weights:
        raise ValueError(f"Checkpoint is not strict equal-weight: {peak_weights}")

    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    raw = load_raw_partition(validation_csv, event_store_dir)
    if tuple(sorted(set(raw.peak_ids.tolist()))) != tuple(sorted(EXPECTED_PEAKS)):
        raise ValueError("Held-out manifest does not contain exactly the three strict tasks")
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, checkpoint["feature_statistics"])
    if not np.all(np.isfinite(values)):
        raise ValueError("Held-out model inputs contain nonfinite values")
    weights = make_event_weights(raw.peak_ids, peak_weights)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"canonical_checkpoint={relative(checkpoint_path)}", flush=True)
    scores = predict(checkpoint, values, args.batch_size, device)
    result_metrics = metrics(raw.labels, scores, weights, raw.peak_ids, EXPECTED_PEAKS)

    score_path = output_dir / "held_out_scores.npz"
    np.savez_compressed(
        score_path,
        labels=raw.labels,
        peak_ids=raw.peak_ids,
        weights=weights,
        ds_cnn_scores=scores,
    )
    model_metadata = {
        "model_kind": "ds_cnn",
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": int(checkpoint["parameter_count"]),
        "representation_config": config.as_dict(),
        "feature_statistics": checkpoint["feature_statistics"],
        "selected_peak_weights": peak_weights,
        "scan_best_epoch": int(checkpoint["scan_best_epoch"]),
        "refit_epochs": int(checkpoint["refit_epochs"]),
        "internal_selection_metrics": checkpoint["scan_best_internal_metrics"],
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PAPER1_EQUAL_WEIGHT_STRICT_DS_CNN_HELD_OUT_EVALUATION",
        "lineage": lineage,
        "selection_complete_before_held_out": True,
        "held_out_scores_used_for_selection": False,
        "target_data_used_for_selection": False,
        "partition": "held_out_validation_file_partition",
        "label_csv": {"path": relative(validation_csv), "sha256": sha256_file(validation_csv)},
        "event_store_dir": relative(event_store_dir),
        "event_count": int(raw.labels.size),
        "pair_count": int(raw.labels.size // 2),
        "representation_qc": representation_qc,
        "models": {"ds_cnn": model_metadata},
        "metrics": {"ds_cnn": result_metrics},
        "score_artifact": {"path": relative(score_path), "sha256": sha256_file(score_path)},
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
        "scientific_boundary": (
            "Same-domain file-held-out strict proxy-label comparison. The first "
            "predeclared equal-weight seed is the sole Paper 1 model; no ensemble "
            "and no held-out or target-data seed selection were used."
        ),
    }
    report_path = output_dir / "held_out_evaluation.json"
    save_json(report_path, report)
    print(json.dumps(result_metrics, indent=2, sort_keys=True), flush=True)
    print(f"report={relative(report_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
