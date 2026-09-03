#!/usr/bin/env python3
"""Score frozen Compact CNN and DS-CNN checkpoints on the held-out partition.

This script is intentionally separate from training.  It must be run only after
model selection and checkpoint provenance are complete; held-out scores are
reported, not used for further tuning.
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

from scripts.train_compact_ds_cnn_performance import (  # noqa: E402
    DEFAULT_EVENT_STORE_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_OUTPUT_DIR,
    MODEL_NAMES,
    SELECTED_PEAK_WEIGHTS,
    build_model,
    load_reference_contract,
    make_event_weights,
    metric_summary,
    sha256_file,
)
from src.ba133_cnn import (  # noqa: E402
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def predict_checkpoint(
    checkpoint_path: Path,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") not in MODEL_NAMES:
        raise ValueError(f"Unexpected model kind in {checkpoint_path}")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError(f"Checkpoint is marked as test-contaminated: {checkpoint_path}")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError(f"Checkpoint was trained with held-out data: {checkpoint_path}")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    model = build_model(checkpoint["model_kind"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    score_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            score_chunks.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(score_chunks)
    metadata = {
        "model_kind": checkpoint["model_kind"],
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": checkpoint["parameter_count"],
        "representation_config": config.as_dict(),
        "feature_statistics": checkpoint["feature_statistics"],
        "selected_peak_weights": checkpoint["selected_peak_weights"],
        "refit_epochs": checkpoint["refit_epochs"],
        "scan_best_epoch": checkpoint["scan_best_epoch"],
        "internal_selection_metrics": checkpoint["scan_best_internal_metrics"],
    }
    del model, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "held_out_evaluation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    label_csv = labels_dir / "label_pairs_validation.csv"
    assert_no_forbidden_path(label_csv)
    assert_development_csv(label_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config, reference_statistics, _ = load_reference_contract(
        PROJECT_ROOT / "outputs/models/three_peak_positive_polarity_20260820/compact_cnn_best.pt"
    )
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"loading held-out validation manifest={label_csv}", flush=True)
    raw = load_raw_partition(label_csv, event_store_dir)
    values, representation_qc = build_representation(raw, reference_config)
    apply_channel_statistics(values, reference_statistics)
    if not np.all(np.isfinite(values)):
        raise ValueError("Held-out representation contains nonfinite values")
    weights = make_event_weights(raw.peak_ids, SELECTED_PEAK_WEIGHTS)
    scores_by_model: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}
    for name in MODEL_NAMES:
        checkpoint_path = model_dir / name / f"{name}_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        print(f"scoring model={name}", flush=True)
        scores, metadata = predict_checkpoint(checkpoint_path, values, args.batch_size, device)
        if metadata["representation_config"] != reference_config.as_dict():
            raise ValueError(f"Representation mismatch for {name}")
        if metadata["feature_statistics"] != reference_statistics:
            raise ValueError(f"Feature-statistics mismatch for {name}")
        scores_by_model[name] = scores
        model_metadata[name] = metadata

    np.savez_compressed(
        output_dir / "held_out_scores.npz",
        labels=raw.labels,
        peak_ids=raw.peak_ids,
        weights=weights,
        compact_cnn_scores=scores_by_model["compact_cnn"],
        ds_cnn_scores=scores_by_model["ds_cnn"],
    )
    metrics = {
        name: metric_summary(raw.labels, scores, weights, raw.peak_ids)
        for name, scores in scores_by_model.items()
    }
    report = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "corrected Compact CNN versus DS-CNN performance comparison",
        "selection_complete_before_held_out": True,
        "held_out_scores_used_for_selection": False,
        "warning_status": "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED",
        "device": str(device),
        "label_csv": {
            "path": label_csv.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(label_csv),
        },
        "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
        "partition": "held_out_validation_file_partition",
        "event_count": int(raw.labels.size),
        "pair_count": int(raw.labels.size // 2),
        "representation_config": reference_config.as_dict(),
        "feature_statistics": reference_statistics,
        "representation_qc": representation_qc,
        "selected_peak_weights": SELECTED_PEAK_WEIGHTS,
        "models": model_metadata,
        "metrics": metrics,
        "score_artifact": {
            "path": (output_dir / "held_out_scores.npz").relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(output_dir / "held_out_scores.npz"),
        },
        "test_partition_used": False,
        "scientific_boundary": (
            "This is an existing source/file-disjoint development validation partition, "
            "not an independent isotope or session campaign."
        ),
    }
    save_json(output_dir / "held_out_evaluation.json", report)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"wrote={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
