#!/usr/bin/env python3
"""Evaluate a frozen three-peak Compact CNN on the held-out validation files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import relative, sha256_file, utc_now  # noqa: E402
from scripts.scan_three_peak_weight_combinations import make_event_weights, metric_summary  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--event-store-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    checkpoint_path = args.checkpoint.resolve()
    label_csv = args.label_csv.resolve()
    event_store_dir = args.event_store_dir.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "compact_cnn" or checkpoint.get("test_partition_used") is not False:
        raise ValueError("Unexpected or test-contaminated Compact checkpoint")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    if config.pulse_polarity != "negative_to_positive":
        raise ValueError("Checkpoint does not use the corrected positive-polarity representation")

    raw = load_raw_partition(label_csv, event_store_dir)
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, checkpoint["feature_statistics"])
    weights = make_event_weights(raw.peak_ids[::2], checkpoint["selected_peak_weights"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = CompactWaveformCNN(config.channel_count, width=int(checkpoint["model_width"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    score_chunks = []
    with torch.inference_mode():
        for (batch,) in loader:
            score_chunks.append(torch.sigmoid(model(batch.to(device, non_blocking=True))).cpu().numpy())
    scores = np.concatenate(score_chunks)
    metrics = metric_summary(raw.labels, scores, weights, raw.peak_ids)

    score_path = output_dir / "held_out_scores.npz"
    np.savez_compressed(
        score_path,
        labels=raw.labels,
        scores=scores,
        peak_ids=raw.peak_ids,
        weights=weights,
    )
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "model_name": "Corrected-polarity three-peak Compact CNN",
        "checkpoint": {"path": relative(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "label_csv": {"path": relative(label_csv), "sha256": sha256_file(label_csv)},
        "event_store_dir": relative(event_store_dir),
        "partition": "validation",
        "event_count": int(raw.labels.size),
        "pair_count": int(raw.labels.size // 2),
        "representation_config": checkpoint["representation_config"],
        "feature_statistics": checkpoint["feature_statistics"],
        "representation_qc": representation_qc,
        "selected_peak_weights": checkpoint["selected_peak_weights"],
        "internal_selection_metrics": checkpoint["selection_internal_metrics"],
        "held_out_metrics": metrics,
        "score_artifact": {"path": relative(score_path), "sha256": sha256_file(score_path)},
        "test_partition_used": False,
        "scientific_boundary": (
            "Held-out existing validation files are source/file-disjoint from training but are not a newly "
            "independent isotope or session campaign."
        ),
    }
    report_path = output_dir / "held_out_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
