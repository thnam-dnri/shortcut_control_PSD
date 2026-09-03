#!/usr/bin/env python3
"""Score one frozen Ba-selected CNN on Ba, Na-22, and Cs-137 validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_source_ablation import grouped_metrics  # noqa: E402
from scripts.train_o2_late_fusion import sha256_file  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    evaluate_model,
    load_raw_partition,
    representation_config_from_checkpoint,
    make_loader,
    set_seed,
)

TARGETS = ("ba133", "na22", "cs137")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/ba133_cnn_optimization/full_data_finalists/both_ma10_global_t10_w750.pt",
    )
    parser.add_argument(
        "--ba-selection-results",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/ba133_cnn_optimization/full_data_finalists/optimization_results.json",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store",
    )
    parser.add_argument(
        "--ba-labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive",
        help="Ba validation domain used during checkpoint selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/ba133_cnn_optimization/frozen_transfer",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.resolve()
    selection_path = args.ba_selection_results.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_trial = selection["trials"][0]
    if selected_trial["checkpoint"] != checkpoint_path.name:
        raise ValueError("Checkpoint is not rank 1 in the frozen Ba selection results")
    if not selection["selection_domain"].startswith("ba133_"):
        raise ValueError("Selection domain is not Ba-only")
    if selection["target_domains_opened"] is not False:
        raise ValueError("Selection results report target-domain access")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    statistics = checkpoint["channel_statistics"]
    model_width = int(checkpoint["model_width"])
    set_seed(int(checkpoint["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactWaveformCNN(config.channel_count, width=model_width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    event_store_dir = args.event_store_dir.resolve()

    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_selection_domain": selection["selection_domain"],
            "checkpoint_frozen_before_target_access": True,
            "target_domains_used_for_model_selection": False,
            "test_partition_used": False,
            "na22_caveat": "The 511-keV photopeak contains significant Compton/background contamination.",
            "cs137_interpretation": "Preferred transfer diagnostic because the 661.7-keV photopeak has much lower Compton contamination.",
        },
        "selected_model": {
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selection_results": selection_path.relative_to(PROJECT_ROOT).as_posix(),
            "selection_results_sha256": sha256_file(selection_path),
            "representation": config.as_dict(),
            "model_width": model_width,
            "parameter_count": int(checkpoint["parameter_count"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "ba_selection_weighted_auroc": float(selected_trial["validation"]["weighted_auroc"]),
        },
        "domains": {},
    }
    score_arrays: dict[str, np.ndarray] = {}
    for source in TARGETS:
        csv_path = (
            args.ba_labels_dir.resolve() / "label_pairs_validation.csv"
            if source == "ba133"
            else PROJECT_ROOT
            / f"outputs/labels/source_ablation/{source}_positive/label_pairs_validation.csv"
        )
        print(f"Loading frozen target={source} ...", flush=True)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        raw = load_raw_partition(csv_path, event_store_dir)
        values, representation_qc = build_representation(raw, config)
        apply_channel_statistics(values, statistics)
        loader = make_loader(values, raw, args.batch_size, False, int(checkpoint["seed"]))
        _, scores = evaluate_model(model, loader, device)
        grouped = grouped_metrics(pair_rows, raw, scores)
        result["domains"][source] = {
            "validation_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(csv_path),
            "representation_qc": representation_qc,
            **grouped,
        }
        score_arrays[f"{source}_labels"] = raw.labels.astype(np.int8)
        score_arrays[f"{source}_weights"] = raw.weights
        score_arrays[f"{source}_peak_ids"] = raw.peak_ids
        score_arrays[f"{source}_scores"] = scores
        print(
            f"target={source} auroc={grouped['overall']['auroc']:.6f} "
            f"weighted={grouped['overall']['weighted_auroc']:.6f}",
            flush=True,
        )
    ba_rescore = result["domains"]["ba133"]["overall"]["weighted_auroc"]
    if not np.isclose(ba_rescore, selected_trial["validation"]["weighted_auroc"], atol=1.0e-12):
        raise ValueError("Frozen Ba rescore does not reproduce model-selection metric")
    metrics_path = output_dir / "frozen_transfer_metrics.json"
    scores_path = output_dir / "frozen_transfer_scores.npz"
    save_json(metrics_path, result)
    np.savez_compressed(scores_path, **score_arrays)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {scores_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
