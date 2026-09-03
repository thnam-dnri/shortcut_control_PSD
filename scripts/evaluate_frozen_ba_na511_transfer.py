#!/usr/bin/env python3
"""Evaluate the frozen Ba-all-peaks plus Na-511 compact CNN by domain."""

from __future__ import annotations

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


def main() -> int:
    checkpoint_path = PROJECT_ROOT / "outputs/models/ba_all_na511_cnn/both_ma10_global_t10_w750.pt"
    selection_path = PROJECT_ROOT / "outputs/models/ba_all_na511_cnn/optimization_results.json"
    output_dir = PROJECT_ROOT / "outputs/models/ba_all_na511_cnn/frozen_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_trial = selection["trials"][0]
    expected_domain = "ba133_all_peaks_plus_na22_511kev_internal_validation_only"
    if selection["selection_domain"] != expected_domain:
        raise ValueError("Unexpected selection domain")
    if selection["target_domains_opened"] is not False:
        raise ValueError("Selection results report target access")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["selection_domain"] != expected_domain:
        raise ValueError("Checkpoint selection-domain mismatch")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    set_seed(int(checkpoint["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactWaveformCNN(config.channel_count, width=int(checkpoint["model_width"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    event_store_dir = PROJECT_ROOT / "processed_data/event_store"
    csv_paths = {
        "combined_ba_na511": PROJECT_ROOT / "outputs/labels/ba_all_na511_positive/label_pairs_validation.csv",
        "ba133": PROJECT_ROOT / "outputs/labels/source_ablation/ba133_positive/label_pairs_validation.csv",
        "na22": PROJECT_ROOT / "outputs/labels/source_ablation/na22_positive/label_pairs_validation.csv",
        "cs137": PROJECT_ROOT / "outputs/labels/source_ablation/cs137_positive/label_pairs_validation.csv",
    }
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection_domain": expected_domain,
            "checkpoint_frozen_before_cs137_access": True,
            "cs137_used_for_model_selection": False,
            "test_partition_used": False,
        },
        "selected_model": {
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selection_results": selection_path.relative_to(PROJECT_ROOT).as_posix(),
            "selection_results_sha256": sha256_file(selection_path),
            "representation": config.as_dict(),
            "model_width": int(checkpoint["model_width"]),
            "parameter_count": int(checkpoint["parameter_count"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "selection_weighted_auroc": float(selected_trial["validation"]["weighted_auroc"]),
        },
        "domains": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for domain, csv_path in csv_paths.items():
        print(f"Loading domain={domain}", flush=True)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        raw = load_raw_partition(csv_path, event_store_dir)
        values, representation_qc = build_representation(raw, config)
        apply_channel_statistics(values, checkpoint["channel_statistics"])
        loader = make_loader(values, raw, 512, False, int(checkpoint["seed"]))
        _, scores = evaluate_model(model, loader, device)
        grouped = grouped_metrics(pair_rows, raw, scores)
        result["domains"][domain] = {
            "validation_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(csv_path),
            "representation_qc": representation_qc,
            **grouped,
        }
        arrays[f"{domain}_labels"] = raw.labels.astype(np.int8)
        arrays[f"{domain}_weights"] = raw.weights
        arrays[f"{domain}_scores"] = scores
        print(domain, grouped["overall"], flush=True)
    reproduced = result["domains"]["combined_ba_na511"]["overall"]["weighted_auroc"]
    if not np.isclose(reproduced, selected_trial["validation"]["weighted_auroc"], atol=1.0e-12):
        raise ValueError("Combined-domain rescore failed")
    metrics_path = output_dir / "frozen_transfer_metrics.json"
    scores_path = output_dir / "frozen_transfer_scores.npz"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(scores_path, **arrays)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
