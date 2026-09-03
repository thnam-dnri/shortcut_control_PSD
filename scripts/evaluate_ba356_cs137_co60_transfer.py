#!/usr/bin/env python3
"""Evaluate Ba-356 models on the Cs-137 photopeak vs Co-60 continuum."""

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

from scripts.evaluate_source_ablation import cnn_scores, metrics  # noqa: E402
from scripts.train_o2_late_fusion import (  # noqa: E402
    build_partition_features,
    sha256_file,
)
from src.ba133_cnn import (  # noqa: E402
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    evaluate_model,
    load_raw_partition,
    representation_config_from_checkpoint,
    make_loader,
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--compact-selection-results", type=Path, required=True)
    parser.add_argument("--late-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cs-labels-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/labels/architecture_pass_warn_20260815_source_ablation/cs137_positive",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def co60_pair_indices(pair_rows: list[dict[str, str]]) -> np.ndarray:
    selected = np.asarray(
        [row["negative_source"] == "co60" for row in pair_rows], dtype=bool
    )
    pair_indices = np.flatnonzero(selected)
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def score_subset(
    pair_rows: list[dict[str, str]],
    labels: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    if labels.size != 2 * len(pair_rows) or scores.size != labels.size:
        raise ValueError("Pair CSV and score lengths disagree")
    if not np.array_equal(
        labels.reshape(-1, 2), np.asarray([[1.0, 0.0]] * len(pair_rows))
    ):
        raise ValueError("Expected positive/negative pair ordering")
    indices = co60_pair_indices(pair_rows)
    selected_rows = [
        row for row in pair_rows if row["negative_source"] == "co60"
    ]
    result = metrics(labels[indices], scores[indices], weights[indices])
    result.update(
        {
            "positive_source": "cs137",
            "positive_peak_id": "cs137_662kev",
            "negative_source": "co60",
            "negative_definition": (
                "Energy-matched Co-60 continuum candidates in the 661.7-keV region"
            ),
            "selected_pair_count": len(selected_rows),
            "maximum_absolute_energy_difference_kev": max(
                abs(
                    float(row["positive_energy_kev"])
                    - float(row["negative_energy_kev"])
                )
                for row in selected_rows
            ),
        }
    )
    return result


def compact_scores(
    checkpoint_path: Path,
    selection_path: Path,
    csv_path: Path,
    event_store_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    raw = load_raw_partition(csv_path, event_store_dir)
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, checkpoint["channel_statistics"])
    model = CompactWaveformCNN(
        config.channel_count, width=int(checkpoint["model_width"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = make_loader(values, raw, batch_size, False, int(checkpoint["seed"]))
    _, scores = evaluate_model(model, loader, device)
    result = {
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_results": relative(selection_path),
        "selection_results_sha256": sha256_file(selection_path),
        "selection_domain": selection["selection_domain"],
        "selection_weighted_auroc": selection["ranking"][0][
            "validation_weighted_auroc"
        ],
        "best_epoch": int(checkpoint["best_epoch"]),
        "representation": config.as_dict(),
        "representation_qc": representation_qc,
    }
    del model, values, raw, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, result


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    cs_labels_dir = args.cs_labels_dir.resolve()
    csv_path = cs_labels_dir / "label_pairs_validation.csv"
    dataset_manifest_path = cs_labels_dir / "label_dataset_manifest.json"
    for path in (csv_path, dataset_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    pair_rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not pair_rows:
        raise ValueError("Cs-137 validation manifest is empty")
    if any(row["positive_source"] != "cs137" for row in pair_rows):
        raise ValueError("Target manifest contains a non-Cs-137 positive source")
    if any(row["peak_id"] != "cs137_662kev" for row in pair_rows):
        raise ValueError("Target manifest contains an unexpected peak")
    if not any(row["negative_source"] == "co60" for row in pair_rows):
        raise ValueError("Target manifest contains no Co-60 continuum pairs")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compact_checkpoint = args.compact_checkpoint.resolve()
    compact_selection = args.compact_selection_results.resolve()
    late_checkpoint = args.late_checkpoint.resolve()
    compact_scores_array, compact_info = compact_scores(
        compact_checkpoint,
        compact_selection,
        csv_path,
        args.event_store_dir.resolve(),
        args.batch_size,
        device,
    )
    late_data = build_partition_features(
        csv_path, event_store_dir=args.event_store_dir.resolve()
    )
    late_scores_array = cnn_scores(late_checkpoint, late_data, device)

    target_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training_positive_domain": "Ba-133 356-keV photopeak only",
            "target_positive_domain": "Cs-137 661.7-keV photopeak ROI",
            "target_negative_domain": "Co-60 energy-matched continuum around 661.7 keV",
            "target_roi": target_manifest["experiment"]["positive_peaks"][0],
            "target_selection_used_for_training_or_checkpoint_selection": False,
            "test_partition_used": False,
            "continuum_truth_warning": target_manifest["matching"][
                "continuum_truth_warning"
            ],
        },
        "target_manifest": {
            "path": relative(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
            "validation_csv": relative(csv_path),
            "validation_csv_sha256": sha256_file(csv_path),
            "total_pair_count": len(pair_rows),
            "co60_pair_count": sum(
                row["negative_source"] == "co60" for row in pair_rows
            ),
            "other_negative_source_counts": {
                source: sum(row["negative_source"] == source for row in pair_rows)
                for source in sorted(
                    {row["negative_source"] for row in pair_rows} - {"co60"}
                )
            },
        },
        "event_store_dir": relative(args.event_store_dir),
        "device": str(device),
        "models": {
            "compact_cnn": {
                **compact_info,
                "metrics_cs137_vs_co60": score_subset(
                    pair_rows,
                    late_data.labels,
                    late_data.weights,
                    compact_scores_array,
                ),
            },
            "o2_late_fusion": {
                "checkpoint": relative(late_checkpoint),
                "checkpoint_sha256": sha256_file(late_checkpoint),
                "metrics_cs137_vs_co60": score_subset(
                    pair_rows,
                    late_data.labels,
                    late_data.weights,
                    late_scores_array,
                ),
            },
        },
    }
    metrics_path = output_dir / "cs137_co60_transfer_metrics.json"
    scores_path = output_dir / "cs137_co60_transfer_scores.npz"
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected_indices = co60_pair_indices(pair_rows)
    np.savez_compressed(
        scores_path,
        pair_indices=(selected_indices.reshape(-1, 2) // 2)[:, 0],
        positive_energy_kev=np.asarray(
            [float(pair_rows[index // 2]["positive_energy_kev"]) for index in selected_indices[::2]]
        ),
        negative_energy_kev=np.asarray(
            [float(pair_rows[index // 2]["negative_energy_kev"]) for index in selected_indices[1::2]]
        ),
        compact_scores=compact_scores_array[selected_indices],
        late_fusion_scores=late_scores_array[selected_indices],
    )
    print(json.dumps(result["models"], indent=2, sort_keys=True))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
