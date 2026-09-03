#!/usr/bin/env python3
"""Evaluate source-specific models on every other positive-source domain.

The script forms a 3x3 transfer matrix from the frozen Ba-133-, Na-22-, and
Cs-137-positive models.  Every target domain uses its existing internal
validation pairs and physically supported, 0.5-keV-matched continuum
negatives.  The locked test pair manifest is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_source_ablation import cnn_scores, grouped_metrics  # noqa: E402
from train_boosting_baselines import tabular_features  # noqa: E402
from train_o2_late_fusion import build_partition_features, sha256_file  # noqa: E402

SOURCES = ("ba133", "na22", "cs137")
MODEL_NAMES = ("hist_gradient_boosting", "xgboost", "o2_late_fusion")


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/source_ablation",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/source_ablation",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/cross_source_transfer",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    label_root = args.label_root.resolve()
    model_root = args.model_root.resolve()
    event_store_dir = args.event_store_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets: dict[str, dict[str, Any]] = {}
    for target_source in SOURCES:
        csv_path = label_root / f"{target_source}_positive/label_pairs_validation.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            pair_rows = list(csv.DictReader(stream))
        print(f"Loading {target_source} validation features ...", flush=True)
        data = build_partition_features(
            csv_path,
            event_store_dir=event_store_dir,
        )
        targets[target_source] = {
            "csv_path": csv_path,
            "csv_sha256": sha256_file(csv_path),
            "pair_rows": pair_rows,
            "data": data,
            "tabular_features": tabular_features(data),
        }

    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training_domains": list(SOURCES),
            "validation_domains": list(SOURCES),
            "validation_partition": "existing internal validation files",
            "target_negative_policy": (
                "Each target uses its frozen source-ablation continuum negatives, "
                "matched one-to-one to positives in 0.5-keV bins."
            ),
            "event_store_dir": relative(event_store_dir),
            "test_partition_used": False,
            "locked_test_csv_read": False,
        },
        "model_files": {},
        "target_inputs": {
            source: {
                "validation_csv": relative(targets[source]["csv_path"]),
                "validation_csv_sha256": targets[source]["csv_sha256"],
                "pair_count": len(targets[source]["pair_rows"]),
                "event_count": int(targets[source]["data"].labels.size),
                "t10_fallback_count": int(targets[source]["data"].t10_fallback_count),
            }
            for source in SOURCES
        },
        "transfer_matrix": {},
    }
    flat_rows: list[dict[str, Any]] = []

    for train_source in SOURCES:
        experiment_root = model_root / f"{train_source}_positive"
        model_paths = {
            "hist_gradient_boosting": experiment_root
            / "boosting/hist_gradient_boosting.joblib",
            "xgboost": experiment_root / "boosting/xgboost.joblib",
            "o2_late_fusion": experiment_root
            / "o2_late_fusion/o2_late_fusion_best.pt",
        }
        for path in model_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        result["model_files"][train_source] = {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in model_paths.items()
        }
        tree_models = {
            name: joblib.load(model_paths[name])
            for name in ("hist_gradient_boosting", "xgboost")
        }
        result["transfer_matrix"][train_source] = {}

        for target_source in SOURCES:
            target = targets[target_source]
            model_results: dict[str, Any] = {}
            for model_name in MODEL_NAMES:
                if model_name == "o2_late_fusion":
                    scores = cnn_scores(
                        model_paths[model_name],
                        target["data"],
                        device,
                    )
                else:
                    scores = tree_models[model_name].predict_proba(
                        target["tabular_features"]
                    )[:, 1]
                model_results[model_name] = grouped_metrics(
                    target["pair_rows"],
                    target["data"],
                    np.asarray(scores),
                )
                overall = model_results[model_name]["overall"]
                flat_rows.append(
                    {
                        "model": model_name,
                        "train_positive_source": train_source,
                        "validation_positive_source": target_source,
                        "same_source_diagonal": train_source == target_source,
                        **overall,
                    }
                )
            result["transfer_matrix"][train_source][target_source] = model_results
            print(
                f"train={train_source} target={target_source}",
                {
                    name: round(model_results[name]["overall"]["auroc"], 6)
                    for name in MODEL_NAMES
                },
                flush=True,
            )

    json_path = output_dir / "cross_source_transfer_matrix.json"
    csv_path = output_dir / "cross_source_transfer_summary.csv"
    save_json(json_path, result)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
