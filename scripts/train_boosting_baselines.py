#!/usr/bin/env python3
"""Train validation-only histogram-gradient and XGBoost waveform baselines.

The script consumes the same file-disjoint train/validation pair manifests as the
CNN.  It deliberately does not read ``label_pairs_test.csv``.  Inputs are compact
shape summaries of the frozen, energy-normalized charge/current representations;
reconstructed energy and source/QC identifiers are excluded from model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_o2_late_fusion import (  # noqa: E402
    PartitionData,
    build_partition_features,
    per_peak_metrics,
    sha256_file,
)

SEED = 20260811
CHARGE_BINS = 30
CURRENT_BINS = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binned_statistics(values: np.ndarray, bins: int) -> np.ndarray:
    """Return bin-wise mean, standard deviation, and extrema."""
    edges = np.linspace(0, values.shape[1], bins + 1, dtype=np.int64)
    summaries: list[np.ndarray] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        section = values[:, start:stop]
        summaries.extend(
            (
                np.mean(section, axis=1),
                np.std(section, axis=1),
                np.max(section, axis=1),
                np.min(section, axis=1),
            )
        )
    return np.column_stack(summaries).astype(np.float32)


def tabular_features(data: PartitionData) -> np.ndarray:
    """Compress aligned branch waveforms into source-agnostic shape features."""
    charge = data.charge
    current = data.current
    global_features = np.column_stack(
        (
            np.mean(charge, axis=1),
            np.std(charge, axis=1),
            np.max(charge, axis=1),
            np.min(charge, axis=1),
            np.sum(charge, axis=1),
            np.argmax(charge, axis=1) / charge.shape[1],
            np.mean(current, axis=1),
            np.std(current, axis=1),
            np.max(current, axis=1),
            np.min(current, axis=1),
            np.argmax(current, axis=1) / current.shape[1],
            np.argmin(current, axis=1) / current.shape[1],
        )
    ).astype(np.float32)
    features = np.column_stack(
        (
            global_features,
            binned_statistics(charge, CHARGE_BINS),
            binned_statistics(current, CURRENT_BINS),
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite tabular feature encountered")
    return features


def metric_summary(data: PartitionData, scores: np.ndarray) -> dict[str, Any]:
    labels = data.labels.astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(
            roc_auc_score(labels, scores, sample_weight=data.weights)
        ),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=data.weights)
        ),
        "per_peak": per_peak_metrics(
            labels,
            scores,
            data.weights,
            data.peak_ids,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "outputs/labels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/boosting_baselines",
    )
    parser.add_argument("--max-events-per-partition", type=int, default=None)
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=None,
        help="Optional consolidated event-store directory for faster waveform reads.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    dataset_manifest = labels_dir / "label_dataset_manifest.json"
    partition_manifest = labels_dir / "file_partition_manifest.json"
    for path in (train_csv, validation_csv, dataset_manifest, partition_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading train waveform representations ...", flush=True)
    event_store_dir = (
        args.event_store_dir.resolve() if args.event_store_dir is not None else None
    )
    train_data = build_partition_features(
        train_csv,
        args.max_events_per_partition,
        event_store_dir=event_store_dir,
    )
    print("Loading validation waveform representations ...", flush=True)
    validation_data = build_partition_features(
        validation_csv,
        args.max_events_per_partition,
        event_store_dir=event_store_dir,
    )
    print("Building tabular shape summaries ...", flush=True)
    train_features = tabular_features(train_data)
    validation_features = tabular_features(validation_data)
    train_labels = train_data.labels.astype(np.int64)

    models: dict[str, Any] = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    metrics: dict[str, Any] = {}
    model_files: dict[str, str] = {}
    for name, model in models.items():
        print(f"Training {name} ...", flush=True)
        fit_kwargs: dict[str, Any] = {"sample_weight": train_data.weights}
        if name == "xgboost":
            fit_kwargs["eval_set"] = [(validation_features, validation_data.labels)]
            fit_kwargs["verbose"] = False
        model.fit(train_features, train_labels, **fit_kwargs)
        train_scores = model.predict_proba(train_features)[:, 1]
        validation_scores = model.predict_proba(validation_features)[:, 1]
        metrics[name] = {
            "train": metric_summary(train_data, train_scores),
            "validation": metric_summary(validation_data, validation_scores),
        }
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_files[name] = model_path.name
        print(
            f"{name}: validation_auroc={metrics[name]['validation']['auroc']:.6f} "
            f"validation_weighted_auroc={metrics[name]['validation']['weighted_auroc']:.6f}",
            flush=True,
        )

    run = {
        "created_utc": utc_now(),
        "seed": SEED,
        "input": {
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv_sha256": sha256_file(validation_csv),
            "test_csv": "outputs/labels/label_pairs_test.csv",
            "test_partition_used": False,
            "label_dataset_manifest_sha256": sha256_file(dataset_manifest),
            "file_partition_manifest_sha256": sha256_file(partition_manifest),
            "train_event_count": int(train_data.labels.size),
            "validation_event_count": int(validation_data.labels.size),
            "max_events_per_partition": args.max_events_per_partition,
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix()
            if event_store_dir is not None
            else None,
        },
        "features": {
            "count": int(train_features.shape[1]),
            "charge_bins": CHARGE_BINS,
            "current_bins": CURRENT_BINS,
            "statistics_per_bin": ["mean", "standard_deviation", "maximum", "minimum"],
            "energy_feature_included": False,
            "source_or_qc_feature_included": False,
            "representation": "frozen energy-normalized O2-style charge/current windows",
        },
        "libraries": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "model_files": model_files,
        "selection_scope": "internal train/validation only; test partition unopened",
        "caveats": [
            "Positive and negative classes originate from different isotope sources.",
            "Exploratory manifests retain WARN/FAIL/UNKNOWN QC files.",
            "Validation metrics are internal baselines, not topology-performance estimates.",
        ],
    }
    save_json(output_dir / "internal_metrics.json", metrics)
    save_json(output_dir / "training_run.json", run)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
