#!/usr/bin/env python3
"""Cross-fit morphology-group and frozen MA20 DS-CNN score fusion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_shared_six_group_ds_cnn import (
    GROUPS,
    PEAK_IDS,
    IndexedWaveforms,
    make_eval_loader,
    predict,
    sha256_file,
)
from src.architecture_candidates import DSCNN
from src.data_access_guards import assert_no_forbidden_path

SEED = 20260822


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        value += float(np.count_nonzero(mask)) / total * abs(
            float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
        )
    return value


def metrics(labels: np.ndarray, peaks: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    per_peak = {
        peak: {
            "auroc": float(roc_auc_score(labels[peaks == peak], scores[peaks == peak])),
            "average_precision": float(
                average_precision_score(labels[peaks == peak], scores[peaks == peak])
            ),
            "events": int(np.count_nonzero(peaks == peak)),
        }
        for peak in PEAK_IDS
    }
    return {
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "macro_peak_auroc": float(np.mean([row["auroc"] for row in per_peak.values()])),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores)),
        "expected_calibration_error_10_bins": expected_calibration_error(labels, scores),
        "prevalence": float(np.mean(labels)),
        "events": int(labels.size),
        "per_peak": per_peak,
    }


def group_metrics(
    labels: np.ndarray,
    peaks: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    return {
        f"group_{group}": metrics(
            labels[groups == group], peaks[groups == group], scores[groups == group]
        )
        for group in GROUPS
    }


def group_features(groups: np.ndarray) -> np.ndarray:
    return np.column_stack([(groups == group).astype(np.float64) for group in GROUPS[1:]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/group_fusion_natural_validation_ma20_20260822",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/shared_six_group_ds_cnn_ma20_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/group_ds_cnn_fusion_20260822",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (cache_dir, model_dir, output_dir):
        assert_no_forbidden_path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    values_path = cache_dir / "natural_file_validation_values.npy"
    metadata_path = cache_dir / "natural_file_validation_metadata.npz"
    assignments_path = cache_dir / "natural_file_validation_assignments.npz"
    dataset = IndexedWaveforms(
        values_path, metadata_path, assignments_path, groups=GROUPS
    )
    loader = make_eval_loader(dataset, args.batch_size)
    training_report = json.loads(
        (model_dir / "experiment_report.json").read_text(encoding="utf-8")
    )
    seed_scores: list[np.ndarray] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for run in training_report["runs"]:
        checkpoint_path = PROJECT_ROOT / run["shared_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["representation_config"]["moving_average"] != 20:
            raise ValueError(f"Non-MA20 checkpoint: {checkpoint_path}")
        model = DSCNN(input_channels=2, width=24).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores = predict(model, loader, device)
        seed_scores.append(scores)
        checkpoint_rows.append(
            {
                "seed": int(run["seed"]),
                "path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(checkpoint_path),
            }
        )
        del model
        torch.cuda.empty_cache()
    ensemble_score = np.mean(np.stack(seed_scores), axis=0)
    np.save(output_dir / "ma20_shared_ensemble_scores.npy", ensemble_score)

    metadata = np.load(metadata_path)
    hdf5 = metadata["hdf5"][dataset.cache_indices].astype(str)
    labels = dataset.labels.astype(np.int8)
    peaks = dataset.peak_ids.astype(str)
    groups = dataset.groups.astype(np.int16)
    clipped = np.clip(ensemble_score.astype(np.float64), 1.0e-6, 1.0 - 1.0e-6)
    ds_logit = np.log(clipped / (1.0 - clipped))[:, None]
    one_hot = group_features(groups)
    feature_sets = {
        "group_only": one_hot,
        "ds_cnn_only": ds_logit,
        "combined": np.column_stack((ds_logit, one_hot)),
    }
    strata = np.asarray(
        [f"{peak}:{label}" for peak, label in zip(peaks, labels)], dtype="U48"
    )
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=SEED
    )
    fold_indices = list(splitter.split(ds_logit, strata, groups=hdf5))
    fold_id = np.full(labels.size, -1, dtype=np.int8)
    oof_scores = {
        name: np.full(labels.size, np.nan, dtype=np.float64)
        for name in feature_sets
    }
    fold_reports: list[dict[str, Any]] = []
    for fold, (fit_indices, held_indices) in enumerate(fold_indices):
        fit_files = set(hdf5[fit_indices])
        held_files = set(hdf5[held_indices])
        if fit_files & held_files:
            raise ValueError("File leakage across fusion folds")
        model_rows: dict[str, Any] = {}
        for name, features in feature_sets.items():
            calibrator = LogisticRegression(
                C=args.regularization_c,
                l1_ratio=0.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight=None,
                random_state=SEED + fold,
            )
            calibrator.fit(features[fit_indices], labels[fit_indices])
            oof_scores[name][held_indices] = calibrator.predict_proba(
                features[held_indices]
            )[:, 1]
            model_rows[name] = {
                "intercept": calibrator.intercept_.tolist(),
                "coefficients": calibrator.coef_.tolist(),
                "iterations": calibrator.n_iter_.tolist(),
            }
        fold_id[held_indices] = fold
        fold_reports.append(
            {
                "fold": fold,
                "fit_events": int(fit_indices.size),
                "held_events": int(held_indices.size),
                "fit_files": len(fit_files),
                "held_files": len(held_files),
                "models": model_rows,
                "evaluation": {
                    name: metrics(
                        labels[held_indices],
                        peaks[held_indices],
                        scores[held_indices],
                    )
                    for name, scores in oof_scores.items()
                },
            }
        )
    if np.any(fold_id < 0):
        raise ValueError("Incomplete fusion fold assignment")
    np.save(output_dir / "fusion_fold_id.npy", fold_id)
    for name, scores in oof_scores.items():
        if np.any(~np.isfinite(scores)):
            raise ValueError(f"Incomplete out-of-fold scores: {name}")
        np.save(output_dir / f"{name}_oof_scores.npy", scores.astype(np.float32))

    results = {
        name: {
            "global": metrics(labels, peaks, scores),
            "by_group": group_metrics(labels, peaks, groups, scores),
        }
        for name, scores in oof_scores.items()
    }
    group_ratios = {
        f"group_{group}": {
            "positive": int(np.count_nonzero((groups == group) & (labels == 1))),
            "negative": int(np.count_nonzero((groups == group) & (labels == 0))),
            "positive_fraction": float(np.mean(labels[groups == group])),
        }
        for group in GROUPS
    }
    combined_gain = (
        results["combined"]["global"]["pooled_auroc"]
        - results["ds_cnn_only"]["global"]["pooled_auroc"]
    )
    fold_auroc_summary = {
        name: {
            "mean_pooled_auroc": float(
                np.mean(
                    [row["evaluation"][name]["pooled_auroc"] for row in fold_reports]
                )
            ),
            "standard_deviation_pooled_auroc": float(
                np.std(
                    [row["evaluation"][name]["pooled_auroc"] for row in fold_reports],
                    ddof=1,
                )
            ),
        }
        for name in feature_sets
    }
    fold_mean_gain = (
        fold_auroc_summary["combined"]["mean_pooled_auroc"]
        - fold_auroc_summary["ds_cnn_only"]["mean_pooled_auroc"]
    )
    fold_gains = [
        row["evaluation"]["combined"]["pooled_auroc"]
        - row["evaluation"]["ds_cnn_only"]["pooled_auroc"]
        for row in fold_reports
    ]
    supporting_folds = int(np.count_nonzero(np.asarray(fold_gains) >= 0.0))
    decision = (
        "GROUP_FUSION_PROVISIONALLY_SUPPORTED_EXTERNAL_VALIDATION_REQUIRED"
        if fold_mean_gain >= 0.004 and supporting_folds >= 2
        else "GROUP_FUSION_NOT_SUPPORTED"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "combined_minus_ds_cnn_pooled_auroc": combined_gain,
        "combined_minus_ds_cnn_mean_fold_auroc": fold_mean_gain,
        "combined_minus_ds_cnn_by_fold": fold_gains,
        "folds_with_nonnegative_gain": supporting_folds,
        "fold_auroc_summary": fold_auroc_summary,
        "results": results,
        "group_class_ratios": group_ratios,
        "cross_fitting": {
            "folds": args.folds,
            "grouping": "complete HDF5 file",
            "stratification": "peak_id plus class label",
            "regularization": f"L2 logistic regression C={args.regularization_c}",
            "fold_reports": fold_reports,
        },
        "base_model": {
            "kind": "three-seed frozen shared MA20 DS-CNN ensemble",
            "checkpoints": checkpoint_rows,
            "ensemble_score_sha256": sha256_file(
                output_dir / "ma20_shared_ensemble_scores.npy"
            ),
        },
        "input": {
            "dataset_report_sha256": sha256_file(cache_dir / "dataset_report.json"),
            "metadata_sha256": sha256_file(metadata_path),
            "assignments_sha256": sha256_file(assignments_path),
        },
        "claim_boundary": (
            "Out-of-fold development proxy on an unbalanced source/ROI candidate "
            "pool. Ratios reflect available source exposures and selection, not "
            "measured deployment interaction prevalence."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "combined_minus_ds_cnn_pooled_auroc": combined_gain,
                "combined_minus_ds_cnn_mean_fold_auroc": fold_mean_gain,
                "combined_minus_ds_cnn_by_fold": fold_gains,
                "fold_auroc_summary": fold_auroc_summary,
                "global_results": {
                    name: row["global"] for name, row in results.items()
                },
                "group_class_ratios": group_ratios,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
