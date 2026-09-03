#!/usr/bin/env python3
"""Measure scalar waveform/acquisition nuisance baselines for frozen development data.

This diagnostic intentionally uses scalar/context features that are forbidden as
inputs to the deployable waveform classifiers. It quantifies how much apparent
discrimination is available from energy, noise, QC, source, session, and file
identity alone. Only train/validation pair manifests and their versioned event
stores are opened; test, Th-232, and Eu-152 are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_access_guards import assert_development_csv, assert_development_partition  # noqa: E402

NUMERIC_FEATURES = [
    "event_reconstructed_energy_kev",
    "event_shaped_energy_unit",
    "event_pulse_extremum_adc",
    "event_pulse_extremum_index",
    "event_trigger_time_s",
    "event_noise_mean_adc",
    "event_noise_max_adc",
    "event_noise_std_adc",
    "pair_abs_energy_delta_kev",
    "pair_match_bin_low_kev",
]
QC_CATEGORICAL = ["event_qc_status"]
SOURCE_CATEGORICAL = ["event_source"]
PROVENANCE_CATEGORICAL = ["event_source", "event_qc_status", "event_session", "event_hdf5"]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["hdf5"]: row for row in csv.DictReader(stream)}


def read_pair_rows(path: Path) -> list[dict[str, str]]:
    assert_development_csv(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    if any(row.get("partition") not in {"train", "validation"} for row in rows):
        raise ValueError(f"Locked or unsupported partition in {path}")
    for row in rows:
        assert_development_partition(row["partition"])
    return rows


def build_event_metadata(rows: list[dict[str, str]], registry: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    weights: list[float] = []
    for pair_index, pair in enumerate(rows):
        positive_energy = float(pair["positive_energy_kev"])
        negative_energy = float(pair["negative_energy_kev"])
        pair_features = {
            "pair_abs_energy_delta_kev": abs(positive_energy - negative_energy),
            "pair_match_bin_low_kev": float(pair["match_bin_low_kev"]),
        }
        for side in ("positive", "negative"):
            hdf5 = pair[f"{side}_hdf5"]
            domain = registry.get(hdf5, {})
            records.append(
                {
                    "pair_index": pair_index,
                    "event_hdf5": hdf5,
                    "event_source_row": int(pair[f"{side}_row"]),
                    "event_source": pair[f"{side}_source"],
                    "event_qc_status": pair.get(f"{side}_qc_status", "UNKNOWN"),
                    "event_session": domain.get("canonical_session_id", "UNKNOWN"),
                    "pair_positive_source": pair["positive_source"],
                    "pair_negative_source": pair["negative_source"],
                    **pair_features,
                }
            )
            labels.append(int(pair[f"{side}_label"]))
            weights.append(float(pair["source_weight"]))
    return pd.DataFrame.from_records(records), np.asarray(labels, dtype=np.int8), np.asarray(weights, dtype=np.float64)


def attach_store_scalars(data: pd.DataFrame, event_store_dir: Path, partition: str) -> None:
    lookup_path = event_store_dir / f"event_lookup_{partition}.csv"
    store_path = event_store_dir / f"{partition}_events.h5"
    required = set(zip(data["event_hdf5"], data["event_source_row"]))
    lookup: dict[tuple[str, int], int] = {}
    with lookup_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            if key in required:
                lookup[key] = int(row["store_index"])
    missing = required - set(lookup)
    if missing:
        raise KeyError(f"Event store missing {len(missing)} scalar references")
    store_rows = np.asarray(
        [lookup[(hdf5, int(row))] for hdf5, row in zip(data["event_hdf5"], data["event_source_row"])],
        dtype=np.int64,
    )
    order = np.argsort(store_rows)
    values: dict[str, np.ndarray] = {}
    with h5py.File(store_path, "r") as handle:
        for name in (
            "reconstructed_energy_kev",
            "shaped_energy_unit",
            "pulse_extremum_adc",
            "pulse_extremum_index",
            "trigger_time_s",
        ):
            output = np.empty(len(store_rows), dtype=np.float64)
            output[order] = np.asarray(handle[name][store_rows[order]], dtype=np.float64)
            values[name] = output
        noise = np.empty((len(store_rows), 5), dtype=np.float64)
        noise[order] = np.asarray(handle["noise_rms_adc"][store_rows[order], :5], dtype=np.float64)
    data["event_reconstructed_energy_kev"] = values["reconstructed_energy_kev"]
    data["event_shaped_energy_unit"] = values["shaped_energy_unit"]
    data["event_pulse_extremum_adc"] = values["pulse_extremum_adc"]
    data["event_pulse_extremum_index"] = values["pulse_extremum_index"]
    data["event_trigger_time_s"] = values["trigger_time_s"]
    data["event_noise_mean_adc"] = np.mean(noise, axis=1)
    data["event_noise_max_adc"] = np.max(noise, axis=1)
    data["event_noise_std_adc"] = np.std(noise, axis=1)


def metric_summary(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_count": int(labels.size),
        "positive_count": int(np.count_nonzero(labels == 1)),
        "negative_count": int(np.count_nonzero(labels == 0)),
    }
    if np.unique(labels).size < 2:
        result.update({"auroc": None, "weighted_auroc": None, "average_precision": None})
        return result
    result.update(
        {
            "auroc": float(roc_auc_score(labels, scores)),
            "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    )
    return result


def make_numeric_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_categorical_model(
    columns: list[str],
    seed: int,
    include_numeric: bool = True,
) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if include_numeric:
        transformers.append(("numeric", StandardScaler(), NUMERIC_FEATURES))
    transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), columns))
    transformer = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("features", transformer),
            ("classifier", LogisticRegression(max_iter=300, solver="liblinear", random_state=seed)),
        ]
    )


def fit_and_score(
    model_name: str,
    train: pd.DataFrame,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation: pd.DataFrame,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if model_name == "physical_nuisance":
        model: Any = make_numeric_model(seed)
        model.fit(train[NUMERIC_FEATURES], train_labels, sample_weight=train_weights)
        train_scores = model.predict_proba(train[NUMERIC_FEATURES])[:, 1]
        validation_scores = model.predict_proba(validation[NUMERIC_FEATURES])[:, 1]
    elif model_name == "physical_plus_qc":
        model = make_categorical_model(QC_CATEGORICAL, seed)
        model.fit(train, train_labels, classifier__sample_weight=train_weights)
        train_scores = model.predict_proba(train)[:, 1]
        validation_scores = model.predict_proba(validation)[:, 1]
    elif model_name == "source_only":
        model = make_categorical_model(SOURCE_CATEGORICAL, seed, include_numeric=False)
        model.fit(train, train_labels, classifier__sample_weight=train_weights)
        train_scores = model.predict_proba(train)[:, 1]
        validation_scores = model.predict_proba(validation)[:, 1]
    elif model_name == "provenance_upper_bound":
        model = make_categorical_model(PROVENANCE_CATEGORICAL, seed)
        model.fit(train, train_labels, classifier__sample_weight=train_weights)
        train_scores = model.predict_proba(train)[:, 1]
        validation_scores = model.predict_proba(validation)[:, 1]
    else:
        raise ValueError(model_name)
    return {
        "train": metric_summary(train_labels, train_scores, train_weights),
        "validation": metric_summary(validation_labels, validation_scores, validation_weights),
        "validation_scores": validation_scores,
        "model": model,
    }


def univariate_scalar_metrics(
    train: pd.DataFrame,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation: pd.DataFrame,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Report one-dimensional scalar AUROC with orientation fixed by train."""

    result: dict[str, dict[str, Any]] = {}
    for feature in NUMERIC_FEATURES:
        train_values = train[feature].to_numpy(dtype=np.float64)
        validation_values = validation[feature].to_numpy(dtype=np.float64)
        train_finite = np.isfinite(train_values)
        validation_finite = np.isfinite(validation_values)
        if (
            np.count_nonzero(train_finite) < 2
            or np.count_nonzero(validation_finite) < 2
            or np.unique(train_labels[train_finite]).size < 2
            or np.unique(validation_labels[validation_finite]).size < 2
            or np.unique(train_values[train_finite]).size < 2
        ):
            result[feature] = {"status": "degenerate"}
            continue
        raw_train = float(roc_auc_score(train_labels[train_finite], train_values[train_finite], sample_weight=train_weights[train_finite]))
        orientation = 1.0 if raw_train >= 0.5 else -1.0
        raw_validation = float(roc_auc_score(validation_labels[validation_finite], validation_values[validation_finite], sample_weight=validation_weights[validation_finite]))
        oriented_train = raw_train if orientation > 0 else 1.0 - raw_train
        oriented_validation = raw_validation if orientation > 0 else 1.0 - raw_validation
        result[feature] = {
            "status": "valid",
            "train_raw_auroc": raw_train,
            "validation_raw_auroc": raw_validation,
            "orientation_from_train": "increasing" if orientation > 0 else "decreasing",
            "train_oriented_auroc": oriented_train,
            "validation_oriented_auroc": oriented_validation,
        }
    return result


def grouped_validation_metrics(
    validation: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
    group_column: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group in sorted({str(value) for value in validation[group_column]}):
        mask = validation[group_column].to_numpy() == group
        result[group] = metric_summary(labels[mask], scores[mask], weights[mask])
    return result


def grouped_session_loso(
    data: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    groups = data["event_session"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        return {"status": "not_enough_groups", "group_count": int(unique_groups.size), "folds": []}
    splitter = GroupKFold(n_splits=min(5, unique_groups.size))
    folds: list[dict[str, Any]] = []
    X = data
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X, labels, groups), start=1):
        if np.unique(labels[train_index]).size < 2 or np.unique(labels[test_index]).size < 2:
            folds.append(
                {
                    "fold": fold_index,
                    "status": "degenerate_class_coverage",
                    "train_groups": sorted(set(groups[train_index])),
                    "test_groups": sorted(set(groups[test_index])),
                }
            )
            continue
        train_part = X.iloc[train_index]
        test_part = X.iloc[test_index]
        if model_name == "physical_nuisance":
            model: Any = make_numeric_model(seed + fold_index)
            model.fit(train_part[NUMERIC_FEATURES], labels[train_index], sample_weight=weights[train_index])
            scores = model.predict_proba(test_part[NUMERIC_FEATURES])[:, 1]
        else:
            columns = QC_CATEGORICAL if model_name == "physical_plus_qc" else SOURCE_CATEGORICAL
            model = make_categorical_model(
                columns,
                seed + fold_index,
                include_numeric=model_name != "source_only",
            )
            model.fit(train_part, labels[train_index], classifier__sample_weight=weights[train_index])
            scores = model.predict_proba(test_part)[:, 1]
        folds.append(
            {
                "fold": fold_index,
                "status": "valid",
                "train_groups": sorted(set(groups[train_index])),
                "test_groups": sorted(set(groups[test_index])),
                "metrics": metric_summary(labels[test_index], scores, weights[test_index]),
            }
        )
    valid = [fold["metrics"] for fold in folds if fold["status"] == "valid"]
    return {
        "status": "complete" if valid else "no_valid_folds",
        "group_count": int(unique_groups.size),
        "folds": folds,
        "valid_fold_macro_auroc": float(np.mean([item["auroc"] for item in valid])) if valid else None,
    }


def load_dataset(
    label_dir: Path,
    event_store_dir: Path,
    registry: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, dict[str, str]]:
    train_csv = label_dir / "label_pairs_train.csv"
    validation_csv = label_dir / "label_pairs_validation.csv"
    train_rows = read_pair_rows(train_csv)
    validation_rows = read_pair_rows(validation_csv)
    train, train_labels, train_weights = build_event_metadata(train_rows, registry)
    validation, validation_labels, validation_weights = build_event_metadata(validation_rows, registry)
    attach_store_scalars(train, event_store_dir, "train")
    attach_store_scalars(validation, event_store_dir, "validation")
    return train, train_labels, train_weights, validation, validation_labels, validation_weights, {
        "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
        "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
        "train_csv_sha256": sha256_file(train_csv),
        "validation_csv_sha256": sha256_file(validation_csv),
    }


def audit_dataset(
    name: str,
    label_dir: Path,
    event_store_dir: Path,
    registry: dict[str, dict[str, str]],
    seed: int,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train, train_labels, train_weights, validation, validation_labels, validation_weights, provenance = load_dataset(
        label_dir, event_store_dir, registry
    )
    result: dict[str, Any] = {
        "provenance": provenance,
        "train_event_count": int(train_labels.size),
        "univariate_scalar_metrics": univariate_scalar_metrics(
            train,
            train_labels,
            train_weights,
            validation,
            validation_labels,
            validation_weights,
        ),
        "validation_event_count": int(validation_labels.size),
        "negative_sources": sorted({str(value) for value in validation["pair_negative_source"]}),
        "models": {},
    }
    for feature, metrics in result["univariate_scalar_metrics"].items():
        if metrics.get("status") == "valid":
            summary_rows.append(
                {
                    "dataset": name,
                    "model": "univariate_scalar",
                    "group_type": "numeric_feature",
                    "group": feature,
                    "event_count": int(validation_labels.size),
                    "positive_count": int(np.count_nonzero(validation_labels == 1)),
                    "negative_count": int(np.count_nonzero(validation_labels == 0)),
                    "auroc": metrics["validation_oriented_auroc"],
                    "weighted_auroc": None,
                    "average_precision": None,
                }
            )
    for model_name in (
        "physical_nuisance",
        "physical_plus_qc",
        "source_only",
        "provenance_upper_bound",
    ):
        print(f"Nuisance model {model_name} on {name} ...", flush=True)
        fitted = fit_and_score(
            model_name,
            train,
            train_labels,
            train_weights,
            validation,
            validation_labels,
            validation_weights,
            seed,
        )
        validation_scores = fitted.pop("validation_scores")
        fitted.pop("model")
        fitted["by_negative_source"] = grouped_validation_metrics(
            validation,
            validation_labels,
            validation_weights,
            validation_scores,
            "pair_negative_source",
        )
        fitted["by_positive_source"] = grouped_validation_metrics(
            validation,
            validation_labels,
            validation_weights,
            validation_scores,
            "pair_positive_source",
        )
        fitted["session_grouped_loso"] = grouped_session_loso(
            pd.concat([train, validation], ignore_index=True),
            np.concatenate([train_labels, validation_labels]),
            np.concatenate([train_weights, validation_weights]),
            model_name,
            seed,
        ) if model_name != "provenance_upper_bound" else {"status": "not_run_provenance_upper_bound"}
        result["models"][model_name] = fitted
        overall = fitted["validation"]
        summary_rows.append(
            {
                "dataset": name,
                "model": model_name,
                "group_type": "overall",
                "group": "all",
                **overall,
            }
        )
        for group_type, groups in (
            ("negative_source", fitted["by_negative_source"]),
            ("positive_source", fitted["by_positive_source"]),
        ):
            for group, metrics in groups.items():
                summary_rows.append(
                    {
                        "dataset": name,
                        "model": model_name,
                        "group_type": group_type,
                        "group": group,
                        **metrics,
                    }
                )
    return result


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "model", "group_type", "group", "event_count",
        "positive_count", "negative_count", "auroc", "weighted_auroc", "average_precision",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-labels-dir", type=Path, required=True)
    parser.add_argument("--primary-event-store-dir", type=Path, required=True)
    parser.add_argument("--source-ablation-root", type=Path, required=True)
    parser.add_argument("--source-ablation-event-store-dir", type=Path, required=True)
    parser.add_argument("--domain-registry", type=Path, default=PROJECT_ROOT / "outputs/protocol/domain_registry.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.domain_registry.resolve()
    registry = load_registry(registry_path)
    datasets = [
        (
            "primary",
            args.primary_labels_dir.resolve(),
            args.primary_event_store_dir.resolve(),
        )
    ]
    for source in ("ba133", "na22", "cs137"):
        datasets.append(
            (
                f"source_ablation_{source}",
                args.source_ablation_root.resolve() / f"{source}_positive",
                args.source_ablation_event_store_dir.resolve(),
            )
        )
    summary_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "EXPLORATORY_SCALAR_WAVEFORM_ACQUISITION_SHORTCUT_AUDIT",
        "created_utc": utc_now(),
        "test_partition_used": False,
        "locked_test_event_rows_read": False,
        "th232_read": False,
        "eu152_read": False,
        "feature_boundary": {
            "deployable_waveform_features_used": False,
            "scalar_waveform_acquisition_features": NUMERIC_FEATURES,
            "qc_features": QC_CATEGORICAL,
            "source_features": SOURCE_CATEGORICAL,
            "provenance_upper_bound_features": PROVENANCE_CATEGORICAL,
        },
        "session_claim_boundary": "Grouped folds are diagnostics on train+validation; the frozen file split is not session-held-out.",
        "datasets": {},
    }
    for name, label_dir, event_store_dir in datasets:
        result["datasets"][name] = audit_dataset(
            name,
            label_dir,
            event_store_dir,
            registry,
            args.seed,
            summary_rows,
        )
    write_json(output_dir / "nuisance_shortcut_audit.json", result)
    write_summary_csv(output_dir / "nuisance_shortcut_summary.csv", summary_rows)
    print(json.dumps({
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "datasets": list(result["datasets"]),
        "models": ["physical_nuisance", "physical_plus_qc", "source_only", "provenance_upper_bound"],
        "test_partition_used": False,
        "locked_test_event_rows_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
