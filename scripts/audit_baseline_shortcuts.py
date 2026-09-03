#!/usr/bin/env python3
"""Audit source, QC, session, and nuisance dependence of frozen baselines.

The audit scores only train/validation manifests. It never opens the locked-test
CSV, test HDF5 rows, Th-232, or Eu-152. The primary validation set is used for
per-peak/QC/session diagnostics. Rebuilt source-ablation validation sets are used
for negative-source transfer diagnostics. These are exploratory audits, not
session-held-out validation, because the current immutable file split has
sessions spanning partitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_multiscale_cnn as residual  # noqa: E402
import train_o2_late_fusion as o2  # noqa: E402
from src.data_access_guards import assert_development_csv, assert_development_partition  # noqa: E402
from src.ba133_cnn import (
    CompactWaveformCNN,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)


MODEL_NAMES = ("o2_late_fusion", "registered_residual_cnn", "compact_cnn")


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


def read_domain_registry(path: Path) -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            registry[row["hdf5"]] = row
    return registry


def read_event_metadata(csv_path: Path) -> dict[str, np.ndarray]:
    """Expand pair rows in the same positive/negative order used by all trainers."""

    assert_development_csv(csv_path)
    values: dict[str, list[Any]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for pair_index, row in enumerate(csv.DictReader(stream)):
            for side in ("positive", "negative"):
                values["hdf5"].append(row[f"{side}_hdf5"])
                values["source_row"].append(int(row[f"{side}_row"]))
                values["label"].append(int(row[f"{side}_label"]))
                values["weight"].append(float(row["source_weight"]))
                values["peak_id"].append(row["peak_id"])
                values["event_source"].append(row[f"{side}_source"])
                values["event_qc_status"].append(row.get(f"{side}_qc_status", "UNKNOWN"))
                values["pair_positive_source"].append(row["positive_source"])
                values["pair_negative_source"].append(row["negative_source"])
                values["pair_index"].append(pair_index)
    if not values["label"] or set(values["label"]) != {0, 1}:
        raise ValueError(f"Expected both labels in {csv_path}")
    return {
        "hdf5": np.asarray(values["hdf5"], dtype="U256"),
        "source_row": np.asarray(values["source_row"], dtype=np.int64),
        "label": np.asarray(values["label"], dtype=np.int8),
        "weight": np.asarray(values["weight"], dtype=np.float64),
        "peak_id": np.asarray(values["peak_id"], dtype="U64"),
        "event_source": np.asarray(values["event_source"], dtype="U32"),
        "event_qc_status": np.asarray(values["event_qc_status"], dtype="U16"),
        "pair_positive_source": np.asarray(values["pair_positive_source"], dtype="U32"),
        "pair_negative_source": np.asarray(values["pair_negative_source"], dtype="U32"),
        "pair_index": np.asarray(values["pair_index"], dtype=np.int64),
    }


def attach_domains(
    metadata: dict[str, np.ndarray],
    registry: dict[str, dict[str, str]],
) -> None:
    session = []
    mapping_status = []
    for hdf5 in metadata["hdf5"].tolist():
        row = registry.get(str(hdf5), {})
        session.append(row.get("canonical_session_id", "UNKNOWN"))
        mapping_status.append(row.get("session_mapping_status", "UNKNOWN"))
    metadata["event_session"] = np.asarray(session, dtype="U256")
    metadata["session_mapping_status"] = np.asarray(mapping_status, dtype="U64")


def load_event_noise(
    metadata: dict[str, np.ndarray],
    event_store_dir: Path,
) -> np.ndarray:
    """Load five-section noise RMS from the versioned event store."""

    partition = "validation"
    assert_development_partition(partition)
    lookup_path = event_store_dir / "event_lookup_validation.csv"
    store_path = event_store_dir / "validation_events.h5"
    required = {(str(hdf5), int(row)) for hdf5, row in zip(metadata["hdf5"], metadata["source_row"])}
    lookup: dict[tuple[str, int], int] = {}
    with lookup_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            if key in required:
                lookup[key] = int(row["store_index"])
    missing = required - set(lookup)
    if missing:
        raise KeyError(f"Event store missing {len(missing)} noise references")
    store_rows = np.asarray(
        [lookup[(str(hdf5), int(row))] for hdf5, row in zip(metadata["hdf5"], metadata["source_row"])],
        dtype=np.int64,
    )
    order = np.argsort(store_rows)
    noise = np.empty((len(store_rows), 5), dtype=np.float32)
    with h5py.File(store_path, "r") as handle:
        noise[order] = handle["noise_rms_adc"][store_rows[order], :5]
    return np.mean(noise, axis=1, dtype=np.float64)


def model_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = np.ones(labels.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y = labels[selected]
    p = scores[selected]
    w = weights[selected]
    result: dict[str, Any] = {
        "event_count": int(y.size),
        "positive_count": int(np.count_nonzero(y == 1)),
        "negative_count": int(np.count_nonzero(y == 0)),
        "score_mean": float(np.mean(p)) if p.size else None,
        "score_std": float(np.std(p)) if p.size else None,
    }
    if np.unique(y).size < 2:
        result.update({"auroc": None, "weighted_auroc": None, "average_precision": None})
        return result
    result.update(
        {
            "auroc": float(roc_auc_score(y, p)),
            "weighted_auroc": float(roc_auc_score(y, p, sample_weight=w)),
            "average_precision": float(average_precision_score(y, p)),
        }
    )
    return result


def grouped_metrics(
    metadata: dict[str, np.ndarray],
    scores: np.ndarray,
    key: str,
) -> dict[str, dict[str, Any]]:
    labels = metadata["label"]
    weights = metadata["weight"]
    groups: dict[str, dict[str, Any]] = {}
    for value in sorted({str(item) for item in metadata[key]}):
        groups[value] = model_metrics(labels, scores, weights, metadata[key] == value)
    return groups


def label_qc_metrics(metadata: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for status in sorted({str(item) for item in metadata["event_qc_status"]}):
        for label in (0, 1):
            mask = (metadata["event_qc_status"] == status) & (metadata["label"] == label)
            values = scores[mask]
            result[f"{status}|label_{label}"] = {
                "event_count": int(values.size),
                "score_mean": float(np.mean(values)) if values.size else None,
                "score_std": float(np.std(values)) if values.size else None,
            }
    return result


def session_score_summary(metadata: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for session in sorted({str(item) for item in metadata["event_session"]}):
        for label in (0, 1):
            mask = (metadata["event_session"] == session) & (metadata["label"] == label)
            values = scores[mask]
            if not values.size:
                continue
            result[f"{session}|label_{label}"] = {
                "event_count": int(values.size),
                "score_mean": float(np.mean(values)),
                "score_std": float(np.std(values)),
                "event_sources": dict(
                    Counter(str(value) for value in metadata["event_source"][mask])
                ),
                "qc_statuses": dict(
                    Counter(str(value) for value in metadata["event_qc_status"][mask])
                ),
            }
    return result


def nuisance_correlation(metadata: dict[str, np.ndarray], scores: np.ndarray, noise: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    finite = np.isfinite(noise) & np.isfinite(scores)
    if np.count_nonzero(finite) >= 3 and np.unique(noise[finite]).size > 1:
        correlation = spearmanr(noise[finite], scores[finite])
        result["all_events"] = {
            "count": int(np.count_nonzero(finite)),
            "spearman_r": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        }
    else:
        result["all_events"] = {"count": int(np.count_nonzero(finite)), "spearman_r": None, "p_value": None}
    for label in (0, 1):
        mask = finite & (metadata["label"] == label)
        if np.count_nonzero(mask) >= 3 and np.unique(noise[mask]).size > 1:
            correlation = spearmanr(noise[mask], scores[mask])
            result[f"label_{label}"] = {
                "count": int(np.count_nonzero(mask)),
                "spearman_r": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }
    return result


def batched_scores(model: torch.nn.Module, inputs: tuple[np.ndarray, ...], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs[0]), batch_size):
            stop = min(start + batch_size, len(inputs[0]))
            tensors = [torch.from_numpy(values[start:stop]).to(device) for values in inputs]
            logits = model(*tensors)
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def load_o2_scores(csv_path: Path, event_store_dir: Path, checkpoint_path: Path, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    data = o2.build_partition_features(csv_path, event_store_dir=event_store_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    statistics = checkpoint["feature_statistics"]
    charge = ((data.charge - statistics["charge_mean"]) / statistics["charge_std"]).astype(np.float32)
    current = ((data.current - statistics["current_mean"]) / statistics["current_std"]).astype(np.float32)
    model = o2.O2LateFusion()
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scores = batched_scores(model, (charge, current), device, batch_size)
    return scores, {"parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def load_registered_scores(csv_path: Path, event_store_dir: Path, checkpoint_path: Path, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    data = residual.build_partition_features(csv_path, None, event_store_dir=event_store_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    statistics = checkpoint["feature_statistics"]
    current = np.gradient(data.charge, 4.0, axis=1).astype(np.float32)
    values = np.stack((data.charge, current), axis=1)
    means = np.asarray(statistics["channel_mean"], dtype=np.float32)
    standard_deviations = np.asarray(statistics["channel_standard_deviation"], dtype=np.float32)
    values = ((values - means[None, :, None]) / standard_deviations[None, :, None]).astype(np.float32)
    model = residual.MultiscaleResidualCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scores = batched_scores(model, (values,), device, batch_size)
    return scores, {"parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def load_compact_scores(csv_path: Path, event_store_dir: Path, checkpoint_path: Path, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    raw = load_raw_partition(csv_path, event_store_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, checkpoint["channel_statistics"])
    model = CompactWaveformCNN(config.channel_count, width=int(checkpoint["model_width"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scores = batched_scores(model, (values,), device, batch_size)
    return scores, {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "representation_config": config.as_dict(),
        "representation_qc": representation_qc,
    }


def score_model(
    model_name: str,
    csv_path: Path,
    event_store_dir: Path,
    checkpoint_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "o2_late_fusion":
        return load_o2_scores(csv_path, event_store_dir, checkpoint_path, batch_size)
    if model_name == "registered_residual_cnn":
        return load_registered_scores(csv_path, event_store_dir, checkpoint_path, batch_size)
    if model_name == "compact_cnn":
        return load_compact_scores(csv_path, event_store_dir, checkpoint_path, batch_size)
    raise ValueError(model_name)


def audit_one_dataset(
    dataset_name: str,
    csv_path: Path,
    event_store_dir: Path,
    registry: dict[str, dict[str, str]],
    checkpoints: dict[str, Path],
    batch_size: int,
    output_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = read_event_metadata(csv_path)
    attach_domains(metadata, registry)
    noise = load_event_noise(metadata, event_store_dir)
    dataset_result: dict[str, Any] = {
        "csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
        "csv_sha256": sha256_file(csv_path),
        "pair_count": int(np.count_nonzero(metadata["label"] == 1)),
        "event_count": int(metadata["label"].size),
        "negative_sources": sorted({str(value) for value in metadata["pair_negative_source"]}),
        "models": {},
    }
    for model_name, checkpoint_path in checkpoints.items():
        print(f"Scoring {model_name} on {dataset_name} ...", flush=True)
        scores, model_info = score_model(
            model_name,
            csv_path,
            event_store_dir,
            checkpoint_path,
            batch_size,
        )
        if scores.size != metadata["label"].size:
            raise ValueError(f"Score/metadata length mismatch for {dataset_name}/{model_name}")
        overall = model_metrics(metadata["label"], scores, metadata["weight"])
        result = {
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parameter_count": model_info.pop("parameter_count"),
            **model_info,
            "overall": overall,
            "by_peak": grouped_metrics(metadata, scores, "peak_id"),
            "by_pair_positive_source": grouped_metrics(metadata, scores, "pair_positive_source"),
            "by_pair_negative_source": grouped_metrics(metadata, scores, "pair_negative_source"),
            "by_event_source": grouped_metrics(metadata, scores, "event_source"),
            "by_event_qc_status_and_label": label_qc_metrics(metadata, scores),
            "session_score_summary": session_score_summary(metadata, scores),
            "noise_rms_spearman": nuisance_correlation(metadata, scores, noise),
        }
        dataset_result["models"][model_name] = result
        for group_type, groups in (
            ("overall", {"all": overall}),
            ("peak", result["by_peak"]),
            ("pair_positive_source", result["by_pair_positive_source"]),
            ("pair_negative_source", result["by_pair_negative_source"]),
        ):
            for group, metrics in groups.items():
                output_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "group_type": group_type,
                        "group": group,
                        "event_count": metrics.get("event_count"),
                        "positive_count": metrics.get("positive_count"),
                        "negative_count": metrics.get("negative_count"),
                        "auroc": metrics.get("auroc"),
                        "weighted_auroc": metrics.get("weighted_auroc"),
                        "average_precision": metrics.get("average_precision"),
                        "score_mean": metrics.get("score_mean"),
                        "score_std": metrics.get("score_std"),
                    }
                )
    return dataset_result


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "model", "group_type", "group", "event_count",
        "positive_count", "negative_count", "auroc", "weighted_auroc",
        "average_precision", "score_mean", "score_std",
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
    parser.add_argument("--o2-checkpoint", type=Path, required=True)
    parser.add_argument("--registered-checkpoint", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = read_domain_registry(args.domain_registry.resolve())
    checkpoints = {
        "o2_late_fusion": args.o2_checkpoint.resolve(),
        "registered_residual_cnn": args.registered_checkpoint.resolve(),
        "compact_cnn": args.compact_checkpoint.resolve(),
    }
    for path in [*checkpoints.values(), args.domain_registry.resolve()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    primary_csv = args.primary_labels_dir.resolve() / "label_pairs_validation.csv"
    if not primary_csv.is_file():
        raise FileNotFoundError(primary_csv)
    dataset_paths: list[tuple[str, Path, Path]] = [("primary_validation", primary_csv, args.primary_event_store_dir.resolve())]
    source_root = args.source_ablation_root.resolve()
    for source in ("ba133", "na22", "cs137"):
        csv_path = source_root / f"{source}_positive" / "label_pairs_validation.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        dataset_paths.append((f"source_ablation_{source}", csv_path, args.source_ablation_event_store_dir.resolve()))

    results: dict[str, Any] = {
        "schema_version": 1,
        "status": "EXPLORATORY_SHORTCUT_AUDIT",
        "created_utc": utc_now(),
        "test_partition_used": False,
        "locked_test_event_rows_read": False,
        "th232_read": False,
        "eu152_read": False,
        "session_claim_boundary": "File-disjoint within-session interpolation only; no session-held-out claim.",
        "domain_registry": args.domain_registry.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "models": {
            name: {
                "checkpoint": path.relative_to(PROJECT_ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(path),
            }
            for name, path in checkpoints.items()
        },
        "datasets": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for dataset_name, csv_path, event_store_dir in dataset_paths:
        results["datasets"][dataset_name] = audit_one_dataset(
            dataset_name,
            csv_path,
            event_store_dir,
            registry,
            checkpoints,
            args.batch_size,
            summary_rows,
        )
    write_json(output_dir / "shortcut_audit.json", results)
    write_summary_csv(output_dir / "shortcut_audit_summary.csv", summary_rows)
    print(json.dumps({
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "datasets": list(results["datasets"]),
        "models": list(checkpoints),
        "test_partition_used": False,
        "locked_test_event_rows_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
