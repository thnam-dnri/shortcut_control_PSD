#!/usr/bin/env python3
"""Train three-seed M0/M3 engineering screens using a frozen morphology catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.ba133_cnn import (
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
    set_seed,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path
from src.waveform_morphology import FiLMDSCNN, build_unconditioned_ds_cnn
from train_compact_ds_cnn_performance import (
    EXPECTED_REPRESENTATION_NAME,
    PEAK_WEIGHT_KEYS,
    SELECTED_PEAK_WEIGHTS,
    load_reference_contract,
)

SEEDS = (20260821, 20260822, 20260823)
REFIT_SEEDS = {"m0": 20261821, "m3": 20261822}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pairs, 2 * pairs + 1)).reshape(-1)


def validate_split(fit: np.ndarray, internal: np.ndarray, total_pairs: int) -> None:
    if np.intersect1d(fit, internal).size:
        raise ValueError("Fit/internal pair overlap")
    if not np.array_equal(
        np.sort(np.concatenate((fit, internal))), np.arange(total_pairs)
    ):
        raise ValueError("Fit/internal split does not cover the train manifest")


def event_weights(peak_ids: np.ndarray) -> np.ndarray:
    pair_peaks = peak_ids[::2]
    if not np.array_equal(pair_peaks, peak_ids[1::2]):
        raise ValueError("Pair members have different peak IDs")
    counts = Counter(pair_peaks.tolist())
    pair_weights = np.asarray(
        [
            SELECTED_PEAK_WEIGHTS[PEAK_WEIGHT_KEYS[peak]] / counts[peak]
            for peak in pair_peaks
        ],
        dtype=np.float32,
    )
    return np.repeat(pair_weights, 2)


def make_loader(
    values: np.ndarray,
    posterior: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, ...]]:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(values),
            torch.from_numpy(posterior),
            torch.from_numpy(labels.astype(np.float32, copy=False)),
            torch.from_numpy(weights.astype(np.float32, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def forward(model: nn.Module, kind: str, values: Tensor, posterior: Tensor) -> Tensor:
    if kind == "m3":
        return model(values, posterior)
    return model(values)


def train_epoch(
    model: nn.Module,
    kind: str,
    loader: DataLoader[tuple[Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    weight_sum = 0.0
    for values, posterior, labels, weights in loader:
        values = values.to(device, non_blocking=True)
        posterior = posterior.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, kind, values, posterior)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += float((losses * weights).sum().item())
        weight_sum += float(weights.sum().item())
    return loss_sum / weight_sum


def predict(
    model: nn.Module,
    kind: str,
    loader: DataLoader[tuple[Tensor, ...]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, posterior, _labels, _weights in loader:
            logits = forward(
                model,
                kind,
                values.to(device, non_blocking=True),
                posterior.to(device, non_blocking=True),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak in sorted(set(peak_ids.tolist())):
        mask = peak_ids == peak
        per_peak[peak] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(
                average_precision_score(labels[mask], scores[mask])
            ),
            "event_count": int(np.count_nonzero(mask)),
        }
    aurocs = [float(value["auroc"]) for value in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "worst_peak_auroc": float(np.min(aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(
            roc_auc_score(labels, scores, sample_weight=weights)
        ),
        "score_standard_deviation": float(np.std(scores)),
        "per_peak": per_peak,
    }


def build_model(kind: str, components: int) -> nn.Module:
    if kind == "m0":
        return build_unconditioned_ds_cnn(width=24)
    if kind == "m3":
        return FiLMDSCNN(posterior_dimensions=components, width=24)
    raise ValueError(kind)


def scan_model(
    kind: str,
    seed: int,
    values: np.ndarray,
    posterior: np.ndarray,
    labels: np.ndarray,
    peak_ids: np.ndarray,
    weights: np.ndarray,
    fit_events: np.ndarray,
    internal_events: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    components: int,
) -> dict[str, Any]:
    set_seed(seed)
    fit_loader = make_loader(
        values[fit_events],
        posterior[fit_events],
        labels[fit_events],
        weights[fit_events],
        args.batch_size,
        True,
        seed,
    )
    internal_loader = make_loader(
        values[internal_events],
        posterior[internal_events],
        labels[internal_events],
        weights[internal_events],
        args.batch_size,
        False,
        seed,
    )
    model = build_model(kind, components).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_epoch = -1
    best_macro = -np.inf
    best_state: dict[str, Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, kind, fit_loader, optimizer, device)
        scores = predict(model, kind, internal_loader, device)
        result = metrics(
            labels[internal_events],
            scores,
            weights[internal_events],
            peak_ids[internal_events],
        )
        history.append({"epoch": epoch, "fit_loss": loss, "internal": result})
        print(
            f"{kind} seed={seed} epoch={epoch} loss={loss:.6f} "
            f"macro={result['macro_auroc']:.6f} "
            f"worst={result['worst_peak_auroc']:.6f}",
            flush=True,
        )
        if result["macro_auroc"] > best_macro + 1.0e-4:
            best_epoch = epoch
            best_macro = float(result["macro_auroc"])
            best_metrics = result
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"No checkpoint selected for {kind}, seed {seed}")
    checkpoint_path = output_dir / "scan" / kind / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_kind": kind,
            "model_state_dict": best_state,
            "posterior_dimensions": components,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_metrics": best_metrics,
            "parameter_count": parameter_count,
            "held_out_partition_loaded": False,
            "test_partition_used": False,
            "th232_used": False,
            "eu152_used": False,
        },
        checkpoint_path,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_internal_metrics": best_metrics,
        "parameter_count": parameter_count,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
    }


def refit_model(
    kind: str,
    epochs: int,
    seed: int,
    values: np.ndarray,
    posterior: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    components: int,
) -> dict[str, Any]:
    set_seed(seed)
    loader = make_loader(
        values, posterior, labels, weights, args.batch_size, True, seed
    )
    model = build_model(kind, components).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = [
        train_epoch(model, kind, loader, optimizer, device)
        for _epoch in range(epochs)
    ]
    path = output_dir / "refit" / f"{kind}_refit.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_kind": kind,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "posterior_dimensions": components,
            "refit_seed": seed,
            "refit_epochs": epochs,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "representation_config": args.representation_config,
            "feature_statistics": args.feature_statistics,
            "catalogue_report_sha256": args.catalogue_report_sha256,
            "test_partition_used": False,
            "held_out_partition_loaded": False,
            "th232_used": False,
            "eu152_used": False,
        },
        path,
    )
    return {
        "checkpoint": path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(path),
        "refit_seed": seed,
        "refit_epochs": epochs,
        "training_loss_history": history,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "cuda"
        or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/morphology_catalogue_20260821",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/morphology_catalogue_20260821",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/three_peak_positive_polarity_20260820/compact_cnn_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    feature_dir = args.feature_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = experiment_dir / "models"
    train_csv = labels_dir / "label_pairs_train.csv"
    split_path = labels_dir / "train_internal_split_indices.npz"
    audit_path = experiment_dir / "audit/audit_report.json"
    catalogue_report_path = experiment_dir / "catalogue/catalogue_report.json"
    for path in (
        labels_dir,
        event_store_dir,
        feature_dir,
        experiment_dir,
        args.reference_checkpoint,
        train_csv,
        split_path,
        audit_path,
        catalogue_report_path,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with audit_path.open() as stream:
        audit = json.load(stream)
    with catalogue_report_path.open() as stream:
        catalogue = json.load(stream)
    if audit["decision"] != "CATALOGUE_TECHNICAL_PASS":
        raise RuntimeError(f"M3 blocked by {audit['decision']}")
    components = int(catalogue["selected_components"])
    args.catalogue_report_sha256 = sha256_file(catalogue_report_path)

    config, feature_statistics, _reference = load_reference_contract(
        args.reference_checkpoint.resolve()
    )
    if config.name != EXPECTED_REPRESENTATION_NAME:
        raise ValueError(config.name)
    args.representation_config = config.as_dict()
    args.feature_statistics = feature_statistics
    device = resolve_device(args.device)
    print(f"device={device} components={components}", flush=True)
    raw = load_raw_partition(train_csv, event_store_dir)
    labels = raw.labels.copy()
    peak_ids = raw.peak_ids.copy()
    values, representation_qc = build_representation(raw, config)
    recomputed = fit_channel_statistics(values)
    for field in ("means", "standard_deviations"):
        if not np.allclose(
            recomputed[field], feature_statistics[field], rtol=1.0e-5, atol=1.0e-7
        ):
            raise ValueError(f"Frozen representation {field} mismatch")
    apply_channel_statistics(values, feature_statistics)
    del raw

    split = np.load(split_path)
    fit_pairs = np.sort(split["fit_pair_indices"].astype(np.int64))
    internal_pairs = np.sort(split["internal_pair_indices"].astype(np.int64))
    validate_split(fit_pairs, internal_pairs, labels.size // 2)
    fit_events = event_indices(fit_pairs)
    internal_events = event_indices(internal_pairs)
    weights = event_weights(peak_ids)
    fit_assignments = np.load(
        experiment_dir / "catalogue/fit_assignments.npz"
    )
    internal_assignments = np.load(
        experiment_dir / "catalogue/internal_assignments.npz"
    )
    posterior = np.empty((labels.size, components), dtype=np.float32)
    posterior[fit_events] = fit_assignments["probability"]
    posterior[internal_events] = internal_assignments["probability"]
    invalid = ~np.all(np.isfinite(posterior), axis=1)
    posterior[invalid] = 1.0 / components

    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL_SHORTCUT_SENSITIVE_ENGINEERING_SCREEN",
        "catalogue_components": components,
        "catalogue_interpretation_status": audit["interpretation_status"],
        "invalid_posterior_rows_uniform_imputed": int(np.count_nonzero(invalid)),
        "representation_qc": representation_qc,
        "training": {
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "input_hashes": {
            "train_csv": sha256_file(train_csv),
            "split": sha256_file(split_path),
            "catalogue_report": args.catalogue_report_sha256,
            "audit_report": sha256_file(audit_path),
            "reference_checkpoint": sha256_file(args.reference_checkpoint.resolve()),
        },
        "models": {"m0": [], "m3": []},
        "held_out_partition_loaded": False,
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    for kind in ("m0", "m3"):
        for seed in SEEDS:
            summary["models"][kind].append(
                scan_model(
                    kind,
                    seed,
                    values,
                    posterior,
                    labels,
                    peak_ids,
                    weights,
                    fit_events,
                    internal_events,
                    args,
                    device,
                    output_dir,
                    components,
                )
            )
    mean_metrics: dict[str, dict[str, float]] = {}
    for kind in ("m0", "m3"):
        mean_metrics[kind] = {
            metric: float(
                np.mean(
                    [
                        run["best_internal_metrics"][metric]
                        for run in summary["models"][kind]
                    ]
                )
            )
            for metric in ("macro_auroc", "worst_peak_auroc")
        }
    delta_macro = (
        mean_metrics["m3"]["macro_auroc"] - mean_metrics["m0"]["macro_auroc"]
    )
    delta_worst = (
        mean_metrics["m3"]["worst_peak_auroc"]
        - mean_metrics["m0"]["worst_peak_auroc"]
    )
    noncollapsed = all(
        run["best_internal_metrics"]["score_standard_deviation"] > 1.0e-6
        for run in summary["models"]["m3"]
    )
    gate = delta_macro >= 0.004 and delta_worst >= -0.002 and noncollapsed
    summary["internal_gate"] = {
        "mean_metrics": mean_metrics,
        "m3_minus_m0_macro_auroc": delta_macro,
        "m3_minus_m0_worst_peak_auroc": delta_worst,
        "all_m3_scores_noncollapsed": noncollapsed,
        "pass": gate,
        "decision": (
            "MORPHOLOGY_CONDITIONING_INTERNAL_GATE_PASS"
            if gate
            else "MORPHOLOGY_CONDITIONING_NOT_SUPPORTED"
        ),
    }
    if gate:
        summary["refit"] = {}
        for kind in ("m0", "m3"):
            epochs = int(
                np.rint(
                    np.median(
                        [run["best_epoch"] for run in summary["models"][kind]]
                    )
                )
            )
            summary["refit"][kind] = refit_model(
                kind,
                epochs,
                REFIT_SEEDS[kind],
                values,
                posterior,
                labels,
                weights,
                args,
                device,
                output_dir,
                components,
            )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["internal_gate"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
