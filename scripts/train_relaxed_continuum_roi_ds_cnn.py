#!/usr/bin/env python3
"""Train a joint DS-CNN on a relaxed-continuum waveform cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN
from src.ba133_cnn import set_seed
from src.data_access_guards import assert_no_forbidden_path

SEEDS = (20260822, 20260823, 20260824)
THREE_PEAK_WEIGHTS = {
    "ba133_356kev": 0.4,
    "na22_511kev": 0.4,
    "cs137_662kev": 0.2,
}
ALL_BA_PEAK_WEIGHTS = {
    "ba133_276kev": 0.1,
    "ba133_303kev": 0.1,
    "ba133_356kev": 0.1,
    "ba133_384kev": 0.1,
    "na22_511kev": 0.4,
    "cs137_662kev": 0.2,
}
PEAK_CENTERS_FWHM = {
    "ba133_276kev": (276.146, 3.986),
    "ba133_303kev": (303.139, 3.971),
    "ba133_356kev": (355.709, 3.941),
    "ba133_384kev": (383.978, 4.120),
    "na22_511kev": (510.926, 4.447),
    "cs137_662kev": (661.668, 3.749),
}
COMMON_THREE_PEAKS = tuple(THREE_PEAK_WEIGHTS)


def resolve_peak_weights(peak_ids: set[str]) -> dict[str, float]:
    if peak_ids == set(THREE_PEAK_WEIGHTS):
        return THREE_PEAK_WEIGHTS
    if peak_ids == set(ALL_BA_PEAK_WEIGHTS):
        return ALL_BA_PEAK_WEIGHTS
    raise ValueError(f"Unsupported cache peak set: {sorted(peak_ids)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CachedWaveforms(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        values_path: Path,
        metadata_path: Path,
        peak_weights: dict[str, float],
    ) -> None:
        self.values = np.load(values_path, mmap_mode="r")
        metadata = np.load(metadata_path)
        self.labels = metadata["label"].astype(np.float32)
        self.peak_ids = metadata["peak_id"].astype(str)
        unknown = set(self.peak_ids.tolist()) - set(peak_weights)
        if unknown:
            raise ValueError(f"Metadata contains peaks without weights: {sorted(unknown)}")
        counts = {
            peak: int(np.count_nonzero(self.peak_ids == peak))
            for peak in set(self.peak_ids.tolist())
        }
        self.weights = np.asarray(
            [peak_weights[peak] / counts[peak] for peak in self.peak_ids],
            dtype=np.float32,
        )
        self.energies = metadata["energy_kev"].astype(np.float32)

    def __len__(self) -> int:
        return self.labels.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = torch.from_numpy(np.asarray(self.values[index], dtype=np.float32))
        return (
            values,
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )


def make_loader(
    dataset: CachedWaveforms,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    weight_total = 0.0
    for values, labels, weights in loader:
        values = values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float((losses * weights).sum().item())
        weight_total += float(weights.sum().item())
    return total / weight_total


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            logits = model(values.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metrics(dataset: CachedWaveforms, scores: np.ndarray) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak in sorted(set(dataset.peak_ids.tolist())):
        mask = dataset.peak_ids == peak
        per_peak[peak] = {
            "auroc": float(roc_auc_score(dataset.labels[mask], scores[mask])),
            "average_precision": float(
                average_precision_score(dataset.labels[mask], scores[mask])
            ),
            "event_count": int(np.count_nonzero(mask)),
        }
    result = {
        "macro_auroc": float(np.mean([row["auroc"] for row in per_peak.values()])),
        "worst_peak_auroc": float(np.min([row["auroc"] for row in per_peak.values()])),
        "pooled_auroc": float(roc_auc_score(dataset.labels, scores)),
        "weighted_auroc": float(
            roc_auc_score(dataset.labels, scores, sample_weight=dataset.weights)
        ),
        "pooled_average_precision": float(
            average_precision_score(dataset.labels, scores)
        ),
        "score_standard_deviation": float(np.std(scores)),
        "per_peak": per_peak,
    }
    common = [per_peak[peak]["auroc"] for peak in COMMON_THREE_PEAKS if peak in per_peak]
    result["common_three_macro_auroc"] = (
        float(np.mean(common)) if len(common) == len(COMMON_THREE_PEAKS) else None
    )
    return result


def energy_only_metrics(dataset: CachedWaveforms) -> dict[str, Any]:
    scores = np.empty(len(dataset), dtype=np.float32)
    for peak in set(dataset.peak_ids.tolist()):
        center, fwhm = PEAK_CENTERS_FWHM[peak]
        mask = dataset.peak_ids == peak
        scores[mask] = -np.abs(dataset.energies[mask] - center) / fwhm
    return metrics(dataset, scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--strict-baseline-summary",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/morphology_catalogue_20260821/models/training_summary.json",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822"
        / "experiment_report.json",
        help="Existing t10 experiment used for like-for-like comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    baseline_path = args.strict_baseline_summary.resolve()
    comparison_path = args.comparison_report.resolve()
    for path in (cache_dir, output_dir, baseline_path, comparison_path):
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
    cache_manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )
    train_peak_ids = {
        row["peak_id"] for row in cache_manifest["train_counts"]
    }
    peak_weights = resolve_peak_weights(train_peak_ids)
    datasets = {
        name: CachedWaveforms(
            cache_dir / f"{name}_values.npy",
            cache_dir / f"{name}_metadata.npz",
            peak_weights,
        )
        for name in ("train", "relaxed_file_validation", "strict_internal")
    }
    evaluation_loaders = {
        name: make_loader(dataset, args.batch_size, False, SEEDS[0])
        for name, dataset in datasets.items()
        if name != "train"
    }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    strict_baseline = float(
        baseline["internal_gate"]["mean_metrics"]["m0"]["macro_auroc"]
    )
    runs: list[dict[str, Any]] = []
    print(f"device={device} train_events={len(datasets['train'])}", flush=True)
    for seed in SEEDS:
        set_seed(seed)
        loader = make_loader(datasets["train"], args.batch_size, True, seed)
        model = DSCNN(input_channels=2, width=24).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        history: list[float] = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, optimizer, device)
            history.append(loss)
            print(f"seed={seed} epoch={epoch}/{args.epochs} loss={loss:.6f}", flush=True)
        evaluations: dict[str, Any] = {}
        for name, eval_loader in evaluation_loaders.items():
            score = predict(model, eval_loader, device)
            evaluations[name] = metrics(datasets[name], score)
            np.save(output_dir / f"seed_{seed}_{name}_scores.npy", score)
        checkpoint_path = output_dir / f"seed_{seed}.pt"
        torch.save(
            {
                "model_kind": "ds_cnn",
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "seed": seed,
                "epochs": args.epochs,
                "selection": "fixed epoch count; no evaluation-driven checkpoint selection",
                "test_partition_used": False,
            },
            checkpoint_path,
        )
        runs.append(
            {
                "seed": seed,
                "training_loss": history,
                "evaluation": evaluations,
                "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
        del model, optimizer, loader
        torch.cuda.empty_cache()
    mean = {
        evaluation: {
            metric: float(
                np.mean([run["evaluation"][evaluation][metric] for run in runs])
            )
            for metric in (
                "macro_auroc",
                "worst_peak_auroc",
                "pooled_auroc",
                "common_three_macro_auroc",
            )
        }
        for evaluation in ("relaxed_file_validation", "strict_internal")
    }
    relaxed_gain = mean["relaxed_file_validation"]["common_three_macro_auroc"] - strict_baseline
    strict_gain = mean["strict_internal"]["common_three_macro_auroc"] - strict_baseline
    baseline_peak_means = {
        peak: float(
            np.mean(
                [
                    run["evaluation"]["strict_internal"]["per_peak"][peak]["auroc"]
                    for run in comparison["runs"]
                ]
            )
        )
        for peak in COMMON_THREE_PEAKS
    }
    candidate_peak_means = {
        peak: float(
            np.mean(
                [
                    run["evaluation"]["strict_internal"]["per_peak"][peak]["auroc"]
                    for run in runs
                ]
            )
        )
        for peak in COMMON_THREE_PEAKS
    }
    comparison_strict_macro = float(np.mean(list(baseline_peak_means.values())))
    comparison_delta = mean["strict_internal"]["common_three_macro_auroc"] - comparison_strict_macro
    comparison_peak_deltas = {
        peak: candidate_peak_means[peak] - baseline_peak_means[peak]
        for peak in COMMON_THREE_PEAKS
    }
    supported = comparison_delta >= 0.004 and all(
        delta >= -0.005 for delta in comparison_peak_deltas.values()
    )
    decision = (
        "ALL_BA_T10_TRAINING_SUPPORTED"
        if supported
        else "ALL_BA_T10_TRAINING_NOT_SUPPORTED"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "all_ba_comparison_gate": {
            "primary_partition": "strict_internal common three peaks",
            "minimum_macro_auroc_gain": 0.004,
            "maximum_allowed_per_peak_loss": 0.005,
            "baseline_macro_auroc": comparison_strict_macro,
            "candidate_macro_auroc": mean["strict_internal"]["common_three_macro_auroc"],
            "macro_auroc_delta": comparison_delta,
            "baseline_per_peak": baseline_peak_means,
            "candidate_per_peak": candidate_peak_means,
            "per_peak_deltas": comparison_peak_deltas,
        },
        "predeclared_interpretation": {
            "much_better_threshold_macro_auroc": 0.02,
            "about_same_tolerance_macro_auroc": 0.01,
            "shortcut_rule": (
                "Relaxed file-validation gain >=0.02 over the strict baseline, "
                "with strict-internal gain <0.01."
            ),
        },
        "strict_energy_matched_baseline_mean_macro_auroc": strict_baseline,
        "mean_metrics": mean,
        "delta_vs_strict_baseline": {
            "relaxed_file_validation_macro_auroc": relaxed_gain,
            "strict_internal_macro_auroc": strict_gain,
        },
        "energy_only_baseline": {
            name: energy_only_metrics(datasets[name])
            for name in ("relaxed_file_validation", "strict_internal")
        },
        "runs": runs,
        "training": {
            "peak_weights": peak_weights,
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "checkpoint_selection": "none; fixed six epochs",
        },
        "input": {
            "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
            "strict_baseline_summary_sha256": sha256_file(baseline_path),
            "comparison_report_sha256": sha256_file(comparison_path),
        },
        "claim_boundary": (
            "Development shortcut diagnostic only; the file-validation partition "
            "is not an independent external campaign."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "mean_metrics": mean, "delta": report["delta_vs_strict_baseline"], "energy_only": report["energy_only_baseline"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
