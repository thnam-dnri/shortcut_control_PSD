#!/usr/bin/env python3
"""Train a group-balanced shared DS-CNN and fine-tune it on Group 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN
from src.ba133_cnn import set_seed
from src.data_access_guards import assert_no_forbidden_path

SEEDS = (20260822, 20260823, 20260824)
PEAK_IDS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
PEAK_UNITS = {"ba133_356kev": 2, "na22_511kev": 2, "cs137_662kev": 1}
GROUPS = tuple(range(1, 7))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedWaveforms(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose selected cache rows without copying the waveform array."""

    def __init__(
        self,
        values_path: Path,
        metadata_path: Path,
        assignments_path: Path,
        groups: Sequence[int] = GROUPS,
    ) -> None:
        self.values = np.load(values_path, mmap_mode="r")
        metadata = np.load(metadata_path)
        assignments = np.load(assignments_path)
        selected = assignments["selected"].astype(bool)
        one_based_group = assignments["assignment"].astype(np.int16) + 1
        keep = selected & np.isin(one_based_group, np.asarray(groups))
        self.cache_indices = np.flatnonzero(keep).astype(np.int64)
        self.labels = metadata["label"][self.cache_indices].astype(np.int8)
        self.peak_ids = metadata["peak_id"][self.cache_indices].astype(str)
        self.groups = one_based_group[self.cache_indices]
        self.energies = metadata["energy_kev"][self.cache_indices].astype(np.float32)

    def __len__(self) -> int:
        return self.cache_indices.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_index = self.cache_indices[index]
        values = torch.from_numpy(
            np.asarray(self.values[cache_index], dtype=np.float32)
        )
        return values, torch.tensor(self.labels[index], dtype=torch.float32)


class BalancedStratumBatchSampler(Sampler[list[int]]):
    """Draw exact group/class/peak proportions in every complete batch."""

    def __init__(
        self,
        dataset: IndexedWaveforms,
        groups: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        units_per_group = 2 * sum(PEAK_UNITS.values())
        total_units = len(groups) * units_per_group
        if batch_size % total_units:
            raise ValueError(
                f"batch_size must be divisible by {total_units} for exact balance"
            )
        self.dataset = dataset
        self.active_groups = tuple(groups)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.batch_count = math.ceil(len(dataset) / batch_size)
        self.indices: dict[tuple[int, int, str], np.ndarray] = {}
        for group in self.active_groups:
            for label in (0, 1):
                for peak in PEAK_IDS:
                    mask = (
                        (dataset.groups == group)
                        & (dataset.labels == label)
                        & (dataset.peak_ids == peak)
                    )
                    rows = np.flatnonzero(mask)
                    if rows.size == 0:
                        raise ValueError(f"Empty training stratum {(group, label, peak)}")
                    self.indices[(group, label, peak)] = rows

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        copies = self.batch_size // (
            len(self.active_groups) * 2 * sum(PEAK_UNITS.values())
        )
        for _ in range(self.batch_count):
            batch: list[int] = []
            for group in self.active_groups:
                for label in (0, 1):
                    for peak in PEAK_IDS:
                        count = copies * PEAK_UNITS[peak]
                        rows = self.indices[(group, label, peak)]
                        batch.extend(
                            rng.choice(rows, size=count, replace=True).tolist()
                        )
            rng.shuffle(batch)
            yield batch


def make_train_loader(
    dataset: IndexedWaveforms,
    groups: Sequence[int],
    batch_size: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=BalancedStratumBatchSampler(
            dataset=dataset,
            groups=groups,
            batch_size=batch_size,
            seed=seed,
        ),
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(dataset: IndexedWaveforms, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
    total_loss = 0.0
    total_count = 0
    for values, labels in loader:
        values = values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_count += labels.numel()
    return total_loss / total_count


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels in loader:
            logits = model(values.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "event_count": int(labels.size),
        "positive_count": int(np.count_nonzero(labels == 1)),
        "negative_count": int(np.count_nonzero(labels == 0)),
    }


def metrics(dataset: IndexedWaveforms, scores: np.ndarray) -> dict[str, Any]:
    per_peak = {
        peak: binary_metrics(dataset.labels[dataset.peak_ids == peak], scores[dataset.peak_ids == peak])
        for peak in PEAK_IDS
    }
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in per_peak.values()])),
        "worst_peak_auroc": float(np.min([row["auroc"] for row in per_peak.values()])),
        "pooled": binary_metrics(dataset.labels, scores),
        "per_peak": per_peak,
    }


def evaluate_by_group(
    dataset: IndexedWaveforms,
    scores: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"all_groups": metrics(dataset, scores)}
    for group in GROUPS:
        mask = dataset.groups == group
        subset = _ArrayDatasetView(dataset, mask)
        result[f"group_{group}"] = metrics(subset, scores[mask])
    return result


class _ArrayDatasetView:
    def __init__(self, dataset: IndexedWaveforms, mask: np.ndarray) -> None:
        self.labels = dataset.labels[mask]
        self.peak_ids = dataset.peak_ids[mask]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    seed: int,
    stage: str,
    epochs: int,
    representation_config: dict[str, Any],
    feature_statistics: dict[str, Any],
) -> str:
    torch.save(
        {
            "model_kind": "ds_cnn",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "seed": seed,
            "stage": stage,
            "epochs": epochs,
            "representation_config": representation_config,
            "feature_statistics": feature_statistics,
            "selection": "fixed epoch count; no evaluation-driven checkpoint selection",
            "test_partition_used": False,
        },
        path,
    )
    return sha256_file(path)


def mean_metric(runs: list[dict[str, Any]], stage: str, population: str, metric: str) -> float:
    return float(
        np.mean([run[stage][population][metric] for run in runs])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--group-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_six_group_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/shared_six_group_ds_cnn_20260822",
    )
    parser.add_argument("--shared-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--shared-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    group_dir = args.group_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (cache_dir, group_dir, output_dir):
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

    train_all = IndexedWaveforms(
        cache_dir / "train_values.npy",
        cache_dir / "train_metadata.npz",
        group_dir / "train_assignments.npz",
    )
    train_group_2 = IndexedWaveforms(
        cache_dir / "train_values.npy",
        cache_dir / "train_metadata.npz",
        group_dir / "train_assignments.npz",
        groups=(2,),
    )
    validation_all = IndexedWaveforms(
        cache_dir / "relaxed_file_validation_values.npy",
        cache_dir / "relaxed_file_validation_metadata.npz",
        group_dir / "relaxed_file_validation_assignments.npz",
    )
    validation_group_2 = IndexedWaveforms(
        cache_dir / "relaxed_file_validation_values.npy",
        cache_dir / "relaxed_file_validation_metadata.npz",
        group_dir / "relaxed_file_validation_assignments.npz",
        groups=(2,),
    )
    strict_group_2 = IndexedWaveforms(
        cache_dir / "strict_internal_values.npy",
        cache_dir / "strict_internal_metadata.npz",
        group_dir / "strict_internal_assignments.npz",
        groups=(2,),
    )
    eval_loaders = {
        "validation_all": make_eval_loader(validation_all, args.batch_size),
        "validation_group_2": make_eval_loader(validation_group_2, args.batch_size),
        "strict_group_2": make_eval_loader(strict_group_2, args.batch_size),
    }
    print(
        f"device={device} train_all={len(train_all)} train_group_2={len(train_group_2)} "
        f"validation_group_2={len(validation_group_2)} strict_group_2={len(strict_group_2)}",
        flush=True,
    )

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_seed(seed)
        model = DSCNN(input_channels=2, width=24).to(device)
        shared_loader = make_train_loader(train_all, GROUPS, args.batch_size, seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.shared_learning_rate, weight_decay=args.weight_decay
        )
        shared_loss: list[float] = []
        for epoch in range(1, args.shared_epochs + 1):
            loss = train_epoch(model, shared_loader, optimizer, device)
            shared_loss.append(loss)
            print(f"seed={seed} shared_epoch={epoch}/{args.shared_epochs} loss={loss:.6f}", flush=True)

        shared_scores_all = predict(model, eval_loaders["validation_all"], device)
        shared_scores_group_2 = predict(model, eval_loaders["validation_group_2"], device)
        shared_scores_strict_group_2 = predict(model, eval_loaders["strict_group_2"], device)
        shared_evaluation = {
            "validation_by_group": evaluate_by_group(validation_all, shared_scores_all),
            "validation_group_2": metrics(validation_group_2, shared_scores_group_2),
            "strict_group_2": metrics(strict_group_2, shared_scores_strict_group_2),
        }
        shared_path = output_dir / f"seed_{seed}_shared.pt"
        shared_hash = save_checkpoint(
            shared_path,
            model,
            seed,
            "shared_six_group",
            args.shared_epochs,
            cache_manifest["representation_config"],
            cache_manifest["feature_statistics"],
        )
        np.save(output_dir / f"seed_{seed}_shared_validation_group_2_scores.npy", shared_scores_group_2)
        np.save(output_dir / f"seed_{seed}_shared_strict_group_2_scores.npy", shared_scores_strict_group_2)

        fine_tune_loader = make_train_loader(train_group_2, (2,), args.batch_size, seed + 1000)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.fine_tune_learning_rate, weight_decay=args.weight_decay
        )
        fine_tune_loss: list[float] = []
        for epoch in range(1, args.fine_tune_epochs + 1):
            loss = train_epoch(model, fine_tune_loader, optimizer, device)
            fine_tune_loss.append(loss)
            print(f"seed={seed} group2_epoch={epoch}/{args.fine_tune_epochs} loss={loss:.6f}", flush=True)

        fine_scores_group_2 = predict(model, eval_loaders["validation_group_2"], device)
        fine_scores_strict_group_2 = predict(model, eval_loaders["strict_group_2"], device)
        fine_evaluation = {
            "validation_group_2": metrics(validation_group_2, fine_scores_group_2),
            "strict_group_2": metrics(strict_group_2, fine_scores_strict_group_2),
        }
        fine_path = output_dir / f"seed_{seed}_group_2_fine_tuned.pt"
        fine_hash = save_checkpoint(
            fine_path,
            model,
            seed,
            "group_2_fine_tuned",
            args.fine_tune_epochs,
            cache_manifest["representation_config"],
            cache_manifest["feature_statistics"],
        )
        np.save(output_dir / f"seed_{seed}_fine_tuned_validation_group_2_scores.npy", fine_scores_group_2)
        np.save(output_dir / f"seed_{seed}_fine_tuned_strict_group_2_scores.npy", fine_scores_strict_group_2)
        runs.append(
            {
                "seed": seed,
                "shared_training_loss": shared_loss,
                "fine_tune_training_loss": fine_tune_loss,
                "shared": shared_evaluation,
                "fine_tuned": fine_evaluation,
                "shared_checkpoint": shared_path.relative_to(PROJECT_ROOT).as_posix(),
                "shared_checkpoint_sha256": shared_hash,
                "fine_tuned_checkpoint": fine_path.relative_to(PROJECT_ROOT).as_posix(),
                "fine_tuned_checkpoint_sha256": fine_hash,
            }
        )
        del model, optimizer, shared_loader, fine_tune_loader
        torch.cuda.empty_cache()

    summary: dict[str, Any] = {}
    for population in ("validation_group_2", "strict_group_2"):
        shared_mean = mean_metric(runs, "shared", population, "macro_auroc")
        fine_mean = mean_metric(runs, "fine_tuned", population, "macro_auroc")
        summary[population] = {
            "shared_mean_macro_auroc": shared_mean,
            "fine_tuned_mean_macro_auroc": fine_mean,
            "fine_tuned_minus_shared": fine_mean - shared_mean,
        }
    decision = (
        "GROUP_2_FINE_TUNING_SUPPORTED"
        if summary["validation_group_2"]["fine_tuned_minus_shared"] >= 0.004
        and summary["strict_group_2"]["fine_tuned_minus_shared"] >= 0.0
        else "KEEP_SHARED_MODEL"
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "summary": summary,
        "runs": runs,
        "data": {
            "train_all_events": len(train_all),
            "train_group_2_events": len(train_group_2),
            "validation_group_2_events": len(validation_group_2),
            "validation_group_2_positive": int(np.count_nonzero(validation_group_2.labels == 1)),
            "validation_group_2_negative": int(np.count_nonzero(validation_group_2.labels == 0)),
            "strict_group_2_events": len(strict_group_2),
        },
        "training": {
            "seeds": list(SEEDS),
            "shared_epochs": args.shared_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "batch_size": args.batch_size,
            "shared_learning_rate": args.shared_learning_rate,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
            "weight_decay": args.weight_decay,
            "batch_contract": (
                "Exact equal group and class contribution per complete batch; "
                "within each group/class, Ba356:Na511:Cs662 draw ratio is 2:2:1."
            ),
            "checkpoint_selection": "none; fixed epoch counts",
            "device": str(device),
        },
        "input": {
            "cache_manifest_sha256": sha256_file(cache_dir / "cache_manifest.json"),
            "six_group_dataset_report_sha256": sha256_file(group_dir / "dataset_report.json"),
        },
        "fine_tuning_gate": (
            "Support Group 2 fine-tuning only when mean file-validation macro AUROC "
            "improves by at least 0.004 and strict-internal Group 2 does not decrease."
        ),
        "claim_boundary": (
            "Development-only comparison on file-disjoint and strict-internal pools; "
            "not independent external interaction-truth validation."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
