#!/usr/bin/env python3
"""Train an all-Ba MA10/t10 DS-CNN using only selected morphology groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN
from src.ba133_cnn import set_seed
from src.data_access_guards import assert_no_forbidden_path


SEEDS = (20260822, 20260823, 20260824)
PEAK_WEIGHTS = {
    "ba133_276kev": 0.1,
    "ba133_303kev": 0.1,
    "ba133_356kev": 0.1,
    "ba133_384kev": 0.1,
    "na22_511kev": 0.4,
    "cs137_662kev": 0.2,
}
COMMON_PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
ALL_GROUPS = tuple(range(1, 7))
TRAIN_GROUPS = (4, 6)
UNSEEN_GROUPS = (1, 2, 3, 5)
PEAK_LABELS = {
    "ba133_276kev": "Ba 276",
    "ba133_303kev": "Ba 303",
    "ba133_356kev": "Ba 356",
    "ba133_384kev": "Ba 384",
    "na22_511kev": "Na 511",
    "cs137_662kev": "Cs 662",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class MorphologyWaveforms(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        values_path: Path,
        metadata_path: Path,
        assignments_path: Path,
        groups: Sequence[int] | None,
        training: bool,
    ) -> None:
        self.values = np.load(values_path, mmap_mode="r")
        with np.load(metadata_path, allow_pickle=False) as metadata:
            labels = metadata["label"].astype(np.int8)
            peaks = metadata["peak_id"].astype(str)
        with np.load(assignments_path, allow_pickle=False) as assignments:
            selected = assignments["selected"].astype(bool)
            one_based_groups = assignments["assignment"].astype(np.int16) + 1
        if groups is None:
            keep = np.ones(labels.size, dtype=bool)
        else:
            keep = selected & np.isin(one_based_groups, np.asarray(groups))
        self.cache_indices = np.flatnonzero(keep).astype(np.int64)
        self.labels = labels[self.cache_indices]
        self.peak_ids = peaks[self.cache_indices]
        self.groups = one_based_groups[self.cache_indices]
        self.selected = selected[self.cache_indices]
        if training:
            weights = np.zeros(self.labels.size, dtype=np.float64)
            for peak, peak_weight in PEAK_WEIGHTS.items():
                for label in (0, 1):
                    mask = (self.peak_ids == peak) & (self.labels == label)
                    count = int(np.count_nonzero(mask))
                    if count == 0:
                        raise ValueError(
                            f"Empty morphology training cell: {peak}, label={label}"
                        )
                    weights[mask] = 0.5 * peak_weight / count
            weights *= self.labels.size / np.sum(weights)
            self.weights = weights.astype(np.float32)
        else:
            self.weights = np.ones(self.labels.size, dtype=np.float32)

    def __len__(self) -> int:
        return self.cache_indices.size

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_index = self.cache_indices[index]
        values = torch.from_numpy(
            np.asarray(self.values[cache_index], dtype=np.float32)
        )
        return (
            values,
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )


def make_train_loader(
    dataset: MorphologyWaveforms,
    batch_size: int,
    samples_per_epoch: int,
    seed: int,
) -> DataLoader:
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=samples_per_epoch,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )


def make_eval_loader(
    dataset: MorphologyWaveforms, batch_size: int
) -> DataLoader:
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
    weighted_loss = 0.0
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
        loss = torch.sum(losses * weights) / torch.sum(weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        weighted_loss += float(torch.sum(losses * weights).item())
        weight_total += float(torch.sum(weights).item())
    return weighted_loss / weight_total


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for values, _labels, _weights in loader:
            logits = model(values.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if np.unique(labels).size != 2:
        raise ValueError("Metric subset lacks both labels")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "events": int(labels.size),
        "positive_events": int(np.count_nonzero(labels == 1)),
        "negative_events": int(np.count_nonzero(labels == 0)),
    }


def metric_summary(
    labels: np.ndarray,
    peaks: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    per_peak = {
        peak: binary_metrics(labels[peaks == peak], scores[peaks == peak])
        for peak in sorted(set(peaks.tolist()))
    }
    peak_aurocs = [row["auroc"] for row in per_peak.values()]
    common = [
        per_peak[peak]["auroc"] for peak in COMMON_PEAKS if peak in per_peak
    ]
    return {
        "macro_auroc": float(np.mean(peak_aurocs)),
        "worst_peak_auroc": float(np.min(peak_aurocs)),
        "pooled": binary_metrics(labels, scores),
        "common_three_macro_auroc": (
            float(np.mean(common)) if len(common) == len(COMMON_PEAKS) else None
        ),
        "per_peak": per_peak,
    }


def group_metrics(
    dataset: MorphologyWaveforms,
    scores: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ALL_GROUPS:
        mask = dataset.selected & (dataset.groups == group)
        result[f"group_{group}"] = metric_summary(
            dataset.labels[mask], dataset.peak_ids[mask], scores[mask]
        )
    return result


def mean_run_metric(
    runs: list[dict[str, Any]], partition: str, key: str
) -> float:
    return float(
        np.mean([run["evaluation"][partition][key] for run in runs])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_all_ba_t10_20260823",
    )
    parser.add_argument(
        "--assignment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_all_ba_six_group_20260823",
    )
    parser.add_argument(
        "--all-ba-baseline-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_all_ba_t10_20260823",
    )
    parser.add_argument(
        "--three-peak-baseline-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/all_ba_group_4_6_t10_20260823",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--minimum-strict-gain", type=float, default=0.004)
    parser.add_argument("--maximum-unseen-group-loss", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    assignment_dir = args.assignment_dir.resolve()
    all_ba_baseline_dir = args.all_ba_baseline_dir.resolve()
    three_peak_baseline_dir = args.three_peak_baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (
        cache_dir,
        assignment_dir,
        all_ba_baseline_dir,
        three_peak_baseline_dir,
        output_dir,
    ):
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

    cache_manifest_path = cache_dir / "cache_manifest.json"
    assignment_report_path = assignment_dir / "dataset_report.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    assignment_report = json.loads(
        assignment_report_path.read_text(encoding="utf-8")
    )
    samples_per_epoch = int(cache_manifest["cache"]["train"]["event_count"])
    steps_per_epoch = math.ceil(samples_per_epoch / args.batch_size)
    train_dataset = MorphologyWaveforms(
        cache_dir / "train_values.npy",
        cache_dir / "train_metadata.npz",
        assignment_dir / "train_assignments.npz",
        groups=TRAIN_GROUPS,
        training=True,
    )
    evaluation_datasets = {
        partition: MorphologyWaveforms(
            cache_dir / f"{partition}_values.npy",
            cache_dir / f"{partition}_metadata.npz",
            assignment_dir / f"{partition}_assignments.npz",
            groups=None,
            training=False,
        )
        for partition in ("relaxed_file_validation", "strict_internal")
    }
    evaluation_loaders = {
        partition: make_eval_loader(dataset, args.batch_size)
        for partition, dataset in evaluation_datasets.items()
    }
    print(
        f"device={device} train_groups={TRAIN_GROUPS} unique_train_events={len(train_dataset)} "
        f"samples_per_epoch={samples_per_epoch} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        set_seed(seed)
        train_loader = make_train_loader(
            train_dataset, args.batch_size, samples_per_epoch, seed
        )
        model = DSCNN(input_channels=2, width=24).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        history: list[float] = []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, device)
            history.append(loss)
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} loss={loss:.6f}",
                flush=True,
            )
        evaluation: dict[str, Any] = {}
        for partition, loader in evaluation_loaders.items():
            scores = predict(model, loader, device)
            dataset = evaluation_datasets[partition]
            evaluation[partition] = metric_summary(
                dataset.labels, dataset.peak_ids, scores
            )
            evaluation[partition]["by_group"] = group_metrics(dataset, scores)
            np.save(output_dir / f"seed_{seed}_{partition}_scores.npy", scores)
        checkpoint_path = output_dir / f"seed_{seed}.pt"
        torch.save(
            {
                "model_kind": "ds_cnn",
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "seed": seed,
                "epochs": args.epochs,
                "train_groups": list(TRAIN_GROUPS),
                "peak_weights": PEAK_WEIGHTS,
                "samples_per_epoch": samples_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "representation_config": cache_manifest["representation_config"],
                "feature_statistics": cache_manifest["feature_statistics"],
                "selection": "fixed six epochs and matched optimizer steps",
                "test_partition_used": False,
            },
            checkpoint_path,
        )
        runs.append(
            {
                "seed": seed,
                "training_loss": history,
                "evaluation": evaluation,
                "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
        del model, optimizer, train_loader
        torch.cuda.empty_cache()

    all_ba_report = json.loads(
        (all_ba_baseline_dir / "experiment_report.json").read_text(
            encoding="utf-8"
        )
    )
    three_peak_report = json.loads(
        (three_peak_baseline_dir / "experiment_report.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_group_runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline_scores = np.load(
            all_ba_baseline_dir / f"seed_{seed}_strict_internal_scores.npy"
        )
        dataset = evaluation_datasets["strict_internal"]
        baseline_group_runs.append(group_metrics(dataset, baseline_scores))

    candidate_strict_macro = mean_run_metric(
        runs, "strict_internal", "common_three_macro_auroc"
    )
    all_ba_strict_macro = float(
        np.mean(
            [
                run["evaluation"]["strict_internal"]["common_three_macro_auroc"]
                for run in all_ba_report["runs"]
            ]
        )
    )
    three_peak_strict_macro = float(
        np.mean(
            [
                run["evaluation"]["strict_internal"]["macro_auroc"]
                for run in three_peak_report["runs"]
            ]
        )
    )
    strict_gain = candidate_strict_macro - all_ba_strict_macro
    group_comparison: dict[str, Any] = {}
    for group in ALL_GROUPS:
        key = f"group_{group}"
        baseline_value = float(
            np.mean(
                [row[key]["common_three_macro_auroc"] for row in baseline_group_runs]
            )
        )
        candidate_value = float(
            np.mean(
                [
                    run["evaluation"]["strict_internal"]["by_group"][key][
                        "common_three_macro_auroc"
                    ]
                    for run in runs
                ]
            )
        )
        group_comparison[key] = {
            "all_ba_baseline_macro_auroc": baseline_value,
            "group_4_6_training_macro_auroc": candidate_value,
            "delta": candidate_value - baseline_value,
            "training_group": group in TRAIN_GROUPS,
        }
    unseen_group_gate = all(
        group_comparison[f"group_{group}"]["delta"]
        >= -args.maximum_unseen_group_loss
        for group in UNSEEN_GROUPS
    )
    supported = strict_gain >= args.minimum_strict_gain and unseen_group_gate
    decision = (
        "GROUP_4_6_EXCLUSIVE_TRAINING_SUPPORTED"
        if supported
        else "GROUP_4_6_EXCLUSIVE_TRAINING_NOT_SUPPORTED"
    )

    peak_comparison: dict[str, Any] = {}
    for peak in COMMON_PEAKS:
        baseline_value = float(
            np.mean(
                [
                    run["evaluation"]["strict_internal"]["per_peak"][peak]["auroc"]
                    for run in all_ba_report["runs"]
                ]
            )
        )
        candidate_value = float(
            np.mean(
                [
                    run["evaluation"]["strict_internal"]["per_peak"][peak]["auroc"]
                    for run in runs
                ]
            )
        )
        peak_comparison[peak] = {
            "all_ba_baseline_auroc": baseline_value,
            "group_4_6_training_auroc": candidate_value,
            "delta": candidate_value - baseline_value,
        }

    summary = {
        partition: {
            "macro_auroc": mean_run_metric(runs, partition, "macro_auroc"),
            "common_three_macro_auroc": mean_run_metric(
                runs, partition, "common_three_macro_auroc"
            ),
            "worst_peak_auroc": mean_run_metric(
                runs, partition, "worst_peak_auroc"
            ),
        }
        for partition in ("relaxed_file_validation", "strict_internal")
    }

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "contract": {
            "training_groups": list(TRAIN_GROUPS),
            "global_evaluation": True,
            "representation": cache_manifest["representation_config"],
            "peak_weights": PEAK_WEIGHTS,
            "unique_training_events": len(train_dataset),
            "samples_per_epoch": samples_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": steps_per_epoch * args.epochs,
            "baseline_optimizer_steps_matched": True,
            "morphology_catalogue_refit": False,
            "minimum_index_cut": assignment_report["selection"][
                "raw_negative_minimum_index_inclusive"
            ],
        },
        "gate": {
            "minimum_strict_macro_gain_vs_all_ba": args.minimum_strict_gain,
            "maximum_unseen_group_macro_loss": args.maximum_unseen_group_loss,
            "observed_strict_macro_gain": strict_gain,
            "unseen_group_gate_passed": unseen_group_gate,
            "supported": supported,
        },
        "strict_comparison": {
            "three_peak_t10_macro_auroc": three_peak_strict_macro,
            "all_ba_t10_macro_auroc": all_ba_strict_macro,
            "group_4_6_training_macro_auroc": candidate_strict_macro,
            "delta_vs_all_ba": strict_gain,
            "per_peak": peak_comparison,
            "per_group": group_comparison,
        },
        "mean_metrics": summary,
        "runs": runs,
        "training": {
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "inputs": {
            "cache_manifest": cache_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
            "assignment_report": assignment_report_path.relative_to(PROJECT_ROOT).as_posix(),
            "assignment_report_sha256": sha256_file(assignment_report_path),
            "all_ba_baseline_report_sha256": sha256_file(
                all_ba_baseline_dir / "experiment_report.json"
            ),
            "three_peak_baseline_report_sha256": sha256_file(
                three_peak_baseline_dir / "experiment_report.json"
            ),
        },
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for peak, values in peak_comparison.items():
        rows.append({"scope": "peak", "name": peak, **values})
    for group, values in group_comparison.items():
        rows.append({"scope": "group", "name": group, **values})
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "strict_peak_group_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.36
    peak_x = np.arange(len(COMMON_PEAKS))
    axes[0].bar(
        peak_x - width / 2,
        [peak_comparison[peak]["all_ba_baseline_auroc"] for peak in COMMON_PEAKS],
        width,
        label="all-Ba baseline",
    )
    axes[0].bar(
        peak_x + width / 2,
        [peak_comparison[peak]["group_4_6_training_auroc"] for peak in COMMON_PEAKS],
        width,
        label="Groups 4+6 training",
    )
    axes[0].set_xticks(peak_x, [PEAK_LABELS[peak] for peak in COMMON_PEAKS])
    axes[0].set_ylabel("Strict AUROC")
    axes[0].set_title("Global strict result by peak")
    axes[0].set_ylim(0.59, 0.73)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    group_x = np.arange(len(ALL_GROUPS))
    axes[1].bar(
        group_x - width / 2,
        [
            group_comparison[f"group_{group}"]["all_ba_baseline_macro_auroc"]
            for group in ALL_GROUPS
        ],
        width,
        label="all-Ba baseline",
    )
    axes[1].bar(
        group_x + width / 2,
        [
            group_comparison[f"group_{group}"]["group_4_6_training_macro_auroc"]
            for group in ALL_GROUPS
        ],
        width,
        label="Groups 4+6 training",
    )
    axes[1].set_xticks(group_x, [str(group) for group in ALL_GROUPS])
    axes[1].set_xlabel("Morphology group")
    axes[1].set_ylabel("Strict common-three macro AUROC")
    axes[1].set_title("Global application by morphology group")
    axes[1].set_ylim(0.52, 0.72)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("All-Ba MA10/t10 training restricted to Groups 4 and 6")
    figure.tight_layout()
    figure.savefig(output_dir / "group_4_6_global_comparison.png", dpi=180)
    plt.close(figure)

    markdown = f"""# Groups 4+6 exclusive all-Ba t10 training

## Simple result

- Decision: `{decision}`
- Unique Group 4+6 training events: {len(train_dataset):,}.
- Optimizer steps: {steps_per_epoch * args.epochs:,}, exactly matched to the all-Ba baseline.
- Three-peak t10 strict macro AUROC: {three_peak_strict_macro:.6f}
- All-Ba all-group strict macro AUROC: {all_ba_strict_macro:.6f}
- Groups 4+6-only training, applied globally: {candidate_strict_macro:.6f}
- Change versus all-Ba baseline: {strict_gain:+.6f}; required: at least {args.minimum_strict_gain:+.3f}.

## Strict result by peak

| Peak | all-Ba baseline | Groups 4+6 training | change |
|---|---:|---:|---:|
"""
    for peak in COMMON_PEAKS:
        values = peak_comparison[peak]
        markdown += (
            f"| {PEAK_LABELS[peak]} | {values['all_ba_baseline_auroc']:.6f} | "
            f"{values['group_4_6_training_auroc']:.6f} | {values['delta']:+.6f} |\n"
        )
    markdown += """

## Strict result by morphology group

| Group | Seen in training | all-Ba baseline | Groups 4+6 training | change |
|---:|:---:|---:|---:|---:|
"""
    for group in ALL_GROUPS:
        values = group_comparison[f"group_{group}"]
        markdown += (
            f"| {group} | {'yes' if values['training_group'] else 'no'} | "
            f"{values['all_ba_baseline_macro_auroc']:.6f} | "
            f"{values['group_4_6_training_macro_auroc']:.6f} | "
            f"{values['delta']:+.6f} |\n"
        )
    markdown += f"""

## Interpretation

The exclusive morphology hypothesis {"passes" if supported else "does not pass"}
the global strict gate. The model was evaluated on every event and every assigned
group, including Groups 1, 2, 3, and 5 that were absent from training.

Locked test, Th-232, and Eu-152 were not used.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "strict_macro_gain": strict_gain,
                "unseen_group_gate_passed": unseen_group_gate,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
