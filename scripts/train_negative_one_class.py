#!/usr/bin/env python3
"""Train a Co-60-negative-only Deep-SVDD screen and score held-out classes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    fit_channel_statistics,
    load_raw_partition,
)
from src.one_class_cnn import (  # noqa: E402
    OneClassCompactEncoder,
    clamp_center,
    embedding_diagnostics,
)

PEAK_IDS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
REPRESENTATION = RepresentationConfig(
    name="both_ma10_global_t10_w750_positive_polarity",
    input_mode="both",
    moving_average=10,
    normalization="global",
    anchor="t10",
    pre_samples=250,
    post_samples=500,
    pulse_polarity="negative_to_positive",
    standardization="class0_fit_zscore",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def select_energy_balanced_rows(
    store_path: Path,
    counts_per_bin: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(100.0, 1000.0 + 50.0, 50.0, dtype=np.float32)
    if len(counts_per_bin) != edges.size - 1:
        raise ValueError("Expected one requested count per 50-keV bin")
    rng = np.random.default_rng(seed)
    with h5py.File(store_path, "r") as handle:
        energies = np.asarray(handle["corrected_energy_kev"], dtype=np.float32)
    selected: list[int] = []
    bin_ids: list[str] = []
    selected_energies: list[float] = []
    for bin_index, count in enumerate(counts_per_bin):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        mask = (energies >= lower) & (
            energies <= upper if bin_index == edges.size - 2 else energies < upper
        )
        available = np.flatnonzero(mask)
        if available.size < count:
            raise ValueError(f"Energy bin {lower}-{upper} has only {available.size} rows")
        chosen = rng.choice(available, size=count, replace=False)
        selected.extend(chosen.tolist())
        bin_ids.extend([f"co60_{int(lower)}_{int(upper)}kev"] * count)
        selected_energies.extend(energies[chosen].tolist())
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(bin_ids, dtype="U32"),
        np.asarray(selected_energies, dtype=np.float32),
    )


def load_continuum_rows(
    store_path: Path,
    rows: np.ndarray,
    bin_ids: np.ndarray,
) -> RawPartition:
    order = np.argsort(rows)
    waveforms = np.empty((rows.size, 4500), dtype=np.float32)
    shaped = np.empty(rows.size, dtype=np.float32)
    with h5py.File(store_path, "r") as handle:
        for start in range(0, rows.size, 512):
            stop = min(start + 512, rows.size)
            destinations = order[start:stop]
            selected = rows[destinations]
            waveforms[destinations] = handle["waveform"][selected]
            shaped[destinations] = handle["shaped_energy_unit"][selected]
    return RawPartition(
        waveforms=waveforms,
        shaped_energy=shaped,
        labels=np.zeros(rows.size, dtype=np.float32),
        weights=np.ones(rows.size, dtype=np.float32),
        peak_ids=bin_ids,
    )


def subset_raw(raw: RawPartition, indices: np.ndarray) -> RawPartition:
    return RawPartition(
        waveforms=raw.waveforms[indices],
        shaped_energy=raw.shaped_energy[indices],
        labels=raw.labels[indices],
        weights=raw.weights[indices],
        peak_ids=raw.peak_ids[indices],
    )


def equal_peak_weights(peaks: np.ndarray) -> np.ndarray:
    counts = Counter(peaks.tolist())
    weights = np.asarray(
        [1.0 / (len(counts) * counts[peak]) for peak in peaks], dtype=np.float32
    )
    return weights


def make_loader(
    values: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(weights)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def collect_embeddings(
    model: OneClassCompactEncoder,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            result.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(result)


def anomaly_scores(
    model: OneClassCompactEncoder,
    center: torch.Tensor,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings = collect_embeddings(model, values, batch_size, device)
    center_array = center.detach().cpu().numpy()
    return np.sum(np.square(embeddings - center_array[None, :]), axis=1)


def train_epoch(
    model: OneClassCompactEncoder,
    center: torch.Tensor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    weight_sum = 0.0
    for values, weights in loader:
        values = values.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(values)
        distances = torch.sum(torch.square(embeddings - center[None, :]), dim=1)
        loss = torch.sum(distances * weights) / torch.sum(weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += float(torch.sum(distances * weights).item())
        weight_sum += float(torch.sum(weights).item())
    return loss_sum / weight_sum


def score_metrics(labels: np.ndarray, scores: np.ndarray, peaks: np.ndarray) -> dict[str, Any]:
    per_peak: dict[str, Any] = {}
    for peak_id in PEAK_IDS:
        mask = peaks == peak_id
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "label0_count": int(np.count_nonzero(labels[mask] == 0)),
            "label1_count": int(np.count_nonzero(labels[mask] == 1)),
        }
    aurocs = [per_peak[peak]["auroc"] for peak in PEAK_IDS]
    return {
        "worst_peak_auroc": float(min(aurocs)),
        "macro_auroc": float(np.mean(aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "per_peak": per_peak,
    }


def plot_distributions(
    path: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    peaks: np.ndarray,
) -> None:
    transformed = np.log10(np.maximum(scores, 1.0e-12))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [(None, "All three energies")] + [
        (peak, peak.replace("_", " ")) for peak in PEAK_IDS
    ]
    for axis, (peak_id, title) in zip(axes.ravel(), panels):
        mask = np.ones(labels.size, dtype=bool) if peak_id is None else peaks == peak_id
        lower, upper = np.percentile(transformed[mask], (0.5, 99.5))
        bins = np.linspace(lower, upper, 80)
        axis.hist(
            transformed[mask & (labels == 0)],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Held-out Co-60 label 0",
            color="tab:blue",
        )
        axis.hist(
            transformed[mask & (labels == 1)],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Held-out label 1",
            color="tab:red",
        )
        axis.set_title(title)
        axis.set_xlabel("log10 Deep-SVDD anomaly score")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Negative-only Co-60 one-class CNN: held-out anomaly scores")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_energy_dependence(path: Path, energies: np.ndarray, scores: np.ndarray) -> None:
    edges = np.arange(100.0, 1000.0 + 50.0, 50.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    transformed = np.log10(np.maximum(scores, 1.0e-12))
    for index in range(edges.size - 1):
        mask = (energies >= edges[index]) & (
            energies <= edges[index + 1]
            if index == edges.size - 2
            else energies < edges[index + 1]
        )
        quantiles = np.quantile(transformed[mask], (0.05, 0.5, 0.95))
        lower.append(float(quantiles[0]))
        medians.append(float(quantiles[1]))
        upper.append(float(quantiles[2]))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(centers, medians, marker="o", label="Median")
    axis.fill_between(centers, lower, upper, alpha=0.25, label="5th-95th percentile")
    axis.set_xlabel("Corrected Co-60 energy (keV)")
    axis.set_ylabel("log10 Deep-SVDD anomaly score")
    axis.set_title("Held-out broad Co-60 anomaly score versus energy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--label-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
    )
    result.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    result.add_argument(
        "--continuum-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/co60_continuum_100_1000kev_20260819",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/models/negative_one_class_co60_20260820",
    )
    result.add_argument("--epochs", type=int, default=8)
    result.add_argument("--fit-per-50kev-bin", type=int, default=5000)
    result.add_argument("--internal-per-50kev-bin", type=int, default=1000)
    result.add_argument("--validation-per-50kev-bin", type=int, default=1000)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=1.0e-4)
    result.add_argument("--weight-decay", type=float, default=1.0e-6)
    result.add_argument("--seed", type=int, default=20260820)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = args.label_dir.resolve()
    train_csv = label_dir / "label_pairs_train.csv"
    heldout_csv = label_dir / "label_pairs_validation.csv"
    split_path = label_dir / "train_internal_split_indices.npz"
    heldout_rows = read_rows(heldout_csv)
    continuum_dir = args.continuum_store_dir.resolve()
    continuum_train_path = continuum_dir / "train_events.h5"
    continuum_validation_path = continuum_dir / "validation_events.h5"
    continuum_manifest_path = (
        PROJECT_ROOT
        / "outputs/event_store/co60_continuum_100_1000kev_20260819/continuum_store_manifest.json"
    )
    total_per_bin = args.fit_per_50kev_bin + args.internal_per_50kev_bin
    selected_rows, selected_bins, _selected_energies = select_energy_balanced_rows(
        continuum_train_path,
        (total_per_bin,) * 18,
        args.seed,
    )
    fit_mask = np.concatenate(
        [
            np.concatenate(
                (
                    np.ones(args.fit_per_50kev_bin, dtype=bool),
                    np.zeros(args.internal_per_50kev_bin, dtype=bool),
                )
            )
            for _ in range(18)
        ]
    )
    fit_rows = selected_rows[fit_mask]
    internal_rows = selected_rows[~fit_mask]
    fit_bins = selected_bins[fit_mask]
    internal_bins = selected_bins[~fit_mask]
    if np.intersect1d(fit_rows, internal_rows).size:
        raise ValueError("Fit/internal broad Co-60 overlap")
    print(
        f"fit_co60={fit_rows.size} internal_co60={internal_rows.size}",
        flush=True,
    )

    fit_raw = load_continuum_rows(continuum_train_path, fit_rows, fit_bins)
    internal_raw = load_continuum_rows(
        continuum_train_path, internal_rows, internal_bins
    )
    fit_values, fit_qc = build_representation(fit_raw, REPRESENTATION)
    internal_values, internal_qc = build_representation(internal_raw, REPRESENTATION)
    statistics = fit_channel_statistics(fit_values)
    apply_channel_statistics(fit_values, statistics)
    apply_channel_statistics(internal_values, statistics)
    fit_weights = equal_peak_weights(fit_raw.peak_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = OneClassCompactEncoder().to(device)
    initial_embeddings = collect_embeddings(model, fit_values, args.batch_size, device)
    center = clamp_center(torch.from_numpy(np.mean(initial_embeddings, axis=0)).to(device))
    initial_diagnostics = embedding_diagnostics(initial_embeddings)
    del initial_embeddings
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = make_loader(
        fit_values, fit_weights, args.batch_size, True, args.seed
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_radius = np.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, center, loader, optimizer, device)
        validation_embeddings = collect_embeddings(
            model, internal_values, args.batch_size, device
        )
        validation_scores = np.sum(
            np.square(validation_embeddings - center.detach().cpu().numpy()[None, :]),
            axis=1,
        )
        diagnostics = embedding_diagnostics(validation_embeddings)
        radius95 = float(np.quantile(validation_scores, 0.95))
        collapsed = bool(
            diagnostics["effective_rank"] < 1.5
            or diagnostics["mean_dimension_std"] < 1.0e-4
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "internal_mean_score": float(np.mean(validation_scores)),
                "internal_score_q95": radius95,
                "embedding_diagnostics": diagnostics,
                "collapsed": collapsed,
            }
        )
        print(
            f"epoch={epoch} loss={train_loss:.6f} q95={radius95:.6f} "
            f"rank={diagnostics['effective_rank']:.3f} "
            f"std={diagnostics['mean_dimension_std']:.6f} collapsed={collapsed}",
            flush=True,
        )
        if not collapsed and radius95 < best_radius:
            best_radius = radius95
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("All epochs failed the collapse gate")
    model.load_state_dict(best_state)
    internal_scores = anomaly_scores(
        model, center, internal_values, args.batch_size, device
    )
    internal_threshold95 = float(np.quantile(internal_scores, 0.95))

    raw_heldout = load_raw_partition(heldout_csv, args.event_store_dir.resolve())
    heldout_values, heldout_qc = build_representation(raw_heldout, REPRESENTATION)
    apply_channel_statistics(heldout_values, statistics)
    heldout_scores_all = anomaly_scores(
        model, center, heldout_values, args.batch_size, device
    )
    heldout_negative_sources = np.asarray(
        [source for row in heldout_rows for source in ("positive", row["negative_source"])],
        dtype="U16",
    )
    primary_mask = (raw_heldout.labels == 1) | (
        (raw_heldout.labels == 0) & (heldout_negative_sources == "co60")
    )
    primary_labels = raw_heldout.labels[primary_mask].astype(np.int64)
    primary_scores = heldout_scores_all[primary_mask]
    primary_peaks = raw_heldout.peak_ids[primary_mask]
    primary_metrics = score_metrics(primary_labels, primary_scores, primary_peaks)
    all_negative_metrics = score_metrics(
        raw_heldout.labels.astype(np.int64), heldout_scores_all, raw_heldout.peak_ids
    )
    heldout_co60_scores = primary_scores[primary_labels == 0]
    heldout_positive_scores = primary_scores[primary_labels == 1]
    operating_point = {
        "threshold_from_internal_co60_q95": internal_threshold95,
        "heldout_co60_acceptance": float(np.mean(heldout_co60_scores <= internal_threshold95)),
        "heldout_label1_anomaly_acceptance": float(
            np.mean(heldout_positive_scores > internal_threshold95)
        ),
    }
    potential = bool(
        primary_metrics["macro_auroc"] >= 0.60
        and primary_metrics["worst_peak_auroc"] >= 0.55
        and not history[best_epoch - 1]["collapsed"]
    )

    broad_rows, broad_bins, broad_energies = select_energy_balanced_rows(
        continuum_validation_path,
        (args.validation_per_50kev_bin,) * 18,
        args.seed + 1,
    )
    broad_raw = load_continuum_rows(
        continuum_validation_path, broad_rows, broad_bins
    )
    broad_values, broad_qc = build_representation(broad_raw, REPRESENTATION)
    apply_channel_statistics(broad_values, statistics)
    broad_scores = anomaly_scores(model, center, broad_values, args.batch_size, device)
    broad_bin_quantiles = {}
    for bin_id in sorted(set(broad_bins.tolist())):
        values = broad_scores[broad_bins == bin_id]
        broad_bin_quantiles[bin_id] = {
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
        }

    checkpoint_path = output_dir / "negative_one_class_checkpoint.pt"
    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "center": center.detach().cpu(),
        "model_kind": "deep_svdd_compact_negative_only",
        "embedding_dim": 8,
        "model_width": 24,
        "representation_config": REPRESENTATION.as_dict(),
        "feature_statistics": statistics,
        "selected_epoch": best_epoch,
        "seed": args.seed,
        "training_class": "co60_negative_candidates_only",
        "opposite_class_used_for_training_or_selection": False,
        "test_partition_used": False,
    }
    torch.save(checkpoint, checkpoint_path)
    scores_path = output_dir / "held_out_anomaly_scores.npz"
    np.savez_compressed(
        scores_path,
        labels=primary_labels,
        scores=primary_scores,
        peak_ids=primary_peaks,
        all_event_labels=raw_heldout.labels.astype(np.int64),
        all_event_scores=heldout_scores_all,
        all_event_peak_ids=raw_heldout.peak_ids,
        all_event_negative_sources=heldout_negative_sources,
        broad_validation_rows=broad_rows,
        broad_validation_energies_kev=broad_energies,
        broad_validation_scores=broad_scores,
        broad_validation_bin_ids=broad_bins,
    )
    plot_path = output_dir / "held_out_anomaly_score_distribution.png"
    plot_distributions(plot_path, primary_labels, primary_scores, primary_peaks)
    energy_plot_path = output_dir / "held_out_co60_score_vs_energy.png"
    plot_energy_dependence(energy_plot_path, broad_energies, broad_scores)
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "screen_decision": "POTENTIAL_CONTINUE" if potential else "STOP_NO_DECENT_SEPARATION",
        "predeclared_potential_gate": {
            "minimum_macro_auroc": 0.60,
            "minimum_worst_peak_auroc": 0.55,
            "collapse_forbidden": True,
        },
        "primary_heldout_metrics_co60_label0_vs_all_label1": primary_metrics,
        "sensitivity_heldout_metrics_all_label0_vs_all_label1": all_negative_metrics,
        "negative_only_operating_point": operating_point,
        "training": {
            "source_population": "Co-60 100-1000 keV operational continuum",
            "selection": "deterministic equal count in eighteen 50-keV bins",
            "fit_co60_event_count": int(fit_rows.size),
            "internal_co60_event_count": int(internal_rows.size),
            "fit_per_50kev_bin": args.fit_per_50kev_bin,
            "internal_per_50kev_bin": args.internal_per_50kev_bin,
            "selected_epoch": best_epoch,
            "initial_embedding_diagnostics": initial_diagnostics,
            "history": history,
        },
        "heldout": {
            "co60_label0_count": int(np.count_nonzero(primary_labels == 0)),
            "label1_count": int(np.count_nonzero(primary_labels == 1)),
            "all_label0_count": int(np.count_nonzero(raw_heldout.labels == 0)),
            "broad_co60_validation_sample_count": int(broad_rows.size),
            "broad_co60_validation_per_50kev_bin": args.validation_per_50kev_bin,
        },
        "broad_co60_validation_score_quantiles": broad_bin_quantiles,
        "representation_config": REPRESENTATION.as_dict(),
        "feature_statistics": statistics,
        "representation_qc": {
            "fit": fit_qc,
            "internal": internal_qc,
            "heldout_all": heldout_qc,
            "broad_co60_validation": broad_qc,
        },
        "inputs": {
            "heldout_labels": {"path": relative(heldout_csv), "sha256": sha256_file(heldout_csv)},
            "split": {"path": relative(split_path), "sha256": sha256_file(split_path)},
            "continuum_train_store": {
                "path": relative(continuum_train_path),
                "sha256": sha256_file(continuum_train_path),
            },
            "continuum_validation_store": {
                "path": relative(continuum_validation_path),
                "sha256": sha256_file(continuum_validation_path),
            },
            "continuum_manifest": {
                "path": relative(continuum_manifest_path),
                "sha256": sha256_file(continuum_manifest_path),
            },
        },
        "artifacts": {},
        "scientific_boundary": (
            "Source/ROI labels are not event-level interaction truth; the held-out files are "
            "same-domain development validation, not a locked test or independent campaign."
        ),
        "opposite_class_used_for_training_or_selection": False,
        "test_partition_used": False,
        "external_data_used": False,
    }
    for path in (checkpoint_path, scores_path, plot_path, energy_plot_path):
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "negative_one_class_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["screen_decision"],
                "primary_heldout_metrics": primary_metrics,
                "operating_point": operating_point,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
