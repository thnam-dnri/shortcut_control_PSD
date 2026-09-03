#!/usr/bin/env python3
"""Test whether t50/t90 scores complement t10 in its ambiguous score region."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ("t10", "t50", "t90")
PEAKS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
SEEDS = (20260822, 20260823, 20260824)
PARTITIONS = ("relaxed_file_validation", "strict_internal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t10-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--t50-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_anchor_t50_20260822",
    )
    parser.add_argument(
        "--t90-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_anchor_t90_20260822",
    )
    parser.add_argument(
        "--t10-score-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--t50-score-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/relaxed_continuum_anchor_t50_20260822",
    )
    parser.add_argument(
        "--t90-score-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/relaxed_continuum_anchor_t90_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/ds_cnn_anchor_complementarity_20260823",
    )
    parser.add_argument("--route-low", type=float, default=0.4)
    parser.add_argument("--route-high", type=float, default=0.6)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--minimum-strict-macro-gain", type=float, default=0.01)
    parser.add_argument("--maximum-per-peak-loss", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def logit_features(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    logits: dict[str, np.ndarray] = {}
    for anchor in ANCHORS:
        clipped = np.clip(scores[anchor].astype(np.float64), 1.0e-6, 1.0 - 1.0e-6)
        logits[anchor] = np.log(clipped / (1.0 - clipped))
    return {
        "t10_only": logits["t10"][:, None],
        "t10_t50_t90": np.column_stack([logits[anchor] for anchor in ANCHORS]),
    }


def equal_peak_label_weights(labels: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    weights = np.zeros(labels.size, dtype=np.float64)
    for peak in PEAKS:
        for label in (0, 1):
            mask = (peaks == peak) & (labels == label)
            count = int(np.count_nonzero(mask))
            if count == 0:
                raise ValueError(f"Missing routed cell: {peak}, label={label}")
            weights[mask] = 1.0 / count
    weights *= labels.size / np.sum(weights)
    return weights


def metric_summary(
    labels: np.ndarray, peaks: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak in PEAKS:
        mask = peaks == peak
        if np.unique(labels[mask]).size != 2:
            raise ValueError(f"Metric stratum lacks both labels: {peak}")
        per_peak[peak] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(
                average_precision_score(labels[mask], scores[mask])
            ),
            "events": int(np.count_nonzero(mask)),
            "positive_events": int(np.count_nonzero(mask & (labels == 1))),
            "negative_events": int(np.count_nonzero(mask & (labels == 0))),
        }
    peak_aurocs = [float(per_peak[peak]["auroc"]) for peak in PEAKS]
    return {
        "events": int(labels.size),
        "positive_events": int(np.count_nonzero(labels == 1)),
        "negative_events": int(np.count_nonzero(labels == 0)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "macro_peak_auroc": float(np.mean(peak_aurocs)),
        "worst_peak_auroc": float(np.min(peak_aurocs)),
        "per_peak": per_peak,
    }


def fit_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization_c: float,
    random_state: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    model = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    model.fit(scaled, labels, sample_weight=weights)
    return scaler, model


def predict_calibrator(
    scaler: StandardScaler, model: LogisticRegression, features: np.ndarray
) -> np.ndarray:
    return model.predict_proba(scaler.transform(features))[:, 1]


def model_parameters(
    scaler: StandardScaler, model: LogisticRegression
) -> dict[str, Any]:
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "intercept": model.intercept_.tolist(),
        "coefficients": model.coef_.tolist(),
        "iterations": model.n_iter_.tolist(),
    }


def load_partition(
    partition: str,
    cache_dirs: dict[str, Path],
    score_dirs: dict[str, Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    metadata: dict[str, np.ndarray] | None = None
    scores: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {"metadata": {}, "scores": {}}
    for anchor in ANCHORS:
        metadata_path = cache_dirs[anchor] / f"{partition}_metadata.npz"
        with np.load(metadata_path, allow_pickle=False) as loaded:
            current = {key: loaded[key] for key in loaded.files}
        if metadata is None:
            metadata = current
        else:
            for key in metadata:
                if key not in current or not np.array_equal(metadata[key], current[key]):
                    raise ValueError(f"Metadata mismatch: {partition}, {anchor}, {key}")
        seed_arrays = []
        seed_rows = []
        for seed in SEEDS:
            score_path = score_dirs[anchor] / f"seed_{seed}_{partition}_scores.npy"
            values = np.load(score_path, allow_pickle=False).astype(np.float64)
            seed_arrays.append(values)
            seed_rows.append(
                {
                    "seed": seed,
                    "path": score_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(score_path),
                }
            )
        if metadata is None or any(array.shape != metadata["label"].shape for array in seed_arrays):
            raise ValueError(f"Score shape mismatch: {partition}, {anchor}")
        scores[anchor] = np.mean(np.stack(seed_arrays), axis=0)
        provenance["metadata"][anchor] = {
            "path": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(metadata_path),
        }
        provenance["scores"][anchor] = seed_rows
    if metadata is None:
        raise RuntimeError("No metadata loaded")
    return metadata, scores, provenance


def correlation_summary(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    matrix = np.column_stack([scores[anchor] for anchor in ANCHORS])
    return {
        "anchors": list(ANCHORS),
        "pearson": np.corrcoef(matrix, rowvar=False).tolist(),
        "spearman": spearmanr(matrix, axis=0).statistic.tolist(),
    }


def threshold_rescue(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, int]:
    baseline_correct = (baseline >= 0.5) == labels
    candidate_correct = (candidate >= 0.5) == labels
    return {
        "baseline_errors_corrected": int(np.count_nonzero(~baseline_correct & candidate_correct)),
        "baseline_correct_introduced_errors": int(
            np.count_nonzero(baseline_correct & ~candidate_correct)
        ),
        "net_correct_at_0p5": int(np.count_nonzero(candidate_correct) - np.count_nonzero(baseline_correct)),
    }


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.route_low < args.route_high <= 1.0:
        raise ValueError("Invalid route bounds")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = {
        "t10": args.t10_cache_dir.resolve(),
        "t50": args.t50_cache_dir.resolve(),
        "t90": args.t90_cache_dir.resolve(),
    }
    score_dirs = {
        "t10": args.t10_score_dir.resolve(),
        "t50": args.t50_score_dir.resolve(),
        "t90": args.t90_score_dir.resolve(),
    }

    loaded = {
        partition: load_partition(partition, cache_dirs, score_dirs)
        for partition in PARTITIONS
    }
    routed: dict[str, dict[str, Any]] = {}
    for partition, (metadata, all_scores, provenance) in loaded.items():
        route = (all_scores["t10"] >= args.route_low) & (
            all_scores["t10"] <= args.route_high
        )
        routed[partition] = {
            "metadata": {key: values[route] for key, values in metadata.items()},
            "scores": {anchor: values[route] for anchor, values in all_scores.items()},
            "provenance": provenance,
            "total_events": int(route.size),
            "route_events": int(np.count_nonzero(route)),
            "route_fraction": float(np.mean(route)),
        }

    dev = routed["relaxed_file_validation"]
    dev_labels = dev["metadata"]["label"].astype(np.int8)
    dev_peaks = dev["metadata"]["peak_id"].astype(str)
    dev_files = dev["metadata"]["hdf5"].astype(str)
    dev_features = logit_features(dev["scores"])
    dev_weights = equal_peak_label_weights(dev_labels, dev_peaks)
    strata = np.asarray(
        [f"{peak}:{label}" for peak, label in zip(dev_peaks, dev_labels)], dtype="U48"
    )
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=20260823
    )
    fold_indices = list(
        splitter.split(dev_features["t10_only"], strata, groups=dev_files)
    )
    oof_scores = {
        name: np.full(dev_labels.size, np.nan, dtype=np.float64)
        for name in dev_features
    }
    fold_rows: list[dict[str, Any]] = []
    for fold, (fit_indices, held_indices) in enumerate(fold_indices):
        if set(dev_files[fit_indices]) & set(dev_files[held_indices]):
            raise ValueError("HDF5 file leakage across fusion folds")
        model_rows: dict[str, Any] = {}
        evaluation = {
            "t10_raw": metric_summary(
                dev_labels[held_indices],
                dev_peaks[held_indices],
                dev["scores"]["t10"][held_indices],
            )
        }
        for name, features in dev_features.items():
            scaler, model = fit_calibrator(
                features[fit_indices],
                dev_labels[fit_indices],
                dev_weights[fit_indices],
                args.regularization_c,
                20260823 + fold,
            )
            predictions = predict_calibrator(scaler, model, features[held_indices])
            oof_scores[name][held_indices] = predictions
            evaluation[name] = metric_summary(
                dev_labels[held_indices], dev_peaks[held_indices], predictions
            )
            model_rows[name] = model_parameters(scaler, model)
        fold_rows.append(
            {
                "fold": fold,
                "fit_events": int(fit_indices.size),
                "held_events": int(held_indices.size),
                "fit_files": len(set(dev_files[fit_indices])),
                "held_files": len(set(dev_files[held_indices])),
                "models": model_rows,
                "evaluation": evaluation,
            }
        )
    for name, values in oof_scores.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"Incomplete OOF scores: {name}")
        np.save(output_dir / f"relaxed_routed_{name}_oof_scores.npy", values.astype(np.float32))

    strict = routed["strict_internal"]
    strict_labels = strict["metadata"]["label"].astype(np.int8)
    strict_peaks = strict["metadata"]["peak_id"].astype(str)
    strict_features = logit_features(strict["scores"])
    strict_predictions: dict[str, np.ndarray] = {}
    final_models: dict[str, Any] = {}
    for name, features in dev_features.items():
        scaler, model = fit_calibrator(
            features,
            dev_labels,
            dev_weights,
            args.regularization_c,
            20260826,
        )
        strict_predictions[name] = predict_calibrator(
            scaler, model, strict_features[name]
        )
        np.save(
            output_dir / f"strict_routed_{name}_scores.npy",
            strict_predictions[name].astype(np.float32),
        )
        final_models[name] = model_parameters(scaler, model)

    partition_reports: dict[str, Any] = {}
    for partition in PARTITIONS:
        data = routed[partition]
        labels = data["metadata"]["label"].astype(np.int8)
        peaks = data["metadata"]["peak_id"].astype(str)
        raw_metrics = {
            anchor: metric_summary(labels, peaks, data["scores"][anchor])
            for anchor in ANCHORS
        }
        partition_reports[partition] = {
            "total_events": data["total_events"],
            "route_events": data["route_events"],
            "route_fraction": data["route_fraction"],
            "raw_anchor_metrics": raw_metrics,
            "score_correlations": correlation_summary(data["scores"]),
        }
    partition_reports["relaxed_file_validation"]["oof_fusion_metrics"] = {
        name: metric_summary(dev_labels, dev_peaks, values)
        for name, values in oof_scores.items()
    }
    partition_reports["relaxed_file_validation"]["mean_fold_macro_auroc"] = {
        name: float(
            np.mean(
                [row["evaluation"][name]["macro_peak_auroc"] for row in fold_rows]
            )
        )
        for name in ("t10_raw", "t10_only", "t10_t50_t90")
    }
    partition_reports["strict_internal"]["dev_fitted_fusion_metrics"] = {
        name: metric_summary(strict_labels, strict_peaks, values)
        for name, values in strict_predictions.items()
    }
    partition_reports["strict_internal"]["threshold_0p5_rescue"] = threshold_rescue(
        strict_labels,
        strict["scores"]["t10"],
        strict_predictions["t10_t50_t90"],
    )

    strict_baseline = partition_reports["strict_internal"]["raw_anchor_metrics"]["t10"]
    strict_fused = partition_reports["strict_internal"]["dev_fitted_fusion_metrics"]["t10_t50_t90"]
    strict_macro_gain = (
        strict_fused["macro_peak_auroc"] - strict_baseline["macro_peak_auroc"]
    )
    strict_per_peak_deltas = {
        peak: strict_fused["per_peak"][peak]["auroc"]
        - strict_baseline["per_peak"][peak]["auroc"]
        for peak in PEAKS
    }
    supported = strict_macro_gain >= args.minimum_strict_macro_gain and all(
        delta >= -args.maximum_per_peak_loss
        for delta in strict_per_peak_deltas.values()
    )
    decision = (
        "LATER_ANCHOR_COMPLEMENTARITY_SUPPORTED"
        if supported
        else "LATER_ANCHOR_COMPLEMENTARITY_NOT_SUPPORTED"
    )

    metric_rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        data_report = partition_reports[partition]
        for name, values in data_report["raw_anchor_metrics"].items():
            metric_rows.append(
                {
                    "partition": partition,
                    "method": name,
                    "macro_peak_auroc": values["macro_peak_auroc"],
                    "worst_peak_auroc": values["worst_peak_auroc"],
                    "pooled_auroc": values["pooled_auroc"],
                    **{
                        f"{peak}_auroc": values["per_peak"][peak]["auroc"]
                        for peak in PEAKS
                    },
                }
            )
        fusion_key = (
            "oof_fusion_metrics"
            if partition == "relaxed_file_validation"
            else "dev_fitted_fusion_metrics"
        )
        for name, values in data_report[fusion_key].items():
            metric_rows.append(
                {
                    "partition": partition,
                    "method": name,
                    "macro_peak_auroc": values["macro_peak_auroc"],
                    "worst_peak_auroc": values["worst_peak_auroc"],
                    "pooled_auroc": values["pooled_auroc"],
                    **{
                        f"{peak}_auroc": values["per_peak"][peak]["auroc"]
                        for peak in PEAKS
                    },
                }
            )
    with (output_dir / "anchor_complementarity_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for label, color in ((0, "tab:orange"), (1, "tab:blue")):
        mask = dev_labels == label
        axes[0, 0].hist(
            dev["scores"]["t50"][mask], bins=50, density=True, histtype="step",
            linewidth=1.6, color=color, label=f"label {label}",
        )
        axes[0, 1].hist(
            dev["scores"]["t90"][mask], bins=50, density=True, histtype="step",
            linewidth=1.6, color=color, label=f"label {label}",
        )
    axes[0, 0].set_title("t50 inside t10 [0.4, 0.6]")
    axes[0, 1].set_title("t90 inside t10 [0.4, 0.6]")
    for axis in axes[0]:
        axis.set_xlabel("Ensemble score")
        axis.set_ylabel("Density")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)

    corr = np.asarray(
        partition_reports["strict_internal"]["score_correlations"]["spearman"]
    )
    image = axes[1, 0].imshow(corr, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1, 0].set_xticks(range(3), ANCHORS)
    axes[1, 0].set_yticks(range(3), ANCHORS)
    axes[1, 0].set_title("Strict routed Spearman correlation")
    for row in range(3):
        for column in range(3):
            axes[1, 0].text(column, row, f"{corr[row, column]:.3f}", ha="center", va="center", color="white" if corr[row, column] < 0.65 else "black")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    methods = ("t10", "t50", "t90", "t10_t50_t90")
    values = []
    for name in methods:
        source = (
            partition_reports["strict_internal"]["raw_anchor_metrics"]
            if name in ANCHORS
            else partition_reports["strict_internal"]["dev_fitted_fusion_metrics"]
        )
        values.append(source[name]["macro_peak_auroc"])
    axes[1, 1].bar(range(len(methods)), values, color=("tab:blue", "tab:green", "tab:red", "tab:purple"))
    axes[1, 1].axhline(values[0], color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(range(len(methods)), ("t10", "t50", "t90", "fusion"))
    axes[1, 1].set_ylabel("Macro AUROC")
    axes[1, 1].set_title("Strict energy-matched routed events")
    axes[1, 1].grid(axis="y", alpha=0.2)
    lower = min(values) - 0.01
    upper = max(values) + 0.01
    axes[1, 1].set_ylim(max(0.45, lower), min(1.0, upper))
    figure.suptitle("t10 ambiguous-region anchor complementarity")
    figure.tight_layout()
    figure.savefig(output_dir / "anchor_complementarity.png", dpi=180)
    plt.close(figure)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "gate": {
            "primary_partition": "strict_internal routed by t10 ensemble",
            "minimum_macro_peak_auroc_gain": args.minimum_strict_macro_gain,
            "maximum_allowed_per_peak_loss": args.maximum_per_peak_loss,
            "observed_macro_peak_auroc_gain": strict_macro_gain,
            "observed_per_peak_deltas": strict_per_peak_deltas,
        },
        "contract": {
            "new_cnn_training": False,
            "anchor_seed_ensembles": list(SEEDS),
            "routing_score": "three-seed t10 ensemble",
            "route_interval": [args.route_low, args.route_high],
            "route_interval_inclusive": True,
            "fusion": "L2 logistic regression over t10/t50/t90 logits",
            "fusion_fit": "relaxed file-validation routed rows only",
            "fusion_cross_fit": f"{args.folds}-fold, complete HDF5 files grouped",
            "fusion_sample_weighting": "equal weight for each peak-by-label cell",
            "strict_internal_use": "one-time confirmation of development-fitted fusion",
            "forbidden_partitions_used": False,
        },
        "partitions": partition_reports,
        "folds": fold_rows,
        "final_models": final_models,
        "provenance": {
            partition: routed[partition]["provenance"] for partition in PARTITIONS
        },
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    relaxed_report = partition_reports["relaxed_file_validation"]
    strict_report = partition_reports["strict_internal"]
    markdown = f"""# t10/t50/t90 ambiguous-region complementarity

## Simple result

- Decision: `{decision}`
- No new CNN was trained. Existing three-seed anchor scores were averaged.
- t10 routed {relaxed_report['route_events']:,}/{relaxed_report['total_events']:,} relaxed-validation events ({100.0 * relaxed_report['route_fraction']:.2f}%) and {strict_report['route_events']:,}/{strict_report['total_events']:,} strict events ({100.0 * strict_report['route_fraction']:.2f}%).
- Strict routed t10 macro AUROC: {strict_baseline['macro_peak_auroc']:.6f}
- Strict routed t10+t50+t90 fusion macro AUROC: {strict_fused['macro_peak_auroc']:.6f}
- Gain: {strict_macro_gain:+.6f}; required: at least {args.minimum_strict_macro_gain:+.3f}.

## Strict routed result by peak

| Peak | t10 | fusion | change |
|---|---:|---:|---:|
"""
    for peak, label in zip(PEAKS, ("356 keV", "511 keV", "662 keV")):
        markdown += (
            f"| {label} | {strict_baseline['per_peak'][peak]['auroc']:.6f} | "
            f"{strict_fused['per_peak'][peak]['auroc']:.6f} | "
            f"{strict_per_peak_deltas[peak]:+.6f} |\n"
        )
    markdown += f"""

## Interpretation

The later-anchor scores are useful only if their fusion improves the strict energy-matched ambiguous subset by at least 0.01 macro AUROC and no peak loses more than 0.005. The observed result {"passes" if supported else "does not pass"} that gate. {"A separately trained Stage-2 model is justified as the next controlled experiment." if supported else "Do not train a new t50/t90 Stage-2 CNN from this hypothesis; the saved later-anchor scores do not add enough independent ranking information."}

Locked test, Th-232, and Eu-152 were not used.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"decision": decision, "strict_macro_gain": strict_macro_gain, "strict_per_peak_deltas": strict_per_peak_deltas, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
