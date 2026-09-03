#!/usr/bin/env python3
"""Compare conventional scalar HPGe A/E with the frozen DS-CNN.

The primary scalar follows the experiment's A/E-like definition:

    A/E_waveform = max_t I(t) / max_t Q(t)

where the acquired negative-polarity waveform is converted to positive
polarity, baseline-subtracted, causally MA10-smoothed, and differentiated.
The search interval is samples [1000, 2000).  A second scalar using the stored
positive shaped-energy quantity is reported as a denominator sensitivity
check.  Neither scalar is fitted or selected on the held-out data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (  # noqa: E402
    BASELINE_STOP,
    SAMPLE_PERIOD_NS,
    build_representation,
    load_raw_partition,
    moving_average,
)
from src.cascade_refinement import sha256_file  # noqa: E402
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402


EXPERIMENT_ID = "traditional_ae_vs_ds_cnn_20260821"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_CNN_SCORES = (
    PROJECT_ROOT
    / "outputs/experiments/cascaded_ambiguous_refinement_ds_cnn_20260821/held_out/held_out_scores.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments" / EXPERIMENT_ID
CHUNK_SIZE = 512
SEARCH_START = 1000
SEARCH_STOP = 2000
MOVING_AVERAGE = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_cnn_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        names = set(source.files)
        score_key = "stage1_scores" if "stage1_scores" in names else "ds_cnn_scores"
        required = {"labels", "peak_ids", "weights", score_key}
        if not required.issubset(names):
            raise ValueError(f"CNN score artifact lacks keys: {sorted(required - names)}")
        return {
            "labels": np.asarray(source["labels"], dtype=np.float32),
            "peak_ids": np.asarray(source["peak_ids"]),
            "weights": np.asarray(source["weights"], dtype=np.float64),
            "cnn_scores": np.asarray(source[score_key], dtype=np.float64),
            "score_key": np.asarray(score_key),
        }


def infer_peak_weights(peak_ids: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    peak_ids = np.asarray(peak_ids)
    weights = np.asarray(weights, dtype=np.float64)
    if peak_ids.ndim != 1 or weights.shape != peak_ids.shape or peak_ids.size % 2:
        raise ValueError("CNN peak IDs and weights must be matching paired arrays")
    if not np.array_equal(peak_ids[::2], peak_ids[1::2]):
        raise ValueError("Positive and negative pair members have different peak IDs")
    if not np.allclose(weights[::2], weights[1::2], rtol=0.0, atol=1.0e-12):
        raise ValueError("Positive and negative pair members have different weights")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("CNN event weights must be finite and positive")
    totals = {
        str(peak): float(np.sum(weights[::2][peak_ids[::2] == peak]))
        for peak in sorted(set(peak_ids.tolist()))
    }
    scale = sum(totals.values())
    if not scale > 0.0:
        raise ValueError("CNN peak weights have zero total")
    return {peak: value / scale for peak, value in totals.items()}


def compute_ae_scalars(
    waveforms: np.ndarray,
    shaped_energy: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    waveforms = np.asarray(waveforms, dtype=np.float32)
    shaped_energy = np.asarray(shaped_energy, dtype=np.float32)
    if waveforms.ndim != 2 or waveforms.shape[1] != 4500:
        raise ValueError(f"Expected [N,4500] waveforms, got {waveforms.shape}")
    if shaped_energy.shape != (waveforms.shape[0],):
        raise ValueError("Shaped-energy shape does not match waveforms")
    peak_current = np.empty(waveforms.shape[0], dtype=np.float64)
    peak_charge = np.empty(waveforms.shape[0], dtype=np.float64)
    for start in range(0, waveforms.shape[0], chunk_size):
        stop = min(start + chunk_size, waveforms.shape[0])
        positive = -waveforms[start:stop]
        baseline = np.median(positive[:, :BASELINE_STOP], axis=1).astype(np.float32)
        charge = moving_average(positive - baseline[:, None], MOVING_AVERAGE)
        current = np.gradient(charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
        peak_current[start:stop] = np.max(
            current[:, SEARCH_START:SEARCH_STOP], axis=1
        )
        peak_charge[start:stop] = np.max(
            charge[:, SEARCH_START:SEARCH_STOP], axis=1
        )
    if not np.all(np.isfinite(peak_current)) or not np.all(np.isfinite(peak_charge)):
        raise ValueError("Nonfinite current or charge peak in A/E calculation")
    if not np.all(np.isfinite(shaped_energy)) or np.any(shaped_energy <= 0.0):
        raise ValueError("Invalid shaped energy in A/E calculation")
    valid_waveform_ae = np.isfinite(peak_charge) & (peak_charge > 0.0)
    ae_waveform = np.full(peak_charge.shape, np.nan, dtype=np.float64)
    ae_waveform[valid_waveform_ae] = (
        peak_current[valid_waveform_ae] / peak_charge[valid_waveform_ae]
    )
    return {
        "peak_current": peak_current,
        "peak_charge": peak_charge,
        "ae_waveform": ae_waveform,
        "ae_shaped_energy": peak_current / shaped_energy.astype(np.float64),
        "valid_waveform_ae": valid_waveform_ae,
    }


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    peak_ids = np.asarray(peak_ids)
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(peak_ids.tolist())):
        mask = peak_ids == peak_id
        if np.unique(labels[mask]).size < 2:
            raise ValueError(f"Peak stratum lacks both classes: {peak_id}")
        per_peak[str(peak_id)] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
            "positive_count": int(np.count_nonzero(mask & (labels == 1.0))),
            "negative_count": int(np.count_nonzero(mask & (labels == 0.0))),
        }
    aurocs = [float(value["auroc"]) for value in per_peak.values()]
    high_auroc = float(roc_auc_score(labels, scores))
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "worst_peak_auroc": float(np.min(aurocs)),
        "pooled_auroc": high_auroc,
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "orientation_invariant_pooled_auroc": float(max(high_auroc, 1.0 - high_auroc)),
        "per_peak": per_peak,
    }


def distribution_statistics(
    labels: np.ndarray,
    scores: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in [("overall", np.ones(labels.size, dtype=bool))] + [
        (str(peak), peak_ids == peak) for peak in sorted(set(peak_ids.tolist()))
    ]:
        result[name] = {}
        for label, label_name in ((0.0, "negative"), (1.0, "positive")):
            values = scores[mask & (labels == label)]
            result[name][label_name] = {
                "event_count": int(values.size),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            }
    return result


def plot_distributions(
    output_path: Path,
    labels: np.ndarray,
    ae_scores: np.ndarray,
    cnn_scores: np.ndarray,
    peak_ids: np.ndarray,
    ae_metrics: dict[str, Any],
    cnn_metrics: dict[str, Any],
) -> None:
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    groups = [("overall", np.ones(labels.size, dtype=bool), "All peaks")]
    groups.extend(
        (str(peak), peak_ids == peak, str(peak).replace("_", " "))
        for peak in sorted(set(peak_ids.tolist()))
    )
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for column, (name, mask, title) in enumerate(groups):
        ae_values = ae_scores[mask]
        cnn_values = cnn_scores[mask]
        ae_low, ae_high = np.quantile(ae_values, [0.005, 0.995])
        if not ae_high > ae_low:
            ae_low, ae_high = float(np.min(ae_values)), float(np.max(ae_values) + 1.0)
        ae_bins = np.linspace(ae_low, ae_high, 65)
        cnn_bins = np.linspace(0.0, 1.0, 65)
        for label, color, label_name in (
            (0.0, "#d95f02", "negative"),
            (1.0, "#1b9e77", "positive"),
        ):
            selection = mask & (labels == label)
            axes[0, column].hist(
                ae_scores[selection],
                bins=ae_bins,
                density=True,
                alpha=0.48,
                color=color,
                label=label_name,
            )
            axes[1, column].hist(
                cnn_scores[selection],
                bins=cnn_bins,
                density=True,
                alpha=0.48,
                color=color,
                label=label_name,
            )
        peak_key = "overall" if name == "overall" else name
        ae_value = (
            ae_metrics["pooled_auroc"]
            if name == "overall"
            else ae_metrics["per_peak"][peak_key]["auroc"]
        )
        cnn_value = (
            cnn_metrics["pooled_auroc"]
            if name == "overall"
            else cnn_metrics["per_peak"][peak_key]["auroc"]
        )
        axes[0, column].set_title(f"{title}\nA/E AUROC={ae_value:.4f}")
        axes[1, column].set_title(f"{title}\nDS-CNN AUROC={cnn_value:.4f}")
        axes[0, column].set_xlabel("max(I) / max(Q)")
        axes[1, column].set_xlabel("DS-CNN score")
        axes[0, column].set_ylabel("Density")
        axes[1, column].set_ylabel("Density")
        axes[0, column].legend(fontsize=8)
        axes[1, column].legend(fontsize=8)
    figure.suptitle("Traditional HPGe A/E versus frozen DS-CNN on held-out events", fontsize=15)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_rocs(
    output_path: Path,
    labels: np.ndarray,
    ae_scores: np.ndarray,
    shaped_ae_scores: np.ndarray,
    cnn_scores: np.ndarray,
    peak_ids: np.ndarray,
    ae_metrics: dict[str, Any],
    shaped_metrics: dict[str, Any],
    cnn_metrics: dict[str, Any],
) -> None:
    groups = [("overall", np.ones(labels.size, dtype=bool), "All peaks")]
    groups.extend(
        (str(peak), peak_ids == peak, str(peak).replace("_", " "))
        for peak in sorted(set(peak_ids.tolist()))
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (name, mask, title) in zip(axes.flat, groups):
        for scores, metrics, color, label in (
            (ae_scores, ae_metrics, "#d95f02", "waveform A/E"),
            (shaped_ae_scores, shaped_metrics, "#7570b3", "max(I)/shaped energy"),
            (cnn_scores, cnn_metrics, "#1b9e77", "DS-CNN"),
        ):
            fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
            value = metrics["pooled_auroc"] if name == "overall" else metrics["per_peak"][name]["auroc"]
            axis.plot(fpr, tpr, linewidth=1.8, color=color, label=f"{label} ({value:.4f})")
        axis.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(alpha=0.25)
    figure.suptitle("Held-out ROC comparison: traditional A/E versus DS-CNN", fontsize=15)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_metrics_csv(path: Path, metrics_by_method: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for method, metrics in metrics_by_method.items():
        for peak, values in metrics["per_peak"].items():
            rows.append(
                {
                    "method": method,
                    "peak_id": peak,
                    "auroc": values["auroc"],
                    "average_precision": values["average_precision"],
                    "event_count": values["event_count"],
                    "positive_count": values["positive_count"],
                    "negative_count": values["negative_count"],
                }
            )
        rows.append(
            {
                "method": method,
                "peak_id": "overall",
                "auroc": metrics["pooled_auroc"],
                "average_precision": metrics["pooled_average_precision"],
                "event_count": "",
                "positive_count": "",
                "negative_count": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--cnn-scores", type=Path, default=DEFAULT_CNN_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    cnn_scores_path = args.cnn_scores.resolve()
    output_dir = args.output_dir.resolve()
    validation_csv = labels_dir / "label_pairs_validation.csv"
    for path in (labels_dir, event_store_dir, validation_csv, cnn_scores_path):
        if not path.exists():
            raise FileNotFoundError(path)
    assert_no_forbidden_path(validation_csv)
    assert_development_csv(validation_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn = load_cnn_scores(cnn_scores_path)
    raw = load_raw_partition(validation_csv, event_store_dir)
    if not (
        raw.labels.shape == cnn["labels"].shape == cnn["peak_ids"].shape == cnn["cnn_scores"].shape
    ):
        raise ValueError("CNN score and held-out waveform arrays have inconsistent shapes")
    if not np.array_equal(raw.labels, cnn["labels"]):
        raise ValueError("CNN score labels do not match validation waveform labels")
    if not np.array_equal(raw.peak_ids, cnn["peak_ids"]):
        raise ValueError("CNN score peak IDs do not match validation waveform peak IDs")
    if not np.all(np.isfinite(cnn["cnn_scores"])):
        raise ValueError("CNN scores contain nonfinite values")
    weights = cnn["weights"].astype(np.float64, copy=False)
    selected_peak_weights = infer_peak_weights(raw.peak_ids, weights)

    print("computing conventional scalar A/E on held-out waveforms", flush=True)
    scalars = compute_ae_scalars(raw.waveforms, raw.shaped_energy, args.chunk_size)
    ae_valid = scalars["valid_waveform_ae"]
    ae_valid_scores = scalars["ae_waveform"]
    if not np.any(ae_valid):
        raise ValueError("No valid waveform A/E values")
    # A conventional A/E calculation cannot assign a finite value when the
    # waveform has no positive charge maximum in the search window.  Retain
    # those events and assign the lowest A/E score, while reporting a separate
    # common-valid-event metric so this policy is explicit.
    ae_scores = np.where(ae_valid, ae_valid_scores, 0.0)
    shaped_ae_scores = scalars["ae_shaped_energy"]
    labels = raw.labels
    peak_ids = raw.peak_ids
    ae_metrics = metric_summary(labels, ae_scores, weights, peak_ids)
    ae_valid_metrics = metric_summary(
        labels[ae_valid],
        ae_valid_scores[ae_valid],
        weights[ae_valid],
        peak_ids[ae_valid],
    )
    shaped_metrics = metric_summary(labels, shaped_ae_scores, weights, peak_ids)
    cnn_metrics = metric_summary(labels, cnn["cnn_scores"], weights, peak_ids)
    metrics_by_method = {
        "waveform_A_over_Q": ae_metrics,
        "max_I_over_shaped_energy": shaped_metrics,
        "frozen_DS_CNN": cnn_metrics,
    }
    print(
        json.dumps(
            {
                method: {
                    "macro_auroc": values["macro_auroc"],
                    "worst_peak_auroc": values["worst_peak_auroc"],
                    "pooled_auroc": values["pooled_auroc"],
                }
                for method, values in metrics_by_method.items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    np.savez_compressed(
        output_dir / "held_out_ae_scores.npz",
        labels=labels,
        peak_ids=peak_ids,
        weights=weights,
        peak_current=scalars["peak_current"],
        peak_charge=scalars["peak_charge"],
        waveform_ae_valid=ae_valid,
        shaped_energy=raw.shaped_energy,
        waveform_ae=ae_scores,
        shaped_energy_ae=shaped_ae_scores,
        cnn_scores=cnn["cnn_scores"],
    )
    distribution_path = output_dir / "ae_vs_cnn_distributions.png"
    roc_path = output_dir / "ae_vs_cnn_roc.png"
    plot_distributions(
        distribution_path,
        labels,
        ae_scores,
        cnn["cnn_scores"],
        peak_ids,
        ae_metrics,
        cnn_metrics,
    )
    plot_rocs(
        roc_path,
        labels,
        ae_scores,
        shaped_ae_scores,
        cnn["cnn_scores"],
        peak_ids,
        ae_metrics,
        shaped_metrics,
        cnn_metrics,
    )
    metrics_csv = output_dir / "ae_vs_cnn_metrics.csv"
    write_metrics_csv(metrics_csv, metrics_by_method)

    positive = labels == 1.0
    spearman = {
        "waveform_A_over_Q_vs_cnn": float(
            spearmanr(ae_scores, cnn["cnn_scores"]).statistic
        ),
        "max_I_over_shaped_energy_vs_cnn": float(
            spearmanr(shaped_ae_scores, cnn["cnn_scores"]).statistic
        ),
    }
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "HELD_OUT_COMPARISON",
        "partition": "same-domain development held-out validation file partition",
        "event_count": int(labels.size),
        "pair_count": int(labels.size // 2),
        "input": {
            "validation_csv": relative(validation_csv),
            "validation_csv_sha256": sha256_file(validation_csv),
            "event_store_dir": relative(event_store_dir),
            "cnn_score_artifact": relative(cnn_scores_path),
            "cnn_score_artifact_sha256": sha256_file(cnn_scores_path),
            "cnn_score_key": str(cnn["score_key"]),
        },
        "a_over_e_definition": {
            "primary_name": "waveform_A_over_Q",
            "formula": "max(I(t)) / max(Q(t))",
            "waveform_polarity": "negative acquired polarity converted to positive",
            "baseline": "median samples 0:1000",
            "moving_average": "causal MA10",
            "current": "dQ/dt with 4 ns sample period",
            "search_samples": [SEARCH_START, SEARCH_STOP],
            "denominator_sensitivity": "max(I(t)) / shaped_energy_unit",
            "higher_score_means_label_1": True,
        },
        "selected_peak_weights": selected_peak_weights,
        "metrics": metrics_by_method,
        "waveform_A_over_Q_valid_only_metrics": ae_valid_metrics,
        "invalid_waveform_A_over_Q": {
            "event_count": int(np.count_nonzero(~ae_valid)),
            "fraction": float(np.mean(~ae_valid)),
            "policy_for_all_event_comparison": "retain event and assign A/E score 0.0, the lowest score",
            "label_counts": {
                "label_0": int(np.count_nonzero((~ae_valid) & (labels == 0.0))),
                "label_1": int(np.count_nonzero((~ae_valid) & (labels == 1.0))),
            },
        },
        "distribution_statistics": {
            "waveform_A_over_Q": distribution_statistics(labels, ae_scores, peak_ids),
            "max_I_over_shaped_energy": distribution_statistics(
                labels, shaped_ae_scores, peak_ids
            ),
            "frozen_DS_CNN": distribution_statistics(
                labels, cnn["cnn_scores"], peak_ids
            ),
        },
        "rank_correlation_with_cnn": spearman,
        "held_out_scores_used_for_AE_selection": False,
        "test_partition_used": False,
        "th232_used": False,
        "scientific_boundary": (
            "This compares a conventional scalar A/E-like waveform discriminator "
            "with the frozen DS-CNN on the same-domain development holdout. The "
            "labels are source/energy-matched proxy labels under the existing "
            "scalar-shortcut warning, not independent interaction truth."
        ),
        "artifacts": {},
    }
    score_path = output_dir / "held_out_ae_scores.npz"
    for path in (score_path, distribution_path, roc_path, metrics_csv):
        report["artifacts"][path.name] = {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
    report_path = output_dir / "ae_vs_cnn_report.json"
    report["artifacts"][report_path.name] = {"path": relative(report_path)}
    save_json(report_path, report)
    print(f"distribution_plot={relative(distribution_path)}", flush=True)
    print(f"roc_plot={relative(roc_path)}", flush=True)
    print(f"report={relative(report_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
