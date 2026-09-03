#!/usr/bin/env python3
"""Plot one central discovery waveform and summarize class balance per group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import (
    BASELINE_STOP,
    SAMPLE_PERIOD_NS,
    gather_window,
    moving_average,
    t10_anchor,
)
from src.data_access_guards import assert_no_forbidden_path
from src.waveform_morphology import MorphologyConfig


def load_waveform(path: str, row: int) -> np.ndarray:
    source = Path(path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    assert_no_forbidden_path(source)
    with h5py.File(source, "r") as handle:
        return np.asarray(handle["waveform"][row], dtype=np.float32)


def plot_representation(
    acquired: np.ndarray,
    config: MorphologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = -acquired[None, :]
    baseline = np.median(positive[:, :BASELINE_STOP], axis=1).astype(np.float32)
    charge = moving_average(positive - baseline[:, None], config.moving_average)
    current = np.gradient(charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
    anchors, fallback = t10_anchor(charge)
    if fallback:
        raise ValueError("Representative waveform required a t10 fallback")
    charge_window = gather_window(
        charge, anchors, config.pre_samples, config.post_samples
    )[0]
    current_window = gather_window(
        current, anchors, config.pre_samples, config.post_samples
    )[0]
    charge_scale = float(np.max(np.abs(charge_window)))
    current_scale = float(np.max(np.maximum(current_window, 0.0)))
    if charge_scale <= 0 or current_scale <= 0:
        raise ValueError("Invalid representative waveform scale")
    time_ns = (
        np.arange(config.window_length, dtype=np.float32) - config.pre_samples
    ) * SAMPLE_PERIOD_NS
    return time_ns, charge_window / charge_scale, current_window / current_scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    feature_dir = args.feature_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = experiment_dir / "group_waveforms"
    for path in (feature_dir, experiment_dir, output_dir):
        assert_no_forbidden_path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(feature_dir / "fit_features.npz")
    internal = np.load(feature_dir / "internal_features.npz")
    fit_assign = np.load(experiment_dir / "catalogue/fit_assignments.npz")
    internal_assign = np.load(experiment_dir / "catalogue/internal_assignments.npz")
    components = fit_assign["probability"].shape[1]
    discovery = (
        fit["is_discovery"].astype(bool)
        & fit["valid"].astype(bool)
        & fit_assign["valid"].astype(bool)
    )
    internal_valid = (
        internal["valid"].astype(bool) & internal_assign["valid"].astype(bool)
    )
    config = MorphologyConfig()
    representatives: list[dict[str, object]] = []
    traces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    summaries: list[dict[str, object]] = []

    for component in range(components):
        candidates = np.flatnonzero(
            discovery & (fit_assign["assignment"] == component)
        )
        if candidates.size == 0:
            raise ValueError(f"No discovery rows for component {component}")
        representative = int(
            candidates[
                np.argmax(fit_assign["probability"][candidates, component])
            ]
        )
        waveform = load_waveform(
            str(fit["hdf5"][representative]),
            int(fit["source_row"][representative]),
        )
        time_ns, charge, current = plot_representation(waveform, config)
        traces.append((time_ns, charge, current))

        mask = internal_valid & (internal_assign["assignment"] == component)
        labels = internal["label"][mask].astype(np.int8)
        positive = int(np.count_nonzero(labels == 1))
        negative = int(np.count_nonzero(labels == 0))
        total = positive + negative
        dominant = "positive" if positive > negative else "negative"
        source_counts = pd.Series(internal["source"][mask]).value_counts()
        peak_counts = pd.Series(internal["peak_id"][mask]).value_counts()
        summary = {
            "group": component + 1,
            "internal_events": total,
            "positive_events": positive,
            "negative_events": negative,
            "positive_percent": 100.0 * positive / total,
            "negative_percent": 100.0 * negative / total,
            "dominant_class": dominant,
            "top_source": str(source_counts.index[0]),
            "top_source_events": int(source_counts.iloc[0]),
            "top_peak": str(peak_counts.index[0]),
            "top_peak_events": int(peak_counts.iloc[0]),
            "representative_posterior": float(
                fit_assign["probability"][representative, component]
            ),
            "representative_source": str(fit["source"][representative]),
            "representative_energy_kev": float(fit["energy_kev"][representative]),
            "representative_hdf5": str(fit["hdf5"][representative]),
            "representative_row": int(fit["source_row"][representative]),
        }
        summaries.append(summary)
        representatives.append(
            {
                "group": component + 1,
                "fit_cache_index": representative,
                **summary,
            }
        )

        figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        raw_time = np.arange(waveform.size) * SAMPLE_PERIOD_NS
        axes[0].plot(raw_time, waveform, linewidth=0.8)
        axes[0].set_ylabel("Acquired ADC")
        axes[0].set_title(
            f"Group {component + 1}: central discovery waveform | "
            f"internal positive {summary['positive_percent']:.1f}% / "
            f"negative {summary['negative_percent']:.1f}%"
        )
        axes[1].plot(time_ns, charge, linewidth=1.0)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=0.7)
        axes[1].set_ylabel("MA20 charge\n(normalized)")
        axes[2].plot(time_ns, current, linewidth=1.0)
        axes[2].axvline(0, color="black", linestyle="--", linewidth=0.7)
        axes[2].set_ylabel("MA20 current\n(normalized)")
        axes[2].set_xlabel("Time relative to t10 (ns)")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.savefig(
            output_dir / f"group_{component + 1}_representative.png", dpi=160
        )
        plt.close(figure)

    figure, axes = plt.subplots(
        components, 1, figsize=(11, 2.25 * components), sharex=True,
        constrained_layout=True,
    )
    for component, (axis, trace, summary) in enumerate(
        zip(axes, traces, summaries, strict=True)
    ):
        time_ns, _charge, current = trace
        axis.plot(time_ns, current, color=f"C{component % 10}", linewidth=1.1)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.6)
        axis.set_ylabel(f"Group {component + 1}")
        axis.set_title(
            f"n={summary['internal_events']:,}; "
            f"positive={summary['positive_percent']:.1f}%, "
            f"negative={summary['negative_percent']:.1f}%"
        )
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time relative to t10 (ns)")
    figure.suptitle(
        "Central representative MA20 current waveform for each empirical group",
        fontsize=14,
    )
    figure.savefig(experiment_dir / "representative_waveforms.png", dpi=180)
    plt.close(figure)

    pd.DataFrame(summaries).to_csv(
        experiment_dir / "group_summary.csv", index=False
    )
    (experiment_dir / "group_summary.json").write_text(
        json.dumps(
            {
                "representative_definition": (
                    "Highest frozen GMM posterior among valid discovery rows "
                    "assigned to the component"
                ),
                "class_summary_population": (
                    "All valid internal-audit waveform rows; labels are post-hoc"
                ),
                "groups": representatives,
                "test_partition_used": False,
                "th232_used": False,
                "eu152_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
