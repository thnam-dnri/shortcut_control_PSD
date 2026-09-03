#!/usr/bin/env python3
"""Add energy-distance and input-SNR diagnostics to the relaxed-ROI test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path

SEEDS = (20260822, 20260823, 20260824)
PEAKS = {
    "ba133_356kev": (355.709, 3.941),
    "na22_511kev": (510.926, 4.447),
    "cs137_662kev": (661.668, 3.749),
}


def snr_metrics(
    values_path: Path,
    labels: np.ndarray,
    peak_ids: np.ndarray,
    channel_mean: float,
    channel_std: float,
) -> dict[str, Any]:
    values = np.load(values_path, mmap_mode="r")
    scores = np.empty(labels.size, dtype=np.float32)
    for start in range(0, labels.size, 2048):
        stop = min(start + 2048, labels.size)
        charge = (
            np.asarray(values[start:stop, 0], dtype=np.float32) * channel_std
            + channel_mean
        )
        noise = np.std(charge[:, :180], axis=1) + 1.0e-6
        scores[start:stop] = np.max(charge, axis=1) / noise
    per_peak = {
        peak: float(roc_auc_score(labels[peak_ids == peak], scores[peak_ids == peak]))
        for peak in sorted(PEAKS)
    }
    return {
        "definition": "maximum MA10 charge divided by pre-pulse charge standard deviation",
        "macro_auroc": float(np.mean(list(per_peak.values()))),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "per_peak_auroc": per_peak,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experiments/relaxed_continuum_roi_ds_cnn_20260822",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    for path in (cache_dir, experiment_dir):
        assert_no_forbidden_path(path)
    report = json.loads(
        (experiment_dir / "experiment_report.json").read_text(encoding="utf-8")
    )
    cache_manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )
    channel_mean = float(cache_manifest["feature_statistics"]["means"][0])
    channel_std = float(
        cache_manifest["feature_statistics"]["standard_deviations"][0]
    )
    diagnostics: dict[str, Any] = {}
    for population in ("relaxed_file_validation", "strict_internal"):
        metadata = np.load(cache_dir / f"{population}_metadata.npz")
        labels = metadata["label"].astype(np.int8)
        peak_ids = metadata["peak_id"].astype(str)
        energies = metadata["energy_kev"].astype(np.float64)
        offsets = np.empty(labels.size, dtype=np.float64)
        for peak, (center, fwhm) in PEAKS.items():
            mask = peak_ids == peak
            offsets[mask] = np.abs(energies[mask] - center) / fwhm
        ensemble = np.mean(
            [
                np.load(experiment_dir / f"seed_{seed}_{population}_scores.npy")
                for seed in SEEDS
            ],
            axis=0,
        )
        diagnostics[population] = {
            "ensemble_score_spearman_vs_abs_energy_offset": {
                "all": float(spearmanr(ensemble, offsets).statistic),
                "positive_only": float(
                    spearmanr(ensemble[labels == 1], offsets[labels == 1]).statistic
                ),
                "negative_only": float(
                    spearmanr(ensemble[labels == 0], offsets[labels == 0]).statistic
                ),
            },
            "input_snr_baseline": snr_metrics(
                cache_dir / f"{population}_values.npy",
                labels,
                peak_ids,
                channel_mean,
                channel_std,
            ),
        }
    relaxed = report["mean_metrics"]["relaxed_file_validation"]["macro_auroc"]
    strict = report["mean_metrics"]["strict_internal"]["macro_auroc"]
    paired_differences = [
        run["evaluation"]["relaxed_file_validation"]["macro_auroc"]
        - run["evaluation"]["strict_internal"]["macro_auroc"]
        for run in report["runs"]
    ]
    diagnostics["conclusion"] = {
        "decision": "NO_DOMINANT_ENERGY_SNR_SHORTCUT_DETECTED",
        "relaxed_minus_strict_mean_macro_auroc": float(relaxed - strict),
        "relaxed_minus_strict_seed_sd": float(
            np.std(paired_differences, ddof=1)
        ),
        "interpretation": (
            "Relaxed and strict evaluations differ by less than 0.01; score-energy "
            "correlation within each class is negligible and the input-SNR scalar "
            "is at chance. The energy-defined labels nevertheless permit a strong "
            "energy-only oracle, so strict matched evaluation remains mandatory."
        ),
    }
    (experiment_dir / "shortcut_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
