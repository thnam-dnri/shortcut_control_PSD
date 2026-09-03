#!/usr/bin/env python3
"""Fit a label-blind PCA/GMM morphology catalogue and HDBSCAN diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path
from src.waveform_morphology import FEATURE_NAMES

SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def robust_fit_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, (25, 75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale <= 1.0e-12)] = 1.0
    return (values - median) / scale, median, scale


def mean_pairwise_ari(assignments: list[np.ndarray]) -> float:
    if len(assignments) < 2:
        return 1.0
    return float(
        np.mean(
            [
                adjusted_rand_score(assignments[left], assignments[right])
                for left, right in combinations(range(len(assignments)), 2)
            ]
        )
    )


def ordered_probabilities(
    model: GaussianMixture,
    projected: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(projected)[:, order].astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/morphology_catalogue_20260821",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/morphology_catalogue_20260821",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (feature_dir, output_dir):
        assert_no_forbidden_path(path)
    fit_path = feature_dir / "fit_features.npz"
    internal_path = feature_dir / "internal_features.npz"
    manifest_path = feature_dir / "feature_manifest.json"
    for path in (fit_path, internal_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    catalogue_dir = output_dir / "catalogue"
    if catalogue_dir.exists() and any(catalogue_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(catalogue_dir)
    catalogue_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(fit_path)
    internal = np.load(internal_path)
    discovery_mask = fit["is_discovery"].astype(bool) & fit["valid"].astype(bool)
    discovery = fit["features"][discovery_mask].astype(np.float64)
    if discovery.shape[0] < 1000:
        raise ValueError("Insufficient valid discovery features")
    standardized, median, scale = robust_fit_transform(discovery)
    full_pca = PCA(n_components=len(FEATURE_NAMES), svd_solver="full")
    full_pca.fit(standardized)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    dimensions = int(np.clip(np.searchsorted(cumulative, 0.90) + 1, 2, 8))
    pca = PCA(n_components=dimensions, svd_solver="full")
    projected = pca.fit_transform(standardized)

    candidates: list[dict[str, object]] = []
    best_models: dict[int, GaussianMixture] = {}
    for components in range(1, 9):
        models: list[GaussianMixture] = []
        assignments: list[np.ndarray] = []
        bics: list[float] = []
        fractions: list[list[float]] = []
        for seed in SEEDS:
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=1.0e-5,
                n_init=1,
                max_iter=500,
                random_state=seed,
            )
            labels = model.fit_predict(projected)
            models.append(model)
            assignments.append(labels)
            bics.append(float(model.bic(projected)))
            fractions.append(
                (np.bincount(labels, minlength=components) / labels.size).tolist()
            )
        stability = mean_pairwise_ari(assignments)
        best_index = int(np.argmin(bics))
        best_models[components] = models[best_index]
        minimum_fraction = float(np.min(fractions[best_index]))
        eligible = minimum_fraction >= 0.02 and stability >= 0.80
        candidates.append(
            {
                "components": components,
                "best_seed": SEEDS[best_index],
                "best_bic": bics[best_index],
                "all_bic": bics,
                "mean_pairwise_ari": stability,
                "component_fractions": fractions[best_index],
                "minimum_component_fraction": minimum_fraction,
                "eligible": eligible,
            }
        )
        print(
            f"K={components} bic={bics[best_index]:.1f} "
            f"stability={stability:.4f} min_fraction={minimum_fraction:.4f} "
            f"eligible={eligible}",
            flush=True,
        )
    eligible = [item for item in candidates if bool(item["eligible"])]
    if not eligible:
        selected_components = None
        decision = "STOP_NO_STABLE_CATALOGUE"
    else:
        selected = min(eligible, key=lambda item: float(item["best_bic"]))
        selected_components = int(selected["components"])
        decision = (
            "CATALOGUE_CANDIDATE_SELECTED"
            if selected_components >= 2
            else "STOP_SINGLE_COMPONENT_CATALOGUE"
        )

    hdbscan_results: list[dict[str, object]] = []
    for min_cluster_size in (500, 1000, 1500):
        for min_samples in (20, 50):
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                n_jobs=-1,
            ).fit_predict(projected)
            clustered = labels >= 0
            hdbscan_results.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_count": int(np.unique(labels[clustered]).size),
                    "noise_fraction": float(np.mean(~clustered)),
                }
            )

    report: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "selection_is_label_blind": True,
        "feature_names": list(FEATURE_NAMES),
        "input": {
            "fit_features": fit_path.relative_to(PROJECT_ROOT).as_posix(),
            "fit_features_sha256": sha256_file(fit_path),
            "internal_features": internal_path.relative_to(PROJECT_ROOT).as_posix(),
            "internal_features_sha256": sha256_file(internal_path),
            "feature_manifest_sha256": sha256_file(manifest_path),
            "discovery_valid_event_count": int(discovery.shape[0]),
        },
        "standardizer": {
            "median": median.tolist(),
            "iqr": scale.tolist(),
        },
        "pca": {
            "selection_rule": "smallest dimension with cumulative explained variance >= 0.90, bounded 2..8",
            "selected_dimensions": dimensions,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": float(
                np.sum(pca.explained_variance_ratio_)
            ),
        },
        "gmm_candidates": candidates,
        "selected_components": selected_components,
        "hdbscan_diagnostic": hdbscan_results,
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    if selected_components is not None:
        model = best_models[selected_components]
        order = np.argsort(model.means_[:, 0])
        fit_valid = fit["valid"].astype(bool)
        internal_valid = internal["valid"].astype(bool)
        fit_projected = pca.transform(
            (fit["features"][fit_valid].astype(np.float64) - median) / scale
        )
        internal_projected = pca.transform(
            (internal["features"][internal_valid].astype(np.float64) - median)
            / scale
        )
        fit_probability = np.full(
            (fit["features"].shape[0], selected_components), np.nan, dtype=np.float32
        )
        internal_probability = np.full(
            (internal["features"].shape[0], selected_components),
            np.nan,
            dtype=np.float32,
        )
        fit_probability[fit_valid] = ordered_probabilities(
            model, fit_projected, order
        )
        internal_probability[internal_valid] = ordered_probabilities(
            model, internal_projected, order
        )
        fit_labels = np.full(fit_valid.size, -1, dtype=np.int16)
        internal_labels = np.full(internal_valid.size, -1, dtype=np.int16)
        fit_labels[fit_valid] = np.argmax(fit_probability[fit_valid], axis=1)
        internal_labels[internal_valid] = np.argmax(
            internal_probability[internal_valid], axis=1
        )
        np.savez_compressed(
            catalogue_dir / "fit_assignments.npz",
            probability=fit_probability,
            assignment=fit_labels,
            valid=fit_valid,
            original_event_index=fit["original_event_index"],
        )
        np.savez_compressed(
            catalogue_dir / "internal_assignments.npz",
            probability=internal_probability,
            assignment=internal_labels,
            valid=internal_valid,
            original_event_index=internal["original_event_index"],
        )
        joblib.dump(
            {
                "median": median,
                "scale": scale,
                "pca": pca,
                "gmm": model,
                "component_order": order,
                "feature_names": FEATURE_NAMES,
            },
            catalogue_dir / "catalogue_model.joblib",
        )
    (catalogue_dir / "catalogue_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"decision={decision}", flush=True)
    return 0 if decision == "CATALOGUE_CANDIDATE_SELECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
