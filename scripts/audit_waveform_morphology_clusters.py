#!/usr/bin/env python3
"""Audit frozen morphology assignments for stability and nuisance dependence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contingency(values: np.ndarray, assignments: np.ndarray) -> dict[str, dict[str, int]]:
    table = pd.crosstab(
        pd.Series(values.astype(str), name="group"),
        pd.Series(assignments, name="component"),
    )
    return {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in table.iterrows()
    }


def numeric_summary(
    values: np.ndarray,
    assignments: np.ndarray,
    components: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for component in range(components):
        selected = values[assignments == component]
        finite = selected[np.isfinite(selected)]
        result[str(component)] = {
            "count": int(finite.size),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "iqr": float(np.subtract(*np.percentile(finite, [75, 25])))
            if finite.size
            else float("nan"),
        }
    return result


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
    for path in (feature_dir, experiment_dir):
        assert_no_forbidden_path(path)
    catalogue_dir = experiment_dir / "catalogue"
    audit_dir = experiment_dir / "audit"
    if audit_dir.exists() and any(audit_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = catalogue_dir / "catalogue_report.json"
    with report_path.open() as stream:
        catalogue = json.load(stream)
    if catalogue["decision"] != "CATALOGUE_CANDIDATE_SELECTED":
        raise RuntimeError(f"Catalogue decision is {catalogue['decision']}")
    components = int(catalogue["selected_components"])
    fit = np.load(feature_dir / "fit_features.npz")
    internal = np.load(feature_dir / "internal_features.npz")
    fit_assignment = np.load(catalogue_dir / "fit_assignments.npz")
    internal_assignment = np.load(catalogue_dir / "internal_assignments.npz")
    discovery = fit["is_discovery"].astype(bool) & fit_assignment["valid"]
    internal_valid = internal_assignment["valid"].astype(bool)
    discovery_labels = fit_assignment["assignment"][discovery]
    audit_labels = internal_assignment["assignment"][internal_valid]
    discovery_fraction = (
        np.bincount(discovery_labels, minlength=components) / discovery_labels.size
    )
    internal_fraction = (
        np.bincount(audit_labels, minlength=components) / audit_labels.size
    )
    js_divergence = float(
        jensenshannon(discovery_fraction, internal_fraction, base=2.0) ** 2
    )
    selected_candidate = next(
        item
        for item in catalogue["gmm_candidates"]
        if int(item["components"]) == components
    )
    discovery_valid_fraction = float(
        np.mean(fit["valid"][fit["is_discovery"].astype(bool)])
    )
    internal_valid_fraction = float(np.mean(internal["valid"]))
    gates = {
        "discovery_valid_fraction_at_least_0_99": discovery_valid_fraction >= 0.99,
        "internal_valid_fraction_at_least_0_99": internal_valid_fraction >= 0.99,
        "component_count_2_to_8": 2 <= components <= 8,
        "seed_stability_at_least_0_80": float(
            selected_candidate["mean_pairwise_ari"]
        )
        >= 0.80,
        "minimum_discovery_fraction_at_least_0_02": float(
            np.min(discovery_fraction)
        )
        >= 0.02,
        "minimum_internal_fraction_at_least_0_01": float(np.min(internal_fraction))
        >= 0.01,
        "discovery_internal_js_at_most_0_10": js_divergence <= 0.10,
    }
    technical_pass = all(gates.values())
    nuisance: dict[str, object] = {}
    for field in ("label", "peak_id", "source", "session", "hdf5", "qc_status"):
        values = internal[field][internal_valid]
        nmi = float(normalized_mutual_info_score(values.astype(str), audit_labels))
        nuisance[field] = {
            "normalized_mutual_information": nmi,
            "contingency": contingency(values, audit_labels),
        }
    nuisance_warning = any(
        float(nuisance[field]["normalized_mutual_information"]) > 0.30
        for field in ("label", "source", "session")
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "CATALOGUE_TECHNICAL_PASS"
            if technical_pass
            else "STOP_CATALOGUE_TECHNICAL_GATE_FAILED"
        ),
        "interpretation_status": (
            "NUISANCE_DEPENDENT_CATALOGUE"
            if nuisance_warning
            else "NO_LARGE_NMI_NUISANCE_WARNING"
        ),
        "selected_components": components,
        "validity": {
            "discovery": discovery_valid_fraction,
            "internal": internal_valid_fraction,
        },
        "component_fraction": {
            "discovery": discovery_fraction.tolist(),
            "internal": internal_fraction.tolist(),
            "jensen_shannon_divergence": js_divergence,
        },
        "technical_gates": gates,
        "nuisance": nuisance,
        "numeric_by_internal_component": {
            field: numeric_summary(
                internal[field][internal_valid].astype(np.float64),
                audit_labels,
                components,
            )
            for field in ("energy_kev", "trigger_time_s", "baseline_noise_rms_adc")
        },
        "inputs": {
            "catalogue_report_sha256": sha256_file(report_path),
            "fit_features_sha256": sha256_file(feature_dir / "fit_features.npz"),
            "internal_features_sha256": sha256_file(
                feature_dir / "internal_features.npz"
            ),
        },
        "claim_boundary": (
            "Engineering conditioning may proceed after a technical pass; "
            "source/session/label dependence blocks physical-family interpretation."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (audit_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "interpretation_status": report["interpretation_status"],
                "validity": report["validity"],
                "component_fraction": report["component_fraction"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
