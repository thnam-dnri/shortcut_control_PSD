#!/usr/bin/env python3
"""Estimate P/B and continuum contamination inside the strict training gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815/label_pairs_train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/strict_ds_cnn_reproducibility_20260825/strict_peak_purity"
BIN_WIDTH_KEV = 0.5

PEAKS = (
    ("ba133_276kev", "Ba-133 276", "ba133", 276.146, 3.986),
    ("ba133_303kev", "Ba-133 303", "ba133", 303.139, 3.971),
    ("ba133_356kev", "Ba-133 356", "ba133", 355.709, 3.941),
    ("ba133_384kev", "Ba-133 384", "ba133", 383.978, 4.120),
    ("na22_511kev", "Na-22 511", "na22", 510.926, 4.450),
    ("cs137_662kev", "Cs-137 662", "cs137", 661.668, 3.750),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_source_files(path: Path) -> dict[str, set[Path]]:
    files: dict[str, set[Path]] = {source: set() for *_rest, source, _center, _fwhm in PEAKS}
    expected_peak_sources = {peak_id: source for peak_id, _label, source, _center, _fwhm in PEAKS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["peak_id"] not in expected_peak_sources:
                raise ValueError(f"Unexpected peak in training manifest: {row['peak_id']}")
            if row["positive_source"] != expected_peak_sources[row["peak_id"]]:
                raise ValueError(f"Positive source mismatch for {row['peak_id']}")
            positive_energy = float(row["positive_energy_kev"])
            negative_energy = float(row["negative_energy_kev"])
            if abs(positive_energy - negative_energy) >= 0.5:
                raise ValueError(
                    f"Strict 0.5-keV match failed for {row['pair_id']}: "
                    f"{positive_energy - negative_energy:.6f} keV"
                )
            files[row["positive_source"]].add(Path(row["positive_hdf5"]))
    return files


def load_source_energies(files: set[Path]) -> np.ndarray:
    values: list[np.ndarray] = []
    for path in sorted(files):
        with h5py.File(path, "r") as handle:
            values.append(np.asarray(handle["corrected_energy_kev"], dtype=np.float64))
    if not values:
        return np.empty(0, dtype=np.float64)
    energies = np.concatenate(values)
    return energies[np.isfinite(energies)]


def estimate_peak(
    energies: np.ndarray,
    center: float,
    fwhm: float,
) -> dict[str, Any]:
    bin_count = int(round(6.0 * fwhm / BIN_WIDTH_KEV))
    low = center - 3.0 * fwhm
    edges = low + np.arange(bin_count + 1, dtype=np.float64) * BIN_WIDTH_KEV
    counts, edges = np.histogram(energies, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    roi_half_width = 0.5 * fwhm
    roi = (centers >= center - roi_half_width) & (centers <= center + roi_half_width)
    sidebands = (np.abs(centers - center) >= 0.75 * fwhm) & (
        np.abs(centers - center) <= 2.0 * fwhm
    )
    gross = float(counts[roi].sum())
    sideband_count = float(counts[sidebands].sum())

    sigma = fwhm / 2.354820045

    def peak_response_model(
        values: np.ndarray,
        amplitude: float,
        tail_amplitude: float,
        tail_tau: float,
        continuum: float,
    ) -> np.ndarray:
        gaussian = amplitude * np.exp(-0.5 * ((values - center) / sigma) ** 2)
        low_energy_tail = tail_amplitude * np.exp((values - center) / tail_tau)
        low_energy_tail = np.where(values < center, low_energy_tail, 0.0)
        return gaussian + low_energy_tail + continuum

    fit_mask = np.isfinite(counts)
    far_sideband = np.abs(centers - center) >= 2.5 * fwhm
    continuum_guess = float(np.median(counts[far_sideband])) if np.any(far_sideband) else 1.0
    try:
        fit_parameters, _ = curve_fit(
            peak_response_model,
            centers[fit_mask],
            counts[fit_mask],
            p0=[float(counts.max()), float(counts.max()) * 0.01, 3.0 * fwhm, max(continuum_guess, 1.0)],
            bounds=(
                [0.0, 0.0, 0.2, 0.0],
                [float(counts.max()) * 100.0, float(counts.max()) * 100.0, 100.0, float(counts.max()) * 10.0 + 1.0],
            ),
            sigma=np.sqrt(counts[fit_mask] + 1.0),
            absolute_sigma=True,
            maxfev=300_000,
        )
        amplitude, tail_amplitude, tail_tau, continuum = map(float, fit_parameters)
        gaussian = amplitude * np.exp(-0.5 * ((centers - center) / sigma) ** 2)
        low_energy_tail = tail_amplitude * np.exp((centers - center) / tail_tau)
        low_energy_tail = np.where(centers < center, low_energy_tail, 0.0)
        fit_background = np.full_like(centers, continuum, dtype=np.float64)
        fit_response = gaussian + low_energy_tail
        fit_status = "OK"
    except (RuntimeError, ValueError) as error:
        amplitude = tail_amplitude = tail_tau = continuum = float("nan")
        fit_background = np.full_like(centers, np.nan, dtype=np.float64)
        fit_response = np.full_like(centers, np.nan, dtype=np.float64)
        fit_status = f"FAILED: {error}"

    background = float(fit_background[roi].sum())
    net = gross - background
    sideband_background = sideband_count * (fwhm / (2.5 * fwhm))
    sideband_response = float(fit_response[sidebands].sum())
    sideband_fit_background = float(fit_background[sidebands].sum())
    return {
        "strict_gate_low_kev": center - roi_half_width,
        "strict_gate_high_kev": center + roi_half_width,
        "sideband_definition": "0.75--2.0 FWHM on both sides",
        "strict_gate_gross_counts": gross,
        "sideband_counts": sideband_count,
        "sideband_background_counts": sideband_background,
        "sideband_response_counts": sideband_response,
        "sideband_fit_background_counts": sideband_fit_background,
        "sideband_response_fraction": sideband_response / sideband_count
        if sideband_count > 0
        else None,
        "estimated_background_counts": background,
        "estimated_net_peak_counts": net,
        "peak_to_background": net / background if background > 0 else None,
        "contamination_fraction": background / gross if gross > 0 else None,
        "fit_model": "fixed-center fixed-FWHM Gaussian plus low-energy exponential tail plus constant continuum",
        "fit_status": fit_status,
        "fit_continuum_counts_per_bin": continuum,
        "fit_tail_amplitude_counts_per_bin": tail_amplitude,
        "fit_tail_tau_kev": tail_tau,
        "interpretation": (
            "Continuum estimate after fitting the local photopeak response; it is "
            "not individually proven to be Compton-only."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = args.labels.resolve()
    output_dir = args.output_dir.resolve()
    if not labels.is_file():
        raise FileNotFoundError(labels)
    if "label_pairs_test" in labels.name or "test" in labels.parts:
        raise ValueError("The strict purity audit accepts only a development training manifest")

    files_by_source = load_train_source_files(labels)
    energies_by_source = {
        source: load_source_energies(files)
        for source, files in files_by_source.items()
    }
    rows: list[dict[str, Any]] = []
    for peak_id, label, source, center, fwhm in PEAKS:
        row: dict[str, Any] = {
            "peak_id": peak_id,
            "label": label,
            "source": source,
            "nominal_center_kev": center,
            "fwhm_kev": fwhm,
            "train_source_file_count": len(files_by_source[source]),
            "train_finite_energy_count": int(energies_by_source[source].size),
        }
        row.update(estimate_peak(energies_by_source[source], center, fwhm))
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "strict_peak_purity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 2,
        "status": "DEVELOPMENT_TRAINING_GATE_AUDIT",
        "labels": labels.relative_to(PROJECT_ROOT).as_posix(),
        "labels_sha256": sha256_file(labels),
        "pair_match_rule": "absolute positive-negative corrected-energy difference < 0.5 keV",
        "strict_gate_rule": "fitted center +/- 0.5 FWHM",
        "background_rule": (
            "fixed-center fixed-FWHM Gaussian plus low-energy exponential tail plus "
            "constant continuum fit in +/-3 FWHM"
        ),
        "sideband_diagnostic_rule": (
            "0.75--2.0 FWHM sideband scaling is retained diagnostically only; "
            "it overlaps the fitted photopeak response"
        ),
        "p_to_b_rule": "(gross strict-gate counts - fitted continuum) / fitted continuum",
        "contamination_rule": "estimated background / gross strict-gate counts",
        "locked_test_used": False,
        "rows": rows,
        "csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    (output_dir / "strict_peak_purity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
