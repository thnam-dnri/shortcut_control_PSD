from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "track_hdf5_gain_drift.py"
SPEC = importlib.util.spec_from_file_location("track_hdf5_gain_drift", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_lines(references: list[float], scale: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    peaks = [rng.normal(reference * scale, 2.0, 20_000) for reference in references]
    continuum = rng.uniform(0.0, max(references) * 1.12, 60_000)
    return np.concatenate([*peaks, continuum])


def test_joint_scale_seed_preserves_co60_line_identity() -> None:
    peaks = MODULE.SOURCE_PEAKS["co60"]
    energy = synthetic_lines([peak.reference_kev for peak in peaks], scale=1.06)
    scale, _, _, support = MODULE.estimate_scale(energy, peaks)
    assert scale == pytest.approx(1.06, abs=4e-4)
    assert support == 2


def test_single_line_scale_seed_uses_tight_prior() -> None:
    peaks = MODULE.SOURCE_PEAKS["cs137"]
    energy = synthetic_lines([peaks[0].reference_kev], scale=1.02)
    scale, _, _, support = MODULE.estimate_scale(energy, peaks)
    assert scale == pytest.approx(1.02, abs=4e-4)
    assert support == 1


def test_fit_peak_recovers_injected_centroid_shift() -> None:
    rng = np.random.default_rng(11)
    peak = MODULE.PeakDefinition("test", 661.657, 13.0, True)
    scale = 1.018
    energy = np.concatenate(
        [rng.normal(peak.reference_kev * scale, 2.2, 25_000), rng.uniform(630.0, 690.0, 20_000)]
    )
    result = MODULE.fit_peak(energy, peak, scale, 0.5)
    assert result["fit_status"] in {"OK", "WARN"}
    assert result["fitted_centroid_kev"] == pytest.approx(peak.reference_kev * scale, abs=0.15)
    assert result["centroid_residual_kev"] == pytest.approx(peak.reference_kev * 0.018, abs=0.15)


def test_continuum_only_rejected_as_scale_seed() -> None:
    peaks = MODULE.SOURCE_PEAKS["co60"]
    energy = np.linspace(0.0, 1_500.0, 150_001)
    with pytest.raises(RuntimeError, match="not supported"):
        MODULE.estimate_scale(energy, peaks)


def test_peak_baselines_are_line_specific() -> None:
    rows = [
        {"source": "co60", "peak_name": "co60_1173", "hdf5_file": "a.h5", "acquisition_time": "1", "fit_status": "OK", "fitted_centroid_kev": 1173.0},
        {"source": "co60", "peak_name": "co60_1332", "hdf5_file": "a.h5", "acquisition_time": "1", "fit_status": "OK", "fitted_centroid_kev": 1332.0},
        {"source": "co60", "peak_name": "co60_1173", "hdf5_file": "b.h5", "acquisition_time": "2", "fit_status": "OK", "fitted_centroid_kev": 1174.173},
        {"source": "co60", "peak_name": "co60_1332", "hdf5_file": "b.h5", "acquisition_time": "2", "fit_status": "OK", "fitted_centroid_kev": 1334.664},
    ]
    MODULE.apply_peak_baselines(rows)
    assert rows[2]["peak_baseline_centroid_kev"] == 1173.0
    assert rows[3]["peak_baseline_centroid_kev"] == 1332.0
    assert rows[2]["relative_to_peak_baseline_kev"] == pytest.approx(1.173)
    assert rows[3]["relative_to_peak_baseline_kev"] == pytest.approx(2.664)


def test_centroid_is_stable_under_modest_window_change() -> None:
    rng = np.random.default_rng(19)
    reference = 1274.537
    energy = np.concatenate(
        [rng.normal(reference, 3.0, 30_000), rng.uniform(reference - 30.0, reference + 30.0, 30_000)]
    )
    narrow = MODULE.fit_peak(energy, MODULE.PeakDefinition("narrow", reference, 16.0, True), 1.0, 0.5)
    wide = MODULE.fit_peak(energy, MODULE.PeakDefinition("wide", reference, 22.0, True), 1.0, 0.5)
    difference = abs(narrow["fitted_centroid_kev"] - wide["fitted_centroid_kev"])
    assert difference <= 0.25 * narrow["fwhm_kev"]
