from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_th232_o2_3p_energy_threshold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_th232_o2_3p_energy_threshold_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stretched_threshold_is_monotone_and_has_stable_extrapolation():
    module = _load_module()
    energy = np.linspace(0.0, 3200.0, 1000)
    threshold = module.stretched_exponential_threshold(
        energy, 0.23, 0.41, 240.0, 1.4
    )

    assert np.all(np.diff(threshold) <= 0.0)
    assert threshold[0] == threshold[np.searchsorted(energy, 100.0) - 1]
    assert threshold[-1] > 0.229
    assert threshold[-1] < 0.231


def test_fit_recovers_smooth_threshold_curve(tmp_path):
    module = _load_module()
    path = tmp_path / "threshold.csv"
    energy = np.arange(125.0, 1000.0, 50.0)
    expected = (0.23, 0.41, 240.0, 1.4)
    threshold = module.stretched_exponential_threshold(energy, *expected)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("energy_center_kev", "threshold")
        )
        writer.writeheader()
        writer.writerows(
            {
                "energy_center_kev": e,
                "threshold": t,
            }
            for e, t in zip(energy, threshold)
        )

    result = module.fit_threshold_curve(path)

    assert result["rmse"] < 1.0e-8
    np.testing.assert_allclose(result["parameter_vector"], expected, rtol=1.0e-5)


def test_admission_rejects_energy_bits_but_retains_noise_and_pulse_bits():
    module = _load_module()
    energy = np.asarray([100.0, 100.0, 100.0, np.nan], dtype=np.float32)
    shaped = np.asarray([1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    bits = np.asarray([0b11000, 0b00010, 0, 0], dtype=np.uint16)

    admitted, energy_valid, shaped_valid = module.th232_admission_mask(
        energy, shaped, bits
    )

    np.testing.assert_array_equal(admitted, [True, False, False, False])
    np.testing.assert_array_equal(energy_valid, [True, False, True, False])
    np.testing.assert_array_equal(shaped_valid, [True, True, False, True])


def test_peak_background_metric_improves_when_background_is_reduced():
    module = _load_module()
    window = module.PeakWindow(
        reference_kev=100.0,
        centroid_kev=100.0,
        sigma_kev=1.0,
        roi_low_kev=98.0,
        roi_high_kev=102.0,
        left_low_kev=95.0,
        left_high_kev=97.0,
        right_low_kev=103.0,
        right_high_kev=105.0,
    )
    first = np.full(module.ENERGY_CENTERS.size, 10.0)
    second = np.full(module.ENERGY_CENTERS.size, 5.0)
    peak = (module.ENERGY_CENTERS >= 98.0) & (module.ENERGY_CENTERS < 102.0)
    first[peak] += 20.0
    second[peak] += 20.0

    before = module.peak_background_metrics(first, window)
    after = module.peak_background_metrics(second, window)

    assert after["peak_to_background"] > before["peak_to_background"]
