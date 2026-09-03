from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/plot_th232_shared_exponential_spectra.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_th232_shared_exponential_spectra_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_threshold_function_uses_target_constant_only():
    module = _load_module()
    fit_summary = {
        "origin_kev": 238.903,
        "amplitude": 0.175,
        "tau_kev": 89.0,
        "constants": {"5": 0.27, "10": 0.33, "20": 0.40, "30": 0.46, "45": 0.54},
    }
    energy = np.asarray([238.903, 500.0, 1460.0, 2614.5])
    low_target = module.shared_thresholds(energy, fit_summary, 5)
    high_target = module.shared_thresholds(energy, fit_summary, 45)

    assert np.all(np.isfinite(low_target))
    assert np.allclose(high_target - low_target, 0.27)
    assert np.all((low_target >= 0.0) & (low_target <= 1.0))


def test_shared_threshold_function_has_low_energy_plateau_from_fit_origin():
    module = _load_module()
    fit_summary = {
        "origin_kev": 230.0,
        "amplitude": 0.2,
        "tau_kev": 100.0,
        "constants": {"5": 0.27, "10": 0.33, "20": 0.40, "30": 0.46, "45": 0.54},
    }
    values = module.shared_thresholds(np.asarray([100.0, 220.0, 230.0, 250.0]), fit_summary, 45)

    assert np.allclose(values[:3], values[0])
    assert values[3] < values[2]


def test_shared_threshold_function_uses_power_rise_below_peak():
    module = _load_module()
    origin_kev = 238.903
    fit_summary = {
        "origin_kev": origin_kev,
        "amplitude": 0.2,
        "tau_kev": 90.0,
        "low_energy_power": 3.0,
        "constants": {"5": 0.27, "10": 0.33, "20": 0.40, "30": 0.46, "45": 0.54},
    }
    values = module.shared_thresholds(
        np.asarray([0.0, 100.0, origin_kev, 500.0]), fit_summary, 45
    )

    assert values[0] == 0.0
    assert values[1] < values[2]
    assert values[3] < values[2]
