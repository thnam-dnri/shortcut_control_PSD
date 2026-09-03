from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/fit_th232_threshold_curves_with_uncertainty.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fit_th232_threshold_curves_with_uncertainty_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_simple_fit_functions_are_finite_and_bounded_for_nominal_inputs():
    module = _load_module()
    energy = np.asarray([238.9, 300.0, 409.0, 510.0, 1460.4])
    exponential = module.exponential_decay(energy, 0.25, 0.45, 400.0)
    quadratic = module.quadratic_scaled(energy, 0.1, -0.2, 0.5)

    assert np.all(np.isfinite(exponential))
    assert np.all(np.isfinite(quadratic))
    assert np.all((exponential >= 0.0) & (exponential <= 1.0))


def test_first_threshold_selection_uses_earliest_crossing():
    module = _load_module()
    gains = np.asarray([1.0, 1.16, 1.20, 1.14])
    retentions = np.asarray([1.0, 0.9, 0.7, 0.3])
    threshold = module.select_first_threshold(gains, retentions, 1.15)

    assert np.isclose(threshold, module.SCORE_GRID[1])

