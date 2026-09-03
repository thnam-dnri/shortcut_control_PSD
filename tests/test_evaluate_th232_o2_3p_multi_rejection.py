from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/evaluate_th232_o2_3p_multi_rejection.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("multi_rejection_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pchip_threshold_clamps_outside_calibration_domain():
    module = _load_module()
    energy = np.arange(125.0, 1000.0, 50.0)
    threshold = 0.5 + 0.1 * np.square((energy - 550.0) / 500.0)
    fit = module.fit_threshold_points(energy, threshold)
    values = module.evaluate_fit(np.asarray([0.0, 100.0, 1000.0, 3200.0]), fit)
    assert values[0] == values[1]
    assert values[2] == values[3]


def test_pchip_fit_interpolates_u_shaped_points():
    module = _load_module()
    energy = np.arange(125.0, 1000.0, 50.0)
    expected = 0.5 + 0.2 * np.square((energy - 550.0) / 500.0)
    result = module.fit_threshold_points(energy, expected)
    assert result["family"] == "shape_preserving_pchip"
    np.testing.assert_allclose(module.evaluate_fit(energy, result), expected)


def test_requested_rejection_targets_are_frozen():
    module = _load_module()
    assert module.REJECTION_PERCENTAGES == (30, 50, 70, 90, 99)
