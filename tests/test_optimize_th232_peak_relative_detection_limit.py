from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/optimize_th232_peak_relative_detection_limit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "optimize_th232_peak_relative_detection_limit_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detection_limit_satisfies_defining_equation():
    module = _load_module()
    background = np.asarray([100.0, 500.0])
    estimator_variance = np.asarray([50.0, 250.0])
    critical, detection = module.detection_limit_counts(
        background, estimator_variance
    )
    k = module.norm.ppf(0.95)
    expected = critical + k * np.sqrt(
        detection + background + estimator_variance
    )
    np.testing.assert_allclose(detection, expected)


def test_relative_detection_limit_penalizes_signal_loss():
    module = _load_module()
    category_histograms = {
        "roi": np.asarray([[1500.0, 0.0]]),
        "left": np.asarray([[250.0, 0.0]]),
        "right": np.asarray([[250.0, 0.0]]),
        "left_energy": np.asarray([[250.0 * 90.0, 0.0]]),
        "right_energy": np.asarray([[250.0 * 110.0, 0.0]]),
    }
    window = module.PeakWindow(
        100.0, 100.0, 2.0, 96.0, 104.0, 90.0, 94.0, 106.0, 110.0
    )
    original_bins = module.SCORE_BIN_COUNT
    module.SCORE_BIN_COUNT = 2
    try:
        curve = module.metric_curve(category_histograms, window, "both")
    finally:
        module.SCORE_BIN_COUNT = original_bins
    assert curve["relative_detection_limit"][0] == 1.0
    assert curve["signal_efficiency"][0] == 1.0
    assert not curve["reliable"][1]


def test_select_optimum_uses_lowest_relative_detection_limit():
    module = _load_module()
    curve = {
        "reliable": np.asarray([True, True, True, False]),
        "relative_detection_limit": np.asarray([1.0, 0.92, 0.95, 0.5]),
    }
    assert module.select_optimum(curve) == 1
