from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/evaluate_th232_stable_pb_threshold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_th232_stable_pb_threshold_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_interpolator_and_clamping():
    module = _load_module()
    energies = [238.6, 583.2, 2614.5]
    thresholds = [0.55, 0.36, 0.41]
    interpolator = module.build_interpolator(energies, thresholds)

    eval_energies = np.asarray([100.0, 238.6, 583.2, 2614.5, 3000.0])
    eval_t = module.evaluate_continuous_threshold(
        eval_energies, interpolator, 238.6, 2614.5
    )

    assert len(eval_t) == 5
    # Clamping outside domain
    assert np.isclose(eval_t[0], eval_t[1])
    assert np.isclose(eval_t[3], eval_t[4])
    # Exact knot values
    assert np.isclose(eval_t[1], 0.55)
    assert np.isclose(eval_t[2], 0.36)
    assert np.isclose(eval_t[3], 0.41)


def test_safe_peak_metrics_zero_division_guard():
    module = _load_module()

    class DummyWindow:
        roi_low_kev = 100.0
        roi_high_kev = 110.0
        left_low_kev = 90.0
        left_high_kev = 95.0
        right_low_kev = 115.0
        right_high_kev = 120.0
        reference_kev = 105.0
        centroid_kev = 105.0

    empty_histogram = np.zeros(3200, dtype=np.float64)
    metrics = module.safe_peak_metrics(empty_histogram, DummyWindow())
    assert np.isnan(metrics["peak_to_background"]) or metrics["estimated_background_counts"] == 0.0
