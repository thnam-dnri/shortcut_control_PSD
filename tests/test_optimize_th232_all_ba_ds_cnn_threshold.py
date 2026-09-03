from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/optimize_th232_all_ba_ds_cnn_threshold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "optimize_th232_all_ba_ds_cnn_threshold_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empty_histogram_is_marked_unreliable_instead_of_dividing_by_zero():
    module = _load_module()
    window = module.PeakWindow(100.0, 100.0, 1.0, 98.0, 102.0, 95.0, 97.0, 103.0, 105.0)
    metrics = module.reliable_metrics(
        np.zeros(module.ENERGY_CENTERS.size, dtype=np.float64), window
    )
    assert metrics["estimated_background_counts"] == 0.0
    assert not metrics["statistically_reliable"]


def test_operating_point_selection_respects_each_retention_floor():
    module = _load_module()
    rows = [
        {
            "threshold": 0.2,
            "geometric_mean_pb_improvement": 1.1,
            "minimum_primary_peak_retention": 0.95,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.4,
            "geometric_mean_pb_improvement": 1.3,
            "minimum_primary_peak_retention": 0.82,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.6,
            "geometric_mean_pb_improvement": 1.5,
            "minimum_primary_peak_retention": 0.40,
            "all_peak_statistics_reliable": True,
        },
    ]
    selected = module.select_operating_points(rows)
    assert selected["minimum_retention_90pct"]["threshold"] == 0.2
    assert selected["minimum_retention_80pct"]["threshold"] == 0.4
    assert selected["minimum_retention_00pct"]["threshold"] == 0.6
