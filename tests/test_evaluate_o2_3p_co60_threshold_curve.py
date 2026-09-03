from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_o2_3p_co60_threshold_curve.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_o2_3p_co60_threshold_curve_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constant_pass_threshold_uses_score_greater_equal_semantics():
    module = _load_module()
    scores = np.arange(100, dtype=np.float32) / 100.0

    threshold, passed, fraction = module.closest_constant_pass_threshold(scores, 0.94)

    assert passed == 94
    assert fraction == 0.94
    assert np.count_nonzero(scores >= threshold) == 94


def test_threshold_curve_uses_validation_only_and_includes_final_upper_edge():
    module = _load_module()
    validation_energy = np.asarray([100.0, 125.0, 149.9, 150.0, 175.0, 200.0])
    validation_scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    train_energy = validation_energy.copy()
    train_scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])

    rows = module.build_threshold_rows(
        validation_energy,
        validation_scores,
        train_energy,
        train_scores,
        100.0,
        200.0,
        50.0,
        2.0 / 3.0,
    )

    assert len(rows) == 2
    assert rows[0]["validation_event_count"] == 3
    assert rows[1]["validation_event_count"] == 3
    assert rows[1]["upper_edge_inclusive"] is True
    assert rows[0]["validation_passed_count"] == 2
    assert rows[1]["validation_passed_count"] == 2
    assert rows[0]["train_passing_fraction_at_validation_threshold"] == 1.0
    assert rows[1]["train_passing_fraction_at_validation_threshold"] == 0.0
