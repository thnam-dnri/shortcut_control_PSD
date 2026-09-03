from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/evaluate_th232_ds_cnn.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_th232_ds_cnn_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_requested_positive_retention_points_are_frozen():
    module = _load_module()
    assert module.ACCEPTANCES == (0.99, 0.95, 0.90, 0.80, 0.50, 0.30, 0.10)


def test_weighted_acceptance_threshold_reports_actual_retention():
    module = _load_module()
    result = module.weighted_acceptance_threshold(
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        np.ones(4, dtype=np.float64),
        0.50,
    )
    assert result.name == "50pct"
    assert result.score_threshold == 0.3
    assert result.actual_weighted_acceptance == 0.5
    assert result.actual_unweighted_acceptance == 0.5


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
