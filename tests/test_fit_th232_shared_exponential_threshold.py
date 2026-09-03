from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/fit_th232_shared_exponential_threshold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fit_th232_shared_exponential_threshold_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_shape_changes_between_targets_only_by_constant():
    module = _load_module()
    energy = np.asarray([250.0, 500.0, 1000.0])
    target_indices = np.asarray([0, 1, 2])
    constants = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])
    values = module.shared_exponential_values(
        energy,
        constants,
        amplitude=0.2,
        tau_kev=100.0,
        target_indices=target_indices,
    )
    same_shape = module.shared_exponential_values(
        energy,
        constants + 0.1,
        amplitude=0.2,
        tau_kev=100.0,
        target_indices=target_indices,
    )

    assert np.allclose(same_shape - values, 0.1)


def test_shared_exponential_is_constant_below_fixed_hinge():
    module = _load_module()
    energy = np.asarray([180.0, 220.0, 230.0, 250.0])
    target_indices = np.zeros(energy.size, dtype=np.int64)
    constants = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])
    values = module.shared_exponential_values(
        energy,
        constants,
        amplitude=0.2,
        tau_kev=100.0,
        target_indices=target_indices,
        origin_kev=230.0,
    )

    assert np.allclose(values[:3], values[0])
    assert values[3] < values[2]


def test_shared_exponential_power_rise_is_zero_at_origin():
    module = _load_module()
    origin_kev = 238.903
    energy = np.asarray([0.0, 100.0, origin_kev, 500.0])
    target_indices = np.zeros(energy.size, dtype=np.int64)
    constants = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])
    values = module.shared_exponential_values(
        energy,
        constants,
        amplitude=0.2,
        tau_kev=90.0,
        target_indices=target_indices,
        origin_kev=origin_kev,
        low_energy_power=3.0,
    )

    assert values[0] == 0.0
    assert values[1] < values[2]
    assert values[3] < values[2]
