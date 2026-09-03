from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/scan_th232_flat_pb_grid.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scan_th232_flat_pb_grid_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_get_peak_target_robustness():
    module = _load_module()
    t_grid = np.linspace(0.0, 1.0, 101)
    gains = np.linspace(1.0, 1.30, 101)
    rets = np.linspace(1.0, 0.50, 101)

    class DummyWindow:
        reference_kev = 583.187
        centroid_kev = 583.5

    peak_curves = {583.187: (gains, rets, DummyWindow())}
    th, g, r = module.get_peak_target(583.187, 1.15, peak_curves, t_grid)

    assert 0.0 <= th <= 1.0
    assert np.isclose(g, 1.15, atol=0.01)
    assert 0.0 <= r <= 1.0


def test_get_peak_target_ignores_nonfinite_candidates():
    module = _load_module()
    t_grid = np.asarray([0.10, 0.20, 0.30])
    gains = np.asarray([np.nan, 1.15, 1.30])
    rets = np.asarray([0.0, 0.80, 0.50])

    class DummyWindow:
        reference_kev = 583.187
        centroid_kev = 583.5

    peak_curves = {583.187: (gains, rets, DummyWindow())}
    th, g, r = module.get_peak_target(583.187, 1.15, peak_curves, t_grid)

    assert np.isclose(th, 0.20)
    assert np.isclose(g, 1.15)
    assert np.isclose(r, 0.80)


def test_get_peak_target_prefers_first_crossing_over_late_empty_spectrum():
    module = _load_module()
    t_grid = np.asarray([0.10, 0.20, 0.30, 0.40])
    gains = np.asarray([1.00, 1.151, 1.20, 1.15])
    rets = np.asarray([1.00, 0.90, 0.45, 0.10])

    class DummyWindow:
        reference_kev = 583.187
        centroid_kev = 583.5

    peak_curves = {583.187: (gains, rets, DummyWindow())}
    th, g, r = module.get_peak_target(583.187, 1.15, peak_curves, t_grid)

    assert np.isclose(th, 0.20)
    assert np.isclose(g, 1.151)
    assert np.isclose(r, 0.90)
