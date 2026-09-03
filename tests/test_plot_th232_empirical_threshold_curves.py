from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/plot_th232_empirical_threshold_curves.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_th232_empirical_threshold_curves_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empirical_curve_uses_all_current_anchors_without_parametric_fit():
    module = _load_module()
    rows = []
    for target_pct in module.TARGET_GAINS_PCT:
        for index, spec in enumerate(module.PB_ANCHORS):
            energy = 209.0 + index * 100.0
            rows.append(
                {
                    "target_pb_gain_percent": str(target_pct),
                    "reference_energy_kev": str(spec.reference_kev),
                    "observed_centroid_kev": str(energy),
                    "target_threshold": str(0.2 + 0.01 * index + 0.02 * target_pct / 45.0),
                    "bootstrap_ci_low": str(0.19 + 0.01 * index),
                    "bootstrap_ci_high": str(0.21 + 0.01 * index),
                }
            )

    curves, rows_by_target = module.build_empirical_curves(rows)

    assert len(rows_by_target[module.TARGET_GAINS_PCT[0]]) == len(module.PB_ANCHORS)
    for target_pct in module.TARGET_GAINS_PCT:
        energy, threshold = curves[target_pct]
        assert energy[0] == 209.0
        assert energy[-1] == 209.0 + (len(module.PB_ANCHORS) - 1) * 100.0
        assert np.all(np.isfinite(threshold))
        assert np.all((threshold >= 0.0) & (threshold <= 1.0))

