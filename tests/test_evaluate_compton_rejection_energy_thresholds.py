from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/evaluate_compton_rejection_energy_thresholds.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_compton_rejection_energy_thresholds_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bin_mask_includes_only_the_last_upper_edge():
    module = _load_module()
    values = np.asarray([100.0, 149.999, 150.0, 200.0])
    np.testing.assert_array_equal(
        module.bin_mask(values, 100.0, 150.0, False),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        module.bin_mask(values, 150.0, 200.0, True),
        [False, False, True, True],
    )


def test_polynomial_evaluation_holds_measured_endpoint_values_constant():
    module = _load_module()
    fit = {"coefficients_ascending": [0.3, 0.1]}
    values = module.evaluate_fit(
        np.asarray([0.0, 125.0, 975.0, 3000.0]), fit
    )
    assert values[0] == values[1]
    assert values[2] == values[3]


def test_selected_fit_reports_all_candidate_degrees_and_grouped_error():
    module = _load_module()
    rows = []
    for source, offset in (("co60", 0.0), ("cs137", 0.002)):
        for center in np.arange(125.0, 1025.0, 50.0):
            if source == "cs137" and center > 375.0:
                continue
            x = (center - 550.0) / 450.0
            rows.append(
                {
                    "source": source,
                    "rejection_target_percent": 20,
                    "energy_center_kev": center,
                    "empirical_threshold": 0.35 + 0.04 * x + 0.08 * x * x + offset,
                }
            )
    fit = module.fit_polynomial(rows, 20)
    assert 1 <= fit["selected_degree"] <= module.MAXIMUM_POLYNOMIAL_DEGREE
    assert len(fit["candidate_degrees"]) == module.MAXIMUM_POLYNOMIAL_DEGREE
    assert np.isfinite(fit["grouped_leave_one_energy_out_rmse"])
