from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/plot_th232_revised_pb_threshold_curves.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_th232_revised_pb_threshold_curves_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pb_domain_excludes_2614_retention_only_anchor():
    module = _load_module()

    assert [spec.reference_kev for spec in module.PB_ANCHORS] == [
        209.253,
        238.632,
        300.087,
        409.462,
        510.770,
        583.191,
        727.330,
        911.204,
        1247.080,
        1460.830,
    ]
    assert module.PB_ANCHORS[0].sideband_mode == "both"
    assert module.PB_ANCHORS[-1].reference_kev < module.RETENTION_ONLY_ANCHOR.reference_kev
    assert module.RETENTION_ONLY_ANCHOR.use_pb is False


def test_higher_only_sideband_ignores_lower_sideband_counts():
    module = _load_module()
    window = module.PeakWindow(
        reference_kev=583.191,
        centroid_kev=583.5,
        sigma_kev=1.5,
        roi_low_kev=580.5,
        roi_high_kev=586.5,
        left_low_kev=575.5,
        left_high_kev=578.5,
        right_low_kev=588.5,
        right_high_kev=591.5,
    )
    histogram = np.zeros_like(module.ENERGY_CENTERS)
    histogram[(module.ENERGY_CENTERS >= 580.5) & (module.ENERGY_CENTERS < 586.5)] = 100.0
    histogram[(module.ENERGY_CENTERS >= 575.5) & (module.ENERGY_CENTERS < 578.5)] = 1000.0
    histogram[(module.ENERGY_CENTERS >= 588.5) & (module.ENERGY_CENTERS < 591.5)] = 10.0

    metrics = module.line_aware_metrics(histogram, window, "higher_only")
    right_counts, _ = module.interval_counts(
        histogram, window.right_low_kev, window.right_high_kev
    )
    expected_background = right_counts / (
        window.right_high_kev - window.right_low_kev
    ) * (window.roi_high_kev - window.roi_low_kev)

    assert np.isclose(metrics["estimated_background_counts"], expected_background)
    assert metrics["estimated_background_counts"] < 100.0
