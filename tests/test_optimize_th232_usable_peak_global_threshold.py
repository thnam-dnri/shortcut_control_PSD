from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/optimize_th232_usable_peak_global_threshold.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "optimize_th232_usable_peak_global_threshold_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audited_anchor_set_stops_at_1460_and_excludes_old_regions():
    module = _load_module()
    references = [spec.reference_kev for spec in module.PB_ANCHORS]

    assert references == [
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
    assert 338.320 not in references
    assert 968.971 not in references
    assert 2614.533 not in references


def test_primary_and_209_excluded_sensitivity_can_select_different_thresholds():
    module = _load_module()
    rows = [
        {
            "threshold": 0.30,
            "geometric_mean_pb_improvement": 1.10,
            "minimum_peak_retention": 0.90,
            "all_peak_statistics_reliable": True,
            "sensitivity_geometric_mean_pb_improvement_excluding_209": 1.10,
            "sensitivity_minimum_peak_retention_excluding_209": 0.92,
            "sensitivity_statistics_reliable_excluding_209": True,
        },
        {
            "threshold": 0.45,
            "geometric_mean_pb_improvement": 1.20,
            "minimum_peak_retention": 0.81,
            "all_peak_statistics_reliable": True,
            "sensitivity_geometric_mean_pb_improvement_excluding_209": 1.18,
            "sensitivity_minimum_peak_retention_excluding_209": 0.85,
            "sensitivity_statistics_reliable_excluding_209": True,
        },
        {
            "threshold": 0.50,
            "geometric_mean_pb_improvement": 1.30,
            "minimum_peak_retention": 0.75,
            "all_peak_statistics_reliable": True,
            "sensitivity_geometric_mean_pb_improvement_excluding_209": 1.25,
            "sensitivity_minimum_peak_retention_excluding_209": 0.82,
            "sensitivity_statistics_reliable_excluding_209": True,
        },
    ]

    primary = module.select_operating_points(rows)
    sensitivity = module.select_operating_points(
        rows,
        objective_key="sensitivity_geometric_mean_pb_improvement_excluding_209",
        retention_key="sensitivity_minimum_peak_retention_excluding_209",
        reliability_key="sensitivity_statistics_reliable_excluding_209",
    )

    assert primary["minimum_retention_80pct"]["threshold"] == 0.45
    assert sensitivity["minimum_retention_80pct"]["threshold"] == 0.50


def test_pareto_knee_uses_normalized_gain_minus_retention_loss():
    module = _load_module()
    rows = [
        {
            "threshold": 0.0,
            "geometric_mean_pb_improvement": 1.0,
            "minimum_peak_retention": 1.0,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.4,
            "geometric_mean_pb_improvement": 1.3,
            "minimum_peak_retention": 0.8,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.6,
            "geometric_mean_pb_improvement": 1.4,
            "minimum_peak_retention": 0.4,
            "all_peak_statistics_reliable": True,
        },
    ]

    selected = module.select_pareto_knee(rows)

    assert selected["threshold"] == 0.4
    assert selected["geometric_mean_pb_improvement_pareto_knee_score"] > 0.0


def test_no_brainer_maximizes_pb_subject_to_pooled_retention():
    module = _load_module()
    rows = [
        {
            "threshold": 0.2,
            "geometric_mean_pb_improvement": 1.02,
            "pooled_net_peak_retention": 0.995,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.3,
            "geometric_mean_pb_improvement": 1.04,
            "pooled_net_peak_retention": 0.991,
            "all_peak_statistics_reliable": True,
        },
        {
            "threshold": 0.4,
            "geometric_mean_pb_improvement": 1.08,
            "pooled_net_peak_retention": 0.980,
            "all_peak_statistics_reliable": True,
        },
    ]

    selected = module.select_no_brainer_point(rows)

    assert selected["threshold"] == 0.3
