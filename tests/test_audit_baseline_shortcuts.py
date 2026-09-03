from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_baseline_shortcuts.py"
SPEC = importlib.util.spec_from_file_location("audit_baseline_shortcuts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_model_metrics_reports_weighted_classification_metrics() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.2, 0.7, 0.1], dtype=np.float64)
    weights = np.asarray([1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    metrics = MODULE.model_metrics(labels, scores, weights)
    assert metrics["event_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["weighted_auroc"] == pytest.approx(1.0)


def test_model_metrics_handles_single_class_groups() -> None:
    labels = np.asarray([1, 1], dtype=np.int8)
    scores = np.asarray([0.4, 0.8], dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    metrics = MODULE.model_metrics(labels, scores, weights)
    assert metrics["event_count"] == 2
    assert metrics["auroc"] is None
    assert metrics["weighted_auroc"] is None


def test_qc_and_session_summaries_are_label_stratified() -> None:
    metadata = {
        "label": np.asarray([1, 0, 1, 0], dtype=np.int8),
        "weight": np.ones(4, dtype=np.float64),
        "event_qc_status": np.asarray(["PASS", "PASS", "WARN", "WARN"]),
        "event_session": np.asarray(["s1", "s1", "s2", "s2"]),
        "event_source": np.asarray(["ba133", "co60", "ba133", "co60"]),
    }
    scores = np.asarray([0.8, 0.2, 0.7, 0.3], dtype=np.float64)
    qc = MODULE.label_qc_metrics(metadata, scores)
    sessions = MODULE.session_score_summary(metadata, scores)
    assert qc["PASS|label_1"]["event_count"] == 1
    assert qc["WARN|label_0"]["score_mean"] == pytest.approx(0.3)
    assert sessions["s1|label_0"]["event_sources"] == {"co60": 1}
    assert sessions["s2|label_1"]["score_mean"] == pytest.approx(0.7)
