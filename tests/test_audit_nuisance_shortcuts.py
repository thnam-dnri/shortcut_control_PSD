from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_nuisance_shortcuts.py"
SPEC = importlib.util.spec_from_file_location("audit_nuisance_shortcuts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_development_guard_rejects_locked_and_external_paths() -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        MODULE.assert_development_csv(Path("outputs/labels/label_pairs_test.csv"))
    with pytest.raises(ValueError, match="Forbidden"):
        MODULE.assert_development_csv(Path("processed_data/th232_evaluation/label_pairs_train.csv"))
    with pytest.raises(ValueError, match="Forbidden"):
        MODULE.assert_development_partition("test")


def test_metric_summary_handles_weighted_binary_scores() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64)
    weights = np.asarray([1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    metrics = MODULE.metric_summary(labels, scores, weights)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["weighted_auroc"] == pytest.approx(1.0)


def test_source_only_control_excludes_numeric_features() -> None:
    data = pd.DataFrame(
        {
            "event_source": ["ba133", "co60", "ba133", "co60"],
            **{name: [0.0, 1.0, 0.5, 1.5] for name in MODULE.NUMERIC_FEATURES},
        }
    )
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    weights = np.ones(4, dtype=np.float64)
    model = MODULE.make_categorical_model(
        MODULE.SOURCE_CATEGORICAL,
        seed=1,
        include_numeric=False,
    )
    model.fit(data, labels, classifier__sample_weight=weights)
    assert model.predict_proba(data).shape == (4, 2)
