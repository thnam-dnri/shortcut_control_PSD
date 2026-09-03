from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_compact_ds_cnn_performance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_compact_ds_cnn_performance_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_weights_preserve_equal_positive_negative_weights():
    module = _load_module()
    peak_ids = np.asarray(
        ["ba133_356kev", "ba133_356kev"] * 2
        + ["na22_511kev", "na22_511kev"] * 2
        + ["cs137_662kev", "cs137_662kev"] * 2
    )
    weights = module.make_event_weights(peak_ids, module.SELECTED_PEAK_WEIGHTS)

    assert weights.shape == (12,)
    assert np.allclose(weights[0::2], weights[1::2])
    assert np.isclose(weights[0:4].sum(), 0.8)
    assert np.isclose(weights[4:8].sum(), 0.8)
    assert np.isclose(weights[8:12].sum(), 0.4)


def test_split_validation_requires_complete_disjoint_pair_partition():
    module = _load_module()
    module.validate_split(np.asarray([0, 2]), np.asarray([1, 3]), 4)

    try:
        module.validate_split(np.asarray([0, 1]), np.asarray([1, 2]), 4)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping split was accepted")


def test_both_models_accept_frozen_input_shape():
    module = _load_module()
    values = torch.zeros((3, 2, 750), dtype=torch.float32)

    for name in module.MODEL_NAMES:
        model = module.build_model(name)
        output = model(values)
        assert output.shape == (3,)
        assert torch.isfinite(output).all()
