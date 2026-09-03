from __future__ import annotations

import numpy as np
import torch

from scripts.build_balanced_morphology_group_training_sets import (
    balanced_indices_by_peak,
)
from scripts.merge_morphology_catalogue_groups import merge_assignments
from src.waveform_morphology import (
    FEATURE_NAMES,
    FiLMDSCNN,
    MorphologyConfig,
    describe_current_envelope,
)


def test_merge_assignments_preserves_hard_group_membership():
    probability = np.eye(8, dtype=np.float32)
    assignment = np.arange(8, dtype=np.int16)
    valid = np.ones(8, dtype=bool)

    merged_probability, merged_assignment = merge_assignments(
        probability, assignment, valid
    )

    assert merged_probability.shape == (8, 6)
    assert merged_assignment.tolist() == [0, 1, 2, 3, 4, 3, 5, 5]
    np.testing.assert_allclose(merged_probability.sum(axis=1), 1.0)


def test_merge_assignments_keeps_invalid_assignment_sentinel():
    probability = np.full((2, 8), 0.125, dtype=np.float32)
    assignment = np.asarray([3, -1], dtype=np.int16)
    valid = np.asarray([True, False])

    _, merged_assignment = merge_assignments(probability, assignment, valid)

    assert merged_assignment.tolist() == [3, -1]


def test_balanced_indices_by_peak_balances_each_peak():
    labels = np.asarray([1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int8)
    peaks = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    candidates = np.arange(labels.size)

    selected = balanced_indices_by_peak(
        labels, peaks, candidates, np.random.default_rng(17)
    )

    for peak in ("a", "b"):
        chosen = labels[selected][peaks[selected] == peak]
        assert np.count_nonzero(chosen == 1) == np.count_nonzero(chosen == 0)


def test_descriptor_extracts_finite_features_for_smooth_peak():
    config = MorphologyConfig()
    x = np.arange(config.window_length)
    current = np.exp(-0.5 * np.square((x - 300) / 18.0)).astype(np.float32)

    features, valid, peak_count = describe_current_envelope(current, config)

    assert valid
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert features[0] > 0
    assert peak_count >= 1


def test_descriptor_rejects_flat_zero_current():
    config = MorphologyConfig()

    features, valid, peak_count = describe_current_envelope(
        np.zeros(config.window_length, dtype=np.float32), config
    )

    assert not valid
    assert np.all(np.isnan(features))
    assert peak_count == 0


def test_film_ds_cnn_shape_and_zero_initialized_conditioning():
    torch.manual_seed(7)
    model = FiLMDSCNN(posterior_dimensions=3, width=8)
    values = torch.randn(4, 2, 750)
    first = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    second = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)

    score_first = model(values, first)
    score_second = model(values, second)

    assert score_first.shape == (4,)
    assert torch.allclose(score_first, score_second)


def test_film_ds_cnn_parameter_count_stays_below_protocol_limit():
    model = FiLMDSCNN(posterior_dimensions=7, width=24)

    assert sum(parameter.numel() for parameter in model.parameters()) <= 30000
