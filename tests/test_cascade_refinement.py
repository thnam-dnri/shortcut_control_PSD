"""Tests for cascade refinement representations, event weighting, and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from src.cascade_refinement import (
    CANDIDATES,
    BivariateIsotonicCalibrator,
    CascadeRepresentation,
    event_indices,
    isotonic_fusion,
    make_event_weights,
    moving_average,
    piecewise_fusion,
    soft_gate_fusion,
    weighted_acceptance_threshold,
)


def test_candidates_and_shapes() -> None:
    assert "rep_hires_ma5_t10" in CANDIDATES
    assert "rep_hires_ma4_t_imax" in CANDIDATES
    assert "rep_hires_ma4_t_imax_rise" in CANDIDATES
    assert "rep_hires_ma4_t_imax_ae" in CANDIDATES
    assert "rep_hires_ma4_t_imax_rise_ae" in CANDIDATES

    cand_b = CANDIDATES["rep_hires_ma5_t10"]
    assert cand_b.channel_count == 2
    assert cand_b.window_length == 750

    cand_e = CANDIDATES["rep_hires_ma4_t_imax_rise_ae"]
    assert cand_e.channel_count == 4
    assert cand_e.window_length == 600


def test_event_indices_and_weights() -> None:
    pair_indices = np.array([0, 2], dtype=np.int64)
    ev_idx = event_indices(pair_indices)
    assert np.array_equal(ev_idx, np.array([0, 1, 4, 5]))

    # Test pair-preserving weights
    peaks = np.array(["ba133_356kev", "ba133_356kev", "na22_511kev", "na22_511kev", "cs137_662kev", "cs137_662kev"])
    weights = make_event_weights(peaks)
    assert len(weights) == 6
    assert weights[0] == weights[1]
    assert weights[2] == weights[3]
    assert weights[4] == weights[5]


def test_moving_average_computation() -> None:
    x = np.ones((2, 10), dtype=np.float32)
    ma = moving_average(x, width=4)
    assert ma.shape == (2, 10)
    assert np.allclose(ma[:, 3:], 1.0)


def test_piecewise_fusion_bounds() -> None:
    s_main = np.array([0.2, 0.45, 0.55, 0.8], dtype=np.float32)
    s_stage2 = np.array([0.9, 0.0, 1.0, 0.1], dtype=np.float32)
    
    fused = piecewise_fusion(s_main, s_stage2, tau_low=0.4, tau_high=0.6)
    assert np.isclose(fused[0], 0.2)
    assert np.isclose(fused[3], 0.8)
    assert 0.4 <= fused[1] <= 0.6
    assert 0.4 <= fused[2] <= 0.6


def test_bivariate_isotonic_calibrator() -> None:
    s_main = np.linspace(0.4, 0.6, 20, dtype=np.float32)
    s_stage2 = np.linspace(0.1, 0.9, 20, dtype=np.float32)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.float32)
    weights = np.ones(20, dtype=np.float32)

    calib = BivariateIsotonicCalibrator(bins=4)
    calib.fit(s_main, s_stage2, labels, weights)
    
    preds = calib.predict(s_main, s_stage2)
    assert len(preds) == 20
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_weighted_acceptance_threshold() -> None:
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    res = weighted_acceptance_threshold(scores, weights, acceptance=0.8)
    assert np.isclose(res["score_threshold"], 0.3)
