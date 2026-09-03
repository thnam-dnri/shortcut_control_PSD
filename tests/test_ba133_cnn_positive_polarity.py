from __future__ import annotations

import numpy as np

from src.ba133_cnn import (
    RawPartition,
    RepresentationConfig,
    build_representation,
    representation_config_from_checkpoint,
)


def test_negative_pulse_is_flipped_positive_before_baseline_subtraction():
    waveform = np.zeros((1, 4500), dtype=np.float32)
    waveform[0, 1250:] = -10.0
    raw = RawPartition(
        waveforms=waveform,
        shaped_energy=np.ones(1, dtype=np.float32),
        labels=np.ones(1, dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        peak_ids=np.asarray(["peak"]),
    )
    config = RepresentationConfig(
        name="positive_polarity_test",
        input_mode="both",
        moving_average=10,
        normalization="none",
        anchor="t10",
        pre_samples=250,
        post_samples=500,
        pulse_polarity="negative_to_positive",
        standardization="none",
    )

    values, qc = build_representation(raw, config)

    assert qc["anchor_fallback_count"] == 0
    assert values.shape == (1, 2, 750)
    assert np.mean(values[0, 0, :200]) == 0.0
    assert np.mean(values[0, 0, 400:]) > 9.0
    assert np.max(values[0, 1]) > 0.0


def test_pulse_polarity_is_required_in_representation_contract():
    fields = RepresentationConfig.__dataclass_fields__
    assert fields["pulse_polarity"].default is fields["pulse_polarity"].default_factory


def test_legacy_checkpoint_config_preserves_as_acquired_polarity():
    config = representation_config_from_checkpoint(
        {
            "name": "legacy",
            "input_mode": "both",
            "moving_average": 10,
            "normalization": "global",
            "anchor": "t10",
            "pre_samples": 250,
            "post_samples": 500,
            "downsample": 1,
        }
    )
    assert config.pulse_polarity == "as_acquired"
    assert config.standardization == "train_zscore"


def test_dual_anchor_peak_normalization_produces_positive_unit_peaks():
    waveform = np.zeros((1, 4500), dtype=np.float32)
    waveform[0, 1250:] = -10.0
    raw = RawPartition(
        waveforms=waveform,
        shaped_energy=np.ones(1, dtype=np.float32),
        labels=np.ones(1, dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        peak_ids=np.asarray(["peak"]),
    )
    config = RepresentationConfig(
        name="dual_anchor_test",
        input_mode="both",
        moving_average=10,
        normalization="independent_positive_peak",
        anchor="dual_t10_current_peak",
        pre_samples=250,
        post_samples=250,
        pulse_polarity="negative_to_positive",
        standardization="none",
        representation_schema_version=2,
        endpoint_inclusive=True,
        current_search_start=1100,
        current_search_stop=1500,
        clip_charge_to_unit_interval=True,
    )

    values, qc = build_representation(raw, config)

    assert values.shape == (1, 2, 501)
    assert qc["anchor_fallback_count"] == 0
    assert qc["invalid_scale_count"] == 0
    np.testing.assert_allclose(np.max(values, axis=2), 1.0)
    assert np.min(values[0, 0]) >= 0.0
    assert np.max(values[0, 0]) <= 1.0


def test_shared_t10_charge_peak_normalization_preserves_unclipped_information():
    waveform = np.zeros((1, 4500), dtype=np.float32)
    waveform[0, 1245:1250] = 1.0
    waveform[0, 1250:] = -10.0
    raw = RawPartition(
        waveforms=waveform,
        shaped_energy=np.ones(1, dtype=np.float32),
        labels=np.ones(1, dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        peak_ids=np.asarray(["peak"]),
    )
    config = RepresentationConfig(
        name="shared_t10_test",
        input_mode="both",
        moving_average=10,
        normalization="charge_peak_shared",
        anchor="t10",
        pre_samples=250,
        post_samples=250,
        pulse_polarity="negative_to_positive",
        standardization="fixed_current_peak_scale",
        representation_schema_version=2,
        endpoint_inclusive=True,
    )

    values, qc = build_representation(raw, config)

    assert values.shape == (1, 2, 501)
    assert qc["anchor_fallback_count"] == 0
    assert qc["invalid_scale_count"] == 0
    assert np.min(values[0, 0]) < 0.0
    np.testing.assert_allclose(np.max(values[0, 0]), 1.0)
    assert np.argmax(values[0, 1]) > config.pre_samples
