"""Label-blind waveform-morphology descriptors and conditional DS-CNN helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import Tensor, nn

from src.architecture_candidates import DepthwiseSeparableBlock, _group_norm
from src.ba133_cnn import (
    BASELINE_STOP,
    SAMPLE_PERIOD_NS,
    gather_window,
    moving_average,
    t10_anchor,
)


FEATURE_NAMES = (
    "primary_fwhm_ns",
    "rise_10_90_ns",
    "fall_90_10_ns",
    "rise_fall_asymmetry",
    "crest_flatness",
    "secondary_prominence_ratio",
    "temporal_excess_kurtosis",
    "normalized_peak_compactness",
)


@dataclass(frozen=True)
class MorphologyConfig:
    moving_average: int = 20
    pre_samples: int = 250
    post_samples: int = 500
    baseline_noise_samples: int = 200
    noise_sigma_multiplier: float = 3.0
    minimum_peak_fraction: float = 0.01
    crest_half_width_samples: int = 3
    peak_minimum_distance_samples: int = 5

    @property
    def window_length(self) -> int:
        return self.pre_samples + self.post_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "moving_average": self.moving_average,
            "pre_samples": self.pre_samples,
            "post_samples": self.post_samples,
            "baseline_noise_samples": self.baseline_noise_samples,
            "noise_sigma_multiplier": self.noise_sigma_multiplier,
            "minimum_peak_fraction": self.minimum_peak_fraction,
            "crest_half_width_samples": self.crest_half_width_samples,
            "peak_minimum_distance_samples": self.peak_minimum_distance_samples,
            "sample_period_ns": SAMPLE_PERIOD_NS,
            "feature_names": list(FEATURE_NAMES),
        }


def _left_crossing(values: np.ndarray, peak: int, level: float) -> float | None:
    for index in range(peak - 1, -1, -1):
        low = float(values[index])
        high = float(values[index + 1])
        if low < level <= high and high > low:
            return index + (level - low) / (high - low)
    return None


def _right_crossing(values: np.ndarray, peak: int, level: float) -> float | None:
    for index in range(peak, values.size - 1):
        high = float(values[index])
        low = float(values[index + 1])
        if high >= level > low and high > low:
            return index + (high - level) / (high - low)
    return None


def describe_current_envelope(
    current_window: np.ndarray,
    config: MorphologyConfig,
) -> tuple[np.ndarray, bool, int]:
    """Return eight descriptors, validity, and the number of detected peaks."""

    values = np.asarray(current_window, dtype=np.float64)
    features = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    if values.ndim != 1 or values.size != config.window_length:
        return features, False, 0
    pre = values[: config.baseline_noise_samples]
    center = float(np.median(pre))
    mad = float(np.median(np.abs(pre - center)))
    noise_sigma = 1.4826 * mad
    envelope = np.maximum(values - center, 0.0)
    peak_value = float(np.max(envelope))
    if not np.isfinite(peak_value) or peak_value <= 1.0e-12:
        return features, False, 0
    floor = max(
        config.noise_sigma_multiplier * noise_sigma,
        config.minimum_peak_fraction * peak_value,
    )
    envelope[envelope < floor] = 0.0
    peak = int(np.argmax(envelope))
    peak_value = float(envelope[peak])
    if peak_value <= 0:
        return features, False, 0

    crossings: dict[float, tuple[float | None, float | None]] = {}
    for fraction in (0.1, 0.5, 0.9):
        level = fraction * peak_value
        crossings[fraction] = (
            _left_crossing(envelope, peak, level),
            _right_crossing(envelope, peak, level),
        )
    if any(value is None for pair in crossings.values() for value in pair):
        return features, False, 0
    left10, right10 = crossings[0.1]
    left50, right50 = crossings[0.5]
    left90, right90 = crossings[0.9]
    assert None not in (left10, right10, left50, right50, left90, right90)

    fwhm = (right50 - left50) * SAMPLE_PERIOD_NS
    rise = (left90 - left10) * SAMPLE_PERIOD_NS
    fall = (right10 - right90) * SAMPLE_PERIOD_NS
    fall_from_peak = (right10 - peak) * SAMPLE_PERIOD_NS
    rise_to_peak = (peak - left10) * SAMPLE_PERIOD_NS
    if min(fwhm, rise, fall, fall_from_peak) <= 0:
        return features, False, 0

    lo = max(0, peak - config.crest_half_width_samples)
    hi = min(envelope.size, peak + config.crest_half_width_samples + 1)
    flatness = float(np.mean(envelope[lo:hi]) / peak_value)
    peaks, properties = find_peaks(
        envelope,
        prominence=max(floor, 0.02 * peak_value),
        distance=config.peak_minimum_distance_samples,
    )
    prominences = np.sort(properties.get("prominences", np.asarray([])))[::-1]
    prominence_ratio = (
        float(prominences[1] / prominences[0]) if prominences.size >= 2 else 0.0
    )

    total = float(np.sum(envelope))
    if total <= 0:
        return features, False, int(peaks.size)
    time = np.arange(envelope.size, dtype=np.float64) * SAMPLE_PERIOD_NS
    mean_time = float(np.sum(time * envelope) / total)
    variance = float(np.sum(np.square(time - mean_time) * envelope) / total)
    if variance <= 0:
        return features, False, int(peaks.size)
    kurtosis = float(
        np.sum(np.power(time - mean_time, 4) * envelope)
        / (total * variance * variance)
        - 3.0
    )
    area = float(np.trapezoid(envelope, dx=SAMPLE_PERIOD_NS))
    if area <= 0:
        return features, False, int(peaks.size)

    features[:] = (
        fwhm,
        rise,
        fall,
        rise_to_peak / fall_from_peak,
        flatness,
        prominence_ratio,
        kurtosis,
        peak_value * fwhm / area,
    )
    valid = bool(np.all(np.isfinite(features)))
    return features, valid, int(peaks.size)


def extract_morphology_features(
    acquired_waveforms: np.ndarray,
    config: MorphologyConfig = MorphologyConfig(),
    chunk_size: int = 512,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Extract descriptors with bounded intermediate memory."""

    waveforms = np.asarray(acquired_waveforms, dtype=np.float32)
    result = np.full((waveforms.shape[0], len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    valid = np.zeros(waveforms.shape[0], dtype=bool)
    peak_count = np.zeros(waveforms.shape[0], dtype=np.int16)
    fallback_count = 0
    invalid_scale_count = 0
    for start in range(0, waveforms.shape[0], chunk_size):
        stop = min(start + chunk_size, waveforms.shape[0])
        positive = -waveforms[start:stop]
        baseline = np.median(positive[:, :BASELINE_STOP], axis=1).astype(np.float32)
        charge = positive - baseline[:, None]
        charge = moving_average(charge, config.moving_average)
        current = np.gradient(charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
        anchors, fallbacks = t10_anchor(charge)
        fallback_count += fallbacks
        charge_window = gather_window(
            charge, anchors, config.pre_samples, config.post_samples
        )
        current_window = gather_window(
            current, anchors, config.pre_samples, config.post_samples
        )
        scale = np.sqrt(np.mean(np.square(charge_window), axis=1))
        bad_scale = ~np.isfinite(scale) | (scale <= 1.0e-12)
        invalid_scale_count += int(np.count_nonzero(bad_scale))
        scale[bad_scale] = 1.0
        current_window /= scale[:, None]
        for local in range(stop - start):
            features, is_valid, peaks = describe_current_envelope(
                current_window[local], config
            )
            result[start + local] = features
            valid[start + local] = is_valid and not bad_scale[local]
            peak_count[start + local] = peaks
    return result, {
        "valid": valid,
        "detected_peak_count": peak_count,
        "anchor_fallback_count": fallback_count,
        "invalid_scale_count": invalid_scale_count,
    }


class FiLMDepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable block modulated by a frozen morphology posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        posterior_dimensions: int,
        stride: int = 1,
        kernel: int = 7,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = _group_norm(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.pointwise_norm = _group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.film = nn.Linear(posterior_dimensions, 2 * out_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.activation = nn.GELU()

    def forward(self, values: Tensor, posterior: Tensor) -> Tensor:
        hidden = self.activation(self.depthwise_norm(self.depthwise(values)))
        hidden = self.pointwise_norm(self.pointwise(hidden))
        gamma, beta = self.film(posterior).chunk(2, dim=1)
        hidden = (1.0 + gamma[:, :, None]) * hidden + beta[:, :, None]
        return self.activation(hidden + self.skip(values))


class FiLMDSCNN(nn.Module):
    """DS-CNN with zero-initialized FiLM modulation at every residual block."""

    def __init__(
        self,
        posterior_dimensions: int,
        input_channels: int = 2,
        width: int = 24,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 11, stride=2, padding=5, bias=False),
            _group_norm(width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            (
                FiLMDepthwiseSeparableBlock(width, width, posterior_dimensions),
                FiLMDepthwiseSeparableBlock(
                    width, 2 * width, posterior_dimensions, stride=2
                ),
                FiLMDepthwiseSeparableBlock(
                    2 * width, 3 * width, posterior_dimensions, stride=2
                ),
                FiLMDepthwiseSeparableBlock(
                    3 * width, 3 * width, posterior_dimensions, stride=2
                ),
            )
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(3 * width, 1),
        )

    def forward(self, values: Tensor, posterior: Tensor) -> Tensor:
        hidden = self.stem(values)
        for block in self.blocks:
            hidden = block(hidden, posterior)
        return self.head(hidden).squeeze(1)


def build_unconditioned_ds_cnn(input_channels: int = 2, width: int = 24) -> nn.Module:
    """Local constructor kept beside FiLMDSCNN for controlled comparisons."""

    from src.architecture_candidates import DSCNN

    return DSCNN(input_channels=input_channels, width=width)
