"""Shared representations and utilities for the cascaded DS-CNN experiment.

The Stage-1 representation remains the existing frozen MA-10/t10 contract.  This
module contains only the predeclared high-resolution Stage-2 candidates and the
event-weight/metric helpers needed by the cascade scripts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SAMPLE_PERIOD_NS = 4.0
BASELINE_STOP = 1000
SEARCH_START = 1000
SEARCH_STOP = 2000
CHUNK_SIZE = 512
TAU_LOW = 0.4
TAU_HIGH = 0.6
STAGE2_EPOCHS = 8
STAGE2_BATCH_SIZE = 256
STAGE2_LEARNING_RATE = 8.0e-4
STAGE2_WEIGHT_DECAY = 3.0e-4
STAGE2_SEED = 20260821

PEAK_WEIGHT_KEYS = {
    "ba133_356kev": "ba356",
    "na22_511kev": "na511",
    "cs137_662kev": "cs662",
}
SELECTED_PEAK_WEIGHTS = {"ba356": 0.4, "na511": 0.4, "cs662": 0.2}


@dataclass(frozen=True)
class CascadeRepresentation:
    """One fixed Stage-2 input contract."""

    name: str
    channels: tuple[str, ...]
    moving_average: int
    anchor: str
    pre_samples: int
    post_samples: int
    schema_version: int = 1
    pulse_polarity: str = "negative_to_positive"
    current_peak_definition: str = "signed_maximum_in_samples_1000_2000"
    rise_definition: str = (
        "clip((Q(t)-Q(t10))/(Q(t90)-Q(t10)),0,1); zero_before_t10_one_at_or_after_t90"
    )
    ae_definition: str = "I(t)/max(Q(t)) with max over samples 1000:2000"

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def window_length(self) -> int:
        return self.pre_samples + self.post_samples

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["channels"] = list(self.channels)
        result["channel_count"] = self.channel_count
        result["window_length"] = self.window_length
        return result


CANDIDATES: dict[str, CascadeRepresentation] = {
    "rep_hires_ma4_t_imax": CascadeRepresentation(
        name="rep_hires_ma4_t_imax",
        channels=("charge", "current"),
        moving_average=4,
        anchor="current_peak",
        pre_samples=250,
        post_samples=250,
    ),
    "rep_hires_ma5_t10": CascadeRepresentation(
        name="rep_hires_ma5_t10",
        channels=("charge", "current"),
        moving_average=5,
        anchor="t10",
        pre_samples=250,
        post_samples=500,
    ),
    "rep_hires_ma4_t_imax_rise": CascadeRepresentation(
        name="rep_hires_ma4_t_imax_rise",
        channels=("charge", "current", "rise_profile"),
        moving_average=4,
        anchor="current_peak",
        pre_samples=300,
        post_samples=300,
    ),
    "rep_hires_ma4_t_imax_ae": CascadeRepresentation(
        name="rep_hires_ma4_t_imax_ae",
        channels=("charge", "current", "ae_ratio"),
        moving_average=4,
        anchor="current_peak",
        pre_samples=300,
        post_samples=300,
    ),
    "rep_hires_ma4_t_imax_rise_ae": CascadeRepresentation(
        name="rep_hires_ma4_t_imax_rise_ae",
        channels=("charge", "current", "rise_profile", "ae_ratio"),
        moving_average=4,
        anchor="current_peak",
        pre_samples=300,
        post_samples=300,
    ),
}
CANDIDATE_ORDER = tuple(CANDIDATES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def make_event_weights(peak_ids: np.ndarray) -> np.ndarray:
    """Assign fixed total peak weights while preserving pair-member equality."""

    peak_ids = np.asarray(peak_ids)
    if peak_ids.size % 2:
        raise ValueError("Expected a complete positive/negative pair layout")
    pair_peak_ids = peak_ids[::2]
    if not np.array_equal(pair_peak_ids, peak_ids[1::2]):
        raise ValueError("Positive and negative members of a pair differ in peak ID")
    counts = {peak: int(np.count_nonzero(pair_peak_ids == peak)) for peak in PEAK_WEIGHT_KEYS}
    if set(counts) != set(PEAK_WEIGHT_KEYS) or any(value == 0 for value in counts.values()):
        raise ValueError(f"Unexpected peak IDs or empty peak stratum: {counts}")
    pair_weights = np.asarray(
        [
            SELECTED_PEAK_WEIGHTS[PEAK_WEIGHT_KEYS[peak]] / counts[peak]
            for peak in pair_peak_ids
        ],
        dtype=np.float32,
    )
    return np.repeat(pair_weights, 2)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width < 1:
        raise ValueError("Moving-average width must be positive")
    if width == 1:
        return values.copy()
    cumulative = np.cumsum(values, axis=1, dtype=np.float32)
    result = np.empty_like(values)
    result[:, : width - 1] = cumulative[:, : width - 1] / np.arange(
        1, width, dtype=np.float32
    )[None, :]
    previous = np.concatenate(
        (np.zeros((values.shape[0], 1), dtype=np.float32), cumulative[:, :-width]),
        axis=1,
    )
    result[:, width - 1 :] = (cumulative[:, width - 1 :] - previous) / float(width)
    return result


def _rising_crossing(charge: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    search = charge[:, SEARCH_START:SEARCH_STOP]
    peak_offsets = np.argmax(search, axis=1)
    peak_values = search[np.arange(search.shape[0]), peak_offsets]
    thresholds = fraction * peak_values
    positions = np.full(search.shape[0], SEARCH_START, dtype=np.int64)
    valid_peak = np.isfinite(peak_values) & (peak_values > 0.0)
    crossings = search >= thresholds[:, None]
    before_peak = np.arange(search.shape[1])[None, :] <= peak_offsets[:, None]
    found = np.any(crossings & before_peak, axis=1) & valid_peak
    positions[found] = SEARCH_START + np.argmax(
        crossings[found] & before_peak[found], axis=1
    )
    return positions, int(np.count_nonzero(~found))


def _gather_window(
    values: np.ndarray,
    anchors: np.ndarray,
    pre_samples: int,
    post_samples: int,
) -> np.ndarray:
    offsets = np.arange(-pre_samples, post_samples, dtype=np.int64)
    indices = anchors[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < values.shape[1])
    clipped = np.clip(indices, 0, values.shape[1] - 1)
    gathered = values[np.arange(values.shape[0])[:, None], clipped]
    gathered[~valid] = 0.0
    return gathered.astype(np.float32, copy=False)


def _rise_profile(
    charge: np.ndarray,
    t10: np.ndarray,
    t90: np.ndarray,
    peak_values: np.ndarray,
) -> tuple[np.ndarray, int]:
    sample_indices = np.arange(charge.shape[1], dtype=np.int64)[None, :]
    t10_values = charge[np.arange(charge.shape[0]), t10]
    t90_values = charge[np.arange(charge.shape[0]), t90]
    denominator = t90_values - t10_values
    valid = (
        np.isfinite(peak_values)
        & (peak_values > 0.0)
        & (t90 > t10)
        & np.isfinite(denominator)
        & (denominator > 1.0e-6)
    )
    middle = (sample_indices >= t10[:, None]) & (sample_indices < t90[:, None])
    after = sample_indices >= t90[:, None]
    profile = np.where(
        middle,
        (charge - t10_values[:, None]) / np.maximum(denominator[:, None], 1.0e-6),
        np.where(after, 1.0, 0.0),
    )
    profile = np.clip(profile, 0.0, 1.0).astype(np.float32, copy=False)
    profile[~valid] = 0.0
    return profile, int(np.count_nonzero(~valid))


def build_representation(
    waveforms: np.ndarray,
    representation: CascadeRepresentation,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build one candidate representation from acquired negative-polarity pulses."""

    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2 or waveforms.shape[1] != 4500:
        raise ValueError(f"Expected [N,4500] waveforms, got {waveforms.shape}")
    values = np.empty(
        (waveforms.shape[0], representation.channel_count, representation.window_length),
        dtype=np.float32,
    )
    qc = {
        "anchor_fallback_count": 0,
        "rise_invalid_event_count": 0,
        "ae_invalid_scale_event_count": 0,
        "global_scale_invalid_event_count": 0,
    }
    needs_rise = "rise_profile" in representation.channels
    needs_ae = "ae_ratio" in representation.channels
    for start in range(0, waveforms.shape[0], chunk_size):
        stop = min(start + chunk_size, waveforms.shape[0])
        acquired = waveforms[start:stop]
        positive = -acquired
        baseline = np.median(positive[:, :BASELINE_STOP], axis=1).astype(np.float32)
        charge = moving_average(positive - baseline[:, None], representation.moving_average)
        current = np.gradient(charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
        search = charge[:, SEARCH_START:SEARCH_STOP]
        peak_offsets = np.argmax(search, axis=1)
        peak_values = search[np.arange(search.shape[0]), peak_offsets]
        t10, t10_fallback = _rising_crossing(charge, 0.1)
        qc["anchor_fallback_count"] += t10_fallback
        if needs_rise:
            t90, _ = _rising_crossing(charge, 0.9)
            rise, rise_invalid = _rise_profile(charge, t10, t90, peak_values)
            qc["rise_invalid_event_count"] += rise_invalid
        else:
            rise = None
        if representation.anchor == "t10":
            anchors = t10
        elif representation.anchor == "current_peak":
            current_search = current[:, SEARCH_START:SEARCH_STOP]
            anchors = SEARCH_START + np.argmax(current_search, axis=1)
        else:
            raise ValueError(f"Unknown Stage-2 anchor: {representation.anchor}")
        charge_window = _gather_window(
            charge, anchors, representation.pre_samples, representation.post_samples
        )
        current_window = _gather_window(
            current, anchors, representation.pre_samples, representation.post_samples
        )
        scale = np.sqrt(np.mean(np.square(charge_window), axis=1, dtype=np.float64)).astype(
            np.float32
        )
        invalid_scale = ~np.isfinite(scale) | (scale <= 1.0e-12)
        qc["global_scale_invalid_event_count"] += int(np.count_nonzero(invalid_scale))
        scale[invalid_scale] = 1.0
        charge_window /= scale[:, None]
        current_window /= scale[:, None]
        channel_values: dict[str, np.ndarray] = {
            "charge": charge_window,
            "current": current_window,
        }
        if needs_rise:
            assert rise is not None
            channel_values["rise_profile"] = _gather_window(
                rise, anchors, representation.pre_samples, representation.post_samples
            )
        if needs_ae:
            ae_scale = peak_values.astype(np.float32, copy=True)
            invalid_ae = ~np.isfinite(ae_scale) | (ae_scale <= 1.0e-12)
            qc["ae_invalid_scale_event_count"] += int(np.count_nonzero(invalid_ae))
            ae_scale[invalid_ae] = 1.0
            ae_ratio = current / ae_scale[:, None]
            channel_values["ae_ratio"] = _gather_window(
                ae_ratio, anchors, representation.pre_samples, representation.post_samples
            )
        for channel_index, channel_name in enumerate(representation.channels):
            values[start:stop, channel_index] = channel_values[channel_name]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite values in {representation.name}")
    return values, qc


def fit_channel_statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    means = np.mean(values, axis=(0, 2), dtype=np.float64)
    standard_deviations = np.std(values, axis=(0, 2), dtype=np.float64)
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0.0):
        raise ValueError("Invalid channel statistics")
    return {
        "means": means.tolist(),
        "standard_deviations": standard_deviations.tolist(),
    }


def apply_channel_statistics(values: np.ndarray, statistics: dict[str, list[float]]) -> None:
    means = np.asarray(statistics["means"], dtype=np.float32)
    standard_deviations = np.asarray(statistics["standard_deviations"], dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != means.size:
        raise ValueError("Channel-statistics shape mismatch")
    values -= means[None, :, None]
    values /= standard_deviations[None, :, None]


def validate_representation(values: np.ndarray, representation: CascadeRepresentation) -> None:
    expected = (representation.channel_count, representation.window_length)
    if values.ndim != 3 or tuple(values.shape[1:]) != expected:
        raise ValueError(f"Expected [N,{expected[0]},{expected[1]}], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite values in {representation.name}")


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    peak_ids = np.asarray(peak_ids)
    if not (labels.shape == scores.shape == weights.shape == peak_ids.shape):
        raise ValueError("Metric arrays have inconsistent shapes")
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(peak_ids.tolist())):
        mask = peak_ids == peak_id
        if np.unique(labels[mask]).size < 2:
            raise ValueError(f"Metric stratum lacks both labels: {peak_id}")
        per_peak[str(peak_id)] = {
            "auroc": float(roc_auc_score(labels[mask], scores[mask])),
            "average_precision": float(average_precision_score(labels[mask], scores[mask])),
            "event_count": int(np.count_nonzero(mask)),
            "positive_count": int(np.count_nonzero(mask & (labels == 1.0))),
            "negative_count": int(np.count_nonzero(mask & (labels == 0.0))),
        }
    aurocs = [float(item["auroc"]) for item in per_peak.values()]
    return {
        "macro_auroc": float(np.mean(aurocs)),
        "worst_peak_auroc": float(np.min(aurocs)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak,
    }


def piecewise_fusion(
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
) -> np.ndarray:
    stage1_scores = np.asarray(stage1_scores, dtype=np.float64)
    stage2_scores = np.asarray(stage2_scores, dtype=np.float64)
    if stage1_scores.shape != stage2_scores.shape:
        raise ValueError("Stage-1 and Stage-2 score shapes differ")
    result = stage1_scores.copy()
    ambiguous = np.isfinite(stage2_scores) & (stage1_scores >= tau_low) & (
        stage1_scores <= tau_high
    )
    result[ambiguous] = tau_low + (tau_high - tau_low) * stage2_scores[ambiguous]
    return result


def soft_gate_fusion(
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    temperature: float,
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("Soft-gate temperature must be positive")
    stage1_scores = np.asarray(stage1_scores, dtype=np.float64)
    stage2_scores = np.asarray(stage2_scores, dtype=np.float64)
    result = stage1_scores.copy()
    ambiguous = np.isfinite(stage2_scores) & (stage1_scores >= tau_low) & (
        stage1_scores <= tau_high
    )
    mapped = tau_low + (tau_high - tau_low) * stage2_scores
    lower_gate = 1.0 / (1.0 + np.exp(-(stage1_scores - tau_low) / temperature))
    upper_gate = 1.0 / (1.0 + np.exp(-(tau_high - stage1_scores) / temperature))
    gate = lower_gate * upper_gate
    result[ambiguous] = (
        (1.0 - gate[ambiguous]) * stage1_scores[ambiguous]
        + gate[ambiguous] * mapped[ambiguous]
    )
    return result


class BivariateIsotonicCalibrator:
    """Small monotone 2-D score map fitted only on internal ambiguous events."""

    def __init__(self, bins: int = 12) -> None:
        if bins < 2:
            raise ValueError("Bivariate calibrator needs at least two bins")
        self.bins = bins
        self.stage1_edges: np.ndarray | None = None
        self.stage2_edges: np.ndarray | None = None
        self.grid: np.ndarray | None = None

    def fit(
        self,
        stage1_scores: np.ndarray,
        stage2_scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> "BivariateIsotonicCalibrator":
        stage1_scores = np.asarray(stage1_scores, dtype=np.float64)
        stage2_scores = np.asarray(stage2_scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if not (
            stage1_scores.shape == stage2_scores.shape == labels.shape == weights.shape
        ):
            raise ValueError("Bivariate calibrator arrays have inconsistent shapes")
        if stage1_scores.size == 0 or not np.all(
            np.isfinite(stage1_scores)
            & np.isfinite(stage2_scores)
            & np.isfinite(labels)
            & np.isfinite(weights)
        ):
            raise ValueError("Invalid bivariate calibrator input")
        self.stage1_edges = np.linspace(TAU_LOW, TAU_HIGH, self.bins + 1)
        self.stage2_edges = np.linspace(0.0, 1.0, self.bins + 1)
        x1 = np.clip(stage1_scores, TAU_LOW, TAU_HIGH)
        x2 = np.clip(stage2_scores, 0.0, 1.0)
        index1 = np.clip(np.digitize(x1, self.stage1_edges[1:-1], right=False), 0, self.bins - 1)
        index2 = np.clip(np.digitize(x2, self.stage2_edges[1:-1], right=False), 0, self.bins - 1)
        weighted_sum = np.zeros((self.bins, self.bins), dtype=np.float64)
        weight_sum = np.zeros((self.bins, self.bins), dtype=np.float64)
        for i, j, label, weight in zip(index1, index2, labels, weights):
            weighted_sum[i, j] += label * weight
            weight_sum[i, j] += weight
        global_rate = float(np.sum(weighted_sum) / max(np.sum(weight_sum), 1.0e-12))
        grid = np.divide(
            weighted_sum,
            np.maximum(weight_sum, 1.0e-12),
            out=np.full_like(weighted_sum, global_rate),
            where=weight_sum > 0.0,
        )
        fit_weights = np.maximum(weight_sum, 1.0e-6)
        centers2 = 0.5 * (self.stage2_edges[:-1] + self.stage2_edges[1:])
        centers1 = 0.5 * (self.stage1_edges[:-1] + self.stage1_edges[1:])
        for _ in range(8):
            for i in range(self.bins):
                grid[i] = IsotonicRegression(out_of_bounds="clip").fit_transform(
                    centers2, grid[i], sample_weight=fit_weights[i]
                )
            for j in range(self.bins):
                grid[:, j] = IsotonicRegression(out_of_bounds="clip").fit_transform(
                    centers1, grid[:, j], sample_weight=fit_weights[:, j]
                )
        self.grid = np.clip(grid, 0.0, 1.0)
        return self

    def predict(self, stage1_scores: np.ndarray, stage2_scores: np.ndarray) -> np.ndarray:
        if self.stage1_edges is None or self.stage2_edges is None or self.grid is None:
            raise RuntimeError("Calibrator has not been fitted")
        x1 = np.clip(np.asarray(stage1_scores, dtype=np.float64), TAU_LOW, TAU_HIGH)
        x2 = np.clip(np.asarray(stage2_scores, dtype=np.float64), 0.0, 1.0)
        if x1.shape != x2.shape:
            raise ValueError("Bivariate prediction arrays have inconsistent shapes")
        index1 = np.clip(np.digitize(x1, self.stage1_edges[1:-1], right=False), 0, self.bins - 1)
        index2 = np.clip(np.digitize(x2, self.stage2_edges[1:-1], right=False), 0, self.bins - 1)
        return self.grid[index1, index2]

    def as_dict(self) -> dict[str, Any]:
        if self.stage1_edges is None or self.stage2_edges is None or self.grid is None:
            raise RuntimeError("Calibrator has not been fitted")
        return {
            "type": "bivariate_coordinatewise_isotonic",
            "bins": self.bins,
            "stage1_edges": self.stage1_edges.tolist(),
            "stage2_edges": self.stage2_edges.tolist(),
            "grid": self.grid.tolist(),
        }


def isotonic_fusion(
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bins: int = 12,
) -> tuple[np.ndarray, BivariateIsotonicCalibrator]:
    stage1_scores = np.asarray(stage1_scores, dtype=np.float64)
    stage2_scores = np.asarray(stage2_scores, dtype=np.float64)
    result = stage1_scores.copy()
    ambiguous = np.isfinite(stage2_scores) & (stage1_scores >= TAU_LOW) & (
        stage1_scores <= TAU_HIGH
    )
    calibrator = BivariateIsotonicCalibrator(bins=bins).fit(
        stage1_scores[ambiguous],
        stage2_scores[ambiguous],
        np.asarray(labels)[ambiguous],
        np.asarray(weights)[ambiguous],
    )
    result[ambiguous] = calibrator.predict(stage1_scores[ambiguous], stage2_scores[ambiguous])
    return result, calibrator


def weighted_acceptance_threshold(
    scores: np.ndarray,
    weights: np.ndarray,
    acceptance: float,
) -> dict[str, float | str]:
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 1 or weights.shape != scores.shape or scores.size == 0:
        raise ValueError("Invalid threshold calibration arrays")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Invalid threshold calibration values")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    target = acceptance * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), target, side="left"))
    threshold = float(sorted_scores[min(index, sorted_scores.size - 1)])
    accepted = scores >= threshold
    return {
        "name": f"{int(round(100.0 * acceptance))}pct",
        "requested_weighted_acceptance": float(acceptance),
        "score_threshold": threshold,
        "actual_weighted_acceptance": float(np.sum(weights[accepted]) / np.sum(weights)),
        "actual_unweighted_acceptance": float(np.mean(accepted)),
    }
