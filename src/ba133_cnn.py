"""Configurable waveform representations and compact 1D CNN for Ba transfer."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

SAMPLE_PERIOD_NS = 4.0
BASELINE_STOP = 1000
SEARCH_START = 1000
SEARCH_STOP = 2000
CHUNK_SIZE = 512


@dataclass(frozen=True)
class RepresentationConfig:
    name: str
    input_mode: str
    moving_average: int
    normalization: str
    anchor: str
    pre_samples: int
    post_samples: int
    pulse_polarity: str
    standardization: str
    downsample: int = 1
    representation_schema_version: int = 1
    endpoint_inclusive: bool = False
    current_search_start: int = SEARCH_START
    current_search_stop: int = SEARCH_STOP
    clip_charge_to_unit_interval: bool = False

    @property
    def window_length(self) -> int:
        full_length = self.pre_samples + self.post_samples + int(self.endpoint_inclusive)
        return (full_length + self.downsample - 1) // self.downsample

    @property
    def channel_count(self) -> int:
        return 2 if self.input_mode == "both" else 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def representation_config_from_checkpoint(value: dict[str, Any]) -> RepresentationConfig:
    """Load a serialized config while preserving the pre-polarity legacy contract."""
    fields = dict(value)
    if "pulse_polarity" not in fields:
        fields["pulse_polarity"] = "as_acquired"
    if "standardization" not in fields:
        fields["standardization"] = "train_zscore"
    return RepresentationConfig(**fields)


SCREEN_CONFIGS = (
    RepresentationConfig("charge_ma5_energy_t10_w750", "charge", 5, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("charge_ma10_energy_t10_w750", "charge", 10, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("charge_ma20_energy_t10_w750", "charge", 20, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("current_ma5_energy_t10_w750", "current", 5, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("current_ma10_energy_current_peak_w500", "current", 10, "energy", "current_peak", 250, 250, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma1_energy_t10_w750", "both", 1, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma5_energy_t10_w750", "both", 5, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_energy_t10_w750", "both", 10, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma20_energy_t10_w750", "both", 20, "energy", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_peak_t10_w750", "both", 10, "peak", "t10", 250, 500, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_energy_charge_peak_w750", "both", 10, "energy", "charge_peak", 375, 375, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_energy_trigger_w750", "both", 10, "energy", "trigger", 375, 375, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_energy_t10_w500", "both", 10, "energy", "t10", 150, 350, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_energy_t10_w1000", "both", 10, "energy", "t10", 300, 700, "negative_to_positive", "train_zscore"),
    RepresentationConfig("both_ma10_global_t10_w750", "both", 10, "global", "t10", 250, 500, "negative_to_positive", "train_zscore"),
)


@dataclass
class RawPartition:
    waveforms: np.ndarray
    shaped_energy: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    peak_ids: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_event_references(csv_path: Path, max_events: int | None = None) -> list[tuple[str, int, int, float, str]]:
    records: list[tuple[str, int, int, float, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.extend(
                (
                    (row["positive_hdf5"], int(row["positive_row"]), 1, float(row["source_weight"]), row["peak_id"]),
                    (row["negative_hdf5"], int(row["negative_row"]), 0, float(row["source_weight"]), row["peak_id"]),
                )
            )
            if max_events is not None and len(records) >= max_events:
                break
    if max_events is not None:
        records = records[: max_events - (max_events % 2)]
    if not records or {record[2] for record in records} != {0, 1}:
        raise ValueError(f"Invalid event selection from {csv_path}")
    return records


def load_raw_partition(csv_path: Path, event_store_dir: Path, max_events: int | None = None) -> RawPartition:
    records = read_event_references(csv_path, max_events)
    partition = csv_path.stem.removeprefix("label_pairs_")
    lookup_path = event_store_dir / f"event_lookup_{partition}.csv"
    store_path = event_store_dir / f"{partition}_events.h5"
    required = {(record[0], record[1]) for record in records}
    lookup: dict[tuple[str, int], int] = {}
    with lookup_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_hdf5"], int(row["source_row"]))
            if key in required:
                lookup[key] = int(row["store_index"])
    missing = required - set(lookup)
    if missing:
        raise KeyError(f"Event store missing {len(missing)} requested events")
    store_rows = np.asarray([lookup[(record[0], record[1])] for record in records], dtype=np.int64)
    if np.unique(store_rows).size != store_rows.size:
        raise ValueError("Event reuse detected")
    order = np.argsort(store_rows)
    waveforms = np.empty((len(records), 4500), dtype=np.float32)
    shaped_energy = np.empty(len(records), dtype=np.float32)
    with h5py.File(store_path, "r") as handle:
        for start in range(0, len(records), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(records))
            rows = store_rows[order[start:stop]]
            destinations = order[start:stop]
            waveforms[destinations] = handle["waveform"][rows]
            shaped_energy[destinations] = handle["shaped_energy_unit"][rows]
    return RawPartition(
        waveforms=waveforms,
        shaped_energy=shaped_energy,
        labels=np.asarray([record[2] for record in records], dtype=np.float32),
        weights=np.asarray([record[3] for record in records], dtype=np.float32),
        peak_ids=np.asarray([record[4] for record in records], dtype="U64"),
    )


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return values.copy()
    cumulative = np.cumsum(values, axis=1, dtype=np.float32)
    result = np.empty_like(values)
    result[:, : width - 1] = cumulative[:, : width - 1] / np.arange(1, width, dtype=np.float32)[None, :]
    previous = np.concatenate((np.zeros((values.shape[0], 1), dtype=np.float32), cumulative[:, :-width]), axis=1)
    result[:, width - 1 :] = (cumulative[:, width - 1 :] - previous) / float(width)
    return result


def fraction_anchor(
    charge: np.ndarray, fraction: float
) -> tuple[np.ndarray, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    search = charge[:, SEARCH_START:SEARCH_STOP]
    peak_offsets = np.argmax(search, axis=1)
    peak_values = search[np.arange(search.shape[0]), peak_offsets]
    thresholds = fraction * peak_values
    positions = np.full(search.shape[0], SEARCH_START, dtype=np.int64)
    valid = peak_values > 0
    crossings = search >= thresholds[:, None]
    before_peak = np.arange(search.shape[1])[None, :] <= peak_offsets[:, None]
    found = np.any(crossings & before_peak, axis=1) & valid
    positions[found] = SEARCH_START + np.argmax(crossings[found] & before_peak[found], axis=1)
    return positions, int(np.count_nonzero(~found))


def t10_anchor(charge: np.ndarray) -> tuple[np.ndarray, int]:
    return fraction_anchor(charge, 0.1)


def anchor_positions(charge: np.ndarray, current: np.ndarray, anchor: str) -> tuple[np.ndarray, int]:
    if anchor == "t10":
        return t10_anchor(charge)
    if anchor == "t50":
        return fraction_anchor(charge, 0.5)
    if anchor == "t90":
        return fraction_anchor(charge, 0.9)
    if anchor == "charge_peak":
        return SEARCH_START + np.argmax(charge[:, SEARCH_START:SEARCH_STOP], axis=1), 0
    if anchor == "current_peak":
        return SEARCH_START + np.argmax(np.abs(current[:, SEARCH_START:SEARCH_STOP]), axis=1), 0
    if anchor == "trigger":
        return np.full(charge.shape[0], 1500, dtype=np.int64), 0
    raise ValueError(f"Unknown anchor: {anchor}")


def gather_window(
    values: np.ndarray,
    anchors: np.ndarray,
    pre: int,
    post: int,
    endpoint_inclusive: bool = False,
) -> np.ndarray:
    offsets = np.arange(-pre, post + int(endpoint_inclusive), dtype=np.int64)
    indices = anchors[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < values.shape[1])
    clipped = np.clip(indices, 0, values.shape[1] - 1)
    result = values[np.arange(values.shape[0])[:, None], clipped]
    result[~valid] = 0.0
    return result.astype(np.float32, copy=False)


def build_representation(raw: RawPartition, config: RepresentationConfig) -> tuple[np.ndarray, dict[str, int]]:
    values = np.empty(
        (raw.labels.size, config.channel_count, config.window_length),
        dtype=np.float32,
    )
    fallback_count = 0
    invalid_scale_count = 0
    charge_clipped_sample_count = 0
    charge_clipped_event_count = 0
    for start in range(0, raw.labels.size, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, raw.labels.size)
        acquired_waveforms = raw.waveforms[start:stop]
        if config.pulse_polarity == "negative_to_positive":
            waveforms = -acquired_waveforms
        elif config.pulse_polarity == "as_acquired":
            waveforms = acquired_waveforms
        else:
            raise ValueError(f"Unknown pulse polarity: {config.pulse_polarity}")
        baseline = np.median(waveforms[:, :BASELINE_STOP], axis=1).astype(np.float32)
        charge = waveforms.astype(np.float32, copy=True)
        charge -= baseline[:, None]
        charge = moving_average(charge, config.moving_average)
        current = np.gradient(charge, SAMPLE_PERIOD_NS, axis=1).astype(np.float32)
        if config.anchor == "dual_t10_current_peak":
            charge_anchors, chunk_fallback_count = t10_anchor(charge)
            current_anchors = (
                config.current_search_start
                + np.argmax(
                    current[:, config.current_search_start:config.current_search_stop],
                    axis=1,
                )
            )
            charge_window = gather_window(
                charge,
                charge_anchors,
                config.pre_samples,
                config.post_samples,
                config.endpoint_inclusive,
            )
            current_window = gather_window(
                current,
                current_anchors,
                config.pre_samples,
                config.post_samples,
                config.endpoint_inclusive,
            )
        else:
            anchors, chunk_fallback_count = anchor_positions(
                charge, current, config.anchor
            )
            charge_window = gather_window(
                charge, anchors, config.pre_samples, config.post_samples, config.endpoint_inclusive
            )
            current_window = gather_window(
                current, anchors, config.pre_samples, config.post_samples, config.endpoint_inclusive
            )
        fallback_count += chunk_fallback_count
        if config.normalization == "independent_positive_peak":
            charge_scale = np.max(charge[:, SEARCH_START:SEARCH_STOP], axis=1)
            if config.anchor != "dual_t10_current_peak":
                raise ValueError("Independent peak normalization requires dual anchors")
            current_scale = current[
                np.arange(current.shape[0]), current_anchors
            ].astype(np.float32, copy=True)
            invalid_charge = ~np.isfinite(charge_scale) | (charge_scale <= 1.0e-12)
            invalid_current = ~np.isfinite(current_scale) | (current_scale <= 1.0e-12)
            invalid_scale_count += int(
                np.count_nonzero(invalid_charge) + np.count_nonzero(invalid_current)
            )
            charge_scale[invalid_charge] = 1.0
            current_scale[invalid_current] = 1.0
            charge_window /= charge_scale[:, None]
            current_window /= current_scale[:, None]
            if config.clip_charge_to_unit_interval:
                clipped = (charge_window < 0.0) | (charge_window > 1.0)
                charge_clipped_sample_count += int(np.count_nonzero(clipped))
                charge_clipped_event_count += int(np.count_nonzero(np.any(clipped, axis=1)))
                np.clip(charge_window, 0.0, 1.0, out=charge_window)
        elif config.normalization == "charge_peak_shared":
            charge_scale = np.max(charge[:, SEARCH_START:SEARCH_STOP], axis=1)
            invalid_scale = ~np.isfinite(charge_scale) | (charge_scale <= 1.0e-12)
            invalid_scale_count += int(np.count_nonzero(invalid_scale))
            charge_scale[invalid_scale] = 1.0
            charge_window /= charge_scale[:, None]
            current_window /= charge_scale[:, None]
        else:
            if config.normalization == "energy":
                scale = raw.shaped_energy[start:stop].astype(np.float32, copy=True)
            elif config.normalization == "peak":
                scale = np.max(np.abs(charge_window), axis=1)
            elif config.normalization == "global":
                scale = np.sqrt(np.mean(np.square(charge_window), axis=1))
            elif config.normalization == "none":
                scale = np.ones(charge_window.shape[0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown normalization: {config.normalization}")
            invalid_scale = ~np.isfinite(scale) | (scale <= 1.0e-12)
            invalid_scale_count += int(np.count_nonzero(invalid_scale))
            scale[invalid_scale] = 1.0
            charge_window /= scale[:, None]
            current_window /= scale[:, None]
        if config.downsample > 1:
            charge_window = charge_window[:, :: config.downsample]
            current_window = current_window[:, :: config.downsample]
        if config.input_mode == "charge":
            values[start:stop, 0] = charge_window
        elif config.input_mode == "current":
            values[start:stop, 0] = current_window
        elif config.input_mode == "both":
            values[start:stop, 0] = charge_window
            values[start:stop, 1] = current_window
        else:
            raise ValueError(f"Unknown input mode: {config.input_mode}")
    return values, {
        "anchor_fallback_count": fallback_count,
        "invalid_scale_count": invalid_scale_count,
        "charge_clipped_sample_count": charge_clipped_sample_count,
        "charge_clipped_event_count": charge_clipped_event_count,
    }


def fit_channel_statistics(train_values: np.ndarray) -> dict[str, list[float]]:
    means = np.mean(train_values, axis=(0, 2), dtype=np.float64)
    stds = np.std(train_values, axis=(0, 2), dtype=np.float64)
    if np.any(~np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("Invalid channel statistics")
    return {"means": means.tolist(), "standard_deviations": stds.tolist()}


def apply_channel_statistics(values: np.ndarray, statistics: dict[str, list[float]]) -> None:
    means = np.asarray(statistics["means"], dtype=np.float32)
    stds = np.asarray(statistics["standard_deviations"], dtype=np.float32)
    values -= means[None, :, None]
    values /= stds[None, :, None]


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=padding, dilation=dilation, bias=False),
            group_norm(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=padding, dilation=dilation, bias=False),
            group_norm(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.main(values) + self.skip(values))


class CompactWaveformCNN(nn.Module):
    def __init__(self, input_channels: int, width: int = 24, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, width, 11, stride=2, padding=5, bias=False),
            group_norm(width),
            nn.GELU(),
            ResidualBlock(width, width, 1, 1),
            ResidualBlock(width, 2 * width, 2, 1),
            ResidualBlock(2 * width, 3 * width, 2, 2),
            ResidualBlock(3 * width, 3 * width, 2, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(6 * width, 3 * width),
            nn.LayerNorm(3 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * width, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        features = self.features(values)
        pooled = torch.cat((torch.mean(features, dim=2), torch.amax(features, dim=2)), dim=1)
        return self.head(pooled).squeeze(1)


def make_loader(values: np.ndarray, raw: RawPartition, batch_size: int, shuffle: bool, seed: int) -> DataLoader[tuple[Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(raw.labels), torch.from_numpy(raw.weights)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def metric_summary(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(average_precision_score(labels, scores, sample_weight=weights)),
    }


def evaluate_model(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for values, target, sample_weight in loader:
            values = values.to(device)
            target = target.to(device)
            sample_weight = sample_weight.to(device)
            logits = model(values)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss_sum += float((losses * sample_weight).sum().item())
            labels.append(target.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            weights.append(sample_weight.cpu().numpy())
    label_array = np.concatenate(labels)
    score_array = np.concatenate(scores)
    weight_array = np.concatenate(weights)
    result = metric_summary(label_array, score_array, weight_array)
    result["loss"] = loss_sum / float(np.sum(weight_array))
    return result, score_array


def train_epoch(model: nn.Module, loader: DataLoader[tuple[Tensor, ...]], optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    weight_sum = 0.0
    for values, target, sample_weight in loader:
        values = values.to(device)
        target = target.to(device)
        sample_weight = sample_weight.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (losses * sample_weight).sum() / sample_weight.sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += float((losses * sample_weight).sum().item())
        weight_sum += float(sample_weight.sum().item())
    return loss_sum / weight_sum
