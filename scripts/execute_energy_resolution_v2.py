#!/usr/bin/env python3
"""Execute the V2 offline Co-60 energy-resolution experiment.

The executor reads immutable raw HPGE trees, caches only waveform copies under
``processed_data/``, freezes a file-level optimization/validation split, and
records control, scan, benchmark, and locked-validation artifacts under
``outputs/energy_resolution_experiment/``.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import curve_fit


E1_KEV = 1173.228
E2_KEV = 1332.492
EXPECTED_SAMPLES = 4500
SAMPLE_PERIOD_US = 0.004
RAW_STEP = 2000
FIT_BIN_WIDTH_KEV = 0.25
FIT_WINDOWS_KEV = ((1110.0, 1235.0), (1270.0, 1400.0))
INITIAL_CALIBRATION = (13.670696656992694, -4.2874237270975755)
CONTROL_CONFIG = {
    "baseline": "control_pz_first1000",
    "pole_zero_tau_us": 100.0,
    "rise_samples": 1125,
    "flat_samples": 200,
    "estimator": "global_max",
}
REASON_NONFINITE = 1
REASON_LOWER_RAIL = 2
REASON_UPPER_RAIL = 4
REASON_TIMESTAMP = 8


@dataclass(frozen=True)
class CacheFile:
    raw_path: Path
    waveform_path: Path
    metadata_path: Path
    entries: int
    file_index: int


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def gaussian_linear(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_peak(values: np.ndarray, low: float, high: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= low) & (values <= high)]
    if values.size < 100:
        raise RuntimeError(f"only {values.size} events in fit window {low}:{high}")
    edges = np.arange(low, high + FIT_BIN_WIDTH_KEV, FIT_BIN_WIDTH_KEV)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_index = int(np.argmax(counts))
    mean0 = float(centers[peak_index])
    sigma0 = min(8.0, max(0.8, float(np.std(values[(values > mean0 - 8.0) & (values < mean0 + 8.0)]))))
    amplitude0 = float(max(1.0, counts[peak_index]))
    background0 = float(np.median(counts))
    fitted, covariance = curve_fit(
        gaussian_linear,
        centers,
        counts,
        p0=[amplitude0, mean0, sigma0, background0, 0.0],
        bounds=([0.0, low, 0.15, -np.inf, -np.inf], [np.inf, high, 25.0, np.inf, np.inf]),
        maxfev=100000,
    )
    errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    sigma = float(fitted[2])
    fwhm = float(2.354820045 * sigma)
    area = float(fitted[0] * sigma * math.sqrt(2.0 * math.pi) / FIT_BIN_WIDTH_KEV)
    return {
        "centroid_kev": float(fitted[1]),
        "centroid_error_kev": float(errors[1]),
        "sigma_kev": sigma,
        "fwhm_kev": fwhm,
        "fwhm_error_kev": float(2.354820045 * errors[2]),
        "net_gaussian_events": area,
        "fit_events": int(values.size),
        "fit_window_kev": [low, high],
        "fit_bin_width_kev": FIT_BIN_WIDTH_KEV,
    }


def fit_two_peaks(energy: np.ndarray) -> dict[str, dict[str, float | int]]:
    return {
        "1173_228_kev": fit_peak(energy, *FIT_WINDOWS_KEV[0]),
        "1332_492_kev": fit_peak(energy, *FIT_WINDOWS_KEV[1]),
    }


def objective(fits: dict[str, dict[str, float | int]]) -> float:
    return float(max(fits["1173_228_kev"]["fwhm_kev"], fits["1332_492_kev"]["fwhm_kev"]))


def apply_two_point_calibration(
    approx_energy: np.ndarray,
    approximate_calibration: tuple[float, float],
) -> tuple[tuple[float, float], dict[str, dict[str, float | int]]]:
    fits = fit_two_peaks(approx_energy)
    m1 = float(fits["1173_228_kev"]["centroid_kev"])
    m2 = float(fits["1332_492_kev"]["centroid_kev"])
    if not m2 > m1:
        raise RuntimeError(f"invalid peak order in calibration: {m1}, {m2}")
    # The peak fit is performed in approximate-energy coordinates.  Compose
    # the two-point correction with the approximate amplitude calibration;
    # returning the correction coefficients alone would incorrectly apply
    # keV-per-keV coefficients directly to shaped amplitudes.
    correction_slope = (E2_KEV - E1_KEV) / (m2 - m1)
    correction_intercept = E1_KEV - correction_slope * m1
    approximate_slope, approximate_intercept = approximate_calibration
    slope = correction_slope * approximate_slope
    intercept = correction_slope * approximate_intercept + correction_intercept
    return (float(slope), float(intercept)), fits


def calibration_from_amplitude(
    amplitude: np.ndarray,
    valid: np.ndarray,
    approximate_calibration: tuple[float, float],
) -> tuple[tuple[float, float], dict[str, dict[str, float | int]]]:
    slope, intercept = approximate_calibration
    approx_energy = slope * amplitude[valid] + intercept
    return apply_two_point_calibration(approx_energy, approximate_calibration)


def baseline_correct(inverted: np.ndarray, method: str) -> np.ndarray:
    if method in {"control_pz_first1000", "mean_1000"}:
        baseline = inverted[:, :1000].mean(axis=1, keepdims=True)
        return inverted - baseline
    if method == "mean_1250":
        return inverted - inverted[:, :1250].mean(axis=1, keepdims=True)
    if method == "mean_1375":
        return inverted - inverted[:, :1375].mean(axis=1, keepdims=True)
    if method == "clip_1250":
        window = inverted[:, :1250]
        center = np.median(window, axis=1, keepdims=True)
        deviation = np.abs(window - center)
        scale = 1.4826 * np.median(deviation, axis=1, keepdims=True)
        keep = deviation <= np.maximum(3.0 * scale, 0.5)
        baseline = np.divide(
            np.where(keep, window, 0.0).sum(axis=1, keepdims=True),
            keep.sum(axis=1, keepdims=True),
            out=np.zeros((len(window), 1), dtype=np.float64),
            where=keep.sum(axis=1, keepdims=True) > 0,
        )
        return inverted - baseline
    if method == "clip_1375":
        window = inverted[:, :1375]
        center = np.median(window, axis=1, keepdims=True)
        deviation = np.abs(window - center)
        scale = 1.4826 * np.median(deviation, axis=1, keepdims=True)
        keep = deviation <= np.maximum(3.0 * scale, 0.5)
        baseline = np.divide(
            np.where(keep, window, 0.0).sum(axis=1, keepdims=True),
            keep.sum(axis=1, keepdims=True),
            out=np.zeros((len(window), 1), dtype=np.float64),
            where=keep.sum(axis=1, keepdims=True) > 0,
        )
        return inverted - baseline
    if method == "linear_1250":
        count = 1250
        t = np.arange(count, dtype=np.float64)
        sum_t = float(t.sum())
        sum_t2 = float(np.dot(t, t))
        window = inverted[:, :count]
        sum_y = window.sum(axis=1)
        sum_ty = window @ t
        denominator = count * sum_t2 - sum_t * sum_t
        slope = (count * sum_ty - sum_t * sum_y) / denominator
        intercept = (sum_y - slope * sum_t) / count
        full_t = np.arange(inverted.shape[1], dtype=np.float64)
        return inverted - (intercept[:, None] + slope[:, None] * full_t[None, :])
    raise ValueError(f"unknown baseline method: {method}")


def pole_zero_correct(signal: np.ndarray, tau_us: float | None) -> np.ndarray:
    centered = signal - signal[:, :1000].mean(axis=1, keepdims=True)
    if tau_us is None:
        return centered
    alpha = float(np.exp(-SAMPLE_PERIOD_US / tau_us))
    difference = np.empty_like(centered)
    difference[:, 0] = centered[:, 0]
    difference[:, 1:] = centered[:, 1:] - alpha * centered[:, :-1]
    corrected = np.cumsum(difference, axis=1, dtype=np.float32)
    corrected -= corrected[:, :1000].mean(axis=1, keepdims=True)
    return corrected


def trapezoid_values(signal: np.ndarray, rise: int, flat: int) -> tuple[np.ndarray, int]:
    stop = signal.shape[1]
    positions = stop - 2 * rise - flat + 1
    if positions <= 0:
        raise ValueError(f"invalid shaping support: rise={rise}, flat={flat}")
    prefix = np.empty((signal.shape[0], stop + 1), dtype=np.float32)
    prefix[:, 0] = 0.0
    np.cumsum(signal, axis=1, dtype=np.float32, out=prefix[:, 1:])
    first = prefix[:, rise:rise + positions] - prefix[:, :positions]
    delayed = rise + flat
    second = prefix[:, delayed + rise:delayed + rise + positions] - prefix[:, delayed:delayed + positions]
    return (second - first) / float(rise), positions


def timing_t50(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region = signal[:, 1000:3500]
    peak = np.max(region, axis=1)
    threshold = 0.5 * peak
    crossings = region >= threshold[:, None]
    present = crossings.any(axis=1) & np.isfinite(peak) & (peak > 0.0)
    index = np.argmax(crossings, axis=1).astype(np.float64) + 1000.0
    index[~present] = np.nan
    return index, present


def extract_feature(
    raw: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inverted = -raw.astype(np.float32, copy=False)
    baseline_method = str(config.get("baseline", "control_pz_first1000"))
    baseline_signal = baseline_correct(inverted, baseline_method)
    filtered = pole_zero_correct(baseline_signal, config.get("pole_zero_tau_us"))
    rise = int(config["rise_samples"])
    flat = int(config["flat_samples"])
    values, positions = trapezoid_values(filtered, rise, flat)
    estimator = str(config.get("estimator", "global_max"))
    t50, t50_valid = timing_t50(baseline_signal)
    if estimator == "global_max":
        amplitude = np.max(values, axis=1)
        chosen_position = np.argmax(values, axis=1).astype(np.float64)
        method_valid = np.isfinite(amplitude) & (amplitude > 0.0)
    else:
        offset = float(config["offset_samples"])
        chosen_position = np.zeros(len(t50), dtype=np.int64)
        finite_t50 = np.isfinite(t50)
        chosen_position[finite_t50] = np.rint(t50[finite_t50] + offset).astype(np.int64)
        support = t50_valid & (chosen_position >= 0) & (chosen_position < positions)
        chosen_position = np.clip(chosen_position, 0, max(0, positions - 1))
        row = np.arange(len(values))
        if estimator == "fixed_position":
            amplitude = values[row, chosen_position]
        elif estimator == "flat_top":
            half_width = int(config["flat_half_width_samples"])
            local = np.arange(-half_width, half_width + 1, dtype=np.int64)
            support &= (chosen_position - half_width >= 0) & (chosen_position + half_width < positions)
            safe = np.clip(chosen_position[:, None] + local[None, :], 0, max(0, positions - 1))
            amplitude = np.take_along_axis(values, safe, axis=1).mean(axis=1)
        else:
            raise ValueError(f"unknown estimator: {estimator}")
        method_valid = support & np.isfinite(amplitude) & (amplitude > 0.0)
    baseline_mean = baseline_signal[:, :1000].mean(axis=1)
    baseline_rms = np.sqrt(np.mean(np.square(baseline_signal[:, :500]), axis=1))
    return amplitude.astype(np.float64), method_valid, chosen_position, t50, np.column_stack((baseline_mean, baseline_rms))


def read_raw_batches(path: Path, step: int = RAW_STEP) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with uproot.open(path) as root_file:
        tree = root_file["HPGE"]
        for arrays in tree.iterate(["event"], step_size=step, library="np"):
            event = arrays["event"]
            yield (
                np.asarray(event["waveform"], dtype=np.float32),
                np.asarray(event["event_id"], dtype=np.uint32),
                np.asarray(event["trigger_time_s"], dtype=np.float64),
            )


def cache_raw_file(raw_path: Path, cache_dir: Path, file_index: int) -> CacheFile:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = raw_path.stem
    waveform_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.npz"
    with uproot.open(raw_path) as root_file:
        tree = root_file["HPGE"]
        entries = int(tree.num_entries)
    if waveform_path.exists() and metadata_path.exists():
        with np.load(metadata_path, allow_pickle=False) as metadata:
            if int(metadata["entries"]) == entries and len(metadata["event_id"]) == entries:
                return CacheFile(raw_path, waveform_path, metadata_path, entries, file_index)
    mmap = np.lib.format.open_memmap(
        waveform_path,
        mode="w+",
        dtype=np.float32,
        shape=(entries, EXPECTED_SAMPLES),
    )
    event_ids = np.empty(entries, dtype=np.uint32)
    trigger_times = np.empty(entries, dtype=np.float64)
    reason_bits = np.zeros(entries, dtype=np.uint16)
    cursor = 0
    for waveforms, ids, timestamps in read_raw_batches(raw_path):
        if waveforms.ndim != 2 or waveforms.shape[1] != EXPECTED_SAMPLES:
            raise ValueError(f"{raw_path}: waveform shape {waveforms.shape} != (*, {EXPECTED_SAMPLES})")
        count = len(waveforms)
        end = cursor + count
        mmap[cursor:end] = waveforms
        event_ids[cursor:end] = ids
        trigger_times[cursor:end] = timestamps
        finite = np.isfinite(waveforms).all(axis=1)
        lower = np.any(waveforms <= 0.0, axis=1)
        upper = np.any(waveforms >= 4095.0, axis=1)
        timestamp_valid = np.isfinite(timestamps)
        reason_bits[cursor:end] = (
            (~finite).astype(np.uint16) * REASON_NONFINITE
            | lower.astype(np.uint16) * REASON_LOWER_RAIL
            | upper.astype(np.uint16) * REASON_UPPER_RAIL
            | (~timestamp_valid).astype(np.uint16) * REASON_TIMESTAMP
        )
        cursor = end
    mmap.flush()
    del mmap
    np.savez_compressed(
        metadata_path,
        entries=np.array(entries, dtype=np.int64),
        event_id=event_ids,
        trigger_time_s=trigger_times,
        reason_bits=reason_bits,
    )
    return CacheFile(raw_path, waveform_path, metadata_path, entries, file_index)


def cached_batches(cache_files: list[CacheFile]) -> Iterable[tuple[CacheFile, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    for cache in cache_files:
        waveforms = np.load(cache.waveform_path, mmap_mode="r")
        with np.load(cache.metadata_path, allow_pickle=False) as metadata:
            event_ids = np.asarray(metadata["event_id"])
            timestamps = np.asarray(metadata["trigger_time_s"])
            reason_bits = np.asarray(metadata["reason_bits"])
            for start in range(0, cache.entries, RAW_STEP):
                stop = min(cache.entries, start + RAW_STEP)
                yield (
                    cache,
                    np.asarray(waveforms[start:stop]),
                    event_ids[start:stop],
                    timestamps[start:stop],
                    reason_bits[start:stop],
                )


def collect_features(
    cache_files: list[CacheFile],
    config: dict[str, object],
    keep_diagnostics: bool = False,
) -> dict[str, np.ndarray]:
    amplitudes: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    t50s: list[np.ndarray] = []
    file_indices: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    baseline_diagnostics: list[np.ndarray] = []
    for cache, waveforms, ids, _, reason_bits in cached_batches(cache_files):
        amplitude, method_valid, chosen_position, t50, baseline_diag = extract_feature(waveforms, config)
        admitted = reason_bits == 0
        amplitudes.append(amplitude)
        valid.append(admitted & method_valid)
        positions.append(chosen_position)
        t50s.append(t50)
        file_indices.append(np.full(len(ids), cache.file_index, dtype=np.int16))
        event_ids.append(ids.astype(np.uint32, copy=False))
        if keep_diagnostics:
            baseline_diagnostics.append(baseline_diag)
    result = {
        "amplitude": np.concatenate(amplitudes),
        "valid": np.concatenate(valid),
        "position": np.concatenate(positions),
        "t50": np.concatenate(t50s),
        "file_index": np.concatenate(file_indices),
        "event_id": np.concatenate(event_ids),
    }
    if baseline_diagnostics:
        result["baseline_diagnostics"] = np.concatenate(baseline_diagnostics)
    return result


def _collect_features_one_cache(
    arguments: tuple[CacheFile, dict[str, object], bool],
) -> dict[str, np.ndarray]:
    cache, config, keep_diagnostics = arguments
    return collect_features([cache], config, keep_diagnostics=keep_diagnostics)


def _collect_candidate_cache(
    arguments: tuple[str, CacheFile, dict[str, object]],
) -> tuple[str, dict[str, np.ndarray]]:
    name, cache, config = arguments
    return name, collect_features([cache], config)


def collect_features_parallel(
    cache_files: list[CacheFile],
    config: dict[str, object],
    keep_diagnostics: bool = False,
) -> dict[str, np.ndarray]:
    """Reconstruct independent files concurrently for scan throughput."""
    if len(cache_files) <= 1:
        return collect_features(cache_files, config, keep_diagnostics=keep_diagnostics)
    arguments = [(cache, config, keep_diagnostics) for cache in cache_files]
    with ProcessPoolExecutor(max_workers=len(cache_files)) as executor:
        parts = list(executor.map(_collect_features_one_cache, arguments))
    result = {
        name: np.concatenate([part[name] for part in parts])
        for name in parts[0]
        if name != "baseline_diagnostics"
    }
    if keep_diagnostics and "baseline_diagnostics" in parts[0]:
        result["baseline_diagnostics"] = np.concatenate(
            [part["baseline_diagnostics"] for part in parts]
        )
    return result


def evaluate_candidate_batch(
    caches: list[CacheFile],
    candidates: list[tuple[str, dict[str, object]]],
    approximate_calibration: tuple[float, float],
) -> list[dict[str, object]]:
    """Evaluate a small candidate batch with one worker per file/config pair."""
    if not candidates:
        return []
    tasks = [
        (name, cache, config)
        for name, config in candidates
        for cache in caches
    ]
    with ProcessPoolExecutor(max_workers=min(15, len(tasks))) as executor:
        raw_results = list(executor.map(_collect_candidate_cache, tasks))
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for name, features in raw_results:
        grouped.setdefault(name, {})
        for field, values in features.items():
            grouped[name].setdefault(field, []).append(values)
    summaries: list[dict[str, object]] = []
    for name, config in candidates:
        features = {
            field: np.concatenate(parts)
            for field, parts in grouped[name].items()
        }
        try:
            evaluation = evaluate_optimization(features, approximate_calibration)
        except (RuntimeError, ValueError) as error:
            evaluation = {
                "status": "FAILED",
                "error": str(error),
                "objective_kev": 1.0e9,
                "valid_events": int(features["valid"].sum()),
                "total_events": int(len(features["valid"])),
                "retention": float(features["valid"].mean()),
            }
        summaries.append(candidate_summary(name, config, evaluation))
    return summaries


def save_features(path: Path, features: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **features)


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def evaluate_optimization(
    features: dict[str, np.ndarray],
    approximate_calibration: tuple[float, float],
) -> dict[str, object]:
    amplitude = features["amplitude"]
    valid = features["valid"]
    calibration, approximate_fits = calibration_from_amplitude(amplitude, valid, approximate_calibration)
    energy = calibration[0] * amplitude + calibration[1]
    fits = fit_two_peaks(energy[valid])
    return {
        "status": "OK",
        "calibration": {"slope_kev_per_unit": calibration[0], "intercept_kev": calibration[1]},
        "approximate_fits": approximate_fits,
        "fits": fits,
        "objective_kev": objective(fits),
        "valid_events": int(valid.sum()),
        "total_events": int(len(valid)),
        "retention": float(valid.mean()),
    }


def evaluate_validation(
    features: dict[str, np.ndarray],
    calibration: tuple[float, float],
    valid_override: np.ndarray | None = None,
) -> dict[str, object]:
    valid = features["valid"] if valid_override is None else valid_override
    energy = calibration[0] * features["amplitude"] + calibration[1]
    fits = fit_two_peaks(energy[valid])
    return {
        "fits": fits,
        "objective_kev": objective(fits),
        "valid_events": int(valid.sum()),
        "total_events": int(len(valid)),
        "retention": float(valid.mean()),
    }


def candidate_summary(
    name: str,
    config: dict[str, object],
    optimization: dict[str, object],
) -> dict[str, object]:
    return {"name": name, "config": config, **optimization}


def fit_per_file(features: dict[str, np.ndarray], calibration: tuple[float, float], file_count: int) -> list[dict[str, object]]:
    energy = calibration[0] * features["amplitude"] + calibration[1]
    reports: list[dict[str, object]] = []
    for file_index in range(file_count):
        mask = features["valid"] & (features["file_index"] == file_index)
        if int(mask.sum()) < 100:
            reports.append({"file_index": file_index, "valid_events": int(mask.sum()), "status": "INSUFFICIENT"})
            continue
        fits = fit_two_peaks(energy[mask])
        reports.append({
            "file_index": file_index,
            "valid_events": int(mask.sum()),
            "status": "OK",
            "objective_kev": objective(fits),
            "fits": fits,
        })
    return reports


def plot_gain_drift(reports: list[dict[str, object]], output: Path) -> None:
    indices = [r["file_index"] for r in reports if r.get("status") == "OK"]
    c1 = [r["fits"]["1173_228_kev"]["centroid_kev"] for r in reports if r.get("status") == "OK"]
    c2 = [r["fits"]["1332_492_kev"]["centroid_kev"] for r in reports if r.get("status") == "OK"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(indices, c1, "o-", label="1173.228 keV")
    ax.plot(indices, c2, "o-", label="1332.492 keV")
    ax.axhline(E1_KEV, color="C0", linestyle="--", linewidth=0.8)
    ax.axhline(E2_KEV, color="C1", linestyle="--", linewidth=0.8)
    ax.set(xlabel="File index", ylabel="Centroid (keV)", title="CONTROL_V0 per-file centroid drift")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_scan(entries: list[dict[str, object]], x_key: str, output: Path, title: str, xlabel: str) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        if entry.get("status") != "OK":
            continue
        config = entry.get("config", {})
        group = str(config.get("flat_samples", config.get("pole_zero_tau_us", "all")))
        groups.setdefault(group, []).append(entry)
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, values in groups.items():
        def scan_x(item: dict[str, object]) -> float:
            value = item["config"].get(x_key, 0.0)
            return -1.0 if value is None else float(value)
        values = sorted(values, key=scan_x)
        ax.plot(
            [scan_x(item) for item in values],
            [float(item["objective_kev"]) for item in values],
            "o-",
            label=f"flat/tau={group}",
        )
    ax.set(xlabel=xlabel, ylabel="Optimization max FWHM (keV)", title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_compare(control_energy: np.ndarray, final_energy: np.ndarray, output: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    bins = np.arange(1080.0, 1410.0, 0.25)
    axes[0].hist(control_energy, bins=bins, histtype="step", label="CONTROL_V0")
    axes[0].hist(final_energy, bins=bins, histtype="step", label="Final", alpha=0.8)
    axes[0].set_ylabel("Counts / 0.25 keV")
    axes[0].legend()
    axes[1].hist(control_energy, bins=bins, histtype="step", label="CONTROL_V0")
    axes[1].hist(final_energy, bins=bins, histtype="step", label="Final", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Reconstructed energy (keV)")
    axes[1].set_ylabel("Counts / 0.25 keV")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


FILTER_SAMPLES = 3000
NOISE_SAMPLES = 1000


def collect_optimal_filter_features(
    cache_files: list[CacheFile],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    amplitudes: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    file_indices: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    t50s: list[np.ndarray] = []
    for cache, waveforms, ids, _, reason_bits in cached_batches(cache_files):
        signal = baseline_correct(-waveforms.astype(np.float32, copy=False), "control_pz_first1000")
        window = signal[:, :FILTER_SAMPLES]
        amplitude = window @ weights
        t50, _ = timing_t50(signal)
        amplitudes.append(np.asarray(amplitude, dtype=np.float64))
        valid.append((reason_bits == 0) & np.isfinite(amplitude))
        file_indices.append(np.full(len(ids), cache.file_index, dtype=np.int16))
        event_ids.append(ids.astype(np.uint32, copy=False))
        t50s.append(t50)
    return {
        "amplitude": np.concatenate(amplitudes),
        "valid": np.concatenate(valid),
        "file_index": np.concatenate(file_indices),
        "event_id": np.concatenate(event_ids),
        "t50": np.concatenate(t50s),
    }


def build_optimal_filter(
    opt_caches: list[CacheFile],
    control_opt: dict[str, np.ndarray],
    control_calibration: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    control_energy = control_calibration[0] * control_opt["amplitude"] + control_calibration[1]
    template_parts: list[np.ndarray] = []
    selected_events = 0
    for cache, waveforms, ids, _, _ in cached_batches(opt_caches):
        target = (
            (control_opt["file_index"] == cache.file_index)
            & control_opt["valid"]
            & (control_energy >= E1_KEV - 20.0)
            & (control_energy <= E1_KEV + 20.0)
            & np.isin(control_opt["event_id"], ids)
        )
        selected_ids = control_opt["event_id"][target]
        keep = np.isin(ids, selected_ids)
        if not np.any(keep):
            continue
        signal = baseline_correct(-waveforms[keep].astype(np.float32, copy=False), "control_pz_first1000")
        template_parts.append(signal[:, :FILTER_SAMPLES].mean(axis=0))
        selected_events += int(keep.sum())
    if not template_parts:
        raise RuntimeError("no optimization events available for optimal-filter template")
    template = np.mean(np.stack(template_parts), axis=0).astype(np.float64)
    template -= float(np.mean(template[:NOISE_SAMPLES]))

    nfft = 1 << (FILTER_SAMPLES - 1).bit_length()
    noise_power = np.zeros(nfft // 2 + 1, dtype=np.float64)
    noise_events = 0
    time = np.arange(NOISE_SAMPLES, dtype=np.float64)
    centered_time = time - time.mean()
    denominator = float(np.dot(centered_time, centered_time))
    for cache, waveforms, _, _, reason_bits in cached_batches(opt_caches):
        signal = baseline_correct(-waveforms.astype(np.float32, copy=False), "control_pz_first1000")
        noise = signal[:, :NOISE_SAMPLES].astype(np.float64, copy=False)
        slope = (noise @ centered_time) / denominator
        noise = noise - slope[:, None] * centered_time[None, :]
        padded = np.zeros((len(noise), nfft), dtype=np.float64)
        padded[:, :NOISE_SAMPLES] = noise
        spectrum = np.fft.rfft(padded, axis=1)
        admitted = reason_bits == 0
        if np.any(admitted):
            noise_power += np.square(np.abs(spectrum[admitted])).sum(axis=0)
            noise_events += int(admitted.sum())
    if noise_events < 100:
        raise RuntimeError(f"only {noise_events} events available for optimal-filter noise PSD")
    psd = noise_power / float(noise_events)
    floor = max(float(np.median(psd[1:])) * 1.0e-3, np.finfo(np.float64).eps)
    psd = np.maximum(psd, floor)
    template_spectrum = np.fft.rfft(template, n=nfft)
    filter_spectrum = np.conj(template_spectrum) / psd
    weights = np.fft.irfft(filter_spectrum, n=nfft)[:FILTER_SAMPLES]
    norm = float(np.dot(template, weights))
    if not np.isfinite(norm) or abs(norm) < np.finfo(np.float64).eps:
        raise RuntimeError("optimal-filter normalization is invalid")
    weights = weights / norm
    metadata = {
        "filter_samples": FILTER_SAMPLES,
        "noise_samples": NOISE_SAMPLES,
        "fft_length": nfft,
        "template_events": selected_events,
        "noise_events": noise_events,
        "psd_floor": floor,
        "timing_rule": "fixed trigger alignment; no per-event search",
        "baseline_rule": "mean of first 1000 samples, with linear trend projection for noise PSD",
    }
    return template, psd, {**metadata, "weights": weights}


def rise_time_features(
    cache_files: list[CacheFile],
) -> dict[str, np.ndarray]:
    rise_values: list[np.ndarray] = []
    t10_values: list[np.ndarray] = []
    t50_values: list[np.ndarray] = []
    t90_values: list[np.ndarray] = []
    valid_values: list[np.ndarray] = []
    file_indices: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    for cache, waveforms, ids, _, reason_bits in cached_batches(cache_files):
        signal = baseline_correct(-waveforms.astype(np.float32, copy=False), "control_pz_first1000")
        region = signal[:, 1000:3500]
        peak = np.max(region, axis=1)
        event_valid = (reason_bits == 0) & np.isfinite(peak) & (peak > 0.0)
        crossings: list[np.ndarray] = []
        for fraction in (0.1, 0.5, 0.9):
            present = region >= (fraction * peak)[:, None]
            crossing = np.argmax(present, axis=1).astype(np.float64) + 1000.0
            has_crossing = present.any(axis=1)
            crossing[~has_crossing] = np.nan
            crossings.append(crossing)
            event_valid &= has_crossing
        t10, t50, t90 = crossings
        rise_values.append((t90 - t10) * SAMPLE_PERIOD_US)
        t10_values.append(t10)
        t50_values.append(t50)
        t90_values.append(t90)
        valid_values.append(event_valid)
        file_indices.append(np.full(len(ids), cache.file_index, dtype=np.int16))
        event_ids.append(ids.astype(np.uint32, copy=False))
    return {
        "t10": np.concatenate(t10_values),
        "t50": np.concatenate(t50_values),
        "t90": np.concatenate(t90_values),
        "rise_time_us": np.concatenate(rise_values),
        "valid": np.concatenate(valid_values),
        "file_index": np.concatenate(file_indices),
        "event_id": np.concatenate(event_ids),
    }


def run_benchmarks(
    args: argparse.Namespace,
    opt_caches: list[CacheFile],
    val_caches: list[CacheFile],
) -> None:
    output = args.output_dir
    control_opt, control_val, control_definition = load_control(args)
    control_calibration = (
        float(control_definition["calibration"]["slope_kev_per_unit"]),
        float(control_definition["calibration"]["intercept_kev"]),
    )
    frozen = json.loads((output / "frozen_pipeline_config.json").read_text(encoding="utf-8"))
    benchmark_results: dict[str, object] = {
        "status": "BENCHMARKS_COMPLETE",
        "best_trapezoid_objective_kev": float(frozen["selected_trapezoid"]["objective_kev"]),
    }

    try:
        template, psd, filter_info = build_optimal_filter(opt_caches, control_opt, control_calibration)
        weights = np.asarray(filter_info.pop("weights"), dtype=np.float64)
        optimal_opt = collect_optimal_filter_features(opt_caches, weights)
        optimal_val = collect_optimal_filter_features(val_caches, weights)
        scale_mask = control_opt["valid"] & optimal_opt["valid"]
        if int(scale_mask.sum()) < 100:
            raise RuntimeError("insufficient common events for optimal-filter amplitude scale")
        control_per_opt = np.polyfit(
            optimal_opt["amplitude"][scale_mask],
            control_opt["amplitude"][scale_mask],
            1,
        )
        optimal_initial_calibration = (
            float(INITIAL_CALIBRATION[0] * control_per_opt[0]),
            float(INITIAL_CALIBRATION[0] * control_per_opt[1] + INITIAL_CALIBRATION[1]),
        )
        optimal_opt_eval = evaluate_optimization(optimal_opt, optimal_initial_calibration)
        optimal_calibration = (
            float(optimal_opt_eval["calibration"]["slope_kev_per_unit"]),
            float(optimal_opt_eval["calibration"]["intercept_kev"]),
        )
        optimal_val_eval = evaluate_validation(optimal_val, optimal_calibration)
        save_features(output / "optimal_filter_optimization.npz", optimal_opt)
        save_features(output / "optimal_filter_validation.npz", optimal_val)
        np.savez_compressed(output / "optimal_filter_model.npz", template=template, psd=psd, weights=weights)
        optimal_summary = {
            "status": "OK",
            "config": {
                **filter_info,
                "amplitude_scale_reference": "CONTROL_V0 optimization amplitudes",
                "control_amplitude_per_optimal_amplitude": [float(x) for x in control_per_opt],
                "initial_calibration": {
                    "slope_kev_per_unit": optimal_initial_calibration[0],
                    "intercept_kev": optimal_initial_calibration[1],
                },
            },
            "optimization": optimal_opt_eval,
            "validation": optimal_val_eval,
            "branch": (
                "SUBSTANTIAL_LINEAR_HEADROOM" if float(optimal_opt_eval["objective_kev"]) <=
                float(benchmark_results.get("best_trapezoid_objective_kev", 1.0e9)) - 0.5
                else "LINEAR_BENCHMARK_RECORDED"
            ),
        }
        benchmark_results["optimal_filter"] = optimal_summary
    except (RuntimeError, ValueError) as error:
        benchmark_results["optimal_filter"] = {"status": "FAILED", "error": str(error)}

    opt_rise = rise_time_features(opt_caches)
    val_rise = rise_time_features(val_caches)
    opt_energy = control_calibration[0] * control_opt["amplitude"] + control_calibration[1]
    val_energy = control_calibration[0] * control_val["amplitude"] + control_calibration[1]
    def rise_summary(rise: dict[str, np.ndarray], energy: np.ndarray) -> dict[str, object]:
        peak_mask = rise["valid"] & np.isfinite(energy) & (
            ((energy >= E1_KEV - 15.0) & (energy <= E1_KEV + 15.0))
            | ((energy >= E2_KEV - 15.0) & (energy <= E2_KEV + 15.0))
        )
        residual_target = np.where(energy <= (E1_KEV + E2_KEV) / 2.0, E1_KEV, E2_KEV)
        residual = energy - residual_target
        x = rise["rise_time_us"][peak_mask]
        y = residual[peak_mask]
        if len(x) < 100:
            return {"status": "INSUFFICIENT", "events": int(len(x))}
        x_centered = x - float(np.median(x))
        correlation = float(np.corrcoef(x_centered, y)[0, 1])
        coefficient = float(np.polyfit(x_centered, y, 1)[0])
        corrected = y - coefficient * x_centered
        return {
            "status": "OK",
            "events": int(len(x)),
            "rise_time_median_us": float(np.median(x)),
            "rise_time_p10_us": float(np.quantile(x, 0.1)),
            "rise_time_p90_us": float(np.quantile(x, 0.9)),
            "correlation_with_peak_residual": correlation,
            "linear_correction_kev_per_us": coefficient,
            "residual_rms_before_kev": float(np.std(y, ddof=1)),
            "residual_rms_after_kev": float(np.std(corrected, ddof=1)),
            "correction_decision": "DIAGNOSTIC_ONLY",
        }
    rise_opt_summary = rise_summary(opt_rise, opt_energy)
    rise_val_summary = rise_summary(val_rise, val_energy)
    save_features(output / "rise_time_diagnostic_optimization.npz", {**opt_rise, "energy": opt_energy})
    save_features(output / "rise_time_diagnostic_validation.npz", {**val_rise, "energy": val_energy})
    benchmark_results["rise_time_diagnostic"] = {
        "optimization": rise_opt_summary,
        "validation": rise_val_summary,
        "waveform_definition": "t10, t50, t90 first crossings on baseline-subtracted inverted waveform",
    }
    json_dump(output / "benchmark_results.json", benchmark_results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/energy_resolution_experiment"))
    parser.add_argument("--cache-dir", type=Path, default=Path("processed_data/energy_resolution_experiment/raw_cache"))
    parser.add_argument("--split-index", type=int, default=5)
    parser.add_argument("--stage", choices=("prepare", "control", "scan", "benchmarks", "final", "all"), default="all")
    return parser


def prepare(args: argparse.Namespace) -> tuple[list[CacheFile], list[CacheFile], list[CacheFile]]:
    raw_files = sorted(path.resolve() for path in args.raw_files)
    if not raw_files or any(not path.is_file() for path in raw_files):
        raise FileNotFoundError("one or more raw ROOT files are missing")
    if args.split_index <= 0 or args.split_index >= len(raw_files):
        raise ValueError("split-index must leave files in both partitions")
    caches = [cache_raw_file(path, args.cache_dir, index) for index, path in enumerate(raw_files)]
    metadata = []
    reason_definitions = {
        str(REASON_NONFINITE): "non-finite waveform sample",
        str(REASON_LOWER_RAIL): "lower ADC rail sample",
        str(REASON_UPPER_RAIL): "upper ADC rail sample",
        str(REASON_TIMESTAMP): "invalid trigger timestamp",
    }
    for cache in caches:
        with np.load(cache.metadata_path, allow_pickle=False) as data:
            reasons = np.asarray(data["reason_bits"])
            metadata.append({
                "file_index": cache.file_index,
                "file": str(cache.raw_path),
                "entries": cache.entries,
                "reason_counts": {
                    name: int(np.count_nonzero(reasons & int(name))) for name in reason_definitions
                },
                "qc_status": "recorded in external session QC report",
            })
    manifest = {
        "status": "PREPARED",
        "raw_files": [str(path) for path in raw_files],
        "split_index": args.split_index,
        "optimization_files": [str(path) for path in raw_files[:args.split_index]],
        "validation_files": [str(path) for path in raw_files[args.split_index:]],
        "event_key": "file_name + local event_id",
        "expected_samples": EXPECTED_SAMPLES,
        "sample_period_us": SAMPLE_PERIOD_US,
        "admission_reason_bits": reason_definitions,
        "file_metadata": metadata,
        "external_qc_report": "outputs/data_quality/session_20260808_co60_1m_160748_v2/session_qc_report.json",
        "note": "The external QC report is retained verbatim; event-level reconstruction admission uses the reason bits above.",
    }
    json_dump(args.output_dir / "event_split_manifest.json", manifest)
    return caches, caches[:args.split_index], caches[args.split_index:]


def run_control(args: argparse.Namespace, opt_caches: list[CacheFile], val_caches: list[CacheFile], all_caches: list[CacheFile]) -> None:
    output = args.output_dir
    opt = collect_features(opt_caches, CONTROL_CONFIG, keep_diagnostics=True)
    val = collect_features(val_caches, CONTROL_CONFIG, keep_diagnostics=True)
    all_features = collect_features(all_caches, CONTROL_CONFIG, keep_diagnostics=False)
    calibration, approximate_fits = calibration_from_amplitude(opt["amplitude"], opt["valid"], INITIAL_CALIBRATION)
    opt_eval = evaluate_optimization(opt, INITIAL_CALIBRATION)
    val_eval = evaluate_validation(val, calibration)
    common = val["valid"] & np.ones_like(val["valid"], dtype=bool)
    all_file_reports = fit_per_file(all_features, calibration, len(all_caches))
    save_features(output / "control_v0_optimization.npz", opt)
    save_features(output / "control_v0_validation.npz", val)
    save_features(output / "control_v0_all.npz", all_features)
    definition = {
        "name": "CONTROL_V0",
        "config": CONTROL_CONFIG,
        "raw_input": "HPGE nested event waveform, inverted for negative direct-preamp pulses",
        "calibration": {"slope_kev_per_unit": calibration[0], "intercept_kev": calibration[1]},
        "initial_calibration": {"slope_kev_per_unit": INITIAL_CALIBRATION[0], "intercept_kev": INITIAL_CALIBRATION[1]},
        "optimization": opt_eval,
        "validation": val_eval,
        "per_file": all_file_reports,
        "admission": "finite, 4500 samples, no ADC rail, finite trigger timestamp; method amplitude > 0",
        "fit_protocol": {
            "model": "Gaussian plus linear background",
            "bin_width_kev": FIT_BIN_WIDTH_KEV,
            "windows_kev": [list(window) for window in FIT_WINDOWS_KEV],
        },
    }
    json_dump(output / "control_v0_definition.json", definition)
    plot_gain_drift(all_file_reports, output / "per_file_centroid_drift.png")


def load_control(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    opt = load_features(args.output_dir / "control_v0_optimization.npz")
    val = load_features(args.output_dir / "control_v0_validation.npz")
    definition = json.loads((args.output_dir / "control_v0_definition.json").read_text(encoding="utf-8"))
    return opt, val, definition


def evaluate_candidate(
    args: argparse.Namespace,
    caches: list[CacheFile],
    config: dict[str, object],
    name: str,
    approximate_calibration: tuple[float, float],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    features = collect_features_parallel(caches, config)
    try:
        evaluation = evaluate_optimization(features, approximate_calibration)
    except (RuntimeError, ValueError) as error:
        # A prescribed candidate can legitimately fail the fixed fit-count or
        # support-retention gate.  Preserve that result and continue the
        # optimization rather than silently dropping the candidate or aborting
        # the whole experiment.
        evaluation = {
            "status": "FAILED",
            "error": str(error),
            "objective_kev": 1.0e9,
            "valid_events": int(features["valid"].sum()),
            "total_events": int(len(features["valid"])),
            "retention": float(features["valid"].mean()),
        }
    return candidate_summary(name, config, evaluation), features


def run_scan(args: argparse.Namespace, opt_caches: list[CacheFile]) -> None:
    output = args.output_dir
    control_opt, _, control_definition = load_control(args)
    control_cal = (
        float(control_definition["calibration"]["slope_kev_per_unit"]),
        float(control_definition["calibration"]["intercept_kev"]),
    )
    # The control argmax is frequently at the left support boundary for this
    # 6-us-pretrigger record.  Its raw position is therefore not a usable
    # fixed-timing anchor.  For the fixed estimator, use the physically
    # defined start of the trapezoid plateau relative to t50 (t50 - rise),
    # then scan the narrow prescribed offsets around that center.
    control_offset = -float(CONTROL_CONFIG["rise_samples"])
    scan_entries: list[dict[str, object]] = []

    e1_offsets = [control_offset + offset for offset in (-200, -100, 0, 100, 200)]
    e1_fixed = evaluate_candidate_batch(
        opt_caches,
        [
            (
                f"E1-B-fixed-{offset:.1f}",
                {**CONTROL_CONFIG, "estimator": "fixed_position", "offset_samples": offset},
            )
            for offset in e1_offsets
        ],
        control_cal,
    )
    scan_entries.extend(e1_fixed)
    successful_fixed = [item for item in e1_fixed if item.get("status") == "OK"]
    best_fixed = min(successful_fixed, key=lambda item: float(item["objective_kev"])) if successful_fixed else None
    best_e1_config = dict(best_fixed["config"]) if best_fixed is not None else dict(CONTROL_CONFIG)
    e1_flat = evaluate_candidate_batch(
        opt_caches,
        [
            (
                f"E1-C-flat-{width_ns}ns",
                {**best_e1_config, "estimator": "flat_top", "flat_half_width_samples": half_width},
            )
            for half_width, width_ns in zip((5, 12, 25, 50), (40, 100, 200, 400))
        ],
        control_cal,
    )
    scan_entries.extend(e1_flat)
    control_summary = candidate_summary("CONTROL_V0", CONTROL_CONFIG, evaluate_optimization(control_opt, INITIAL_CALIBRATION))
    scan_entries.append(control_summary)
    successful_e1 = [item for item in [control_summary, *e1_fixed, *e1_flat] if item.get("status") == "OK"]
    best_e1 = min(successful_e1, key=lambda item: float(item["objective_kev"]))
    best_estimator = dict(best_e1["config"])

    shaping_candidates = [
        (
            f"E2-r{rise}-f{flat}",
            {**best_estimator, "pole_zero_tau_us": 100.0, "rise_samples": rise, "flat_samples": flat},
        )
        for rise in (250, 375, 500, 625, 750, 875, 1000, 1125, 1250)
        for flat in (50, 100, 150, 200, 250)
    ]
    shaping_entries: list[dict[str, object]] = []
    for start in range(0, len(shaping_candidates), 3):
        shaping_entries.extend(evaluate_candidate_batch(opt_caches, shaping_candidates[start:start + 3], control_cal))
    scan_entries.extend(shaping_entries)
    best_shaping = min(
        [item for item in shaping_entries if item.get("status") == "OK"],
        key=lambda item: float(item["objective_kev"]),
    )

    pz_candidates = [
        (f"E3-tau-{tau}", {**dict(best_shaping["config"]), "pole_zero_tau_us": tau})
        for tau in (None, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
    ]
    pz_entries: list[dict[str, object]] = []
    for start in range(0, len(pz_candidates), 3):
        pz_entries.extend(evaluate_candidate_batch(opt_caches, pz_candidates[start:start + 3], control_cal))
    scan_entries.extend(pz_entries)
    best_pz = min(
        [item for item in pz_entries if item.get("status") == "OK"],
        key=lambda item: float(item["objective_kev"]),
    )

    baseline_candidates = [
        (f"E4-{baseline}", {**dict(best_pz["config"]), "baseline": baseline})
        for baseline in ("control_pz_first1000", "mean_1250", "clip_1250", "mean_1375", "clip_1375", "linear_1250")
    ]
    baseline_entries: list[dict[str, object]] = []
    for start in range(0, len(baseline_candidates), 3):
        baseline_entries.extend(evaluate_candidate_batch(opt_caches, baseline_candidates[start:start + 3], control_cal))
    scan_entries.extend(baseline_entries)
    best_baseline = min(
        [item for item in baseline_entries if item.get("status") == "OK"],
        key=lambda item: float(item["objective_kev"]),
    )

    selected = min(
        [control_summary, best_e1, best_shaping, best_pz, best_baseline],
        key=lambda item: float(item["objective_kev"]),
    )
    frozen = {
        "selected_trapezoid": selected,
        "control": control_summary,
        "best_experiment_1": best_e1,
        "best_experiment_2": best_shaping,
        "best_experiment_3": best_pz,
        "best_experiment_4": best_baseline,
        "control_offset_samples": control_offset,
    }
    json_dump(output / "optimization_results.json", {
        "status": "OPTIMIZATION_COMPLETE",
        "entries": scan_entries,
        "selected": selected,
        "stage_best": frozen,
    })
    json_dump(output / "frozen_pipeline_config.json", frozen)
    plot_scan(shaping_entries, "rise_samples", output / "fwhm_vs_shaping_time.png", "FWHM versus shaping rise time", "Rise samples")
    plot_scan(pz_entries, "pole_zero_tau_us", output / "fwhm_vs_pz_tau.png", "FWHM versus pole-zero tau", "PZ tau (us)")


def run_final(args: argparse.Namespace, val_caches: list[CacheFile]) -> None:
    output = args.output_dir
    _, val_control, control_definition = load_control(args)
    frozen = json.loads((output / "frozen_pipeline_config.json").read_text(encoding="utf-8"))
    selected = frozen["selected_trapezoid"]
    config = dict(selected["config"])
    calibration = (
        float(selected["calibration"]["slope_kev_per_unit"]),
        float(selected["calibration"]["intercept_kev"]),
    )
    final_features = collect_features_parallel(val_caches, config, keep_diagnostics=True)
    control_valid = val_control["valid"]
    common = control_valid & final_features["valid"]
    control_calibration = (
        float(control_definition["calibration"]["slope_kev_per_unit"]),
        float(control_definition["calibration"]["intercept_kev"]),
    )
    control_end = evaluate_validation(val_control, control_calibration)
    final_end = evaluate_validation(final_features, calibration)
    control_common = evaluate_validation(val_control, control_calibration, common)
    final_common = evaluate_validation(final_features, calibration, common)
    benchmark_data: dict[str, object] = {}
    optimal_common: dict[str, object] | None = None
    optimal_end: dict[str, object] | None = None
    optimal_features_path = output / "optimal_filter_validation.npz"
    benchmark_path = output / "benchmark_results.json"
    if optimal_features_path.exists() and benchmark_path.exists():
        benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
        optimal_entry = benchmark_data.get("optimal_filter", {})
        if optimal_entry.get("status") == "OK":
            optimal_features = load_features(optimal_features_path)
            optimal_calibration = (
                float(optimal_entry["optimization"]["calibration"]["slope_kev_per_unit"]),
                float(optimal_entry["optimization"]["calibration"]["intercept_kev"]),
            )
            optimal_end = evaluate_validation(optimal_features, optimal_calibration)
            optimal_common_mask = val_control["valid"] & optimal_features["valid"]
            optimal_common = evaluate_validation(optimal_features, optimal_calibration, optimal_common_mask)
    save_features(output / "final_validation_features.npz", final_features)
    plot_compare(
        control_calibration[0] * val_control["amplitude"][common] + control_calibration[1],
        calibration[0] * final_features["amplitude"][common] + calibration[1],
        output / "control_vs_final_validation.png",
        "CONTROL_V0 versus selected validation spectrum (common-valid events)",
    )
    results = {
        "status": "VALIDATION_COMPLETE",
        "selected_pipeline": selected,
        "control_end_to_end": control_end,
        "final_end_to_end": final_end,
        "control_common_valid": control_common,
        "final_common_valid": final_common,
        "common_valid_events": int(common.sum()),
        "common_valid_retention": float(common.mean()),
        "validation_files": [str(cache.raw_path) for cache in val_caches],
    }
    if optimal_end is not None and optimal_common is not None:
        results["optimal_filter_end_to_end"] = optimal_end
        results["optimal_filter_common_valid"] = optimal_common
        results["optimal_filter_common_valid_events"] = int((val_control["valid"] & load_features(optimal_features_path)["valid"]).sum())
    optimization_data = json.loads((output / "optimization_results.json").read_text(encoding="utf-8"))
    results["optimization_summary"] = {
        "selected": optimization_data.get("selected"),
        "stage_best": optimization_data.get("stage_best"),
    }
    results["benchmark_summary"] = benchmark_data
    results["external_qc_report"] = "outputs/data_quality/session_20260808_co60_1m_160748_v2/session_qc_report.json"
    json_dump(output / "validation_results.json", results)

    report_lines = [
        "# Co-60 Energy-Resolution Experiment V2 Report",
        "",
        "The raw HPGE waveform data were processed with file-level optimization/validation separation.",
        "",
        "| Method | 1173 FWHM | 1332 FWHM | Max FWHM | Common-valid max | Improvement | Retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_obj = float(control_common["objective_kev"])
    final_obj = float(final_common["objective_kev"])
    def row(name: str, evaluation: dict[str, object], common_eval: dict[str, object], baseline: float = baseline_obj) -> str:
        fit1 = common_eval["fits"]["1173_228_kev"]["fwhm_kev"]
        fit2 = common_eval["fits"]["1332_492_kev"]["fwhm_kev"]
        return f"| {name} | {fit1:.4f} | {fit2:.4f} | {common_eval['objective_kev']:.4f} | {common_eval['objective_kev']:.4f} | {baseline - float(common_eval['objective_kev']):.4f} | {float(evaluation['retention']):.4f} |"
    report_lines.append(row("CONTROL_V0", control_end, control_common))
    report_lines.append(row("Final locked pipeline", final_end, final_common))
    if optimal_end is not None and optimal_common is not None:
        report_lines.append(row("Optimal linear-filter benchmark", optimal_end, optimal_common))
    report_lines.extend([
        "",
        f"Common-valid events: {int(common.sum())} / {len(common)} ({float(common.mean()):.4%})",
        f"Selected optimization objective: {float(selected['objective_kev']):.4f} keV",
        f"Locked validation objective: {final_obj:.4f} keV",
        "",
        "## Optimization-only stage winners",
        "",
    ])
    for stage_name in ("best_experiment_1", "best_experiment_2", "best_experiment_3", "best_experiment_4"):
        stage = frozen.get(stage_name, {})
        report_lines.append(
            f"- {stage_name}: {stage.get('name', 'n/a')} — {float(stage.get('objective_kev', float('nan'))):.4f} keV max FWHM"
        )
    rise_report = benchmark_data.get("rise_time_diagnostic", {}) if benchmark_data else {}
    rise_opt = rise_report.get("optimization", {})
    rise_val = rise_report.get("validation", {})
    report_lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- Optimal linear-filter benchmark: {float(benchmark_data.get('optimal_filter', {}).get('optimization', {}).get('objective_kev', float('nan'))):.4f} keV optimization max FWHM; not selected.",
        f"- Rise-time residual correlation: optimization {float(rise_opt.get('correlation_with_peak_residual', float('nan'))):.4f}, validation {float(rise_val.get('correlation_with_peak_residual', float('nan'))):.4f}; correction remains diagnostic-only.",
        "- External session QC is retained as FAIL because one current Co-60 file exceeded the reference baseline-noise p99 limit; event-level reconstruction admission excluded non-finite, rail, and invalid-timestamp events and the file remains provenance-bearing.",
    ])
    (output / "energy_resolution_experiment_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    json_dump(output / "energy_resolution_experiment_results.json", results)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_caches, opt_caches, val_caches = prepare(args)
    if args.stage in {"prepare"}:
        return
    if args.stage in {"control", "scan", "final", "benchmarks", "all"}:
        if not (args.output_dir / "control_v0_definition.json").exists():
            run_control(args, opt_caches, val_caches, all_caches)
    if args.stage in {"scan", "final", "all"}:
        if not (args.output_dir / "frozen_pipeline_config.json").exists():
            run_scan(args, opt_caches)
    if args.stage in {"benchmarks", "final", "all"}:
        benchmark_path = args.output_dir / "benchmark_results.json"
        rerun_failed_optimal = False
        if benchmark_path.exists():
            prior_benchmarks = json.loads(benchmark_path.read_text(encoding="utf-8"))
            rerun_failed_optimal = prior_benchmarks.get("optimal_filter", {}).get("status") != "OK"
        if not benchmark_path.exists() or rerun_failed_optimal:
            run_benchmarks(args, opt_caches, val_caches)
    if args.stage in {"final", "all"}:
        run_final(args, val_caches)


if __name__ == "__main__":
    main()
