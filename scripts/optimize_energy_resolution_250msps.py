#!/usr/bin/env python3
"""Optimize pole-zero plus trapezoid shaping for 250 MSPS charge data.

The direct-preamp records have a slow recovery tail.  The scan therefore
includes the pole-zero correction used by the FADC500 analysis, followed by
the maximum difference of two moving averages used by ML_test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import curve_fit


E1_KEV, E2_KEV = 1173.228, 1332.492
EXPECTED_SAMPLES = 4500
ANCHOR1_ADC, ANCHOR2_ADC = 86.1664, 97.8905
ANCHOR_HALF_WIDTH_ADC = 0.6

# Candidate tuples are (pole-zero tau [us], rise samples, flat samples).
# The no-PZ point is the direct ML_test/FADC500 4 us/1 us estimator.  The
# remaining candidates cover nearby shaping lengths and tau values while
# keeping the support inside the 4500-sample record.
SHAPING_CANDIDATES = [(None, 1000, 250)]
for _tau in (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 100.0, 150.0):
    for _rise, _flat in ((875, 100), (1000, 100), (1000, 150), (1000, 250),
                         (1125, 100), (1125, 150), (1125, 200), (1250, 100),
                         (1250, 150), (1250, 250)):
        SHAPING_CANDIDATES.append((_tau, _rise, _flat))


def trapezoid_max(signal: np.ndarray, rise: int, flat: int) -> tuple[np.ndarray, np.ndarray]:
    stop = signal.shape[1]
    span = 2 * rise + flat
    positions = stop - span + 1
    prefix = np.empty((signal.shape[0], stop + 1), dtype=np.float64)
    prefix[:, 0] = 0.0
    np.cumsum(signal, axis=1, dtype=np.float64, out=prefix[:, 1:])
    first = prefix[:, rise:rise + positions] - prefix[:, :positions]
    delayed = rise + flat
    second = prefix[:, delayed + rise:delayed + rise + positions] - prefix[:, delayed:delayed + positions]
    values = (second - first) / float(rise)
    positions = np.argmax(values, axis=1).astype(np.int32)
    peak = np.take_along_axis(values, positions[:, None], axis=1)[:, 0]
    return peak, positions


def pole_zero_correct(signal: np.ndarray, tau_us: float | None) -> np.ndarray:
    """Apply the FADC500 first-order pole-zero correction to a batch."""
    centered = signal - signal[:, :1000].mean(axis=1, keepdims=True)
    if tau_us is None:
        return centered
    alpha = float(np.exp(-0.004 / tau_us))  # 4 ns samples at 250 MSPS
    difference = np.empty_like(centered)
    difference[:, 0] = centered[:, 0]
    difference[:, 1:] = centered[:, 1:] - alpha * centered[:, :-1]
    corrected = np.cumsum(difference, axis=1, dtype=np.float32)
    corrected -= corrected[:, :1000].mean(axis=1, keepdims=True)
    return corrected


def gaussian_linear(x, amplitude, mean, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_gaussian(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    half = max(0.5, min(10.0, 5.0 * mad))
    low, high = median - half, median + half
    edges = np.linspace(low, high, 161)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean0 = float(centers[np.argmax(counts)])
    sigma0 = max(0.1, float(np.std(values[(values > median - mad * 3) & (values < median + mad * 3)])))
    # Keep the initial point strictly inside the bounded fit interval.  A
    # narrow/quantized peak can have a robust standard deviation larger than
    # the plotting half-width used above.
    sigma0 = min(max(0.03 * 1.01, sigma0), half * 0.99)
    fitted, covariance = curve_fit(
        gaussian_linear, centers, counts,
        p0=[float(counts.max()), mean0, sigma0, float(np.median(counts)), 0.0],
        bounds=([0.0, low, 0.03, 0.0, -np.inf], [np.inf, high, half, np.inf, np.inf]),
        maxfev=100000,
    )
    errors = np.sqrt(np.diag(covariance))
    return {
        "mean_adc": float(fitted[1]),
        "mean_error_adc": float(errors[1]),
        "sigma_adc": float(fitted[2]),
        "fwhm_adc": float(2.354820045 * fitted[2]),
        "n_events": int(values.size),
    }


def calculate_config(values1: np.ndarray, values2: np.ndarray) -> dict[str, float]:
    fit1, fit2 = fit_gaussian(values1), fit_gaussian(values2)
    slope = (E2_KEV - E1_KEV) / (fit2["mean_adc"] - fit1["mean_adc"])
    intercept = E1_KEV - slope * fit1["mean_adc"]
    fit1["fwhm_kev"] = float(fit1["fwhm_adc"] * slope)
    fit2["fwhm_kev"] = float(fit2["fwhm_adc"] * slope)
    return {
        "peak1_mean_adc": fit1["mean_adc"], "peak2_mean_adc": fit2["mean_adc"],
        "peak1_fwhm_kev": fit1["fwhm_kev"], "peak2_fwhm_kev": fit2["fwhm_kev"],
        "peak1_fwhm_adc": fit1["fwhm_adc"], "peak2_fwhm_adc": fit2["fwhm_adc"],
        "kev_per_adc": float(slope), "intercept_kev": float(intercept),
        "objective_max_fwhm_kev": float(max(fit1["fwhm_kev"], fit2["fwhm_kev"])),
        "peak1_fit": fit1, "peak2_fit": fit2,
    }


def collect_anchor_values(path: Path, scan_limit: int | None,
                          config: tuple[float | None, int, int] | None = None,
                          collect_full: bool = False):
    anchor1_parts, anchor2_parts, full_parts = [], [], []
    selected = 0
    with uproot.open(path) as root_file:
        tree = root_file["HPGE_250MSPS"]
        for arrays in tree.iterate(["waveform", "charge_amplitude_adc", "is_physics_candidate"],
                                   step_size=2000, library="np"):
            waveforms = arrays["waveform"].astype(np.float64, copy=False)
            if waveforms.shape[1] != EXPECTED_SAMPLES:
                raise ValueError(
                    f"Expected {EXPECTED_SAMPLES} samples at 250 MSPS; received {waveforms.shape[1]}"
                )
            amplitude = arrays["charge_amplitude_adc"].astype(np.float64, copy=False)
            physics = arrays["is_physics_candidate"].astype(bool)
            if scan_limit is not None and selected >= scan_limit and not collect_full:
                break
            if scan_limit is not None and not collect_full:
                keep = physics
                available = max(0, scan_limit - selected)
                if int(keep.sum()) > available:
                    indices = np.flatnonzero(keep)[:available]
                    keep = np.zeros_like(keep)
                    keep[indices] = True
                selected += int(keep.sum())
                waveforms, amplitude = waveforms[keep], amplitude[keep]
            else:
                keep = physics
                waveforms, amplitude = waveforms[keep], amplitude[keep]
            if not len(waveforms):
                continue
            if config is None:
                # Keep only the two photopeak neighborhoods for the scan.  This
                # bounds memory and makes the expensive PZ/trapezoid scan fast.
                anchor_mask = ((np.abs(amplitude - ANCHOR1_ADC) <= ANCHOR_HALF_WIDTH_ADC) |
                               (np.abs(amplitude - ANCHOR2_ADC) <= ANCHOR_HALF_WIDTH_ADC))
                anchor1_parts.append((amplitude[anchor_mask], -waveforms[anchor_mask]))
            else:
                tau_us, rise, flat = config
                filtered = pole_zero_correct(-waveforms, tau_us)
                peak, _ = trapezoid_max(filtered, rise, flat)
                anchor1_parts.append(peak[np.abs(amplitude - ANCHOR1_ADC) <= ANCHOR_HALF_WIDTH_ADC])
                anchor2_parts.append(peak[np.abs(amplitude - ANCHOR2_ADC) <= ANCHOR_HALF_WIDTH_ADC])
                if collect_full:
                    full_parts.append(peak)
    if config is None:
        return selected, anchor1_parts
    return np.concatenate(anchor1_parts), np.concatenate(anchor2_parts), np.concatenate(full_parts) if full_parts else None


def plot_spectrum(energy: np.ndarray, results: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(energy, bins=np.arange(0.0, 2502.0, 2.0), histtype="step", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlim(0, 2500)
    ax.set_xlabel("Reconstructed energy (keV)")
    ax.set_ylabel("Counts / 2 keV")
    ax.set_title("Optimized trapezoid reconstruction at 250 MSPS")
    for index, (energy_kev, key) in enumerate(((E1_KEV, "peak1_fwhm_kev"), (E2_KEV, "peak2_fwhm_kev"))):
        ax.axvline(energy_kev, color="tab:red", linestyle="--", linewidth=0.8)
        x = energy_kev - 10 if index == 0 else energy_kev + 10
        ha = "right" if index == 0 else "left"
        ax.text(x, ax.get_ylim()[1] / 5,
                f"{energy_kev:.1f} keV\nFWHM {results[key]:.2f} keV", color="tab:red", ha=ha, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scan-events", type=int, default=50000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_root.stem

    selected, anchor_batches = collect_anchor_values(args.input_root, args.scan_events)
    # anchor_batches contains (box_amplitude, waveform) pairs for the bounded scan subset.
    anchor_amp = np.concatenate([item[0] for item in anchor_batches])
    anchor_wave = np.concatenate([item[1] for item in anchor_batches])
    scan_results = []
    for tau_us, rise, flat in SHAPING_CANDIDATES:
        peak, _ = trapezoid_max(pole_zero_correct(anchor_wave, tau_us), rise, flat)
        mask1 = np.abs(anchor_amp - ANCHOR1_ADC) <= ANCHOR_HALF_WIDTH_ADC
        mask2 = np.abs(anchor_amp - ANCHOR2_ADC) <= ANCHOR_HALF_WIDTH_ADC
        result = calculate_config(peak[mask1], peak[mask2])
        result.update({"pole_zero_tau_us": tau_us, "rise_samples": rise, "flat_samples": flat,
                       "rise_us": rise * 0.004, "flat_us": flat * 0.004})
        scan_results.append(result)
        print(f"tau={tau_us} rise={rise} flat={flat} max_fwhm={result['objective_max_fwhm_kev']:.4f} keV", flush=True)
    best = min(scan_results, key=lambda item: item["objective_max_fwhm_kev"])
    best_config = (best["pole_zero_tau_us"], int(best["rise_samples"]), int(best["flat_samples"]))

    # Full-data validation with the selected shaping.
    values1, values2, all_values = collect_anchor_values(args.input_root, None, best_config, collect_full=True)
    validation = calculate_config(values1, values2)
    energy = validation["kev_per_adc"] * all_values + validation["intercept_kev"]
    spectrum_path = args.output_dir / f"{stem}_optimized_energy_spectrum.png"
    plot_spectrum(energy, validation, spectrum_path)
    hist_path = args.output_dir / f"{stem}_optimized_energy_spectrum.dat"
    edges = np.arange(0.0, 2502.0, 2.0)
    counts, edges = np.histogram(energy, bins=edges)
    np.savetxt(hist_path, np.column_stack((0.5 * (edges[:-1] + edges[1:]), counts)), fmt=("%.6f", "%d"))

    summary = {
        "input_root": str(args.input_root), "scan_events": selected,
        "full_physics_events": int(all_values.size),
        "method": "FADC500 first-order pole-zero correction followed by maximum difference of two moving averages on inverted negative charge waveform",
        "anchor_selection": {"boxcar_peak1_adc": ANCHOR1_ADC, "boxcar_peak2_adc": ANCHOR2_ADC,
                              "half_width_adc": ANCHOR_HALF_WIDTH_ADC},
        "scan_results": scan_results, "selected": best_config,
        "full_validation": validation,
        "outputs": {"spectrum_png": str(spectrum_path), "spectrum_dat": str(hist_path)},
    }
    summary_path = args.output_dir / f"{stem}_optimized_energy_resolution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"selected": best_config, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
