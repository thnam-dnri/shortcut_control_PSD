#!/usr/bin/env python3
"""Plot and reconstruct a negative-going direct-preamp Co-60 run at 250 MSPS."""

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


SAMPLE_RATE_MSPS = 250
SAMPLE_PERIOD_US = 0.004
STORED_SAMPLES = 4500
CO60_ENERGIES_KEV = np.array([1173.228, 1332.492])


def gaussian_linear(x, amplitude, mean, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_photopeak(amplitudes: np.ndarray, low: float, high: float) -> dict[str, float]:
    edges = np.arange(low, high + 0.1, 0.1)
    counts, _ = np.histogram(amplitudes, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    initial_mean = float(centers[np.argmax(counts)])
    initial = [float(counts.max()), initial_mean, 0.5, float(np.median(counts)), 0.0]
    fitted, covariance = curve_fit(
        gaussian_linear,
        centers,
        counts,
        p0=initial,
        bounds=([0.0, low, 0.05, 0.0, -np.inf], [np.inf, high, 3.0, np.inf, np.inf]),
        maxfev=100000,
    )
    errors = np.sqrt(np.diag(covariance))
    return {
        "mean_adc": float(fitted[1]),
        "mean_error_adc": float(errors[1]),
        "sigma_adc": float(fitted[2]),
        "fwhm_adc": float(2.354820045 * fitted[2]),
    }


def baseline_and_amplitude(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_region = waveforms[:, 250:1100]
    baseline = baseline_region.mean(axis=1)
    baseline_rms = baseline_region.std(axis=1)
    early_plateau = waveforms[:, 1350:1600].mean(axis=1)
    amplitude = baseline - early_plateau
    return baseline, baseline_rms, amplitude


def plot_first_100(waveforms: np.ndarray, event_ids: np.ndarray, output_dir: Path, stem: str) -> None:
    baseline, _, _ = baseline_and_amplitude(waveforms)
    centered = waveforms - baseline[:, None]
    time_us = np.arange(waveforms.shape[1]) * SAMPLE_PERIOD_US

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(time_us, centered.T, linewidth=0.45, alpha=0.28)
    ax.axvspan(5.4, 6.4, color="tab:orange", alpha=0.09, label="early plateau window")
    ax.set(title="First 100 accepted Co-60 direct-preamp waveforms",
           xlabel="Time from stored-window start (us)", ylabel="ADC code relative to pre-trigger baseline")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_first100_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(10, 10, figsize=(20, 18), sharex=True, sharey=True)
    y_low, y_high = np.quantile(centered, [0.001, 0.999])
    margin = 0.08 * max(1.0, y_high - y_low)
    for index, ax in enumerate(axes.flat):
        ax.plot(time_us, centered[index], linewidth=0.55, color="tab:blue")
        ax.set_title(f"event {int(event_ids[index])}", fontsize=6)
        ax.set_xlim(0.0, 18.0)
        ax.set_ylim(y_low - margin, y_high + margin)
        ax.tick_params(labelsize=5, length=2)
    fig.suptitle("First 100 accepted waveforms (baseline subtracted)", fontsize=16)
    fig.supxlabel("Time (us)")
    fig.supylabel("Relative ADC code")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
    fig.savefig(output_dir / f"{stem}_first100_grid.png", dpi=160)
    plt.close(fig)


def plot_spectrum(amplitudes: np.ndarray, calibration: np.ndarray, peaks: list[dict[str, float]],
                  output_dir: Path, stem: str) -> None:
    slope, intercept = calibration
    energies = slope * amplitudes + intercept
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].hist(energies, bins=np.arange(0.0, 2502.0, 2.0), histtype="step",
                 linewidth=1.1, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0.0, 2500.0)
    axes[0].set(title="Co-60 reconstructed spectrum: all board-accepted waveforms",
                xlabel="Reconstructed energy (keV)", ylabel="Counts / 2 keV")
    for energy, peak in zip(CO60_ENERGIES_KEV, peaks):
        axes[0].axvline(energy, color="tab:red", linestyle="--", linewidth=0.9)
        axes[0].text(energy + 15, axes[0].get_ylim()[1] / 4,
                     f"{energy:.1f} keV\n{peak['mean_adc']:.3f} ADC", fontsize=8, color="tab:red")
    axes[0].grid(alpha=0.2)
    axes[1].hist(energies, bins=np.arange(-250.0, 46025.0, 25.0), histtype="step",
                 linewidth=1.0, color="tab:purple")
    axes[1].set_yscale("log")
    axes[1].set_xlim(-250.0, 46000.0)
    axes[1].set(title="Extended range including saturated/reset-like records",
                xlabel="Reconstructed energy-equivalent (keV)", ylabel="Counts / 25 keV")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_energy_spectrum.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scope-reference-mv", type=float, default=100.0,
                        help="Approximate scope amplitude assigned to the 1332-keV peak")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_root.stem

    with uproot.open(args.input_root) as root_file:
        tree = root_file["HPGE"]
        first = tree["event"].array(entry_start=0, entry_stop=100, library="np")
        if first["waveform"].shape[1] != STORED_SAMPLES:
            raise ValueError(
                f"Expected {STORED_SAMPLES} samples at {SAMPLE_RATE_MSPS} MSPS; "
                f"received {first['waveform'].shape[1]}"
            )
        plot_first_100(first["waveform"].astype(np.float64), first["event_id"], args.output_dir, stem)
        amplitudes_parts = []
        baseline_rms_parts = []
        negative_dominant = 0
        lower_rail_events = 0
        entries = 0
        for arrays in tree.iterate(["event"], step_size=3000, library="np"):
            event = arrays["event"]
            waveforms = event["waveform"].astype(np.float64, copy=False)
            if waveforms.shape[1] != STORED_SAMPLES:
                raise ValueError(
                    f"Expected {STORED_SAMPLES} samples at {SAMPLE_RATE_MSPS} MSPS; "
                    f"received {waveforms.shape[1]}"
                )
            baseline, baseline_rms, amplitude = baseline_and_amplitude(waveforms)
            negative_excursion = baseline - waveforms.min(axis=1)
            positive_excursion = waveforms.max(axis=1) - baseline
            negative_dominant += int(np.count_nonzero(negative_excursion > positive_excursion))
            lower_rail_events += int(np.count_nonzero(np.any(waveforms <= 0.0, axis=1)))
            entries += len(waveforms)
            amplitudes_parts.append(amplitude)
            baseline_rms_parts.append(baseline_rms)

    amplitudes = np.concatenate(amplitudes_parts)
    baseline_rms = np.concatenate(baseline_rms_parts)
    peaks = [fit_photopeak(amplitudes, 82.0, 89.0), fit_photopeak(amplitudes, 94.0, 101.0)]
    calibration = np.polyfit([peak["mean_adc"] for peak in peaks], CO60_ENERGIES_KEV, 1)
    scope_mv_per_adc = args.scope_reference_mv / peaks[1]["mean_adc"]
    plot_spectrum(amplitudes, calibration, peaks, args.output_dir, stem)

    summary = {
        "input_root": str(args.input_root),
        "entries_all_board_accepted": entries,
        "reconstruction": {"polarity": "negative", "sample_rate_msps": SAMPLE_RATE_MSPS,
                           "stored_samples": STORED_SAMPLES,
                           "baseline_samples": [250, 1100],
                           "early_plateau_samples": [1350, 1600],
                           "amplitude_definition": "mean(baseline)-mean(early_plateau)",
                           "sample_period_ns": 4.0},
        "baseline_rms_adc_quantiles": {"p10": float(np.quantile(baseline_rms, 0.10)),
                                       "median": float(np.median(baseline_rms)),
                                       "p90": float(np.quantile(baseline_rms, 0.90))},
        "negative_dominant_fraction": negative_dominant / entries,
        "lower_rail_event_count": lower_rail_events,
        "photopeak_fits": {"1173_228_kev": peaks[0], "1332_492_kev": peaks[1]},
        "energy_calibration": {"kev_per_adc": float(calibration[0]),
                               "intercept_kev": float(calibration[1]),
                               "equation": "energy_kev = kev_per_adc * amplitude_adc + intercept_kev"},
        "scope_based_voltage_conversion_estimate": {
            "assumption": "the approximately 100 mV scope pulse corresponds to the fitted 1332.492 keV peak",
            "reference_mv": args.scope_reference_mv, "mv_per_adc": float(scope_mv_per_adc),
            "adc_per_mv": float(1.0 / scope_mv_per_adc),
            "status": "approximate; requires simultaneous scope/FADC calibration for traceability"},
        "threshold_context": {
            "scout_threshold_adc": 10,
            "note": "trigger threshold acts on the comparator waveform and is not converted using the boxcar energy amplitude",
        },
    }
    summary_path = args.output_dir / f"{stem}_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
