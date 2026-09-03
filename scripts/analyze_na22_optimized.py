#!/usr/bin/env python3
"""Reconstruct energy spectra with the established 250-MSPS method."""

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


EXPECTED_SAMPLES = 4_500
SAMPLE_PERIOD_US = 0.004
POLE_ZERO_TAU_US = 100.0
RISE_SAMPLES = 1_125
FLAT_SAMPLES = 200
# Applied to each continuous shaped-unit value before histogramming.  This is
# the provisional multi-source affine recalibration derived from the resolved
# Ba-133, Na-22, Cs-137, and Co-60 photopeaks.
CALIBRATION_KEV_PER_UNIT = 13.620393994376382
CALIBRATION_INTERCEPT_KEV = 0.8706343947031012
CALIBRATION_METHOD = "event-level provisional affine fit to resolved Ba-133, Na-22, Cs-137, and Co-60 photopeaks"
CALIBRATION_REFERENCE = "outputs/ba133_optimized_20260807/ba133_candidate_peak_estimate.json; outputs/cs137_optimized_20260807/cs137_661kev_optimized_estimate.json; outputs/na22_optimized_20260808/na22_optimized_analysis_summary.json; Co-60 anchor peaks"
PEAKS = {
    "annihilation_511_kev": {"expected_kev": 510.99895, "fit_window": [480.0, 540.0]},
    "na22_gamma_1274_5_kev": {"expected_kev": 1274.537, "fit_window": [1230.0, 1310.0]},
}


def trapezoid_max(signal: np.ndarray, rise: int, flat: int) -> np.ndarray:
    """Return the maximum difference of two moving averages per waveform."""

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
    peak_positions = np.argmax(values, axis=1).astype(np.int32)
    return np.take_along_axis(values, peak_positions[:, None], axis=1)[:, 0]


def pole_zero_correct(signal: np.ndarray, tau_us: float) -> np.ndarray:
    """Apply the established FADC500 first-order pole-zero correction."""

    centered = signal - signal[:, :1_000].mean(axis=1, keepdims=True)
    alpha = float(np.exp(-SAMPLE_PERIOD_US / tau_us))
    difference = np.empty_like(centered)
    difference[:, 0] = centered[:, 0]
    difference[:, 1:] = centered[:, 1:] - alpha * centered[:, :-1]
    corrected = np.cumsum(difference, axis=1, dtype=np.float32)
    corrected -= corrected[:, :1_000].mean(axis=1, keepdims=True)
    return corrected


def gaussian_linear(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_peak(values: np.ndarray, expected_kev: float, low: float, high: float) -> dict[str, float | int | str]:
    """Fit a Gaussian plus linear background and return its net area."""

    bin_width = 0.25
    edges = np.arange(low, high + bin_width, bin_width)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    baseline_count = float(np.median(np.r_[counts[:10], counts[-10:]]))
    peak_height = max(1.0, float(counts.max()) - baseline_count)
    mean0 = float(centers[np.argmax(counts)])
    fitted, covariance = curve_fit(
        gaussian_linear,
        centers,
        counts,
        p0=[peak_height, mean0, 2.0, baseline_count, 0.0],
        bounds=([0.0, low, 0.05, 0.0, -np.inf], [np.inf, high, 20.0, np.inf, np.inf]),
        maxfev=100_000,
    )
    errors = np.sqrt(np.diag(covariance))
    amplitude, mean, sigma, offset, slope = [float(value) for value in fitted]
    fwhm = 2.354820045 * sigma
    area = amplitude * sigma * np.sqrt(2.0 * np.pi) / bin_width
    observed = int(np.count_nonzero(np.abs(values - mean) <= 0.5 * fwhm))
    model_at_centroid = float(gaussian_linear(np.array([mean]), *fitted)[0])
    background_at_centroid = offset
    return {
        "expected_energy_kev": expected_kev,
        "fitted_centroid_kev": mean,
        "centroid_error_kev": float(errors[1]),
        "sigma_kev": sigma,
        "fwhm_kev": fwhm,
        "fwhm_error_kev": float(2.354820045 * errors[2]),
        "net_gaussian_events": area,
        "observed_events_within_fwhm": observed,
        "background_at_centroid_counts_per_bin": background_at_centroid,
        "fit_model_count_at_centroid": model_at_centroid,
        "fit_window_kev": [low, high],
        "fit_bin_width_kev": bin_width,
        "fit_status": "OK",
    }


def plot_spectrum(
    energy: np.ndarray,
    fits: dict[str, dict[str, float | int | str]],
    output_path: Path,
    source_label: str,
    histogram_max_kev: float,
) -> None:
    """Save linear and logarithmic spectrum views."""

    edges = np.arange(0.0, histogram_max_kev + 1.0, 1.0)
    counts, _ = np.histogram(energy, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    for ax, log_scale in zip(axes, (False, True)):
        ax.step(centers, counts, where="mid", linewidth=0.9, color="tab:blue")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(0.0, histogram_max_kev)
        ax.set_ylabel("Counts / 1 keV")
        ax.grid(alpha=0.2)
        for key, fit in fits.items():
            expected = float(fit["expected_energy_kev"])
            ax.axvline(expected, color="tab:red", linestyle="--", linewidth=0.8)
            label = f"{expected:.1f} keV"
            if fit.get("fit_status") == "OK":
                label += f"; area {float(fit['net_gaussian_events']):.0f}"
            ax.text(expected + 8.0, ax.get_ylim()[1] / 3.0, label, color="tab:red", fontsize=8)
    axes[-1].set_xlabel("Reconstructed energy (keV)")
    axes[0].set_title(f"{source_label} energy spectrum: established 250-MSPS PZ + trapezoid reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/na22_optimized_20260808"))
    parser.add_argument("--source-label", default="Na22")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--histogram-max-kev", type=float, default=1500.0)
    args = parser.parse_args()
    if args.histogram_max_kev <= 0.0:
        parser.error("--histogram-max-kev must be positive")
    output_prefix = args.output_prefix or args.source_label.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    energies: list[np.ndarray] = []
    input_entries = 0
    selected_entries = 0
    lower_rail_entries = 0
    upper_rail_entries = 0
    nonfinite_entries = 0
    nonpositive_shaped_entries = 0

    for input_root in sorted(args.input_roots):
        with uproot.open(input_root) as source:
            tree = source["HPGE"]
            input_entries += int(tree.num_entries)
            for arrays in tree.iterate(["event"], step_size=2_000, library="np"):
                event = arrays["event"]
                waveforms = np.asarray(event["waveform"], dtype=np.float64)
                if waveforms.ndim != 2 or waveforms.shape[1] != EXPECTED_SAMPLES:
                    raise ValueError(f"{input_root}: expected waveform shape (*, {EXPECTED_SAMPLES}), got {waveforms.shape}")
                finite = np.isfinite(waveforms).all(axis=1)
                lower_rail = np.any(waveforms <= 0.0, axis=1)
                upper_rail = np.any(waveforms >= 4095.0, axis=1)
                nonfinite_entries += int(np.count_nonzero(~finite))
                lower_rail_entries += int(np.count_nonzero(lower_rail))
                upper_rail_entries += int(np.count_nonzero(upper_rail))
                keep = finite & ~lower_rail & ~upper_rail
                if not np.any(keep):
                    continue
                shaped = trapezoid_max(
                    pole_zero_correct(-waveforms[keep], POLE_ZERO_TAU_US),
                    RISE_SAMPLES,
                    FLAT_SAMPLES,
                )
                nonpositive_shaped_entries += int(np.count_nonzero(shaped <= 0.0))
                keep_shaped = shaped > 0.0
                if not np.any(keep_shaped):
                    continue
                reconstructed = (
                    CALIBRATION_KEV_PER_UNIT * shaped[keep_shaped]
                    + CALIBRATION_INTERCEPT_KEV
                )
                reconstructed = reconstructed[np.isfinite(reconstructed)]
                selected_entries += int(reconstructed.size)
                energies.append(reconstructed)

    if not energies:
        raise RuntimeError(f"no valid {args.source_label} reconstructed energies")
    energy = np.concatenate(energies)
    fits: dict[str, dict[str, float | int | str]] = {}
    if args.source_label.lower() == "na22":
        for key, peak in PEAKS.items():
            low, high = peak["fit_window"]
            try:
                fits[key] = fit_peak(energy, peak["expected_kev"], low, high)
            except (RuntimeError, ValueError) as error:
                fits[key] = {
                    "expected_energy_kev": peak["expected_kev"],
                    "fit_window_kev": [low, high],
                    "fit_status": f"FAILED: {error}",
                }

    output_stem = args.output_dir / output_prefix
    spectrum_png = output_stem.with_name(f"{output_prefix}_energy_spectrum.png")
    spectrum_csv = output_stem.with_name(f"{output_prefix}_energy_spectrum_1kev.csv")
    summary_json = output_stem.with_name(f"{output_prefix}_optimized_analysis_summary.json")
    plot_spectrum(energy, fits, spectrum_png, args.source_label, args.histogram_max_kev)
    edges = np.arange(0.0, args.histogram_max_kev + 1.0, 1.0)
    counts, _ = np.histogram(energy, bins=edges)
    np.savetxt(spectrum_csv, np.column_stack((0.5 * (edges[:-1] + edges[1:]), counts)), fmt=("%.6f", "%d"))

    summary = {
        "status": "ESTIMATE",
        "source": args.source_label,
        "input_files": [str(path) for path in sorted(args.input_roots)],
        "input_entries": input_entries,
        "reconstructed_entries": selected_entries,
        "cuts": {
            "finite_waveform": True,
            "exclude_lower_rail": True,
            "exclude_upper_rail": True,
            "exclude_nonpositive_shaped_value": True,
            "legacy_charge_amplitude_cut_250_adc": False,
        },
        "cut_counts": {
            "nonfinite_entries": nonfinite_entries,
            "lower_rail_entries": lower_rail_entries,
            "upper_rail_entries": upper_rail_entries,
            "nonpositive_shaped_entries": nonpositive_shaped_entries,
        },
        "reconstruction": {
            "method": "FADC500 first-order pole-zero correction followed by maximum difference of two moving averages on inverted negative-pulse waveforms",
            "pole_zero_tau_us": POLE_ZERO_TAU_US,
            "rise_samples": RISE_SAMPLES,
            "flat_samples": FLAT_SAMPLES,
            "rise_us": RISE_SAMPLES * SAMPLE_PERIOD_US,
            "flat_us": FLAT_SAMPLES * SAMPLE_PERIOD_US,
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "reference_file": CALIBRATION_REFERENCE,
            "kev_per_shaped_unit": CALIBRATION_KEV_PER_UNIT,
            "intercept_kev": CALIBRATION_INTERCEPT_KEV,
        },
        "peak_fits": fits,
        "spectrum_outputs": {
            "png": str(spectrum_png),
            "csv": str(spectrum_csv),
            "csv_columns": ["energy_kev_bin_center", "count"],
            "csv_header": False,
            "bin_width_kev": 1.0,
            "histogram_range_kev": [0.0, args.histogram_max_kev],
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
