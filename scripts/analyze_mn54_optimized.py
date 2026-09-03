#!/usr/bin/env python3
"""Reconstruct and plot a bounded Mn-54 spectrum at 250 MSPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from analyze_na22_optimized import pole_zero_correct, trapezoid_max


EXPECTED_SAMPLES = 4_500
SAMPLE_PERIOD_US = 0.004
POLE_ZERO_TAU_US = 100.0
RISE_SAMPLES = 1_125
FLAT_SAMPLES = 200
CALIBRATION_KEV_PER_UNIT = 13.620393994376382
CALIBRATION_INTERCEPT_KEV = 0.8706343947031012
EXPECTED_LINE_KEV = 834.848
HISTOGRAM_MAX_KEV = 5_000.0
MAIN_RANGE_MAX_KEV = 1_500.0


def reconstruct(input_root: Path) -> tuple[np.ndarray, dict[str, int]]:
    energies: list[np.ndarray] = []
    counts = {
        "input_entries": 0,
        "nonfinite_entries": 0,
        "lower_rail_entries": 0,
        "upper_rail_entries": 0,
        "nonpositive_shaped_entries": 0,
    }

    with uproot.open(input_root) as root_file:
        tree = root_file["HPGE"]
        for arrays in tree.iterate(["event"], step_size=2_000, library="np"):
            event = arrays["event"]
            waveforms = np.asarray(event["waveform"], dtype=np.float64)
            if waveforms.ndim != 2 or waveforms.shape[1] != EXPECTED_SAMPLES:
                raise ValueError(
                    f"expected waveform shape (*, {EXPECTED_SAMPLES}), got {waveforms.shape}"
                )

            counts["input_entries"] += len(waveforms)
            finite = np.isfinite(waveforms).all(axis=1)
            lower_rail = np.any(waveforms <= 0.0, axis=1)
            upper_rail = np.any(waveforms >= 4095.0, axis=1)
            counts["nonfinite_entries"] += int(np.count_nonzero(~finite))
            counts["lower_rail_entries"] += int(np.count_nonzero(lower_rail))
            counts["upper_rail_entries"] += int(np.count_nonzero(upper_rail))

            keep = finite & ~lower_rail & ~upper_rail
            if not np.any(keep):
                continue

            shaped = trapezoid_max(
                pole_zero_correct(-waveforms[keep], POLE_ZERO_TAU_US),
                RISE_SAMPLES,
                FLAT_SAMPLES,
            )
            nonpositive = shaped <= 0.0
            counts["nonpositive_shaped_entries"] += int(np.count_nonzero(nonpositive))
            valid = ~nonpositive & np.isfinite(shaped)
            if np.any(valid):
                energies.append(
                    CALIBRATION_KEV_PER_UNIT * shaped[valid]
                    + CALIBRATION_INTERCEPT_KEV
                )

    if not energies:
        raise RuntimeError("no valid reconstructed energies")
    return np.concatenate(energies), counts


def line_window_summary(energy: np.ndarray) -> dict[str, float | int | str]:
    peak_low = EXPECTED_LINE_KEV - 15.0
    peak_high = EXPECTED_LINE_KEV + 15.0
    left_low, left_high = EXPECTED_LINE_KEV - 55.0, EXPECTED_LINE_KEV - 25.0
    right_low, right_high = EXPECTED_LINE_KEV + 25.0, EXPECTED_LINE_KEV + 55.0
    peak_count = int(np.count_nonzero((energy >= peak_low) & (energy < peak_high)))
    left_count = int(np.count_nonzero((energy >= left_low) & (energy < left_high)))
    right_count = int(np.count_nonzero((energy >= right_low) & (energy < right_high)))
    sideband_width = (left_high - left_low) + (right_high - right_low)
    peak_width = peak_high - peak_low
    background = (left_count + right_count) * peak_width / sideband_width
    net = peak_count - background
    variance = peak_count + (peak_width / sideband_width) ** 2 * (left_count + right_count)
    significance = net / np.sqrt(variance) if variance > 0.0 else 0.0
    status = "NO_SIGNIFICANT_EXCESS" if significance < 3.0 else "CANDIDATE_EXCESS"
    return {
        "expected_line_kev": EXPECTED_LINE_KEV,
        "search_window_kev": [peak_low, peak_high],
        "search_window_events": peak_count,
        "sidebands_kev": [[left_low, left_high], [right_low, right_high]],
        "sideband_events": left_count + right_count,
        "estimated_background_in_search_window": float(background),
        "estimated_net_events": float(net),
        "sideband_significance_sigma": float(significance),
        "status": status,
    }


def plot_spectrum(
    energy: np.ndarray,
    line_summary: dict[str, float | int | str],
    output_path: Path,
) -> None:
    edges = np.arange(0.0, HISTOGRAM_MAX_KEV + 2.0, 2.0)
    counts, _ = np.histogram(energy, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected = EXPECTED_LINE_KEV
    peak_low, peak_high = [float(value) for value in line_summary["search_window_kev"]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    axes[0].step(centers, counts, where="mid", linewidth=0.9, color="tab:blue")
    axes[0].axvline(expected, color="tab:red", linestyle="--", linewidth=1.0,
                    label=f"Mn-54 line: {expected:.3f} keV")
    axes[0].axvspan(peak_low, peak_high, color="tab:red", alpha=0.08,
                    label="±15 keV search window")
    axes[0].set_xlim(0.0, MAIN_RANGE_MAX_KEV)
    axes[0].set_ylabel("Counts / 2 keV")
    axes[0].set_title("Mn-54 reconstructed energy spectrum: 15-minute run")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].step(centers, counts, where="mid", linewidth=0.9, color="tab:purple")
    axes[1].axvline(expected, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.0, HISTOGRAM_MAX_KEV)
    axes[1].set_xlabel("Reconstructed energy (keV)")
    axes[1].set_ylabel("Counts / 2 keV (log)")
    axes[1].set_title("Extended range; high-energy tail retained for inspection")
    axes[1].grid(alpha=0.2)

    annotation = (
        f"{line_summary['status']}\n"
        f"search: {line_summary['search_window_events']} events\n"
        f"sideband background: {line_summary['estimated_background_in_search_window']:.1f}\n"
        f"net: {line_summary['estimated_net_events']:.1f} "
        f"({line_summary['sideband_significance_sigma']:.1f}σ)"
    )
    axes[0].text(
        0.98,
        0.97,
        annotation,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mn54_optimized_20260813"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    energy, counts = reconstruct(args.input_root)
    line_summary = line_window_summary(energy)
    edges = np.arange(0.0, HISTOGRAM_MAX_KEV + 2.0, 2.0)
    histogram_counts, _ = np.histogram(energy, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spectrum_png = args.output_dir / "mn54_energy_spectrum.png"
    spectrum_csv = args.output_dir / "mn54_energy_spectrum_2kev.csv"
    summary_json = args.output_dir / "mn54_optimized_analysis_summary.json"
    plot_spectrum(energy, line_summary, spectrum_png)
    np.savetxt(
        spectrum_csv,
        np.column_stack((centers, histogram_counts)),
        fmt=("%.6f", "%d"),
    )

    summary = {
        "status": "ESTIMATE",
        "source": "Mn54",
        "input_root": str(args.input_root),
        **counts,
        "reconstructed_entries": int(energy.size),
        "energies_above_histogram_range": int(np.count_nonzero(energy >= HISTOGRAM_MAX_KEV)),
        "reconstruction": {
            "method": "FADC500 first-order pole-zero correction followed by maximum difference of two moving averages on inverted negative-pulse waveforms",
            "pole_zero_tau_us": POLE_ZERO_TAU_US,
            "rise_samples": RISE_SAMPLES,
            "flat_samples": FLAT_SAMPLES,
            "rise_us": RISE_SAMPLES * SAMPLE_PERIOD_US,
            "flat_us": FLAT_SAMPLES * SAMPLE_PERIOD_US,
        },
        "calibration": {
            "method": "provisional multi-source affine calibration used by the established optimized pipeline",
            "kev_per_shaped_unit": CALIBRATION_KEV_PER_UNIT,
            "intercept_kev": CALIBRATION_INTERCEPT_KEV,
        },
        "line_check": line_summary,
        "spectrum_outputs": {
            "png": str(spectrum_png),
            "csv": str(spectrum_csv),
            "bin_width_kev": 2.0,
            "histogram_range_kev": [0.0, HISTOGRAM_MAX_KEV],
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
