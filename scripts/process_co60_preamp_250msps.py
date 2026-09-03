#!/usr/bin/env python3
"""Process direct-preamp charge waveforms and derive 250 MSPS current proxies."""

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


CO60_ENERGIES_KEV = np.array([1173.228, 1332.492])
DOWNSAMPLE_FACTOR = 1
OUTPUT_SAMPLE_PERIOD_NS = 4.0
BASELINE_SLICE = slice(250, 1100)
PLATEAU_SLICE = slice(1350, 1600)
CURRENT_SEARCH_SLICE = slice(1100, 1500)
CURRENT_GATE_OFFSETS = np.arange(-500, 501, dtype=np.int32)


def gaussian_linear(x, amplitude, mean, sigma, offset, slope):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * (x - mean)


def fit_peak(values: np.ndarray, low: float, high: float) -> dict[str, float]:
    edges = np.arange(low, high + 0.1, 0.1)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean0 = float(centers[np.argmax(counts)])
    fitted, covariance = curve_fit(
        gaussian_linear,
        centers,
        counts,
        p0=[float(counts.max()), mean0, 0.5, float(np.median(counts)), 0.0],
        bounds=([0.0, low, 0.05, 0.0, -np.inf], [np.inf, high, 3.0, np.inf, np.inf]),
        maxfev=100000,
    )
    error = np.sqrt(np.diag(covariance))
    return {"mean_adc": float(fitted[1]), "mean_error_adc": float(error[1]),
            "sigma_adc": float(fitted[2]), "fwhm_adc": float(2.354820045 * fitted[2])}


def plot_current(first_current: np.ndarray, first_ids: np.ndarray, output_dir: Path, stem: str) -> None:
    sample_offset = CURRENT_GATE_OFFSETS
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sample_offset, first_current.T, linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(title="First 100 gated current-proxy waveforms at 250 MSPS",
           xlabel="Sample offset from current peak (4 ns/sample)",
           ylabel="Current proxy (ADC-code change per sample)")
    ax.set_xlim(-500, 500)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_250msps_current_first100_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(10, 10, figsize=(20, 18), sharex=True, sharey=True)
    low, high = np.quantile(first_current, [0.001, 0.999])
    margin = 0.08 * max(1.0, high - low)
    for index, ax in enumerate(axes.flat):
        ax.plot(sample_offset, first_current[index], linewidth=0.55)
        ax.axvline(0, color="black", alpha=0.2, linewidth=0.4)
        ax.set_title(f"event {int(first_ids[index])}", fontsize=6)
        ax.set_xlim(-500, 500)
        ax.set_ylim(low - margin, high + margin)
        ax.tick_params(labelsize=5, length=2)
    fig.suptitle("First 100 current-proxy waveforms, peak-aligned and gated ±500 samples", fontsize=15)
    fig.supxlabel("Sample offset from peak (4 ns/sample)")
    fig.supylabel("ADC-code change per sample")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
    fig.savefig(output_dir / f"{stem}_250msps_current_first100_grid.png", dpi=160)
    plt.close(fig)


def plot_spectrum(amplitudes: np.ndarray, calibration: np.ndarray, fits: list[dict[str, float]],
                  output_dir: Path, stem: str) -> None:
    slope, intercept = calibration
    energy = slope * amplitudes + intercept
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(energy, bins=np.arange(0.0, 2502.0, 2.0), histtype="step", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlim(0, 2500)
    ax.set(title="Co-60 energy spectrum from direct 250 MSPS charge waveforms",
           xlabel="Reconstructed energy (keV)", ylabel="Counts / 2 keV")
    for index, (energy_kev, fit) in enumerate(zip(CO60_ENERGIES_KEV, fits)):
        resolution = 100.0 * fit["fwhm_kev"] / energy_kev
        ax.axvline(energy_kev, color="tab:red", linestyle="--", linewidth=0.8)
        text_x = energy_kev - 12 if index == 0 else energy_kev + 12
        alignment = "right" if index == 0 else "left"
        ax.text(text_x, ax.get_ylim()[1] / 5,
                f"{energy_kev:.1f} keV\nFWHM {fit['fwhm_kev']:.2f} keV ({resolution:.3f}%)",
                fontsize=8, color="tab:red", ha=alignment)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_250msps_energy_spectrum.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_root.stem
    charge_path = args.processed_dir / f"{stem}_250msps_charge.root"
    current_path = args.processed_dir / f"{stem}_250msps_current_gated.root"

    amplitudes_all = []
    amplitudes_physics = []
    first_current = []
    first_current_ids = []
    total_entries = physics_entries = lower_rail_entries = large_entries = 0

    with uproot.open(args.input_root) as source, \
            uproot.recreate(charge_path, compression=uproot.ZSTD(3)) as charge_file, \
            uproot.recreate(current_path, compression=uproot.ZSTD(3)) as current_file:
        charge_tree = charge_file.mktree("HPGE_250MSPS", {
            "event_id": "uint32", "trigger_time_s": "float32",
            "waveform": "4500 * float32", "charge_amplitude_adc": "float32",
            "is_physics_candidate": "bool"})
        current_tree = current_file.mktree("HPGE_CURRENT_GATED", {
            "event_id": "uint32", "trigger_time_s": "float32", "peak_index_250msps": "int32",
            "charge_amplitude_adc": "float32", "current_waveform": "1001 * float32"})

        for arrays in source["HPGE"].iterate(["event"], step_size=1500, library="np"):
            event = arrays["event"]
            raw = event["waveform"].astype(np.float32, copy=False)
            if raw.shape[1] != 4500:
                raise ValueError(f"Expected 4500 samples at 250 MSPS; received {raw.shape[1]}")
            charge = raw.copy()
            baseline = charge[:, BASELINE_SLICE].mean(axis=1)
            amplitude = baseline - charge[:, PLATEAU_SLICE].mean(axis=1)
            lower_rail = np.any(raw <= 0.0, axis=1)
            large = amplitude >= 250.0
            physics = (~lower_rail) & (~large) & (amplitude > 0.0)

            charge_tree.extend({"event_id": event["event_id"],
                                "trigger_time_s": event["trigger_time_s"], "waveform": charge,
                                "charge_amplitude_adc": amplitude.astype(np.float32),
                                "is_physics_candidate": physics})
            selected = charge[physics]
            selected_ids = event["event_id"][physics]
            selected_times = event["trigger_time_s"][physics]
            selected_amplitudes = amplitude[physics]
            current = np.zeros_like(selected)
            current[:, 1:-1] = -0.5 * (selected[:, 2:] - selected[:, :-2])
            peak_index = CURRENT_SEARCH_SLICE.start + np.argmax(current[:, CURRENT_SEARCH_SLICE], axis=1)
            gate_indices = peak_index[:, None] + CURRENT_GATE_OFFSETS[None, :]
            gated = np.take_along_axis(current, gate_indices, axis=1).astype(np.float32)
            current_tree.extend({"event_id": selected_ids, "trigger_time_s": selected_times,
                                 "peak_index_250msps": peak_index.astype(np.int32),
                                 "charge_amplitude_adc": selected_amplitudes.astype(np.float32),
                                 "current_waveform": gated})

            needed = 100 - len(first_current_ids)
            if needed > 0:
                first_current.extend(gated[:needed])
                first_current_ids.extend(selected_ids[:needed])
            total_entries += len(raw)
            physics_entries += int(np.count_nonzero(physics))
            lower_rail_entries += int(np.count_nonzero(lower_rail))
            large_entries += int(np.count_nonzero(large & ~lower_rail))
            amplitudes_all.append(amplitude)
            amplitudes_physics.append(selected_amplitudes)

    amplitudes_all = np.concatenate(amplitudes_all)
    amplitudes_physics = np.concatenate(amplitudes_physics)
    fits = [fit_peak(amplitudes_physics, 82.0, 89.0), fit_peak(amplitudes_physics, 94.0, 101.0)]
    calibration = np.polyfit([fit["mean_adc"] for fit in fits], CO60_ENERGIES_KEV, 1)
    for fit in fits:
        fit["fwhm_kev"] = float(fit["fwhm_adc"] * calibration[0])
    plot_spectrum(amplitudes_physics, calibration, fits, args.output_dir, stem)
    plot_current(np.asarray(first_current), np.asarray(first_current_ids), args.output_dir, stem)

    summary = {
        "input_root": str(args.input_root), "downsample_factor": 1,
        "input_sample_rate_msps": 250, "output_sample_rate_msps": 250,
        "input_stored_samples": 4500,
        "decimation_filter": "none; input is acquired directly at the 250 MSPS target",
        "total_board_accepted_entries": total_entries, "physics_candidate_entries": physics_entries,
        "nonphysics_rule": "exclude lower-rail records, amplitude >=250 ADC, and nonpositive reconstructed amplitudes",
        "lower_rail_entries": lower_rail_entries, "large_nonrail_entries": large_entries,
        "charge_output_root": str(charge_path), "current_output_root": str(current_path),
        "charge_amplitude": {"baseline_samples_250msps": [250, 1100],
                             "plateau_samples_250msps": [1350, 1600]},
        "photopeak_resolution": {"1173_228_kev": fits[0], "1332_492_kev": fits[1]},
        "energy_calibration": {"kev_per_adc": float(calibration[0]),
                               "intercept_kev": float(calibration[1])},
        "current_conversion": {"definition": "-(charge[n+1]-charge[n-1])/2",
                               "units": "ADC-code change per 4 ns sample",
                               "peak_search_samples_250msps": [1100, 1500],
                               "gate_relative_samples": [-500, 500], "gate_samples": 1001,
                               "gate_duration_us": 4.004},
        "trigger_threshold_adc": 10,
    }
    summary_path = args.output_dir / f"{stem}_250msps_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
