#!/usr/bin/env python3
"""Diagnose why the 2026-08-08 Co-60 session baseline noise is elevated.

Compares the baseline (samples 0:1000, QC uses 0:500 for noise) across:
  - the scout morning file   (20260807_113540, p99 ~1.40 in old reports)
  - the evening reference    (20260807_202046, p99 ~1.044)
  - Aug-8 session files      (160801, 163230 FAIL, 164446)

Metrics per file:
  - pedestal (baseline offset) median over events
  - per-event RMS distribution on corrected samples 0:500 (same as QC)
  - per-sample variance structure: white noise vs sample-correlated pattern
  - FFT power spectrum of the mean noise to identify interference lines
  - autocorrelation at small lags to quantify colouring
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

BASELINE_SLICE = slice(0, 1000)
NOISE_SLICE = slice(0, 500)
N_EVENTS = 500





def analyze_file(path: Path, label: str, n_events: int = 500) -> dict:
    with uproot.open(path) as root_file:
        tree = root_file["HPGE"]
        branch = tree["event"]
        records = branch.array(entry_stop=n_events)
        waveforms = records["waveform"].to_numpy()  # (N, 4500)

    wf = waveforms.astype(np.float64)
    baseline = wf[:, BASELINE_SLICE]          # (N, 1000)
    noise_win = wf[:, NOISE_SLICE]            # (N, 500)

    # pedestal per event = mean of samples 0:500 (matches QC baseline subtraction)
    pedestal = noise_win.mean(axis=1)

    # corrected noise window (same as QC)
    corrected = noise_win - pedestal[:, None]
    rms = np.sqrt(np.mean(corrected**2, axis=1))

    # per-sample spread across events (fixed-pattern / sample-correlated noise)
    sample_std = corrected.std(axis=0)        # (500,)

    # mean corrected waveform -> time-averaged structure
    mean_corrected = corrected.mean(axis=0)

    # FFT power spectrum of the mean waveform (0:500 samples, 250 MSPS)
    n = len(mean_corrected)
    fft = np.fft.rfft(mean_corrected * np.hanning(n))
    freq = np.fft.rfftfreq(n, d=4e-9)
    power = np.abs(fft) ** 2

    # white-noise residual: std of corrected after removing sample-mean
    # (correlated component -> sample_std vs event-to-event white noise)
    residual = corrected - corrected.mean(axis=0)
    residual_rms = np.sqrt(np.mean(residual**2, axis=1))

    # autocorrelation of a whitened event (average of first 100 events, lag 1..8)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=100, replace=False)
    acs = []
    for i in idx:
        x = corrected[i] - corrected[i].mean()
        x = x / (np.sqrt(np.mean(x**2)) + 1e-12)
        acs.append([np.mean(x[:-lag] * x[lag:]) for lag in range(1, 9)])
    acs = np.mean(acs, axis=0)

    # dominant frequencies (skip DC)
    top_freq = sorted(
        ((power[k], freq[k]) for k in range(1, len(freq))),
        reverse=True,
    )[:5]
    top_freq = [(f, p) for p, f in top_freq]

    return {
        "label": label,
        "pedestal_p50": float(np.median(pedestal)),
        "pedestal_p99_minus_p01": float(np.quantile(pedestal, 0.99) - np.quantile(pedestal, 0.01)),
        "rms_p50": float(np.quantile(rms, 0.50)),
        "rms_p95": float(np.quantile(rms, 0.95)),
        "rms_p99": float(np.quantile(rms, 0.99)),
        "residual_rms_p99": float(np.quantile(residual_rms, 0.99)),
        "sample_std_mean": float(sample_std.mean()),
        "sample_std_max": float(sample_std.max()),
        "sample_std_min": float(sample_std.min()),
        "mean_corrected_ptp": float(np.ptp(mean_corrected)),
        "top_freqs": top_freq,
        "autocorr_lag1_8": [float(a) for a in acs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--events", type=int, default=500)
    args = parser.parse_args()

    n_events = args.events

    print(f"{'file':<28} {'ped50':>8} {'ped99-01':>8} {'r50':>6} {'r95':>6} {'r99':>6} {'res99':>6} {'sstdM':>6} {'sstdmx':>6} {'mcptp':>6}  top-freqs (MHz:pow)")
    for path in args.files:
        r = analyze_file(path, path.stem[:28], n_events=n_events)
        r["n_events"] = n_events
        print(
            f"{r['label']:<28} {r['pedestal_p50']:>8.2f} {r['pedestal_p99_minus_p01']:>8.3f} "
            f"{r['rms_p50']:>6.3f} {r['rms_p95']:>6.3f} {r['rms_p99']:>6.3f} {r['residual_rms_p99']:>6.3f} "
            f"{r['sample_std_mean']:>6.3f} {r['sample_std_max']:>6.3f} {r['mean_corrected_ptp']:>6.3f}"
        )
        freq_str = ", ".join(f"{f:.1f}:{p:.0f}" for f, p in r["top_freqs"])
        print(f"    freqs: {freq_str}")
        print(f"    ac(1..8): {', '.join(f'{a:.3f}' for a in r['autocorr_lag1_8'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
