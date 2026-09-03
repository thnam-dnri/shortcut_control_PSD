from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "preprocess_root_to_hdf5.py"
SPEC = importlib.util.spec_from_file_location("preprocess_root_to_hdf5", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_baseline_slices_invariants() -> None:
    slices = MODULE.BASELINE_SLICES
    assert len(slices) == 5
    assert slices[0] == slice(0, 200)
    assert slices[1] == slice(200, 400)
    assert slices[2] == slice(400, 600)
    assert slices[3] == slice(600, 800)
    assert slices[4] == slice(800, 1000)
    # Verify baseline region ends strictly at or before 1000 (before pulse onset ~1100-1250)
    assert slices[-1].stop <= 1000


def test_clean_waveform_with_pulse_onset_not_flagged_as_noisy() -> None:
    # Construct a synthetic negative-pulse waveform:
    # Pre-trigger baseline at 2000 ADC with std ~0.5 ADC (clean)
    # Pulse rising edge starts at sample 1120, reaching minimum at 1400 with amplitude -800 ADC
    t = np.arange(4500)
    baseline = 2000.0 + np.random.default_rng(42).normal(0.0, 0.5, size=4500)
    
    # Negative pulse starting at 1120
    pulse = np.zeros(4500)
    onset = 1120
    peak = 1400
    pulse[onset:peak] = -800.0 * ((t[onset:peak] - onset) / (peak - onset)) ** 2
    pulse[peak:] = -800.0 * np.exp(-(t[peak:] - peak) / 2500.0)
    
    waveform = (baseline + pulse).reshape(1, 4500)
    
    noise_sections = MODULE.section_noise_rms(waveform)
    assert noise_sections.shape == (1, 5)
    
    # All 5 sections in 0:1000 should have std around ~0.5, well below 1.4 ADC threshold
    assert np.all(noise_sections <= 1.4)
    
    # Contrast with old window 1000:1200 which includes the onset slope
    old_sixth_section = waveform[:, 1000:1200].std(axis=1)
    assert old_sixth_section[0] > 10.0  # Slope creates huge false noise RMS


def test_noisy_baseline_correctly_flagged() -> None:
    # Baseline with excessive noise in section 2 (std ~2.5 ADC)
    t = np.arange(4500)
    waveform = np.full((1, 4500), 2000.0)
    rng = np.random.default_rng(123)
    waveform[0, :] += rng.normal(0.0, 0.5, size=4500)
    waveform[0, 200:400] += rng.normal(0.0, 2.5, size=200)  # Inject noise into section 1
    
    noise_sections = MODULE.section_noise_rms(waveform)
    assert noise_sections[0, 0] <= 1.4
    assert noise_sections[0, 1] > 1.4
    assert np.any(noise_sections > 1.4)
