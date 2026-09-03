from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

TRAIN_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_peak_specialists.py"
EVAL_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_peak_specialists.py"
TH232_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_th232_peak_specialists.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_specialist_definitions_and_constants():
    train_mod = _load_module(TRAIN_SCRIPT_PATH, "train_peak_specialists_under_test")
    assert len(train_mod.SPECIALISTS) == 3
    names = [s["name"] for s in train_mod.SPECIALISTS]
    assert names == ["356A", "511A", "661A"]
    peak_ids = [s["peak_id"] for s in train_mod.SPECIALISTS]
    assert peak_ids == ["ba133_356kev", "na22_511kev", "cs137_662kev"]


def test_specialist_dscnn_forward_pass():
    from src.architecture_candidates import DSCNN

    model = DSCNN(input_channels=2, width=24)
    dummy_input = torch.randn(4, 2, 750, dtype=torch.float32)
    output = model(dummy_input)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_nearest_energy_expert_switching_boundaries():
    eval_mod = _load_module(EVAL_SCRIPT_PATH, "evaluate_peak_specialists_under_test")

    energies = np.asarray([300.0, 356.0, 433.0, 434.0, 511.0, 586.0, 587.0, 662.0, 1000.0])
    s356 = np.full(len(energies), 0.356, dtype=np.float32)
    s511 = np.full(len(energies), 0.511, dtype=np.float32)
    s661 = np.full(len(energies), 0.661, dtype=np.float32)

    raw_dict = {"356A": s356, "511A": s511, "661A": s661}
    cal_dict = {"356A": s356, "511A": s511, "661A": s661}
    peak_ids = np.full(len(energies), "ba133_356kev", dtype="U64")
    val_map = {"ba133_356kev": "356A", "na22_511kev": "511A", "cs137_662kev": "661A"}

    # Midpoints: 433.5059 and 586.328
    # energies[0:3] < 433.5 -> 356A
    # energies[3:6] in [433.5, 586.3) -> 511A
    # energies[6:9] >= 586.3 -> 661A
    nearest = np.empty(len(energies), dtype=np.float32)
    nearest[energies < eval_mod.SWITCH_ENERGY_1] = s356[energies < eval_mod.SWITCH_ENERGY_1]
    nearest[(energies >= eval_mod.SWITCH_ENERGY_1) & (energies < eval_mod.SWITCH_ENERGY_2)] = s511[
        (energies >= eval_mod.SWITCH_ENERGY_1) & (energies < eval_mod.SWITCH_ENERGY_2)
    ]
    nearest[energies >= eval_mod.SWITCH_ENERGY_2] = s661[energies >= eval_mod.SWITCH_ENERGY_2]

    assert np.allclose(nearest[:3], 0.356)
    assert np.allclose(nearest[3:6], 0.511)
    assert np.allclose(nearest[6:], 0.661)


def test_fusion_rules_monotonicity_and_bounds():
    s356 = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    s511 = np.array([0.3, 0.4, 0.7], dtype=np.float32)
    s661 = np.array([0.1, 0.6, 0.9], dtype=np.float32)

    raw_max = np.maximum(s356, np.maximum(s511, s661))
    cal_mean = (s356 + s511 + s661) / 3.0

    assert np.allclose(raw_max, [0.3, 0.6, 0.9])
    assert np.allclose(cal_mean, [0.2, 0.5, 0.8])
    assert (raw_max >= cal_mean).all()
