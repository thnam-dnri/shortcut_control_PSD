from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_architecture_quantization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_architecture_quantization_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symmetric_quantization_is_bounded_and_deterministic():
    module = _load_module()
    values = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0])
    scale = module.symmetric_scale(values, 8)

    result = module.quantize_dequantize(values, scale, 8)

    assert np.all(np.isfinite(result.numpy()))
    assert torch.max(torch.abs(result)) <= torch.max(torch.abs(values))
    assert torch.equal(result, module.quantize_dequantize(values, scale, 8))


def test_quantization_parser_defaults_to_warning_candidate_set():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.bits == [8, 16]
    assert args.calibration_events == 4096
    assert args.model_dir.name == "architecture_candidates_warning_balanced_20260816"
    assert args.output.name == "quantization_screen.json"
