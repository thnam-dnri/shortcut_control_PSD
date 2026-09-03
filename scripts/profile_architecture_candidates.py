#!/usr/bin/env python3
"""Profile architecture candidates on synthetic tensors only.

This is a hardware-pipeline exercise. It intentionally does not load labels,
waveforms, locked data, Th-232, or Eu-152 and must not be used to rank scientific
classification performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.architecture_candidates import build_candidate  # noqa: E402
CANDIDATES = ("ds_cnn", "tcn", "multi_rate_hpge", "cnn_gru")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = output_shape(item)
            if shape is not None:
                return shape
    return None


def profile_model(model: nn.Module, input_tensor: torch.Tensor) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    hooks = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        shape = output_shape(output)
        if shape is None:
            return
        macs = 0
        if isinstance(module, nn.Conv1d):
            macs = int(
                np.prod(shape[1:])
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
            )
        elif isinstance(module, nn.Linear):
            macs = int(module.in_features * module.out_features)
        elif isinstance(module, nn.GRU):
            sequence = inputs[0]
            sequence_length = int(sequence.shape[1] if module.batch_first else sequence.shape[0])
            directions = 2 if module.bidirectional else 1
            macs = int(
                sequence_length
                * module.num_layers
                * directions
                * 3
                * (module.input_size * module.hidden_size + module.hidden_size * module.hidden_size)
            )
        records.append(
            {
                "module": module.__class__.__name__,
                "output_shape": shape,
                "macs_per_batch_item": macs,
                "activation_bytes": int(np.prod(shape) * input_tensor.element_size()),
            }
        )

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.GRU)):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        output = model(input_tensor)
    for handle in hooks:
        handle.remove()
    return {
        "output_shape": tuple(int(item) for item in output.shape),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "macs_per_event": int(sum(item["macs_per_batch_item"] for item in records)),
        "activation_bytes_sum": int(sum(item["activation_bytes"] for item in records)),
        "peak_single_activation_bytes": int(max((item["activation_bytes"] for item in records), default=0)),
        "layers": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--samples", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.channels, args.samples, args.batch_size, args.width) < 1:
        raise ValueError("channels, samples, batch-size, and width must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    input_tensor = torch.randn(args.batch_size, args.channels, args.samples)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "HARDWARE_PIPELINE_PROFILE_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_input": {
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
            "sample_period_ns": 4.0,
            "source_data_loaded": False,
        },
        "hardware_contract": {
            "status": "UNDEFINED",
            "target_device": None,
            "clock_hz": None,
            "input_precision": None,
            "weight_precision": None,
            "activation_precision": None,
            "throughput_requirement": None,
            "latency_budget": None,
            "resource_limits": None,
            "note": "MACs and PyTorch parameter counts are not synthesis results.",
        },
        "candidates": {},
    }
    for name in CANDIDATES:
        model = build_candidate(name, input_channels=args.channels, width=args.width)
        model.eval()
        result["candidates"][name] = profile_model(model, input_tensor)
    write_json(output_dir / "architecture_profile.json", result)
    print(json.dumps({
        "output": (output_dir / "architecture_profile.json").relative_to(PROJECT_ROOT).as_posix(),
        "input_shape": list(input_tensor.shape),
        "candidates": {
            name: {
                "parameters": values["parameter_count"],
                "macs_per_event": values["macs_per_event"],
                "output_shape": values["output_shape"],
            }
            for name, values in result["candidates"].items()
        },
        "source_data_loaded": False,
        "status": result["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
