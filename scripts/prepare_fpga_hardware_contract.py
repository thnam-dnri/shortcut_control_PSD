#!/usr/bin/env python3
"""Prepare a warning-gated analytic FPGA contract from existing profiles.

This script reads synthetic architecture profiles and development-only
quantization results. It does not load waveforms or labels and it deliberately
leaves target-specific FPGA fields unresolved until the hardware contract is
authorized and defined.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access_guards import assert_no_forbidden_path


WARNING_STATUS = "HARDWARE_CONTRACT_WARNING_TARGET_UNDEFINED"


def load_json(path: Path) -> dict[str, Any]:
    assert_no_forbidden_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/architecture_profile/architecture_candidates_20260816/architecture_profile.json",
    )
    parser.add_argument(
        "--quantization",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_warning_balanced_20260816/quantization_screen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/protocol/fpga_hardware_contract_balanced_warning_20260816.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile_path = args.profile.resolve()
    quantization_path = args.quantization.resolve()
    output_path = args.output.resolve()
    for path in (profile_path, quantization_path, output_path):
        assert_no_forbidden_path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    if not quantization_path.is_file():
        raise FileNotFoundError(quantization_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    profile = load_json(profile_path)
    quantization = load_json(quantization_path)
    if profile.get("status") != "HARDWARE_PIPELINE_PROFILE_ONLY":
        raise ValueError("Unexpected architecture profile status")
    if quantization.get("warning_status") != "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED":
        raise ValueError("Quantization artifact is missing the active shortcut warning")

    candidates: dict[str, Any] = {}
    for name, values in profile["candidates"].items():
        parameters = int(values["parameter_count"])
        macs = int(values["macs_per_event"])
        peak_activation = int(values["peak_single_activation_bytes"])
        quantized = quantization["candidates"].get(name, {})
        candidates[name] = {
            "parameter_count": parameters,
            "macs_per_event": macs,
            "float32_weight_bytes_estimate": parameters * 4,
            "int16_weight_bytes_estimate": parameters * 2,
            "int8_weight_bytes_estimate": parameters,
            "float32_peak_activation_bytes_profiled": peak_activation,
            "int16_peak_activation_bytes_estimate": peak_activation // 2,
            "int8_peak_activation_bytes_estimate": peak_activation // 4,
            "quantization_screen": {
                bits: {
                    "auroc": quantized["by_bits"][bits]["metrics"]["auroc"],
                    "weighted_auroc": quantized["by_bits"][bits]["metrics"][
                        "weighted_auroc"
                    ],
                    "delta_from_float32": quantized["by_bits"][bits][
                        "delta_from_float32"
                    ],
                }
                for bits in ("8", "16")
            },
        }

    result = {
        "schema_version": 1,
        "status": WARNING_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "profile": profile_path.relative_to(PROJECT_ROOT).as_posix(),
            "quantization": quantization_path.relative_to(PROJECT_ROOT).as_posix(),
            "synthetic_profile_source_data_loaded": profile["synthetic_input"][
                "source_data_loaded"
            ],
            "quantization_test_partition_used": quantization["input"][
                "test_partition_used"
            ],
        },
        "interface_assumptions": {
            "representation": "both_ma10_energy_t10_w750",
            "input_shape": [2, 750],
            "sample_period_ns": 4.0,
            "candidate_precision_screen_bits": [8, 16],
        },
        "target_contract": {
            "target_device": None,
            "speed_grade": None,
            "clock_hz": None,
            "throughput_events_per_second": None,
            "latency_budget_ns": None,
            "interface": None,
            "input_precision": None,
            "weight_precision": None,
            "activation_precision": None,
            "resource_limits": {
                "dsp": None,
                "lut": None,
                "ff": None,
                "bram": None,
                "uram": None,
            },
        },
        "required_acceptance_evidence": [
            "target-specific synthesis utilization",
            "timing closure at the frozen clock",
            "streaming latency and initiation interval/throughput",
            "software-versus-RTL numerical agreement",
            "frozen external AUROC and spectral P/B/retention result",
        ],
        "candidates": candidates,
        "caveats": [
            "Parameter/MAC/storage figures are analytic estimates, not FPGA resource measurements.",
            "The shortcut warning remains active and no scientific architecture winner is selected here.",
            "A target-specific synthesis step is blocked only by the undefined contract fields above; implementation work may continue after they are frozen.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
