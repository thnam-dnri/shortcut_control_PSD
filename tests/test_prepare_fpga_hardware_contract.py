from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_fpga_hardware_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_fpga_hardware_contract_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_parser_points_to_warning_artifacts():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.profile.name == "architecture_profile.json"
    assert args.quantization.name == "quantization_screen.json"
    assert args.output.name == "fpga_hardware_contract_balanced_warning_20260816.json"
    assert module.WARNING_STATUS == "HARDWARE_CONTRACT_WARNING_TARGET_UNDEFINED"
