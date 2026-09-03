from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_stage2_peak_combination_screen.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_stage2_peak_combination_screen_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_four_architecture_peak_screen():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.screen_root.name == "architecture_candidates_peak_combinations_warning_20260817"
    assert args.combinations.split(",") == list(module.COMBINATIONS)
    assert args.output.name == "stage2_peak_combination_summary.json"


def test_architecture_and_combination_constants_are_complete():
    module = _load_module()

    assert module.ARCHITECTURES == (
        "ds_cnn",
        "tcn",
        "multi_rate_hpge",
        "cnn_gru",
    )
    assert module.COMBINATIONS == (
        "ba_low",
        "ba_high",
        "ba_low_na511",
        "ba_high_na511",
    )
