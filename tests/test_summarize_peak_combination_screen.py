from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "summarize_peak_combination_screen.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_peak_combination_screen_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_internal_peak_combination_report():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.labels_root.name == "peak_combinations"
    assert args.model_root.name == "peak_combinations"
    assert args.output.name == "peak_combination_screen_summary.json"
    assert args.combinations.split(",") == list(module.DEFAULT_COMBINATIONS)


def test_selected_compact_trial_follows_validation_ranking():
    module = _load_module()
    result = {
        "ranking": [{"config": "best", "validation_weighted_auroc": 0.7}],
        "trials": [
            {"config": {"name": "other"}, "best_epoch": 1},
            {"config": {"name": "best"}, "best_epoch": 2},
        ],
    }

    assert module.selected_compact_trial(result)["best_epoch"] == 2
