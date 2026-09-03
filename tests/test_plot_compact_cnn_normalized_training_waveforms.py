from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/plot_compact_cnn_normalized_training_waveforms.py"


def test_compact_plotter_uses_same_three_positive_peaks():
    spec = importlib.util.spec_from_file_location("plot_compact_inputs_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert tuple(item[0] for item in module.PEAKS) == (
        "ba133_356kev",
        "na22_511kev",
        "cs137_662kev",
    )
