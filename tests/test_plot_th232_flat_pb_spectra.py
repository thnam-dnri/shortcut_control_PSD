from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts/plot_th232_flat_pb_spectra.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_th232_flat_pb_spectra_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_gains_list():
    module = _load_module()
    assert module.TARGET_GAINS_PCT == [5, 10, 20, 30, 45]
    assert "No cut" in module.COLORS
    assert "+45% P/B" in module.COLORS
