from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_architecture_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_architecture_candidates_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_representation_contract():
    module = _load_module()

    assert module.REPRESENTATION_NAME == "both_ma10_energy_t10_w750"
    assert module.REPRESENTATION.channel_count == 2
    assert module.REPRESENTATION.window_length == 750


def test_parser_defaults_keep_warning_and_locked_boundaries():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.labels_dir.name == "architecture_pass_warn_20260815"
    assert args.event_store_dir.name == "architecture_pass_warn_20260815"
    assert args.output_dir.name == "architecture_candidates_warning_balanced_20260816"
    assert args.device == "auto"


def test_peak_balanced_subset_uses_all_peaks():
    module = _load_module()
    raw = module.RawPartition(
        waveforms=np.arange(48, dtype=np.float32).reshape(12, 4),
        shaped_energy=np.ones(12, dtype=np.float32),
        labels=np.asarray([1, 0] * 6, dtype=np.float32),
        weights=np.ones(12, dtype=np.float32),
        peak_ids=np.asarray(["p1", "p1"] * 2 + ["p2", "p2"] * 2 + ["p3", "p3"] * 2),
    )

    selected, metadata = module.select_peak_balanced_subset(raw, 6, 7)

    assert selected.labels.size == 6
    assert metadata["mode"] == "equal_peak_pair_subset"
    assert metadata["peak_pair_counts"] == {"p1": 1, "p2": 1, "p3": 1}


def test_warning_metadata_is_explicit():
    module = _load_module()

    metadata = module.warning_metadata("ds_cnn")

    assert metadata["status"] == "SCALAR_SHORTCUT_WARNING_EXTERNAL_VALIDATION_REQUIRED"
    assert metadata["test_partition_used"] is False
    assert metadata["candidate"] == "ds_cnn"
    assert "external_auroc" in metadata["external_return_metrics"]
    assert "external_spectral_peak_to_background" in metadata["external_return_metrics"]
