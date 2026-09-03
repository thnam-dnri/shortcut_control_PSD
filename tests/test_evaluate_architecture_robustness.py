from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from src.ba133_cnn import RawPartition


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_architecture_robustness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_architecture_robustness_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_partition() -> RawPartition:
    return RawPartition(
        waveforms=np.zeros((4, 4), dtype=np.float32),
        shaped_energy=np.ones(4, dtype=np.float32),
        labels=np.asarray([1, 0, 1, 0], dtype=np.float32),
        weights=np.ones(4, dtype=np.float32),
        peak_ids=np.asarray(["p1", "p1", "p2", "p2"]),
    )


def test_grouped_metrics_preserves_pair_order():
    module = _load_module()
    raw = _raw_partition()
    scores = np.asarray([0.9, 0.1, 0.2, 0.8], dtype=np.float32)

    result = module.grouped_metrics(["first", "second"], raw, scores)

    assert result["first"]["event_count"] == 2
    assert result["second"]["pair_count"] == 1
    assert result["first"]["auroc"] == 1.0
    assert result["second"]["auroc"] == 0.0


def test_pair_group_metadata_maps_sessions_and_qc():
    module = _load_module()
    rows = [
        {
            "peak_id": "p1",
            "positive_source": "ba133",
            "negative_source": "co60",
            "positive_hdf5": "positive.h5",
            "negative_hdf5": "negative.h5",
            "positive_qc_status": "PASS",
            "negative_qc_status": "WARN",
        }
    ]
    registry = {
        "positive.h5": {"canonical_session_id": "session_pos", "acquisition_block_id": "block_pos"},
        "negative.h5": {"canonical_session_id": "session_neg", "acquisition_block_id": "block_neg"},
    }

    result = module.build_pair_groups(rows, registry)

    assert result["positive_session"] == ["session_pos"]
    assert result["negative_session"] == ["session_neg"]
    assert result["qc_pair"] == ["PASS__WARN"]
    assert result["positive_acquisition_block"] == ["block_pos"]
    assert result["negative_acquisition_block"] == ["block_neg"]


def test_parser_defaults_are_development_only():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.labels_dir.name == "architecture_pass_warn_20260815"
    assert args.event_store_dir.name == "architecture_pass_warn_20260815"
    assert args.model_dir.name == "architecture_candidates_warning_balanced_20260816"
    assert args.output.name == "robustness_validation.json"
    assert args.minimum_group_pairs == 100
