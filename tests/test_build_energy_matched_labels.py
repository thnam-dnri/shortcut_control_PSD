from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_energy_matched_labels.py"
SPEC = importlib.util.spec_from_file_location("build_energy_matched_labels", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_peak_definition_half_fwhm_roi() -> None:
    peak = MODULE.PeakDefinition(
        peak_id="cs137_662kev",
        source="cs137",
        nominal_energy_kev=661.657,
        fitted_center_kev=661.668,
        fwhm_kev=3.749,
        training_label=True,
        note="test",
    )
    assert peak.roi_half_width_kev == pytest.approx(0.5 * 3.749)
    assert peak.roi_low_kev == pytest.approx(661.668 - 0.5 * 3.749)
    assert peak.roi_high_kev == pytest.approx(661.668 + 0.5 * 3.749)
    as_dict = peak.as_dict()
    assert as_dict["roi_definition"] == "fitted_center +/- 0.5 FWHM"


def test_partition_counts() -> None:
    counts_10 = MODULE.partition_counts(10)
    assert counts_10 == {"train": 6, "validation": 2, "test": 2}
    assert sum(counts_10.values()) == 10

    counts_25 = MODULE.partition_counts(25)
    assert counts_25["train"] >= 15
    assert counts_25["validation"] >= 1
    assert counts_25["test"] >= 1
    assert sum(counts_25.values()) == 25


def test_ks_distance_identical_and_distinct() -> None:
    first = np.array([1.0, 2.0, 3.0, 4.0])
    assert MODULE.two_sample_ks_distance(first, first) == pytest.approx(0.0)
    second = np.array([10.0, 11.0, 12.0, 13.0])
    assert MODULE.two_sample_ks_distance(first, second) == pytest.approx(1.0)


def test_draft_allowlist_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "qc_allowlist.json"
    path.write_text(
        '{"status": "DRAFT_ADJUDICATION_REQUIRED", "approved_policy": null}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not frozen"):
        MODULE.load_approved_allowlist(path)


def test_frozen_allowlist_inherits_immutable_partition(tmp_path: Path) -> None:
    allowlist = tmp_path / "qc_allowlist.json"
    allowlist.write_text(
        '{"status": "FROZEN", "approved_policy": "pass_warn", '
        '"approved_hdf5": ["processed_data/a.h5"]}',
        encoding="utf-8",
    )
    partition = tmp_path / "file_partition_manifest.json"
    partition.write_text(
        '{"files": [{"hdf5": "processed_data/a.h5", "partition": "validation"}]}',
        encoding="utf-8",
    )
    data, admitted = MODULE.load_approved_allowlist(allowlist)
    assert data["approved_policy"] == "pass_warn"
    assert admitted == {"processed_data/a.h5"}
    assert MODULE.load_frozen_partition_map(partition) == {
        "processed_data/a.h5": "validation"
    }
