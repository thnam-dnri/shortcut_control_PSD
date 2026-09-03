from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_source_continuum_event_store.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_source_continuum_event_store_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source(path: Path, energies: list[float]) -> None:
    count = len(energies)
    with h5py.File(path, "w") as handle:
        handle.attrs["source_label"] = "co60"
        handle.create_dataset(
            "waveform",
            data=np.arange(count * 4500, dtype=np.float32).reshape(count, 4500),
        )
        handle.create_dataset(
            "noise_rms_adc", data=np.ones((count, 6), dtype=np.float32)
        )
        handle.create_dataset("event_id", data=np.arange(count, dtype=np.uint32))
        handle.create_dataset(
            "reconstructed_energy_kev", data=np.asarray(energies, dtype=np.float32)
        )
        handle.create_dataset(
            "corrected_energy_kev", data=np.asarray(energies, dtype=np.float32)
        )
        handle.create_dataset(
            "shaped_energy_unit", data=np.asarray(energies, dtype=np.float32) / 10
        )
        handle.create_dataset(
            "pulse_extremum_adc", data=-np.ones(count, dtype=np.float32)
        )
        handle.create_dataset(
            "pulse_extremum_index", data=np.arange(count, dtype=np.int32)
        )
        handle.create_dataset(
            "trigger_time_s", data=np.arange(count, dtype=np.float32)
        )


def _entry(module, path: Path, partition: str) -> dict[str, object]:
    return {
        "source": "co60",
        "partition": partition,
        "hdf5": str(path),
        "hdf5_sha256": module.sha256_file(path),
        "complete_input": True,
        "processing_status": "OK",
        "qc_status": "PASS",
    }


def test_selection_requires_development_only_manifest(tmp_path):
    module = _load_module()
    train = tmp_path / "train.h5"
    validation = tmp_path / "validation.h5"
    test = tmp_path / "test.h5"
    for path in (train, validation, test):
        _write_source(path, [100.0])
    development_entries = [
        _entry(module, train, "train"),
        _entry(module, validation, "validation"),
    ]

    selected = module.select_source_files(
        {"files": development_entries}, "co60", ("train", "validation")
    )

    assert [item["hdf5"] for item in selected["train"]] == [str(train)]
    assert [item["hdf5"] for item in selected["validation"]] == [str(validation)]
    with pytest.raises(ValueError, match="development-only"):
        module.select_source_files(
            {"files": development_entries + [_entry(module, test, "test")]},
            "co60",
            ("train", "validation"),
        )
    with pytest.raises(ValueError, match="locked"):
        module.select_source_files(
            {"files": development_entries}, "co60", ("test",)
        )


def test_partition_store_retains_full_inclusive_band_and_provenance(tmp_path):
    module = _load_module()
    source = tmp_path / "co60.h5"
    _write_source(source, [99.9, 100.0, 500.0, 1000.0, 1000.1, float("nan")])
    entry = _entry(module, source, "train")
    output = tmp_path / "train_events.h5"

    result = module.build_partition_store(
        "train", [entry], output, "co60", 100.0, 1000.0, chunk_events=2
    )

    assert result["event_count"] == 3
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["source_row"][:], [1, 2, 3])
        np.testing.assert_allclose(
            handle["corrected_energy_kev"][:], [100.0, 500.0, 1000.0]
        )
        np.testing.assert_array_equal(handle["source_file_index"][:], [0, 0, 0])
        assert handle["waveform"].shape == (3, 4500)
        assert handle["noise_rms_adc"].shape == (3, 5)
        assert bool(handle.attrs["test_partition_used"]) is False
        assert handle["source_files/selected_event_count"][0] == 3
        assert handle["source_files/path"].asstr()[0] == str(source)
