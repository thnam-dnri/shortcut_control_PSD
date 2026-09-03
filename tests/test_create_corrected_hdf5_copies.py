from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "create_corrected_hdf5_copies.py"
SPEC = importlib.util.spec_from_file_location("create_corrected_hdf5_copies", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def rows_for(source: str, peaks: list[tuple[str, float, float]], status: str = "OK") -> list[dict[str, str]]:
    return [
        {
            "source": source,
            "peak_name": name,
            "reference_energy_kev": str(reference),
            "fitted_centroid_kev": str(observed),
            "fit_status": status,
        }
        for name, reference, observed in peaks
    ]


def write_peak_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_robust_three_line_model_ignores_outlier() -> None:
    rows = rows_for(
        "th232",
        [("a", 100.0, 110.0), ("b", 500.0, 510.0), ("c", 1000.0, 1020.0)],
    )
    model = MODULE.build_calibration_models(rows)["th232"]
    assert model.model_label == "PRELIMINARY"
    assert model.scale == pytest.approx(1.01, abs=0.01)
    assert model.offset_kev == pytest.approx(8.0, abs=4.0)
    assert len(model.lines) == 3


def test_exact_two_line_affine_model() -> None:
    rows = rows_for("co60", [("a", 100.0, 112.0), ("b", 500.0, 532.0)])
    model = MODULE.build_calibration_models(rows)["co60"]
    assert model.scale == pytest.approx(1.05)
    assert model.offset_kev == pytest.approx(7.0)
    assert all(float(line["residual_kev"]) == pytest.approx(0.0) for line in model.lines)


def test_cs137_one_line_uses_fixed_intercept() -> None:
    reference = 661.657
    observed = 1.002 * reference + MODULE.ORIGINAL_CALIBRATION_INTERCEPT_KEV * (1 - 1.002)
    model = MODULE.build_calibration_models(
        rows_for("cs137", [("cs", reference, observed)])
    )["cs137"]
    assert model.model_label == "CONSTRAINED"
    assert model.fixed_intercept is True
    assert model.pivot_intercept_kev == pytest.approx(MODULE.ORIGINAL_CALIBRATION_INTERCEPT_KEV)
    assert model.offset_kev == pytest.approx(MODULE.ORIGINAL_CALIBRATION_INTERCEPT_KEV * (1.0 - 1.002))
    assert model.scale == pytest.approx(1.002)
    assert model.correct(np.asarray([observed]))[0] == pytest.approx(reference)


def test_full_copy_preserves_original_and_corrects_nonfinite_values(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "waveform_hdf5_corrected"
    source_path = input_dir / "co60" / "sample.h5"
    source_path.parent.mkdir(parents=True)
    energies = np.array([112.0, 532.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    with h5py.File(source_path, "w") as handle:
        handle.attrs["source_label"] = "co60"
        handle.attrs["processing_status"] = "OK"
        handle.create_dataset("reconstructed_energy_kev", data=energies, chunks=(2,))
        handle.create_dataset("event_id", data=np.arange(5, dtype=np.uint32))
    original_hash = MODULE.sha256_file(source_path)
    peak_csv = tmp_path / "peak_positions.csv"
    write_peak_csv(peak_csv, rows_for("co60", [("a", 100.0, 112.0), ("b", 500.0, 532.0)]))
    results_json = tmp_path / "results.json"
    results_json.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    manifest = MODULE.create_copies(input_dir, output_dir, peak_csv, results_json, manifest_path)
    copied = output_dir / "co60" / "sample.h5"
    assert MODULE.sha256_file(source_path) == original_hash
    assert copied.exists()
    with h5py.File(copied, "r") as handle:
        corrected = handle["corrected_energy_kev"][:]
        np.testing.assert_allclose(corrected[:2], [100.0, 500.0], rtol=0, atol=1e-5)
        assert np.isnan(corrected[2])
        assert np.isposinf(corrected[3])
        assert np.isneginf(corrected[4])
        assert handle["corrected_energy_kev"].attrs["correction_formula"] == "(reconstructed_energy_kev - offset_kev) / scale"
        assert handle.attrs["corrected_energy_dataset"] == "corrected_energy_kev"
    assert manifest["records"][0]["source_sha256"] == original_hash
    assert manifest["records"][0]["output_sha256"] == MODULE.sha256_file(copied)
    assert not list(output_dir.rglob("*.partial"))


def test_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    source = input_dir / "sample.h5"
    input_dir.mkdir()
    with h5py.File(source, "w") as handle:
        handle.attrs["source_label"] = "co60"
        handle.attrs["processing_status"] = "INPUT_INVALID"
    output_dir = tmp_path / "waveform_hdf5_corrected"
    output_dir.mkdir()
    (output_dir / "sentinel").write_text("owned", encoding="utf-8")
    peak_csv = tmp_path / "peak_positions.csv"
    write_peak_csv(peak_csv, rows_for("co60", [("a", 100.0, 110.0), ("b", 500.0, 520.0)]))
    results_json = tmp_path / "results.json"
    results_json.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        MODULE.create_copies(input_dir, output_dir, peak_csv, results_json, tmp_path / "manifest.json")
    assert (output_dir / "sentinel").read_text(encoding="utf-8") == "owned"
