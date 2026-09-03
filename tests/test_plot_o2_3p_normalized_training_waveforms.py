from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/plot_o2_3p_normalized_training_waveforms.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_o2_inputs_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_positive_sampling_is_deterministic_and_peak_balanced(tmp_path):
    module = _load_module()
    path = tmp_path / "labels.csv"
    fields = (
        "partition",
        "positive_label",
        "peak_id",
        "pair_id",
        "positive_hdf5",
        "positive_row",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for peak_id, _label, _color in module.PEAKS:
            for index in range(20):
                writer.writerow(
                    {
                        "partition": "train",
                        "positive_label": 1,
                        "peak_id": peak_id,
                        "pair_id": f"{peak_id}_{index}",
                        "positive_hdf5": f"{peak_id}.h5",
                        "positive_row": index,
                    }
                )
    first = module.select_positive_rows(path, 5, 17)
    second = module.select_positive_rows(path, 5, 17)
    assert {key: [row["pair_id"] for row in value] for key, value in first.items()} == {
        key: [row["pair_id"] for row in value] for key, value in second.items()
    }
    assert all(len(rows) == 5 for rows in first.values())
