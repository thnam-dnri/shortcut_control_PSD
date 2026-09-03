from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_peak_combination_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_peak_combination_labels_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combination_specs_and_parser_are_development_only():
    module = _load_module()

    assert module.COMBINATIONS["ba_low"] == {
        "ba133": ("ba133_276kev", "ba133_303kev")
    }
    assert module.COMBINATIONS["ba_high_na511"] == {
        "ba133": ("ba133_356kev", "ba133_384kev"),
        "na22": ("na22_511kev",),
    }
    args = module.build_parser().parse_args([])
    assert args.output_root.name == "peak_combinations"
    assert args.combination == "all"


def test_build_combination_filters_rows_and_reweights_peaks(tmp_path):
    module = _load_module()
    source_root = tmp_path / "source_ablation"
    output_root = tmp_path / "peak_combinations"
    fieldnames = [
        "pair_id",
        "positive_hdf5",
        "positive_row",
        "positive_energy_kev",
        "peak_id",
        "negative_hdf5",
        "negative_row",
        "negative_energy_kev",
        "negative_source",
        "source_weight",
    ]
    rows = {
        "ba133_positive": [
            {
                "pair_id": "old0",
                "positive_hdf5": "ba_a.h5",
                "positive_row": "0",
                "positive_energy_kev": "276.0",
                "peak_id": "ba133_276kev",
                "negative_hdf5": "co_a.h5",
                "negative_row": "0",
                "negative_energy_kev": "276.2",
                "negative_source": "co60",
                "source_weight": "1.0",
            },
            {
                "pair_id": "old1",
                "positive_hdf5": "ba_a.h5",
                "positive_row": "1",
                "positive_energy_kev": "303.0",
                "peak_id": "ba133_303kev",
                "negative_hdf5": "co_a.h5",
                "negative_row": "1",
                "negative_energy_kev": "303.1",
                "negative_source": "co60",
                "source_weight": "1.0",
            },
        ],
        "na22_positive": [
            {
                "pair_id": "old2",
                "positive_hdf5": "na_a.h5",
                "positive_row": "0",
                "positive_energy_kev": "511.0",
                "peak_id": "na22_511kev",
                "negative_hdf5": "co_b.h5",
                "negative_row": "0",
                "negative_energy_kev": "511.2",
                "negative_source": "co60",
                "source_weight": "1.0",
            }
        ],
    }
    for source_dir_name, source_rows in rows.items():
        source_dir = source_root / source_dir_name
        source_dir.mkdir(parents=True)
        (source_dir / "file_partition_manifest.json").write_text("{}\n")
        for partition in module.PARTITIONS:
            with (source_dir / f"label_pairs_{partition}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(source_rows)

    result = module.build_combination(
        "ba_low",
        module.COMBINATIONS["ba_low"],
        source_root,
        output_root,
        seed=7,
        overwrite=False,
    )

    assert result["partitions"]["train"]["pair_count"] == 2
    output_csv = output_root / "ba_low" / "label_pairs_train.csv"
    with output_csv.open(newline="", encoding="utf-8") as stream:
        output_rows = list(csv.DictReader(stream))
    assert {row["peak_id"] for row in output_rows} == {
        "ba133_276kev",
        "ba133_303kev",
    }
    assert {row["source_weight"] for row in output_rows} == {"1.0"}
    manifest = json.loads(
        (output_root / "ba_low" / "label_dataset_manifest.json").read_text()
    )
    assert manifest["test_partition_used"] is False
    assert manifest["external_data_used"] is False
