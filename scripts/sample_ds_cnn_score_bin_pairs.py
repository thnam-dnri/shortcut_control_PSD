#!/usr/bin/env python3
"""Export raw waveform pairs from score bins of the frozen joint 3-peak DS-CNN.

The default source is the development training partition so that every
requested score bin can provide 1,000 unique matched pairs without consuming
the held-out validation pool. A pair is eligible for a bin when either member
of the matched positive/negative pair has an event-level DS-CNN score in that
bin. Both raw acquired waveforms and both individual scores are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_compact_ds_cnn_performance import (  # noqa: E402
    predict_checkpoint,
    resolve_device,
)
from src.ba133_cnn import (  # noqa: E402
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)
from src.cascade_refinement import sha256_file  # noqa: E402
from src.data_access_guards import (  # noqa: E402
    assert_development_csv,
    assert_no_forbidden_path,
)


EXPERIMENT_ID = "joint_3peak_ds_cnn_score_bins_20260821"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)
DEFAULT_HDF5 = (
    PROJECT_ROOT
    / "processed_data/visual_inspection"
    / EXPERIMENT_ID
    / "pairs_by_score_bin.h5"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs/visual_inspection" / EXPERIMENT_ID
SCORE_BINS = ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8))
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
WAVEFORM_LENGTH = 4500
SAMPLE_PERIOD_NS = 4.0


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_pair_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No pair rows found in {path}")
    required = {
        "pair_id",
        "peak_id",
        "positive_hdf5",
        "positive_row",
        "positive_label",
        "negative_hdf5",
        "negative_row",
        "negative_label",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Pair manifest lacks required columns: {sorted(missing)}")
    return rows


def load_store_indices(
    lookup_path: Path,
    pair_rows: list[dict[str, str]],
) -> np.ndarray:
    lookup: dict[tuple[str, int], int] = {}
    with lookup_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            lookup[(row["source_hdf5"], int(row["source_row"]))] = int(row["store_index"])
    indices = np.empty((len(pair_rows), 2), dtype=np.int64)
    for pair_index, row in enumerate(pair_rows):
        for side_index, side in enumerate(("positive", "negative")):
            key = (row[f"{side}_hdf5"], int(row[f"{side}_row"]))
            if key not in lookup:
                raise KeyError(f"Event-store lookup missing {key}")
            indices[pair_index, side_index] = lookup[key]
    flattened = indices.reshape(-1)
    if np.unique(flattened).size != flattened.size:
        raise ValueError("The pair manifest reuses an event-store row")
    return indices


def score_bin_mask(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if high == SCORE_BINS[-1][1]:
        return (scores >= low) & (scores <= high)
    return (scores >= low) & (scores < high)


def select_pairs(
    pair_scores: np.ndarray,
    pairs_per_bin: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if pair_scores.ndim != 2 or pair_scores.shape[1] != 2:
        raise ValueError(f"Expected pair scores with shape [N,2], got {pair_scores.shape}")
    if not np.all(np.isfinite(pair_scores)):
        raise ValueError("DS-CNN scores contain nonfinite values")
    generator = np.random.default_rng(seed)
    candidate_indices_by_bin: list[np.ndarray] = []
    for low, high in SCORE_BINS:
        member_mask = score_bin_mask(pair_scores, low, high)
        candidate_indices_by_bin.append(np.flatnonzero(np.any(member_mask, axis=1)))

    # Allocate the scarcest bins first, then emit the records in the natural
    # ascending-bin order. This keeps all six visual-inspection groups
    # globally disjoint even when a pair's two member scores cross a bin edge.
    allocation_order = sorted(
        range(len(SCORE_BINS)), key=lambda index: candidate_indices_by_bin[index].size
    )
    used_pair_indices = np.zeros(pair_scores.shape[0], dtype=bool)
    selected_by_bin: dict[int, np.ndarray] = {}
    selections: list[dict[str, Any]] = []
    availability: dict[str, dict[str, int]] = {}
    for bin_index in allocation_order:
        low, high = SCORE_BINS[bin_index]
        member_mask = score_bin_mask(pair_scores, low, high)
        candidate_indices = candidate_indices_by_bin[bin_index]
        available_indices = candidate_indices[~used_pair_indices[candidate_indices]]
        bin_name = f"{low:.1f}_{high:.1f}"
        availability[bin_name] = {
            "candidate_pair_count": int(candidate_indices.size),
            "available_after_previous_bins": int(available_indices.size),
            "requested_pair_count": int(pairs_per_bin),
        }
        if available_indices.size < pairs_per_bin:
            raise ValueError(
                f"Score bin [{low:.1f}, {high:.1f}] has only {available_indices.size} "
                f"unused candidate pairs after disjoint allocation; requested {pairs_per_bin}"
            )
        chosen = np.sort(generator.choice(available_indices, size=pairs_per_bin, replace=False))
        used_pair_indices[chosen] = True
        selected_by_bin[bin_index] = chosen
        availability[bin_name]["selected_pair_count"] = int(chosen.size)

    for bin_index, (low, high) in enumerate(SCORE_BINS):
        member_mask = score_bin_mask(pair_scores, low, high)
        chosen = selected_by_bin[bin_index]
        for rank, pair_index in enumerate(chosen):
            eligible_sides = np.flatnonzero(member_mask[pair_index])
            center = (low + high) / 2.0
            anchor_side = int(
                eligible_sides[np.argmin(np.abs(pair_scores[pair_index, eligible_sides] - center))]
            )
            selections.append(
                {
                    "bin_index": bin_index,
                    "bin_low": low,
                    "bin_high": high,
                    "selection_rank": rank,
                    "pair_index": int(pair_index),
                    "anchor_side": anchor_side,
                    "anchor_label": 1 if anchor_side == 0 else 0,
                    "anchor_score": float(pair_scores[pair_index, anchor_side]),
                }
            )
    return selections, availability


def string_values(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def write_hdf5(
    output_path: Path,
    waveforms: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    selections: list[dict[str, Any]],
    pair_rows: list[dict[str, str]],
    store_indices: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_pair_indices = np.asarray([item["pair_index"] for item in selections], dtype=np.int64)
    selected_store_indices = store_indices[selected_pair_indices]
    selected_rows = [pair_rows[int(index)] for index in selected_pair_indices]
    pair_scores = scores.reshape(-1, 2)[selected_pair_indices]
    pair_labels = labels.reshape(-1, 2)[selected_pair_indices]
    pair_waveforms = waveforms.reshape(-1, 2, WAVEFORM_LENGTH)[selected_pair_indices]
    if pair_waveforms.shape[1:] != (2, WAVEFORM_LENGTH):
        raise ValueError(f"Unexpected selected waveform shape: {pair_waveforms.shape}")

    with h5py.File(output_path, "w") as handle:
        for key, value in metadata.items():
            handle.attrs[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        handle.attrs["pair_count"] = int(len(selections))
        handle.attrs["waveform_length"] = WAVEFORM_LENGTH
        handle.attrs["sample_period_ns"] = SAMPLE_PERIOD_NS
        handle.attrs["representation"] = "raw acquired waveform; no preprocessing"
        handle.attrs["pair_layout"] = "waveform[pair_index, member_index, sample], member 0=positive and member 1=negative"

        chunk_pairs = min(32, max(1, len(selections)))
        handle.create_dataset(
            "waveform",
            data=pair_waveforms.astype(np.float32, copy=False),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(chunk_pairs, 2, WAVEFORM_LENGTH),
        )
        handle.create_dataset("label", data=pair_labels.astype(np.int8, copy=False))
        handle.create_dataset("score", data=pair_scores.astype(np.float32, copy=False))
        handle.create_dataset(
            "pair_mean_score", data=np.mean(pair_scores, axis=1, dtype=np.float64).astype(np.float32)
        )
        handle.create_dataset(
            "pair_min_score", data=np.min(pair_scores, axis=1).astype(np.float32, copy=False)
        )
        handle.create_dataset(
            "pair_max_score", data=np.max(pair_scores, axis=1).astype(np.float32, copy=False)
        )
        handle.create_dataset(
            "score_bin_index",
            data=np.asarray([item["bin_index"] for item in selections], dtype=np.int8),
        )
        handle.create_dataset(
            "score_bin_low",
            data=np.asarray([item["bin_low"] for item in selections], dtype=np.float32),
        )
        handle.create_dataset(
            "score_bin_high",
            data=np.asarray([item["bin_high"] for item in selections], dtype=np.float32),
        )
        handle.create_dataset(
            "selection_rank",
            data=np.asarray([item["selection_rank"] for item in selections], dtype=np.int32),
        )
        handle.create_dataset(
            "anchor_member_index",
            data=np.asarray([item["anchor_side"] for item in selections], dtype=np.int8),
        )
        handle.create_dataset(
            "anchor_label",
            data=np.asarray([item["anchor_label"] for item in selections], dtype=np.int8),
        )
        handle.create_dataset(
            "anchor_score",
            data=np.asarray([item["anchor_score"] for item in selections], dtype=np.float32),
        )
        handle.create_dataset("source_store_index", data=selected_store_indices)
        handle.create_dataset(
            "source_row",
            data=np.asarray(
                [
                    [int(row["positive_row"]), int(row["negative_row"])]
                    for row in selected_rows
                ],
                dtype=np.int64,
            ),
        )
        handle.create_dataset(
            "event_id",
            data=np.asarray(
                [
                    [int(row["positive_event_id"]), int(row["negative_event_id"])]
                    for row in selected_rows
                ],
                dtype=np.int64,
            ),
        )
        handle.create_dataset(
            "energy_kev",
            data=np.asarray(
                [
                    [float(row["positive_energy_kev"]), float(row["negative_energy_kev"])]
                    for row in selected_rows
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset(
            "pair_id", data=string_values([row["pair_id"] for row in selected_rows]), dtype=STRING_DTYPE
        )
        handle.create_dataset(
            "peak_id", data=string_values([row["peak_id"] for row in selected_rows]), dtype=STRING_DTYPE
        )
        handle.create_dataset(
            "positive_hdf5",
            data=string_values([row["positive_hdf5"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "negative_hdf5",
            data=string_values([row["negative_hdf5"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "positive_source",
            data=string_values([row["positive_source"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "negative_source",
            data=string_values([row["negative_source"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "positive_qc_status",
            data=string_values([row["positive_qc_status"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )
        handle.create_dataset(
            "negative_qc_status",
            data=string_values([row["negative_qc_status"] for row in selected_rows]),
            dtype=STRING_DTYPE,
        )


def write_selection_csv(
    path: Path,
    selections: list[dict[str, Any]],
    pair_rows: list[dict[str, str]],
    pair_scores: np.ndarray,
    store_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not selections:
        raise ValueError("No selections to write")
    extra_fields = [
        "score_bin_index",
        "score_bin_low",
        "score_bin_high",
        "selection_rank",
        "pair_index",
        "anchor_label",
        "anchor_score",
        "positive_score",
        "negative_score",
        "pair_mean_score",
        "pair_min_score",
        "pair_max_score",
        "positive_store_index",
        "negative_store_index",
    ]
    fieldnames = list(pair_rows[0]) + extra_fields
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for selection in selections:
            pair_index = int(selection["pair_index"])
            scores = pair_scores[pair_index]
            row = dict(pair_rows[pair_index])
            row.update(
                {
                    "score_bin_index": selection["bin_index"],
                    "score_bin_low": selection["bin_low"],
                    "score_bin_high": selection["bin_high"],
                    "selection_rank": selection["selection_rank"],
                    "pair_index": pair_index,
                    "anchor_label": selection["anchor_label"],
                    "anchor_score": f"{selection['anchor_score']:.9g}",
                    "positive_score": f"{scores[0]:.9g}",
                    "negative_score": f"{scores[1]:.9g}",
                    "pair_mean_score": f"{np.mean(scores):.9g}",
                    "pair_min_score": f"{np.min(scores):.9g}",
                    "pair_max_score": f"{np.max(scores):.9g}",
                    "positive_store_index": int(store_indices[pair_index, 0]),
                    "negative_store_index": int(store_indices[pair_index, 1]),
                }
            )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pairs-per-bin", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pairs_per_bin < 1:
        raise ValueError("pairs-per-bin must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_hdf5 = args.output_hdf5.resolve()
    report_dir = args.report_dir.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    lookup_path = event_store_dir / "event_lookup_train.csv"
    store_path = event_store_dir / "train_events.h5"
    for path in (labels_dir, event_store_dir, checkpoint_path, train_csv, lookup_path, store_path):
        if not path.exists():
            raise FileNotFoundError(path)
    assert_no_forbidden_path(train_csv)
    assert_development_csv(train_csv)
    if (output_hdf5.exists() or report_dir.exists()) and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace the requested visual-inspection artifacts"
        )
    if output_hdf5.exists():
        output_hdf5.unlink()
    report_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = read_pair_rows(train_csv)
    store_indices = load_store_indices(lookup_path, pair_rows)
    raw = load_raw_partition(train_csv, event_store_dir)
    if raw.waveforms.shape != (2 * len(pair_rows), WAVEFORM_LENGTH):
        raise ValueError(f"Unexpected loaded waveform shape: {raw.waveforms.shape}")
    if not np.array_equal(raw.labels[0::2], np.ones(len(pair_rows), dtype=np.float32)):
        raise ValueError("Training pair positive rows are not ordered as label 1")
    if not np.array_equal(raw.labels[1::2], np.zeros(len(pair_rows), dtype=np.float32)):
        raise ValueError("Training pair negative rows are not ordered as label 0")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "ds_cnn":
        raise ValueError("The requested checkpoint is not the joint DS-CNN")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("The checkpoint is marked as test-contaminated")
    if checkpoint.get("held_out_partition_loaded") is not False:
        raise ValueError("The checkpoint was trained with held-out data")
    config = representation_config_from_checkpoint(checkpoint["representation_config"])
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    print(f"loading development training pairs={len(pair_rows)}", flush=True)
    print(f"building frozen DS-CNN representation={config.name}", flush=True)
    values, representation_qc = build_representation(raw, config)
    apply_channel_statistics(values, checkpoint["feature_statistics"])
    if not np.all(np.isfinite(values)):
        raise ValueError("Model representation contains nonfinite values")
    print("scoring frozen joint 3-peak DS-CNN", flush=True)
    scores, model_metadata = predict_checkpoint(checkpoint_path, values, args.batch_size, device)
    if not np.all(np.isfinite(scores)):
        raise ValueError("DS-CNN scores contain nonfinite values")
    pair_scores = scores.reshape(len(pair_rows), 2)
    pair_labels = raw.labels.reshape(len(pair_rows), 2)
    if not np.array_equal(pair_labels, np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(pair_rows), 1))):
        raise ValueError("Loaded labels do not preserve positive/negative pair order")

    selections, availability = select_pairs(pair_scores, args.pairs_per_bin, args.seed)
    selected_pair_indices = np.asarray([item["pair_index"] for item in selections], dtype=np.int64)
    selected_event_indices = np.column_stack(
        (2 * selected_pair_indices, 2 * selected_pair_indices + 1)
    ).reshape(-1)
    selected_waveforms = raw.waveforms[selected_event_indices].reshape(
        selected_pair_indices.size, 2, WAVEFORM_LENGTH
    )
    output_metadata = {
        "schema_version": "1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "development_training",
        "source_label_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
        "source_event_store": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
        "score_checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "score_checkpoint_sha256": sha256_file(checkpoint_path),
        "score_definition": "frozen joint 3-peak DS-CNN sigmoid output per waveform",
        "score_bin_definition": "pair is eligible when either member score is in the bin; left-inclusive, right-exclusive except final bin",
        "score_bins": SCORE_BINS,
        "pairs_per_bin": args.pairs_per_bin,
        "selection_seed": args.seed,
        "held_out_used": False,
        "test_partition_used": False,
    }
    write_hdf5(
        output_hdf5,
        raw.waveforms,
        raw.labels,
        scores,
        selections,
        pair_rows,
        store_indices,
        output_metadata,
    )
    pair_scores_for_csv = pair_scores
    selection_csv = report_dir / "selection_manifest.csv"
    write_selection_csv(selection_csv, selections, pair_rows, pair_scores_for_csv, store_indices)

    selected_pair_ids = [pair_rows[int(index)]["pair_id"] for index in selected_pair_indices]
    report = {
        **output_metadata,
        "event_count": int(selected_event_indices.size),
        "pair_count": int(selected_pair_indices.size),
        "source_pair_count": int(len(pair_rows)),
        "source_event_count": int(raw.labels.size),
        "source_label_csv_sha256": sha256_file(train_csv),
        "source_lookup_csv_sha256": sha256_file(lookup_path),
        "source_store_sha256": sha256_file(store_path),
        "checkpoint_metadata": model_metadata,
        "representation_qc": representation_qc,
        "availability": availability,
        "selected_unique_pair_count": int(np.unique(selected_pair_indices).size),
        "selected_unique_pair_id_count": int(len(set(selected_pair_ids))),
        "peak_counts": {
            peak: int(sum(pair_rows[int(index)]["peak_id"] == peak for index in selected_pair_indices))
            for peak in sorted({row["peak_id"] for row in pair_rows})
        },
        "artifacts": {
            "hdf5": {
                "path": output_hdf5.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(output_hdf5),
            },
            "selection_manifest": {
                "path": selection_csv.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(selection_csv),
            },
        },
        "scientific_boundary": (
            "This is a development-training visual-inspection sample. It is not a model- or threshold-selection artifact, "
            "and the held-out, locked-test, Th-232, and Eu-152 data were not used."
        ),
    }
    report_path = report_dir / "selection_report.json"
    save_json(report_path, report)
    print(json.dumps(availability, indent=2, sort_keys=True), flush=True)
    print(f"wrote_hdf5={output_hdf5}", flush=True)
    print(f"wrote_report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
