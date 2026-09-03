#!/usr/bin/env python3
"""Plot raw and MA20 charge waveforms for manual review of one internal group."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ba133_cnn import BASELINE_STOP, load_raw_partition, moving_average
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, default=6)
    parser.add_argument("--minimum-index", type=int, default=650)
    parser.add_argument("--maximum-index", type=int, default=1800)
    parser.add_argument("--waveforms-per-figure", type=int, default=10)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT
        / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/morphology_catalogue_20260821",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/morphology_catalogue_20260821",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.group < 1:
        raise ValueError("group must be at least 1")
    if not 0 <= args.minimum_index <= args.maximum_index < 4500:
        raise ValueError("invalid accepted extremum-index range")
    if args.waveforms_per_figure < 1:
        raise ValueError("waveforms-per-figure must be positive")

    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    feature_dir = args.feature_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = (
        experiment_dir
        / f"internal_group_{args.group}_remaining_review"
    )
    train_csv = labels_dir / "label_pairs_train.csv"
    for path in (
        labels_dir,
        event_store_dir,
        feature_dir,
        experiment_dir,
        train_csv,
        output_dir,
    ):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(feature_dir / "internal_features.npz")
    assignments = np.load(
        experiment_dir / "catalogue/internal_assignments.npz"
    )
    component = args.group - 1
    if component >= assignments["probability"].shape[1]:
        raise ValueError(f"group {args.group} does not exist")

    print("Loading development train waveforms ...", flush=True)
    raw = load_raw_partition(train_csv, event_store_dir)
    original_indices = features["original_event_index"].astype(np.int64)
    internal_waveforms = raw.waveforms[original_indices]
    extremum_indices = np.argmin(internal_waveforms, axis=1).astype(np.int32)
    selected_mask = (
        assignments["valid"].astype(bool)
        & (assignments["assignment"] == component)
        & (extremum_indices >= args.minimum_index)
        & (extremum_indices <= args.maximum_index)
    )
    selected = np.flatnonzero(selected_mask)
    if selected.size == 0:
        raise ValueError("No waveforms satisfy the review selection")

    records: list[dict[str, object]] = []
    page_count = math.ceil(selected.size / args.waveforms_per_figure)
    sample_index = np.arange(4500)
    for page, start in enumerate(
        range(0, selected.size, args.waveforms_per_figure), start=1
    ):
        page_rows = selected[start : start + args.waveforms_per_figure]
        figure, axes = plt.subplots(
            len(page_rows),
            2,
            figsize=(16, 2.35 * len(page_rows)),
            squeeze=False,
            constrained_layout=True,
        )
        for row_number, cache_index in enumerate(page_rows):
            waveform = internal_waveforms[cache_index]
            positive = -waveform[None, :]
            baseline = np.median(
                positive[:, :BASELINE_STOP], axis=1
            ).astype(np.float32)
            charge = moving_average(positive - baseline[:, None], 20)[0]
            charge_scale = float(np.max(charge))
            normalized_charge = (
                charge / charge_scale if charge_scale > 0 else charge
            )
            extremum = int(extremum_indices[cache_index])
            label = int(features["label"][cache_index])
            class_name = "positive" if label == 1 else "negative"
            source = str(features["source"][cache_index])
            energy = float(features["energy_kev"][cache_index])
            event_index = int(original_indices[cache_index])

            left, right = axes[row_number]
            left.plot(sample_index, waveform, linewidth=0.7)
            right.plot(sample_index, normalized_charge, linewidth=0.8)
            for axis in (left, right):
                axis.axvspan(
                    args.minimum_index,
                    args.maximum_index,
                    color="green",
                    alpha=0.08,
                )
                axis.axvline(args.minimum_index, color="green", linewidth=0.6)
                axis.axvline(args.maximum_index, color="green", linewidth=0.6)
                axis.axvline(extremum, color="red", linestyle="--", linewidth=0.8)
                axis.grid(alpha=0.18)
            left.set_ylabel(f"Event {start + row_number + 1}")
            left.set_title(
                f"Raw negative waveform | min={extremum} | "
                f"{class_name}, {source}, {energy:.2f} keV",
                fontsize=9,
            )
            right.set_title("Positive-polarity baseline-subtracted MA20 charge", fontsize=9)
            if row_number == len(page_rows) - 1:
                left.set_xlabel("Sample index")
                right.set_xlabel("Sample index")
            records.append(
                {
                    "review_order": start + row_number + 1,
                    "figure": f"group_{args.group}_remaining_{page:03d}.png",
                    "panel_row": row_number + 1,
                    "internal_cache_index": int(cache_index),
                    "original_train_event_index": event_index,
                    "pair_index": int(features["pair_index"][cache_index]),
                    "pair_member": int(features["pair_member"][cache_index]),
                    "label": label,
                    "class_name": class_name,
                    "source": source,
                    "energy_kev": energy,
                    "raw_minimum_index": extremum,
                    "hdf5": str(features["hdf5"][cache_index]),
                    "source_row": int(features["source_row"][cache_index]),
                }
            )
        figure.suptitle(
            f"Internal Group {args.group}: remaining waveforms "
            f"({start + 1}-{start + len(page_rows)} of {selected.size})",
            fontsize=14,
        )
        figure.savefig(
            output_dir / f"group_{args.group}_remaining_{page:03d}.png",
            dpi=150,
        )
        plt.close(figure)
        print(f"wrote page {page}/{page_count}", flush=True)

    table = pd.DataFrame(records)
    table.to_csv(output_dir / "selection_manifest.csv", index=False)
    summary = {
        "group": args.group,
        "population": "internal catalogue assignment",
        "selection": (
            f"valid group assignment and {args.minimum_index} <= "
            f"argmin(raw negative waveform) <= {args.maximum_index}"
        ),
        "selected_waveform_count": int(selected.size),
        "waveforms_per_figure": args.waveforms_per_figure,
        "figure_count": page_count,
        "positive_count": int(np.count_nonzero(features["label"][selected] == 1)),
        "negative_count": int(np.count_nonzero(features["label"][selected] == 0)),
        "excluded_outside_index_count": int(
            np.count_nonzero(
                assignments["valid"].astype(bool)
                & (assignments["assignment"] == component)
                & ~(
                    (extremum_indices >= args.minimum_index)
                    & (extremum_indices <= args.maximum_index)
                )
            )
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
