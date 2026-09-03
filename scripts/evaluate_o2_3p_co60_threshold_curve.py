#!/usr/bin/env python3
"""Infer O2-3P scores and calibrate a constant-pass Co-60 threshold curve.

Thresholds are derived independently in fixed corrected-energy bins from the
development validation continuum. The train continuum is scored only to report
its passing fraction at those frozen validation-derived thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_o2_late_fusion import (  # noqa: E402
    O2LateFusion,
    extract_o2_features,
)


MODEL_NAME = "O2-3P Late Fusion"
DEFAULT_TARGET_PASSING_FRACTION = 16218.0 / 17298.0
ALLOWED_PARTITIONS = ("validation", "train")
CSV_FIELDS = (
    "energy_low_kev",
    "energy_high_kev",
    "energy_center_kev",
    "upper_edge_inclusive",
    "threshold",
    "target_passing_fraction",
    "validation_event_count",
    "validation_score_min",
    "validation_score_median",
    "validation_score_max",
    "validation_threshold_tie_count",
    "validation_passed_count",
    "validation_passing_fraction",
    "validation_rejection_fraction",
    "train_event_count",
    "train_passed_count_at_validation_threshold",
    "train_passing_fraction_at_validation_threshold",
    "train_rejection_fraction_at_validation_threshold",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[O2LateFusion, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "late_fusion":
        raise ValueError("Checkpoint is not a late-fusion model")
    if checkpoint.get("architecture") != "O2_style_charge_current_late_fusion":
        raise ValueError("Unexpected checkpoint architecture")
    if checkpoint.get("test_partition_used") is not False:
        raise ValueError("Checkpoint does not preserve the locked-test boundary")
    statistics = checkpoint.get("feature_statistics", {})
    required = {"charge_mean", "charge_std", "current_mean", "current_std"}
    if set(statistics) != required:
        raise ValueError(f"Unexpected feature statistics: {sorted(statistics)}")
    if statistics["charge_std"] <= 0.0 or statistics["current_std"] <= 0.0:
        raise ValueError("Checkpoint feature standard deviations must be positive")
    model = O2LateFusion().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def validate_continuum_store(handle: h5py.File, partition: str) -> int:
    if str(handle.attrs.get("partition", "")) != partition:
        raise ValueError(f"Store partition mismatch: expected {partition}")
    if str(handle.attrs.get("source", "")).lower() != "co60":
        raise ValueError("Continuum store is not Co-60")
    if bool(handle.attrs.get("test_partition_used", True)):
        raise ValueError("Continuum store indicates test access")
    if bool(handle.attrs.get("external_data_used", True)):
        raise ValueError("Continuum store indicates external-data access")
    required = {"waveform", "shaped_energy_unit", "corrected_energy_kev"}
    missing = sorted(required - set(handle.keys()))
    if missing:
        raise ValueError(f"Missing continuum datasets: {missing}")
    event_count = int(handle["waveform"].shape[0])
    if handle["waveform"].shape != (event_count, 4500):
        raise ValueError(f"Unexpected waveform shape: {handle['waveform'].shape}")
    for name in ("shaped_energy_unit", "corrected_energy_kev"):
        if handle[name].shape != (event_count,):
            raise ValueError(f"Unexpected {name} shape: {handle[name].shape}")
    return event_count


def infer_partition(
    model: O2LateFusion,
    checkpoint: dict[str, Any],
    store_path: Path,
    expected_store_sha256: str,
    partition: str,
    output_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    current_store_hash = sha256_file(store_path)
    if current_store_hash != expected_store_sha256:
        raise ValueError(f"Continuum store hash mismatch: {store_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    statistics = checkpoint["feature_statistics"]
    fallback_count = 0
    try:
        with h5py.File(store_path, "r") as source, h5py.File(partial, "w") as output:
            event_count = validate_continuum_store(source, partition)
            output.attrs.update(
                {
                    "schema_version": "1",
                    "model_name": MODEL_NAME,
                    "partition": partition,
                    "source_store": relative(store_path),
                    "source_store_sha256": current_store_hash,
                    "event_count": event_count,
                    "score_definition": "sigmoid of O2 late-fusion logit",
                    "join_rule": "score row index equals source-store row index",
                    "created_utc": utc_now(),
                    "test_partition_used": False,
                    "external_data_used": False,
                }
            )
            scores = output.create_dataset(
                "score",
                shape=(event_count,),
                dtype=np.float32,
                chunks=(min(batch_size, event_count),),
            )
            output.create_dataset(
                "corrected_energy_kev",
                shape=(event_count,),
                dtype=np.float32,
                chunks=(min(batch_size, event_count),),
            )
            with torch.inference_mode():
                for start in range(0, event_count, batch_size):
                    stop = min(start + batch_size, event_count)
                    waveforms = np.asarray(source["waveform"][start:stop], dtype=np.float32)
                    shaped_energy = np.asarray(
                        source["shaped_energy_unit"][start:stop], dtype=np.float32
                    )
                    charge, current, batch_fallback = extract_o2_features(
                        waveforms, shaped_energy
                    )
                    fallback_count += batch_fallback
                    charge -= np.float32(statistics["charge_mean"])
                    charge /= np.float32(statistics["charge_std"])
                    current -= np.float32(statistics["current_mean"])
                    current /= np.float32(statistics["current_std"])
                    charge_tensor = torch.from_numpy(charge).to(device, non_blocking=True)
                    current_tensor = torch.from_numpy(current).to(device, non_blocking=True)
                    batch_scores = torch.sigmoid(
                        model(charge_tensor, current_tensor)
                    ).cpu().numpy()
                    scores[start:stop] = batch_scores.astype(np.float32, copy=False)
                    output["corrected_energy_kev"][start:stop] = source[
                        "corrected_energy_kev"
                    ][start:stop]
                    if stop == event_count or stop % (100 * batch_size) == 0:
                        print(
                            f"{partition}: inferred={stop}/{event_count} "
                            f"t10_fallbacks={fallback_count}",
                            flush=True,
                        )
            output.attrs["t10_fallback_count"] = fallback_count
            output.flush()
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {
        "partition": partition,
        "source_store": relative(store_path),
        "source_store_sha256": current_store_hash,
        "score_file": relative(output_path),
        "score_file_sha256": sha256_file(output_path),
        "event_count": event_count,
        "t10_fallback_count": fallback_count,
    }


def closest_constant_pass_threshold(
    scores: np.ndarray, target_passing_fraction: float
) -> tuple[float, int, float]:
    values = np.sort(np.asarray(scores, dtype=np.float32))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be nonempty and finite")
    desired_passed = int(round(target_passing_fraction * values.size))
    desired_rejected = values.size - desired_passed
    candidates = {
        float(np.nextafter(values[0], -np.inf, dtype=np.float32)),
        float(np.nextafter(values[-1], np.inf, dtype=np.float32)),
    }
    for index in range(max(0, desired_rejected - 2), min(values.size, desired_rejected + 3)):
        value = values[index]
        candidates.add(float(value))
        candidates.add(float(np.nextafter(value, np.inf, dtype=np.float32)))
        if index > 0 and values[index - 1] < value:
            candidates.add(float((np.float64(values[index - 1]) + np.float64(value)) / 2.0))
    ranked = []
    for threshold in candidates:
        rejected = int(np.searchsorted(values, threshold, side="left"))
        passed = int(values.size - rejected)
        ranked.append((abs(passed - desired_passed), -passed, threshold, passed))
    _, _, threshold, passed = min(ranked)
    return threshold, passed, passed / values.size


def bin_mask(
    energies: np.ndarray, low: float, high: float, final_bin: bool
) -> np.ndarray:
    if final_bin:
        return (energies >= low) & (energies <= high)
    return (energies >= low) & (energies < high)


def build_threshold_rows(
    validation_energy: np.ndarray,
    validation_scores: np.ndarray,
    train_energy: np.ndarray,
    train_scores: np.ndarray,
    minimum_energy_kev: float,
    maximum_energy_kev: float,
    bin_width_kev: float,
    target_passing_fraction: float,
) -> list[dict[str, Any]]:
    span = maximum_energy_kev - minimum_energy_kev
    bin_count = int(round(span / bin_width_kev))
    if bin_count <= 0 or not np.isclose(bin_count * bin_width_kev, span):
        raise ValueError("Energy span must be an integer multiple of bin width")
    rows = []
    for index in range(bin_count):
        low = minimum_energy_kev + index * bin_width_kev
        high = low + bin_width_kev
        final_bin = index == bin_count - 1
        validation_mask = bin_mask(validation_energy, low, high, final_bin)
        train_mask = bin_mask(train_energy, low, high, final_bin)
        threshold, validation_passed, validation_fraction = (
            closest_constant_pass_threshold(
                validation_scores[validation_mask], target_passing_fraction
            )
        )
        selected_validation_scores = validation_scores[validation_mask]
        selected_train_scores = train_scores[train_mask]
        train_passed = int(np.count_nonzero(selected_train_scores >= threshold))
        rows.append(
            {
                "energy_low_kev": low,
                "energy_high_kev": high,
                "energy_center_kev": (low + high) / 2.0,
                "upper_edge_inclusive": final_bin,
                "threshold": threshold,
                "target_passing_fraction": target_passing_fraction,
                "validation_event_count": int(selected_validation_scores.size),
                "validation_score_min": float(np.min(selected_validation_scores)),
                "validation_score_median": float(np.median(selected_validation_scores)),
                "validation_score_max": float(np.max(selected_validation_scores)),
                "validation_threshold_tie_count": int(
                    np.count_nonzero(selected_validation_scores == threshold)
                ),
                "validation_passed_count": validation_passed,
                "validation_passing_fraction": validation_fraction,
                "validation_rejection_fraction": 1.0 - validation_fraction,
                "train_event_count": int(selected_train_scores.size),
                "train_passed_count_at_validation_threshold": train_passed,
                "train_passing_fraction_at_validation_threshold": train_passed
                / selected_train_scores.size,
                "train_rejection_fraction_at_validation_threshold": 1.0
                - train_passed / selected_train_scores.size,
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    centers = np.asarray([row["energy_center_kev"] for row in rows])
    thresholds = np.asarray([row["threshold"] for row in rows])
    validation_pass = 100.0 * np.asarray(
        [row["validation_passing_fraction"] for row in rows]
    )
    train_pass = 100.0 * np.asarray(
        [row["train_passing_fraction_at_validation_threshold"] for row in rows]
    )
    target = 100.0 * float(rows[0]["target_passing_fraction"])
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
    axes[0].plot(centers, thresholds, marker="o", linewidth=1.8)
    axes[0].set_ylabel("O2-3P score threshold")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Co-60 constant-pass threshold versus corrected energy")
    axes[1].axhline(target, color="black", linestyle="--", label=f"target {target:.4f}%")
    axes[1].plot(centers, validation_pass, marker="o", label="validation calibration")
    axes[1].plot(centers, train_pass, marker="s", label="train diagnostic")
    axes[1].set_xlabel("Corrected energy bin center (keV)")
    axes[1].set_ylabel("Passing fraction (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuum-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/event_store/co60_continuum_100_1000kev_20260819/continuum_store_manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/three_peak_weight_scan_20260819/late_fusion_best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-energy-kev", type=float, default=100.0)
    parser.add_argument("--maximum-energy-kev", type=float, default=1000.0)
    parser.add_argument("--bin-width-kev", type=float, default=50.0)
    parser.add_argument(
        "--target-passing-fraction",
        type=float,
        default=DEFAULT_TARGET_PASSING_FRACTION,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.target_passing_fraction < 1.0:
        raise ValueError("Target passing fraction must be between zero and one")
    if args.bin_width_kev <= 0.0 or args.batch_size <= 0:
        raise ValueError("Bin width and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    continuum_manifest_path = args.continuum_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    continuum_manifest = json.loads(continuum_manifest_path.read_text(encoding="utf-8"))
    if continuum_manifest["selection"].get("test_partition_used") is not False:
        raise ValueError("Continuum manifest does not preserve locked-test isolation")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_hash = sha256_file(checkpoint_path)
    print(f"device={device} checkpoint_sha256={checkpoint_hash}", flush=True)
    inference = {}
    for partition in ALLOWED_PARTITIONS:
        store_record = continuum_manifest["stores"][partition]
        store_path = PROJECT_ROOT / store_record["store_file"]
        inference[partition] = infer_partition(
            model,
            checkpoint,
            store_path,
            store_record["store_sha256"],
            partition,
            output_dir / f"{partition}_scores.h5",
            args.batch_size,
            device,
        )
    arrays = {}
    for partition in ALLOWED_PARTITIONS:
        with h5py.File(output_dir / f"{partition}_scores.h5", "r") as handle:
            arrays[partition] = {
                "scores": np.asarray(handle["score"], dtype=np.float32),
                "energy": np.asarray(handle["corrected_energy_kev"], dtype=np.float32),
            }
    rows = build_threshold_rows(
        arrays["validation"]["energy"],
        arrays["validation"]["scores"],
        arrays["train"]["energy"],
        arrays["train"]["scores"],
        args.minimum_energy_kev,
        args.maximum_energy_kev,
        args.bin_width_kev,
        args.target_passing_fraction,
    )
    csv_path = output_dir / "threshold_curve_50kev.csv"
    plot_path = output_dir / "threshold_curve_50kev.png"
    save_csv(csv_path, rows)
    save_plot(plot_path, rows)
    report = {
        "schema_version": "1",
        "created_utc": utc_now(),
        "model_name": MODEL_NAME,
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selected_peak_weights": checkpoint["selected_peak_weights"],
        "feature_statistics": checkpoint["feature_statistics"],
        "continuum_manifest": relative(continuum_manifest_path),
        "continuum_manifest_sha256": sha256_file(continuum_manifest_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "calibration": {
            "threshold_partition": "validation",
            "pass_rule": "score >= threshold",
            "energy_dataset": "corrected_energy_kev",
            "minimum_energy_kev": args.minimum_energy_kev,
            "maximum_energy_kev": args.maximum_energy_kev,
            "bin_width_kev": args.bin_width_kev,
            "target_passing_fraction": args.target_passing_fraction,
            "baseline_fraction_definition": "16218 passing label-0 events / 17298 held-out label-0 events",
            "train_role": "diagnostic only; does not set thresholds",
        },
        "inference": inference,
        "threshold_bins": rows,
        "artifacts": {
            "csv": relative(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "plot": relative(plot_path),
            "plot_sha256": sha256_file(plot_path),
        },
        "test_partition_used": False,
        "external_data_used": False,
        "scientific_boundary": (
            "Thresholds are development-validation calibrations on existing Co-60 "
            "sessions, not independent external-validation or claim-grade cuts."
        ),
    }
    report_path = output_dir / "threshold_curve_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
