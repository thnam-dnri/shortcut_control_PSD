#!/usr/bin/env python3
"""Evaluate frozen Peak-Specialist DS-CNN models and fusion rules on Th-232.

Applies frozen 356A, 511A, 661A specialists, joint DS-CNN, and frozen fusion rules
to all 30 files in the external Th-232 evaluation cache (2.88M admitted events).
Calculates Peak-to-Background (P/B) ratio and net peak retention across 7 reference lines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
    ENERGY_MAX_KEV,
    ENERGY_MIN_KEV,
    REFERENCE_PEAKS_KEV,
    PeakWindow,
    fit_peak_windows,
    interval_counts,
    peak_background_metrics,
    th232_admission_mask,
)
from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RawPartition,
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    representation_config_from_checkpoint,
)

DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/peak_specialist_ds_cnn_20260820"
DEFAULT_TH232_DIR = (
    PROJECT_ROOT / "processed_data/waveform_hdf5_corrected/th232_evaluation_20260813"
)
DEFAULT_JOINT_BASELINE = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)

SPECIALIST_NAMES = ("356A", "511A", "661A")
ACCEPTANCES = (0.99, 0.95, 0.90, 0.80, 0.50, 0.30, 0.10)
EXPECTED_TH232_FILE_COUNT = 30
BATCH_SIZE = 2048

SWITCH_ENERGY_1 = 433.505925
SWITCH_ENERGY_2 = 586.327975


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def threshold_name(acceptance: float) -> str:
    return f"{int(round(100.0 * acceptance))}pct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--th232-dir", type=Path, default=DEFAULT_TH232_DIR)
    parser.add_argument("--joint-baseline-checkpoint", type=Path, default=DEFAULT_JOINT_BASELINE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    print(f"Using compute device: {device}")

    # Load specialist checkpoints
    specialist_models: dict[str, nn.Module] = {}
    specialist_ckpts: dict[str, Any] = {}
    for spec_name in SPECIALIST_NAMES:
        ckpt_path = args.experiment_dir / f"model_{spec_name}_checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = DSCNN(input_channels=2, width=24).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        specialist_models[spec_name] = m
        specialist_ckpts[spec_name] = ckpt

    # Load joint baseline
    joint_ckpt = torch.load(args.joint_baseline_checkpoint, map_location="cpu", weights_only=False)
    joint_model = DSCNN(input_channels=2, width=24).to(device)
    joint_model.load_state_dict(joint_ckpt["model_state_dict"])
    joint_model.eval()

    rep_config = representation_config_from_checkpoint(specialist_ckpts["356A"]["representation_config"])
    feature_stats = specialist_ckpts["356A"]["feature_statistics"]

    # Load internal calibration isotonic regressors from internal scores
    internal_npz_path = args.experiment_dir / "internal_specialist_scores.npz"
    if not internal_npz_path.is_file():
        raise FileNotFoundError(f"Missing internal scores NPZ: {internal_npz_path}")
    internal_data = np.load(internal_npz_path)
    calibrators: dict[str, IsotonicRegression] = {}
    for spec_name in SPECIALIST_NAMES:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(internal_data[f"score_{spec_name}"], internal_data["true_label"])
        calibrators[spec_name] = iso

    # Load held-out scores to calibrate acceptance thresholds on development positive events
    held_out_npz_path = args.experiment_dir / "held_out_specialist_scores.npz"
    if not held_out_npz_path.is_file():
        raise FileNotFoundError(f"Missing held-out scores NPZ: {held_out_npz_path}")
    held_out_data = np.load(held_out_npz_path)
    pos_mask = held_out_data["true_label"] == 1

    # Load evaluation summary to get selected fusion rule and validation best map
    eval_summary_path = args.experiment_dir / "evaluation_summary.json"
    if not eval_summary_path.is_file():
        raise FileNotFoundError(f"Missing evaluation summary: {eval_summary_path}")
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    selected_fusion_rule = eval_summary["selected_fusion_rule"]

    # Determine thresholds for models/rules:
    # 1. joint_ds_cnn
    # 2. 356A, 511A, 661A
    # 3. nearest_energy_expert
    # 4. calibrated_mean
    # 5. calibrated_max
    # 6. selected_fusion_rule
    threshold_catalog: dict[str, dict[str, float]] = {}

    def get_held_out_pos_scores(model_name: str) -> np.ndarray:
        if model_name in SPECIALIST_NAMES:
            return held_out_data[f"score_{model_name}"][pos_mask]
        elif model_name == "joint_ds_cnn":
            return held_out_data["score_joint_ds_cnn"][pos_mask]
        elif model_name == "nearest_energy_expert":
            energies = held_out_data["energy_kev"][pos_mask]
            s356 = held_out_data["score_356A"][pos_mask]
            s511 = held_out_data["score_511A"][pos_mask]
            s661 = held_out_data["score_661A"][pos_mask]
            res = np.empty(len(energies), dtype=np.float32)
            res[energies < SWITCH_ENERGY_1] = s356[energies < SWITCH_ENERGY_1]
            res[(energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)] = s511[
                (energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)
            ]
            res[energies >= SWITCH_ENERGY_2] = s661[energies >= SWITCH_ENERGY_2]
            return res
        elif model_name == "nearest_energy_expert_calibrated":
            energies = held_out_data["energy_kev"][pos_mask]
            c356 = held_out_data["score_calibrated_356A"][pos_mask]
            c511 = held_out_data["score_calibrated_511A"][pos_mask]
            c661 = held_out_data["score_calibrated_661A"][pos_mask]
            res = np.empty(len(energies), dtype=np.float32)
            res[energies < SWITCH_ENERGY_1] = c356[energies < SWITCH_ENERGY_1]
            res[(energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)] = c511[
                (energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)
            ]
            res[energies >= SWITCH_ENERGY_2] = c661[energies >= SWITCH_ENERGY_2]
            return res
        elif model_name == "calibrated_mean":
            c356 = held_out_data["score_calibrated_356A"][pos_mask]
            c511 = held_out_data["score_calibrated_511A"][pos_mask]
            c661 = held_out_data["score_calibrated_661A"][pos_mask]
            return (c356 + c511 + c661) / 3.0
        elif model_name == "calibrated_max":
            c356 = held_out_data["score_calibrated_356A"][pos_mask]
            c511 = held_out_data["score_calibrated_511A"][pos_mask]
            c661 = held_out_data["score_calibrated_661A"][pos_mask]
            return np.maximum(c356, np.maximum(c511, c661))
        elif model_name == "selected_fusion_rule":
            return held_out_data["selected_fusion_score"][pos_mask]
        raise ValueError(f"Unknown model: {model_name}")

    evaluated_rules_on_th232 = [
        "joint_ds_cnn",
        "356A",
        "511A",
        "661A",
        "nearest_energy_expert",
        "calibrated_mean",
        "calibrated_max",
    ]
    if selected_fusion_rule not in evaluated_rules_on_th232:
        evaluated_rules_on_th232.append(selected_fusion_rule)

    for rule_name in evaluated_rules_on_th232:
        pos_scores = get_held_out_pos_scores(rule_name)
        threshold_catalog[rule_name] = {}
        for acc in ACCEPTANCES:
            thr = float(np.quantile(pos_scores, 1.0 - acc))
            threshold_catalog[rule_name][threshold_name(acc)] = thr

    # Process Th-232 files or load existing H5 cache
    scores_h5_path = args.experiment_dir / "th232_specialist_scores.h5"

    if scores_h5_path.exists():
        print(f"Loading existing Th-232 scores from {scores_h5_path}...")
        with h5py.File(scores_h5_path, "r") as h5in:
            energies_arr = h5in["energy_kev"][:]
            s356_arr = h5in["score_356A"][:]
            s511_arr = h5in["score_511A"][:]
            s661_arr = h5in["score_661A"][:]
            s_joint_arr = h5in["score_joint_ds_cnn"][:]
            c356_arr = h5in["score_calibrated_356A"][:]
            c511_arr = h5in["score_calibrated_511A"][:]
            c661_arr = h5in["score_calibrated_661A"][:]
            nearest_arr = h5in["score_nearest_expert"][:]
        total_admitted = len(energies_arr)
        total_scanned = 3_000_000
    else:
        th232_files = sorted(args.th232_dir.glob("th232_preamp_250msps_*.h5"))
        if len(th232_files) != EXPECTED_TH232_FILE_COUNT:
            raise ValueError(f"Expected {EXPECTED_TH232_FILE_COUNT} Th-232 files, found {len(th232_files)}")

        print(f"\nProcessing {len(th232_files)} Th-232 evaluation files...")

        all_energies: list[np.ndarray] = []
        all_raw_356A: list[np.ndarray] = []
        all_raw_511A: list[np.ndarray] = []
        all_raw_661A: list[np.ndarray] = []
        all_raw_joint: list[np.ndarray] = []

        total_admitted = 0
        total_scanned = 0

        for file_idx, fpath in enumerate(th232_files, 1):
            with h5py.File(fpath, "r") as handle:
                wf = handle["waveform"][:]
                energy = handle["corrected_energy_kev"][:]
                shaped = handle["shaped_energy_unit"][:]
                qc = handle["qc_rejection_bits"][:]

            total_scanned += len(energy)
            mask, _, _ = th232_admission_mask(energy, shaped, qc)
            admitted_count = int(np.count_nonzero(mask))
            total_admitted += admitted_count

            if admitted_count == 0:
                continue

            admitted_wf = wf[mask]
            admitted_energy = energy[mask]
            admitted_shaped = shaped[mask]

            all_energies.append(admitted_energy)

            # Batch scoring
            file_scores_356A: list[np.ndarray] = []
            file_scores_511A: list[np.ndarray] = []
            file_scores_661A: list[np.ndarray] = []
            file_scores_joint: list[np.ndarray] = []

            for start in range(0, admitted_count, args.batch_size):
                stop = min(start + args.batch_size, admitted_count)
                count = stop - start

                raw_chunk = RawPartition(
                    waveforms=admitted_wf[start:stop],
                    shaped_energy=admitted_shaped[start:stop],
                    labels=np.zeros(count, dtype=np.float32),
                    weights=np.ones(count, dtype=np.float32),
                    peak_ids=np.full(count, "th232", dtype="U16"),
                )
                v_chunk, _ = build_representation(raw_chunk, rep_config)
                apply_channel_statistics(v_chunk, feature_stats)

                t_chunk = torch.from_numpy(v_chunk).to(device, non_blocking=True)
                with torch.inference_mode():
                    file_scores_356A.append(
                        torch.sigmoid(specialist_models["356A"](t_chunk)).cpu().numpy()
                    )
                    file_scores_511A.append(
                        torch.sigmoid(specialist_models["511A"](t_chunk)).cpu().numpy()
                    )
                    file_scores_661A.append(
                        torch.sigmoid(specialist_models["661A"](t_chunk)).cpu().numpy()
                    )
                    file_scores_joint.append(
                        torch.sigmoid(joint_model(t_chunk)).cpu().numpy()
                    )

            all_raw_356A.append(np.concatenate(file_scores_356A))
            all_raw_511A.append(np.concatenate(file_scores_511A))
            all_raw_661A.append(np.concatenate(file_scores_661A))
            all_raw_joint.append(np.concatenate(file_scores_joint))

            if file_idx % 5 == 0 or file_idx == len(th232_files):
                print(f"  Processed {file_idx}/{len(th232_files)} files ({total_admitted:,} admitted events)")

        # Concatenate all admitted event data
        energies_arr = np.concatenate(all_energies)
        s356_arr = np.concatenate(all_raw_356A)
        s511_arr = np.concatenate(all_raw_511A)
        s661_arr = np.concatenate(all_raw_661A)
        s_joint_arr = np.concatenate(all_raw_joint)

        print(f"Total scanned: {total_scanned:,}, Total admitted: {total_admitted:,}")

        # Compute calibrated scores
        c356_arr = calibrators["356A"].transform(s356_arr)
        c511_arr = calibrators["511A"].transform(s511_arr)
        c661_arr = calibrators["661A"].transform(s661_arr)

        nearest_arr = np.empty(len(energies_arr), dtype=np.float32)
        nearest_arr[energies_arr < SWITCH_ENERGY_1] = s356_arr[energies_arr < SWITCH_ENERGY_1]
        nearest_arr[(energies_arr >= SWITCH_ENERGY_1) & (energies_arr < SWITCH_ENERGY_2)] = s511_arr[
            (energies_arr >= SWITCH_ENERGY_1) & (energies_arr < SWITCH_ENERGY_2)
        ]
        nearest_arr[energies_arr >= SWITCH_ENERGY_2] = s661_arr[energies_arr >= SWITCH_ENERGY_2]

        with h5py.File(scores_h5_path, "w") as h5out:
            h5out.create_dataset("energy_kev", data=energies_arr, compression="gzip")
            h5out.create_dataset("score_356A", data=s356_arr, compression="gzip")
            h5out.create_dataset("score_511A", data=s511_arr, compression="gzip")
            h5out.create_dataset("score_661A", data=s661_arr, compression="gzip")
            h5out.create_dataset("score_joint_ds_cnn", data=s_joint_arr, compression="gzip")
            h5out.create_dataset("score_calibrated_356A", data=c356_arr, compression="gzip")
            h5out.create_dataset("score_calibrated_511A", data=c511_arr, compression="gzip")
            h5out.create_dataset("score_calibrated_661A", data=c661_arr, compression="gzip")
            h5out.create_dataset("score_nearest_expert", data=nearest_arr, compression="gzip")
        print(f"Saved {scores_h5_path}")

    # Compute score arrays for all rules
    score_dict: dict[str, np.ndarray] = {
        "joint_ds_cnn": s_joint_arr,
        "356A": s356_arr,
        "511A": s511_arr,
        "661A": s661_arr,
        "calibrated_mean": (c356_arr + c511_arr + c661_arr) / 3.0,
        "calibrated_max": np.maximum(c356_arr, np.maximum(c511_arr, c661_arr)),
        "nearest_energy_expert": nearest_arr,
    }

    nearest_cal_arr = np.empty(len(energies_arr), dtype=np.float32)
    nearest_cal_arr[energies_arr < SWITCH_ENERGY_1] = c356_arr[energies_arr < SWITCH_ENERGY_1]
    nearest_cal_arr[(energies_arr >= SWITCH_ENERGY_1) & (energies_arr < SWITCH_ENERGY_2)] = c511_arr[
        (energies_arr >= SWITCH_ENERGY_1) & (energies_arr < SWITCH_ENERGY_2)
    ]
    nearest_cal_arr[energies_arr >= SWITCH_ENERGY_2] = c661_arr[energies_arr >= SWITCH_ENERGY_2]
    score_dict["nearest_energy_expert_calibrated"] = nearest_cal_arr

    # ==========================================
    # Peak-to-Background & Spectrum Evaluation
    # ==========================================
    print("\n--- Fitting Peak Windows and Calculating P/B on Th-232 ---")
    no_cut_hist, _ = np.histogram(energies_arr, bins=ENERGY_EDGES)
    peak_windows = fit_peak_windows(no_cut_hist)

    def safe_peak_background_metrics(histogram: np.ndarray, window: PeakWindow) -> dict[str, float]:
        roi_counts, _ = interval_counts(histogram, window.roi_low_kev, window.roi_high_kev)
        left_counts, left_energy = interval_counts(
            histogram, window.left_low_kev, window.left_high_kev
        )
        right_counts, right_energy = interval_counts(
            histogram, window.right_low_kev, window.right_high_kev
        )
        left_center = (
            left_energy / left_counts
            if left_counts > 0.0
            else 0.5 * (window.left_low_kev + window.left_high_kev)
        )
        right_center = (
            right_energy / right_counts
            if right_counts > 0.0
            else 0.5 * (window.right_low_kev + window.right_high_kev)
        )
        left_density = left_counts / (window.left_high_kev - window.left_low_kev)
        right_density = right_counts / (window.right_high_kev - window.right_low_kev)
        fraction = (
            (window.centroid_kev - left_center) / (right_center - left_center)
            if (right_center != left_center)
            else 0.5
        )
        background_density = left_density + fraction * (right_density - left_density)
        background_counts = max(float(background_density * (window.roi_high_kev - window.roi_low_kev)), 0.0)
        net_peak_counts = roi_counts - background_counts
        pb = net_peak_counts / background_counts if background_counts > 0.0 else 0.0
        return {
            "roi_counts": roi_counts,
            "estimated_background_counts": background_counts,
            "net_peak_counts": net_peak_counts,
            "peak_to_background": pb,
        }

    def compute_all_peak_metrics(e_arr: np.ndarray) -> dict[float, dict[str, float]]:
        hist, _ = np.histogram(e_arr, bins=ENERGY_EDGES)
        return {p.reference_kev: safe_peak_background_metrics(hist, p) for p in peak_windows}

    # Calculate baseline (no cut) metrics for all reference peaks
    baseline_metrics = compute_all_peak_metrics(energies_arr)

    specialist_results_rows: list[dict[str, Any]] = []
    fusion_results_rows: list[dict[str, Any]] = []

    # Evaluate each rule across cuts
    for rule_name in evaluated_rules_on_th232:
        rule_scores = score_dict[rule_name]

        # No cut row
        for p_idx, p_win in enumerate(peak_windows):
            base_p = baseline_metrics[p_win.reference_kev]
            row_dict = {
                "model_or_rule": rule_name,
                "cut_name": "no_cut",
                "target_acceptance": 1.0,
                "score_threshold": -np.inf,
                "total_admitted_events": len(energies_arr),
                "passing_events": len(energies_arr),
                "passing_fraction": 1.0,
                "reference_energy_kev": p_win.reference_kev,
                "peak_centroid_kev": p_win.centroid_kev,
                "peak_sigma_kev": p_win.sigma_kev,
                "net_peak_counts": base_p["net_peak_counts"],
                "peak_retention": 1.0,
                "background_counts": base_p["estimated_background_counts"],
                "continuum_rejection": 0.0,
                "peak_to_background": base_p["peak_to_background"],
                "pb_relative_improvement": 0.0,
            }
            if rule_name in SPECIALIST_NAMES or rule_name == "joint_ds_cnn":
                specialist_results_rows.append(row_dict)
            else:
                fusion_results_rows.append(row_dict)

        for acc in ACCEPTANCES:
            cut_label = threshold_name(acc)
            thr = threshold_catalog[rule_name][cut_label]
            pass_mask = rule_scores >= thr
            passing_energies = energies_arr[pass_mask]
            cut_metrics = compute_all_peak_metrics(passing_energies)

            for p_idx, p_win in enumerate(peak_windows):
                base_p = baseline_metrics[p_win.reference_kev]
                cut_p = cut_metrics[p_win.reference_kev]
                pb_gain = (
                    (cut_p["peak_to_background"] - base_p["peak_to_background"]) / base_p["peak_to_background"]
                    if base_p["peak_to_background"] > 0
                    else 0.0
                )
                retention = (
                    cut_p["net_peak_counts"] / base_p["net_peak_counts"]
                    if base_p["net_peak_counts"] > 0
                    else 0.0
                )
                rej = (
                    1.0 - (cut_p["estimated_background_counts"] / base_p["estimated_background_counts"])
                    if base_p["estimated_background_counts"] > 0
                    else 0.0
                )

                row_dict = {
                    "model_or_rule": rule_name,
                    "cut_name": cut_label,
                    "target_acceptance": acc,
                    "score_threshold": thr,
                    "total_admitted_events": len(energies_arr),
                    "passing_events": int(np.count_nonzero(pass_mask)),
                    "passing_fraction": float(np.mean(pass_mask)),
                    "reference_energy_kev": p_win.reference_kev,
                    "peak_centroid_kev": p_win.centroid_kev,
                    "peak_sigma_kev": p_win.sigma_kev,
                    "net_peak_counts": cut_p["net_peak_counts"],
                    "peak_retention": retention,
                    "background_counts": cut_p["estimated_background_counts"],
                    "continuum_rejection": rej,
                    "peak_to_background": cut_p["peak_to_background"],
                    "pb_relative_improvement": pb_gain,
                }
                if rule_name in SPECIALIST_NAMES or rule_name == "joint_ds_cnn":
                    specialist_results_rows.append(row_dict)
                else:
                    fusion_results_rows.append(row_dict)

    # Save CSVs
    spec_csv_path = args.experiment_dir / "th232_specialist_results.csv"
    with spec_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(specialist_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(specialist_results_rows)
    print(f"Saved {spec_csv_path}")

    fusion_csv_path = args.experiment_dir / "th232_fusion_results.csv"
    with fusion_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fusion_results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fusion_results_rows)
    print(f"Saved {fusion_csv_path}")

    th232_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_scanned_events": total_scanned,
        "total_admitted_events": total_admitted,
        "threshold_catalog": threshold_catalog,
        "reference_peaks_kev": list(REFERENCE_PEAKS_KEV),
        "peak_windows": [asdict(pw) for pw in peak_windows],
    }
    save_json(args.experiment_dir / "th232_specialist_evaluation.json", th232_summary)
    print("Th-232 evaluation finished successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
