#!/usr/bin/env python3
"""Evaluate frozen Peak-Specialist DS-CNN models, score calibration, and fusion rules.

Evaluates on internal validation (to fit isotonic calibration and select the best fusion rule)
and on the file-disjoint held-out validation partition (to compare against the joint DS-CNN baseline).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture_candidates import DSCNN  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    RepresentationConfig,
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
    representation_config_from_checkpoint,
)
from src.data_access_guards import assert_development_csv, assert_no_forbidden_path  # noqa: E402

SEED = 20260820
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/peak_specialist_ds_cnn_20260820"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
DEFAULT_EVENT_STORE_DIR = (
    PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
DEFAULT_JOINT_BASELINE = (
    PROJECT_ROOT / "outputs/models/compact_ds_cnn_performance_20260820/ds_cnn/ds_cnn_best.pt"
)

SPECIALIST_NAMES = ("356A", "511A", "661A")
PEAK_IDS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
PEAK_TO_ENERGY = {
    "ba133_356kev": 356.0129,
    "na22_511kev": 510.99895,
    "cs137_662kev": 661.657,
}
PEAK_WEIGHTS = {"ba133_356kev": 0.4, "na22_511kev": 0.4, "cs137_662kev": 0.2}

# Energy boundaries for nearest-energy expert:
# Midpoint between 356.0129 and 510.99895 = 433.505925 keV
# Midpoint between 510.99895 and 661.657 = 586.327975 keV
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


def event_indices(pair_indices: np.ndarray) -> np.ndarray:
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    return np.column_stack((2 * pair_indices, 2 * pair_indices + 1)).reshape(-1)


def parse_event_details(csv_path: Path) -> dict[str, np.ndarray]:
    event_ids: list[str] = []
    energies: list[float] = []
    sources: list[str] = []
    peak_ids: list[str] = []
    labels: list[int] = []

    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # positive event
            event_ids.append(str(row["positive_event_id"]))
            energies.append(float(row["positive_energy_kev"]))
            sources.append(str(row["positive_source"]))
            peak_ids.append(str(row["peak_id"]))
            labels.append(1)
            # negative event
            event_ids.append(str(row["negative_event_id"]))
            energies.append(float(row["negative_energy_kev"]))
            sources.append(str(row["negative_source"]))
            peak_ids.append(str(row["peak_id"]))
            labels.append(0)

    return {
        "event_id": np.asarray(event_ids, dtype="U64"),
        "energy_kev": np.asarray(energies, dtype=np.float32),
        "source": np.asarray(sources, dtype="U32"),
        "peak_id": np.asarray(peak_ids, dtype="U64"),
        "label": np.asarray(labels, dtype=np.int64),
    }


def predict_model(
    model: nn.Module,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, Any]:
    per_peak: dict[str, dict[str, float | int]] = {}
    for peak_id in PEAK_IDS:
        mask = peak_ids == peak_id
        if np.count_nonzero(mask) == 0:
            continue
        sub_labels = labels[mask]
        sub_scores = scores[mask]
        if len(np.unique(sub_labels)) < 2:
            continue
        per_peak[peak_id] = {
            "auroc": float(roc_auc_score(sub_labels, sub_scores)),
            "average_precision": float(average_precision_score(sub_labels, sub_scores)),
            "event_count": int(np.count_nonzero(mask)),
            "pair_count": int(np.count_nonzero(mask) // 2),
        }
    peak_aurocs = [float(item["auroc"]) for item in per_peak.values()]
    peak_aps = [float(item["average_precision"]) for item in per_peak.values()]

    # Loss weights for weighted auroc
    weights = np.asarray([PEAK_WEIGHTS[pid] for pid in peak_ids], dtype=np.float32)

    return {
        "macro_auroc": float(np.mean(peak_aurocs)) if peak_aurocs else 0.0,
        "worst_peak_auroc": float(np.min(peak_aurocs)) if peak_aurocs else 0.0,
        "macro_average_precision": float(np.mean(peak_aps)) if peak_aps else 0.0,
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "per_peak": per_peak,
    }


def compute_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    target_acceptances: tuple[float, ...] = (0.99, 0.95, 0.90, 0.80, 0.50, 0.30, 0.10),
) -> list[dict[str, float]]:
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    records: list[dict[str, float]] = []
    for target_acc in target_acceptances:
        threshold = float(np.quantile(pos_scores, 1.0 - target_acc))
        actual_acc = float(np.mean(pos_scores >= threshold))
        cont_pass = float(np.mean(neg_scores >= threshold))
        cont_rej = 1.0 - cont_pass
        records.append(
            {
                "target_acceptance": target_acc,
                "threshold": threshold,
                "actual_photopeak_acceptance": actual_acc,
                "continuum_rejection": cont_rej,
                "continuum_passing_fraction": cont_pass,
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--event-store-dir", type=Path, default=DEFAULT_EVENT_STORE_DIR)
    parser.add_argument("--joint-baseline-checkpoint", type=Path, default=DEFAULT_JOINT_BASELINE)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.labels_dir / "label_pairs_train.csv"
    val_csv = args.labels_dir / "label_pairs_validation.csv"
    split_path = args.labels_dir / "train_internal_split_indices.npz"

    assert_development_csv(train_csv)
    assert_development_csv(val_csv)
    assert_no_forbidden_path(train_csv)
    assert_no_forbidden_path(val_csv)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    print(f"Using compute device: {device}")

    # Load specialist checkpoints
    specialist_models: dict[str, nn.Module] = {}
    specialist_configs: dict[str, Any] = {}
    for spec_name in SPECIALIST_NAMES:
        ckpt_path = args.experiment_dir / f"model_{spec_name}_checkpoint.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing specialist checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = DSCNN(input_channels=2, width=24).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        specialist_models[spec_name] = m
        specialist_configs[spec_name] = ckpt

    # Load joint baseline
    joint_ckpt = torch.load(args.joint_baseline_checkpoint, map_location="cpu", weights_only=False)
    joint_model = DSCNN(input_channels=2, width=24).to(device)
    joint_model.load_state_dict(joint_ckpt["model_state_dict"])
    joint_model.eval()

    rep_config = representation_config_from_checkpoint(specialist_configs["356A"]["representation_config"])
    feature_stats = specialist_configs["356A"]["feature_statistics"]

    # ==========================================
    # 1. Internal Validation Inference & Tuning
    # ==========================================
    print("\n--- Processing Internal Validation Partition ---")
    raw_train = load_raw_partition(train_csv, args.event_store_dir)
    values_train, _ = build_representation(raw_train, rep_config)
    apply_channel_statistics(values_train, feature_stats)

    split = np.load(split_path)
    internal_pairs = split["internal_pair_indices"]
    internal_events = event_indices(internal_pairs)

    values_internal = values_train[internal_events]
    train_event_meta = parse_event_details(train_csv)
    meta_internal = {k: v[internal_events] for k, v in train_event_meta.items()}
    labels_internal = meta_internal["label"]
    peak_ids_internal = meta_internal["peak_id"]
    energies_internal = meta_internal["energy_kev"]

    raw_scores_internal: dict[str, np.ndarray] = {}
    for spec_name in SPECIALIST_NAMES:
        raw_scores_internal[spec_name] = predict_model(
            specialist_models[spec_name], values_internal, args.batch_size, device
        )
    raw_scores_internal["joint_ds_cnn"] = predict_model(
        joint_model, values_internal, args.batch_size, device
    )

    # Fit Isotonic Calibration strictly on internal validation per specialist
    print("Fitting Isotonic Regressions on internal validation...")
    calibrators: dict[str, IsotonicRegression] = {}
    cal_scores_internal: dict[str, np.ndarray] = {}
    for spec_name in SPECIALIST_NAMES:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_scores_internal[spec_name], labels_internal)
        calibrators[spec_name] = iso
        cal_scores_internal[spec_name] = iso.transform(raw_scores_internal[spec_name])

    # Fit Logistic Regression lightweight fusion on internal validation
    print("Fitting Lightweight Logistic Regression fusion on internal validation...")
    X_fusion_internal = np.column_stack(
        [
            cal_scores_internal["356A"],
            cal_scores_internal["511A"],
            cal_scores_internal["661A"],
            energies_internal / 1000.0,
        ]
    )
    fusion_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    fusion_lr.fit(X_fusion_internal, labels_internal)

    # Define fusion / selection rule functions
    def compute_fusion_scores(
        raw_dict: dict[str, np.ndarray],
        cal_dict: dict[str, np.ndarray],
        energies: np.ndarray,
        peak_ids: np.ndarray,
        validation_best_specialist_map: dict[str, str],
    ) -> dict[str, np.ndarray]:
        fused: dict[str, np.ndarray] = {}
        # Rule A: Raw max
        fused["raw_max"] = np.maximum(
            raw_dict["356A"], np.maximum(raw_dict["511A"], raw_dict["661A"])
        )
        # Rule B: Calibrated max
        fused["calibrated_max"] = np.maximum(
            cal_dict["356A"], np.maximum(cal_dict["511A"], cal_dict["661A"])
        )
        # Rule C: Calibrated mean
        fused["calibrated_mean"] = (
            cal_dict["356A"] + cal_dict["511A"] + cal_dict["661A"]
        ) / 3.0
        # Rule D: Nearest-energy expert
        nearest_scores = np.empty(len(energies), dtype=np.float32)
        nearest_scores[energies < SWITCH_ENERGY_1] = raw_dict["356A"][energies < SWITCH_ENERGY_1]
        nearest_scores[(energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)] = raw_dict["511A"][
            (energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)
        ]
        nearest_scores[energies >= SWITCH_ENERGY_2] = raw_dict["661A"][energies >= SWITCH_ENERGY_2]
        fused["nearest_energy_expert"] = nearest_scores

        # Rule D-cal: Nearest-energy expert (calibrated)
        nearest_cal_scores = np.empty(len(energies), dtype=np.float32)
        nearest_cal_scores[energies < SWITCH_ENERGY_1] = cal_dict["356A"][energies < SWITCH_ENERGY_1]
        nearest_cal_scores[(energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)] = cal_dict["511A"][
            (energies >= SWITCH_ENERGY_1) & (energies < SWITCH_ENERGY_2)
        ]
        nearest_cal_scores[energies >= SWITCH_ENERGY_2] = cal_dict["661A"][energies >= SWITCH_ENERGY_2]
        fused["nearest_energy_expert_calibrated"] = nearest_cal_scores

        # Rule E: Validation-selected energy expert
        val_selected = np.empty(len(peak_ids), dtype=np.float32)
        for pid, best_spec in validation_best_specialist_map.items():
            mask = peak_ids == pid
            val_selected[mask] = raw_dict[best_spec][mask]
        fused["validation_selected_expert"] = val_selected

        # Rule F: Lightweight Logistic Regression fusion
        X_eval = np.column_stack(
            [
                cal_dict["356A"],
                cal_dict["511A"],
                cal_dict["661A"],
                energies / 1000.0,
            ]
        )
        fused["logistic_fusion"] = fusion_lr.predict_proba(X_eval)[:, 1]
        return fused

    # Determine validation-selected expert map on internal validation
    val_best_map: dict[str, str] = {}
    for pid in PEAK_IDS:
        mask = peak_ids_internal == pid
        best_spec_for_pid = max(
            SPECIALIST_NAMES,
            key=lambda s: roc_auc_score(labels_internal[mask], raw_scores_internal[s][mask]),
        )
        val_best_map[pid] = best_spec_for_pid
    print(f"Internal Validation Best Specialist Map: {val_best_map}")

    # Evaluate internal validation rules
    fusion_scores_internal = compute_fusion_scores(
        raw_scores_internal, cal_scores_internal, energies_internal, peak_ids_internal, val_best_map
    )

    print("\n--- Internal Validation Evaluation Summary ---")
    internal_comparison: dict[str, Any] = {}
    all_evaluated_models = (
        list(SPECIALIST_NAMES)
        + ["joint_ds_cnn"]
        + list(fusion_scores_internal.keys())
    )
    for model_name in all_evaluated_models:
        if model_name in raw_scores_internal:
            sc = raw_scores_internal[model_name]
        else:
            sc = fusion_scores_internal[model_name]
        m = compute_metrics(labels_internal, sc, peak_ids_internal)
        internal_comparison[model_name] = m
        print(
            f"{model_name:<32}: Worst AUROC = {m['worst_peak_auroc']:.5f} | "
            f"Macro AUROC = {m['macro_auroc']:.5f} | "
            f"Weighted AUROC = {m['weighted_auroc']:.5f} | "
            f"Ba356={m['per_peak']['ba133_356kev']['auroc']:.4f} "
            f"Na511={m['per_peak']['na22_511kev']['auroc']:.4f} "
            f"Cs662={m['per_peak']['cs137_662kev']['auroc']:.4f}"
        )

    # Rank fusion rules by internal priority:
    # 1. highest worst-energy AUROC
    # 2. highest equal-energy macro AUROC
    # 3. simplicity
    fusion_rule_names = list(fusion_scores_internal.keys())
    ranked_fusion_rules = sorted(
        fusion_rule_names,
        key=lambda r: (
            internal_comparison[r]["worst_peak_auroc"],
            internal_comparison[r]["macro_auroc"],
        ),
        reverse=True,
    )
    selected_fusion_rule = ranked_fusion_rules[0]
    print(f"\nSelected Fusion Rule (Internal Validation Priority): {selected_fusion_rule}")

    # Save internal specialist scores NPZ
    np.savez_compressed(
        args.experiment_dir / "internal_specialist_scores.npz",
        event_id=meta_internal["event_id"],
        true_label=labels_internal,
        energy_kev=energies_internal,
        source=meta_internal["source"],
        peak_domain=peak_ids_internal,
        split=np.full(labels_internal.size, "internal_validation", dtype="U32"),
        score_356A=raw_scores_internal["356A"],
        score_511A=raw_scores_internal["511A"],
        score_661A=raw_scores_internal["661A"],
        score_joint_ds_cnn=raw_scores_internal["joint_ds_cnn"],
        score_calibrated_356A=cal_scores_internal["356A"],
        score_calibrated_511A=cal_scores_internal["511A"],
        score_calibrated_661A=cal_scores_internal["661A"],
        selected_fusion_score=fusion_scores_internal[selected_fusion_rule],
    )
    print(f"Saved {args.experiment_dir / 'internal_specialist_scores.npz'}")

    del raw_train, values_train, values_internal

    # ==========================================
    # 2. Held-Out Validation Partition Scoring
    # ==========================================
    print("\n--- Processing Held-Out Validation Partition ---")
    raw_val = load_raw_partition(val_csv, args.event_store_dir)
    values_val, _ = build_representation(raw_val, rep_config)
    apply_channel_statistics(values_val, feature_stats)

    meta_val = parse_event_details(val_csv)
    labels_val = meta_val["label"]
    peak_ids_val = meta_val["peak_id"]
    energies_val = meta_val["energy_kev"]

    raw_scores_val: dict[str, np.ndarray] = {}
    cal_scores_val: dict[str, np.ndarray] = {}
    for spec_name in SPECIALIST_NAMES:
        raw_scores_val[spec_name] = predict_model(
            specialist_models[spec_name], values_val, args.batch_size, device
        )
        cal_scores_val[spec_name] = calibrators[spec_name].transform(raw_scores_val[spec_name])
    raw_scores_val["joint_ds_cnn"] = predict_model(
        joint_model, values_val, args.batch_size, device
    )

    fusion_scores_val = compute_fusion_scores(
        raw_scores_val, cal_scores_val, energies_val, peak_ids_val, val_best_map
    )

    # Save held-out specialist scores NPZ
    np.savez_compressed(
        args.experiment_dir / "held_out_specialist_scores.npz",
        event_id=meta_val["event_id"],
        true_label=labels_val,
        energy_kev=energies_val,
        source=meta_val["source"],
        peak_domain=peak_ids_val,
        split=np.full(labels_val.size, "held_out_validation", dtype="U32"),
        score_356A=raw_scores_val["356A"],
        score_511A=raw_scores_val["511A"],
        score_661A=raw_scores_val["661A"],
        score_joint_ds_cnn=raw_scores_val["joint_ds_cnn"],
        score_calibrated_356A=cal_scores_val["356A"],
        score_calibrated_511A=cal_scores_val["511A"],
        score_calibrated_661A=cal_scores_val["661A"],
        selected_fusion_score=fusion_scores_val[selected_fusion_rule],
    )
    print(f"Saved {args.experiment_dir / 'held_out_specialist_scores.npz'}")

    # ==========================================
    # 3. Cross-Energy Transfer Matrix
    # ==========================================
    print("\n--- Generating Cross-Energy Transfer Matrix ---")
    transfer_rows: list[dict[str, Any]] = []
    for eval_peak_id in PEAK_IDS:
        eval_mask = peak_ids_val == eval_peak_id
        eval_labels = labels_val[eval_mask]
        eval_energy = PEAK_TO_ENERGY[eval_peak_id]
        for spec_name in SPECIALIST_NAMES:
            spec_scores = raw_scores_val[spec_name][eval_mask]
            auroc = float(roc_auc_score(eval_labels, spec_scores))
            auprc = float(average_precision_score(eval_labels, spec_scores))
            # 99% photopeak retention threshold on that evaluation set
            pos_sc = spec_scores[eval_labels == 1]
            neg_sc = spec_scores[eval_labels == 0]
            thr_99 = float(np.quantile(pos_sc, 0.01))
            ret_99 = float(np.mean(pos_sc >= thr_99))
            rej_99 = float(np.mean(neg_sc < thr_99))

            transfer_rows.append(
                {
                    "evaluation_domain": eval_peak_id,
                    "evaluation_energy_kev": eval_energy,
                    "specialist": spec_name,
                    "auroc": auroc,
                    "auprc": auprc,
                    "photopeak_retention_at_99target": ret_99,
                    "continuum_rejection_at_99target": rej_99,
                    "threshold_at_99target": thr_99,
                }
            )

    transfer_csv_path = args.experiment_dir / "cross_energy_transfer_matrix.csv"
    with transfer_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transfer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(transfer_rows)
    print(f"Saved {transfer_csv_path}")

    # ==========================================
    # 4. Fusion Results CSV & Held-Out Comparison
    # ==========================================
    print("\n--- Held-Out Validation Evaluation Summary ---")
    held_out_comparison: dict[str, Any] = {}
    fusion_summary_rows: list[dict[str, Any]] = []

    for model_name in all_evaluated_models:
        if model_name in raw_scores_val:
            sc = raw_scores_val[model_name]
        else:
            sc = fusion_scores_val[model_name]
        m = compute_metrics(labels_val, sc, peak_ids_val)
        held_out_comparison[model_name] = m

        print(
            f"{model_name:<32}: Worst AUROC = {m['worst_peak_auroc']:.5f} | "
            f"Macro AUROC = {m['macro_auroc']:.5f} | "
            f"Weighted AUROC = {m['weighted_auroc']:.5f} | "
            f"Ba356={m['per_peak']['ba133_356kev']['auroc']:.4f} "
            f"Na511={m['per_peak']['na22_511kev']['auroc']:.4f} "
            f"Cs662={m['per_peak']['cs137_662kev']['auroc']:.4f}"
        )

        fusion_summary_rows.append(
            {
                "model_or_rule": model_name,
                "is_selected_fusion_rule": (model_name == selected_fusion_rule),
                "held_out_macro_auroc": m["macro_auroc"],
                "held_out_worst_peak_auroc": m["worst_peak_auroc"],
                "held_out_weighted_auroc": m["weighted_auroc"],
                "held_out_pooled_auroc": m["pooled_auroc"],
                "held_out_ba356_auroc": m["per_peak"]["ba133_356kev"]["auroc"],
                "held_out_na511_auroc": m["per_peak"]["na22_511kev"]["auroc"],
                "held_out_cs662_auroc": m["per_peak"]["cs137_662kev"]["auroc"],
                "held_out_macro_ap": m["macro_average_precision"],
                "held_out_pooled_ap": m["pooled_average_precision"],
                "internal_macro_auroc": internal_comparison[model_name]["macro_auroc"],
                "internal_worst_peak_auroc": internal_comparison[model_name]["worst_peak_auroc"],
            }
        )

    fusion_csv_path = args.experiment_dir / "fusion_results.csv"
    with fusion_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fusion_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fusion_summary_rows)
    print(f"Saved {fusion_csv_path}")

    # ==========================================
    # 5. Operating Points Results CSV
    # ==========================================
    print("\n--- Generating Operating Point Results ---")
    operating_point_rows: list[dict[str, Any]] = []
    for model_name in ("joint_ds_cnn", "356A", "511A", "661A", selected_fusion_rule):
        if model_name in raw_scores_val:
            sc = raw_scores_val[model_name]
        else:
            sc = fusion_scores_val[model_name]
        ops = compute_operating_points(labels_val, sc)
        for op in ops:
            operating_point_rows.append(
                {
                    "model_or_rule": model_name,
                    **op,
                }
            )

    op_csv_path = args.experiment_dir / "operating_point_results.csv"
    with op_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operating_point_rows[0].keys()))
        writer.writeheader()
        writer.writerows(operating_point_rows)
    print(f"Saved {op_csv_path}")

    # Save comprehensive evaluation summary JSON
    eval_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_fusion_rule": selected_fusion_rule,
        "validation_selected_expert_map": val_best_map,
        "internal_comparison": internal_comparison,
        "held_out_comparison": held_out_comparison,
        "switch_energies_kev": [SWITCH_ENERGY_1, SWITCH_ENERGY_2],
    }
    save_json(args.experiment_dir / "evaluation_summary.json", eval_summary)
    print("Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
