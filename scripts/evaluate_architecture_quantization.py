#!/usr/bin/env python3
"""Run a development-only weight/input precision screen for architecture candidates.

This is a preliminary Stage 4 emulation, not FPGA synthesis or a production
fixed-point implementation. Candidate weights and standardized input samples
are symmetrically quantized and dequantized in PyTorch. The input scale is
calibrated only from the development training partition; validation remains
untouched for model/threshold selection and locked/external data are not read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.architecture_candidates import build_candidate  # noqa: E402
from src.ba133_cnn import (  # noqa: E402
    apply_channel_statistics,
    build_representation,
    load_raw_partition,
)
from src.data_access_guards import (  # noqa: E402
    assert_development_csv,
    assert_no_forbidden_path,
)
from train_architecture_candidates import (  # noqa: E402
    CANDIDATES,
    REPRESENTATION,
    WARNING_STATUS,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quantization_range(bits: int) -> int:
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    return (1 << (bits - 1)) - 1


def symmetric_scale(values: torch.Tensor | np.ndarray, bits: int) -> float:
    limit = quantization_range(bits)
    if isinstance(values, torch.Tensor):
        maximum = float(torch.max(torch.abs(values)).item())
    else:
        maximum = float(np.max(np.abs(values)))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return 1.0
    return maximum / float(limit)


def quantize_dequantize(values: torch.Tensor, scale: float, bits: int) -> torch.Tensor:
    limit = quantization_range(bits)
    quantized = torch.clamp(torch.round(values / scale), -limit, limit)
    return quantized * scale


def quantize_model_weights(model: nn.Module, bits: int) -> dict[str, float]:
    scales: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            scale = symmetric_scale(parameter, bits)
            parameter.copy_(quantize_dequantize(parameter, scale, bits))
            scales[name] = scale
    return scales


def metric_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(average_precision_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "event_count": int(labels.size),
        "pair_count": int(labels.size // 2),
    }


def score_model(
    model: nn.Module,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def per_peak_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    peak_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for peak_id in sorted(set(peak_ids.tolist())):
        event_indices = np.flatnonzero(peak_ids == peak_id)
        result[peak_id] = metric_summary(
            labels[event_indices], scores[event_indices], weights[event_indices]
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/labels/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/event_store/architecture_pass_warn_20260815",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_warning_balanced_20260816",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/models/architecture_candidates_warning_balanced_20260816/quantization_screen.json",
    )
    parser.add_argument("--candidate", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--calibration-events", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-validation-events", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> int:
    args = build_parser().parse_args()
    if args.calibration_events < 2 or args.batch_size < 1:
        raise ValueError("calibration-events must be at least 2 and batch-size positive")
    for bits in args.bits:
        quantization_range(bits)
    if args.max_validation_events is not None and args.max_validation_events < 2:
        raise ValueError("max-validation-events must be at least 2")

    labels_dir = args.labels_dir.resolve()
    event_store_dir = args.event_store_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    train_csv = labels_dir / "label_pairs_train.csv"
    validation_csv = labels_dir / "label_pairs_validation.csv"
    comparison_path = model_dir / "comparison.json"
    for path in (
        train_csv,
        validation_csv,
        event_store_dir,
        model_dir,
        comparison_path,
    ):
        assert_no_forbidden_path(path)
    assert_development_csv(train_csv)
    assert_development_csv(validation_csv)
    if not args.overwrite and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("warning_status") != WARNING_STATUS:
        raise ValueError("Candidate comparison is not marked with the active warning")
    width = int(comparison["training"]["width"])
    for candidate in args.candidate:
        checkpoint = model_dir / candidate / f"{candidate}_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        assert_no_forbidden_path(checkpoint)

    print("Loading calibration train event store ...", flush=True)
    calibration_raw = load_raw_partition(
        train_csv,
        event_store_dir,
        args.calibration_events,
    )
    print("Loading validation event store ...", flush=True)
    validation_raw = load_raw_partition(
        validation_csv,
        event_store_dir,
        args.max_validation_events,
    )
    print("Building representations ...", flush=True)
    calibration_values, calibration_stats = build_representation(
        calibration_raw, REPRESENTATION
    )
    validation_values, validation_stats = build_representation(
        validation_raw, REPRESENTATION
    )

    reference_checkpoint = torch.load(
        model_dir / args.candidate[0] / f"{args.candidate[0]}_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    feature_statistics = reference_checkpoint["feature_statistics"]
    apply_channel_statistics(calibration_values, feature_statistics)
    apply_channel_statistics(validation_values, feature_statistics)
    calibration_scale = symmetric_scale(calibration_values, max(args.bits))
    device = resolve_device(args.device)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PROVISIONAL_STAGE_4_QUANTIZATION_WITH_SHORTCUT_WARNING",
        "warning_status": WARNING_STATUS,
        "created_utc": utc_now(),
        "device": str(device),
        "representation": REPRESENTATION.as_dict(),
        "input": {
            "labels_dir": labels_dir.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv": train_csv.relative_to(PROJECT_ROOT).as_posix(),
            "validation_csv": validation_csv.relative_to(PROJECT_ROOT).as_posix(),
            "train_csv_sha256": sha256_file(train_csv),
            "validation_csv_sha256": sha256_file(validation_csv),
            "event_store_dir": event_store_dir.relative_to(PROJECT_ROOT).as_posix(),
            "calibration_events": int(calibration_raw.labels.size),
            "validation_events": int(validation_raw.labels.size),
            "max_validation_events": args.max_validation_events,
            "test_partition_used": False,
            "th232_used_for_selection": False,
            "eu152_used_for_selection": False,
        },
        "calibration": {
            "source": "development_train_only",
            "representation_build_statistics": calibration_stats,
            "validation_representation_build_statistics": validation_stats,
            "feature_statistics_source": "candidate_train_only",
            "input_symmetric_scale_by_bits": {
                str(bits): symmetric_scale(calibration_values, bits) for bits in args.bits
            },
        },
        "configuration": {
            "candidates": list(args.candidate),
            "bits": list(args.bits),
            "batch_size": args.batch_size,
            "model_width": width,
            "emulation": "symmetric weight and input quantize-dequantize in PyTorch",
        },
        "caveats": [
            "This is not FPGA synthesis, HLS conversion, timing closure, or a production fixed-point implementation.",
            "Only model parameters and standardized input samples are quantized; intermediate activation scales are not yet hardware-calibrated.",
            "The validation result remains development-only and shortcut-sensitive.",
            "No threshold was selected and no locked/external data were opened.",
        ],
        "candidates": {},
    }

    for candidate in args.candidate:
        checkpoint_path = model_dir / candidate / f"{candidate}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        float_model = build_candidate(candidate, input_channels=2, width=width).to(device)
        float_model.load_state_dict(checkpoint["model_state_dict"])
        float_scores = score_model(
            float_model, validation_values, args.batch_size, device
        )
        float_metrics = metric_summary(
            validation_raw.labels, float_scores, validation_raw.weights
        )
        candidate_result: dict[str, Any] = {
            "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            "float32": {
                **float_metrics,
                "per_peak": per_peak_metrics(
                    validation_raw.labels,
                    float_scores,
                    validation_raw.weights,
                    validation_raw.peak_ids,
                ),
            },
            "by_bits": {},
        }
        for bits in args.bits:
            model = build_candidate(candidate, input_channels=2, width=width).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            weight_scales = quantize_model_weights(model, bits)
            input_scale = symmetric_scale(calibration_values, bits)
            quantized_validation = quantize_dequantize(
                torch.from_numpy(validation_values), input_scale, bits
            ).numpy()
            quantized_scores = score_model(
                model, quantized_validation, args.batch_size, device
            )
            quantized_metrics = metric_summary(
                validation_raw.labels,
                quantized_scores,
                validation_raw.weights,
            )
            candidate_result["by_bits"][str(bits)] = {
                "input_scale": input_scale,
                "weight_scale_count": len(weight_scales),
                "metrics": {
                    **quantized_metrics,
                    "per_peak": per_peak_metrics(
                        validation_raw.labels,
                        quantized_scores,
                        validation_raw.weights,
                        validation_raw.peak_ids,
                    ),
                },
                "delta_from_float32": {
                    "auroc": quantized_metrics["auroc"] - float_metrics["auroc"],
                    "weighted_auroc": quantized_metrics["weighted_auroc"]
                    - float_metrics["weighted_auroc"],
                },
            }
        result["candidates"][candidate] = candidate_result
        print(
            candidate,
            {
                bits: round(
                    float(candidate_result["by_bits"][str(bits)]["metrics"]["auroc"]),
                    6,
                )
                for bits in args.bits
            },
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
