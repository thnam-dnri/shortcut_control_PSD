#!/usr/bin/env python3
"""Compare trained MA10 and MA20 shared DS-CNNs on identical evaluation rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_shared_six_group_ds_cnn import (
    GROUPS,
    IndexedWaveforms,
    evaluate_by_group,
    make_eval_loader,
    predict,
    sha256_file,
)
from src.architecture_candidates import DSCNN
from src.data_access_guards import assert_no_forbidden_path


def mean_by_group(runs: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        group: float(np.mean([run[key][group]["macro_auroc"] for run in runs]))
        for group in ("all_groups", *(f"group_{value}" for value in GROUPS))
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_six_group_20260822",
    )
    parser.add_argument(
        "--ma10-cache",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_20260822",
    )
    parser.add_argument(
        "--ma20-cache",
        type=Path,
        default=PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_ma20_20260822",
    )
    parser.add_argument(
        "--ma10-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/shared_six_group_ds_cnn_20260822",
    )
    parser.add_argument(
        "--ma20-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/shared_six_group_ds_cnn_ma20_20260822",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/experiments/shared_six_group_ds_cnn_ma10_vs_ma20_20260822",
    )
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "group_dir": args.group_dir,
            "ma10_cache": args.ma10_cache,
            "ma20_cache": args.ma20_cache,
            "ma10_output": args.ma10_output,
            "ma20_output": args.ma20_output,
            "output_dir": args.output_dir,
        }.items()
    }
    for path in paths.values():
        assert_no_forbidden_path(path)
    output_dir = paths["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    metadata_hashes = {
        width: sha256_file(paths[f"ma{width}_cache"] / "strict_internal_metadata.npz")
        for width in (10, 20)
    }
    if len(set(metadata_hashes.values())) != 1:
        raise ValueError("MA10 and MA20 strict evaluation rows differ")

    representations: dict[str, Any] = {}
    for width in (10, 20):
        cache_dir = paths[f"ma{width}_cache"]
        training_dir = paths[f"ma{width}_output"]
        training_report = json.loads(
            (training_dir / "experiment_report.json").read_text(encoding="utf-8")
        )
        dataset = IndexedWaveforms(
            cache_dir / "strict_internal_values.npy",
            cache_dir / "strict_internal_metadata.npz",
            paths["group_dir"] / "strict_internal_assignments.npz",
        )
        loader = make_eval_loader(dataset, args.batch_size)
        runs: list[dict[str, Any]] = []
        for training_run in training_report["runs"]:
            seed = int(training_run["seed"])
            checkpoint_path = PROJECT_ROOT / training_run["shared_checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = DSCNN(input_channels=2, width=24).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            scores = predict(model, loader, device)
            score_path = output_dir / f"ma{width}_seed_{seed}_strict_scores.npy"
            np.save(score_path, scores)
            runs.append(
                {
                    "seed": seed,
                    "strict_by_group": evaluate_by_group(dataset, scores),
                    "score_path": score_path.relative_to(PROJECT_ROOT).as_posix(),
                    "score_sha256": sha256_file(score_path),
                }
            )
            del model
            torch.cuda.empty_cache()
        validation_runs = [
            {"validation_by_group": run["shared"]["validation_by_group"]}
            for run in training_report["runs"]
        ]
        cache_manifest = json.loads(
            (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
        )
        representations[f"ma{width}"] = {
            "representation_config": cache_manifest["representation_config"],
            "feature_statistics": cache_manifest["feature_statistics"],
            "validation_mean_macro_auroc": mean_by_group(
                validation_runs, "validation_by_group"
            ),
            "strict_mean_macro_auroc": mean_by_group(runs, "strict_by_group"),
            "strict_runs": runs,
            "training_report_sha256": sha256_file(
                training_dir / "experiment_report.json"
            ),
        }

    deltas = {
        population: {
            group: (
                representations["ma20"][f"{population}_mean_macro_auroc"][group]
                - representations["ma10"][f"{population}_mean_macro_auroc"][group]
            )
            for group in representations["ma10"][f"{population}_mean_macro_auroc"]
        }
        for population in ("validation", "strict")
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MA20_REPRESENTATION_EXECUTED",
        "representations": representations,
        "ma20_minus_ma10": deltas,
        "row_identity": {
            "strict_metadata_sha256": metadata_hashes[10],
            "identical": True,
        },
        "claim_boundary": (
            "Development comparison only; file-validation and strict-internal "
            "pools are not independent external interaction-truth validation."
        ),
        "test_partition_used": False,
        "th232_used": False,
        "eu152_used": False,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "ma20_minus_ma10": deltas}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
