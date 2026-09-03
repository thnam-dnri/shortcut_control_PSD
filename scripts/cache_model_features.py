#!/usr/bin/env python3
"""Cache exact train/validation O2 waveform representations for model iteration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_o2_late_fusion import build_partition_features, sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "outputs/labels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "processed_data/model_features",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--event-store-dir",
        type=Path,
        default=None,
        help="Optional consolidated event-store directory for faster waveform reads.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for partition in ("train", "validation"):
        csv_path = labels_dir / f"label_pairs_{partition}.csv"
        print(f"Loading {partition} representations ...", flush=True)
        data = build_partition_features(
            csv_path,
            event_store_dir=args.event_store_dir.resolve()
            if args.event_store_dir is not None
            else None,
        )
        output_path = output_dir / f"o2_features_{partition}.npz"
        np.savez(
            output_path,
            charge=data.charge,
            current=data.current,
            labels=data.labels,
            weights=data.weights,
            peak_ids=data.peak_ids,
            t10_fallback_count=np.asarray(data.t10_fallback_count, dtype=np.int64),
        )
        records[partition] = {
            "cache_file": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "cache_sha256": sha256_file(output_path),
            "csv_file": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "csv_sha256": sha256_file(csv_path),
            "event_count": int(data.labels.size),
            "charge_shape": list(data.charge.shape),
            "current_shape": list(data.current.shape),
            "t10_fallback_count": data.t10_fallback_count,
        }
        print(f"Wrote {output_path}", flush=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation_builder": "scripts/train_o2_late_fusion.py:build_partition_features",
        "test_partition_used": False,
        "event_store_dir": args.event_store_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
        if args.event_store_dir is not None
        else None,
        "partitions": records,
    }
    (output_dir / "feature_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
