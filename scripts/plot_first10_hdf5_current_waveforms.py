#!/usr/bin/env python3
"""Plot the first ten raw waveforms from every processed HDF5 file as current proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SAMPLE_PERIOD_NS = 4.0
DEFAULT_WAVEFORMS_PER_FILE = 10


def charge_to_current(charge: np.ndarray) -> np.ndarray:
    """Convert a negative-polarity charge waveform to a current proxy.

    This is the centered finite-difference convention used by the existing
    250-MSPS processing pipeline: -(charge[n + 1] - charge[n - 1]) / 2.
    The two endpoints are set to zero because they have no centered neighbor.
    """
    charge = np.asarray(charge, dtype=np.float32)
    if charge.ndim != 2 or charge.shape[1] < 3:
        raise ValueError(f"expected a 2-D waveform array with at least 3 samples; received {charge.shape}")
    current = np.zeros_like(charge, dtype=np.float32)
    current[:, 1:-1] = -0.5 * (charge[:, 2:] - charge[:, :-2])
    return current


def attribute_text(value: Any, default: str = "") -> str:
    """Return an HDF5 attribute as readable text."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def optional_float(value: Any) -> float | None:
    """Convert an optional scalar to float for the manifest and plot label."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def plot_current_waveform(
    current: np.ndarray,
    output_path: Path,
    source: str,
    hdf5_name: str,
    row_index: int,
    event_id: int | None,
    reconstructed_energy_kev: float | None,
    sample_period_ns: float,
) -> None:
    """Write one current-waveform figure."""
    sample_count = current.shape[0]
    time_us = np.arange(sample_count, dtype=np.float32) * sample_period_ns / 1_000.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time_us, current, color="tab:blue", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.65)
    ax.set_xlim(float(time_us[0]), float(time_us[-1]))
    ax.set_xlabel(f"Time (µs) ({1_000.0 / sample_period_ns:.1f} MSPS)")
    ax.set_ylabel("Current proxy (ADC-code change per sample)")
    ax.grid(True, alpha=0.25)

    details = [f"row {row_index + 1}"]
    if event_id is not None:
        details.append(f"event {event_id}")
    if reconstructed_energy_kev is not None:
        details.append(f"E = {reconstructed_energy_kev:.2f} keV")
    ax.set_title(f"{hdf5_name}\n{', '.join(details)}", fontsize=10)
    fig.suptitle(f"{source.upper()} raw charge waveform converted to current", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_file(
    hdf5_path: Path,
    output_dir: Path,
    waveforms_per_file: int,
    sample_period_ns: float,
) -> dict[str, Any]:
    """Plot the first waveforms in one HDF5 file and return a manifest record."""
    file_output_dir = output_dir / hdf5_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as input_file:
        if "waveform" not in input_file:
            return {
                "input_hdf5": str(hdf5_path),
                "status": "SKIPPED_MISSING_WAVEFORM_DATASET",
                "waveform_count": 0,
                "plots": [],
            }

        waveform_dataset = input_file["waveform"]
        if waveform_dataset.ndim != 2:
            raise ValueError(f"{hdf5_path}: expected waveform dataset to be 2-D; received {waveform_dataset.shape}")

        waveform_count = int(waveform_dataset.shape[0])
        count = min(waveforms_per_file, waveform_count)
        source = attribute_text(input_file.attrs.get("source_label"), "unknown")
        processing_status = attribute_text(input_file.attrs.get("processing_status"), "unknown")
        event_dataset = input_file.get("event_id")
        energy_dataset = input_file.get("reconstructed_energy_kev")
        event_ids = np.asarray(event_dataset[:count]) if event_dataset is not None else None
        energies = np.asarray(energy_dataset[:count]) if energy_dataset is not None else None
        charge = np.asarray(waveform_dataset[:count], dtype=np.float32)

    if count == 0:
        return {
            "input_hdf5": str(hdf5_path),
            "source": source,
            "processing_status": processing_status,
            "status": "SKIPPED_EMPTY_WAVEFORM_DATASET",
            "waveform_count": waveform_count,
            "plots": [],
        }

    current = charge_to_current(charge)
    plots: list[dict[str, Any]] = []
    for row_index in range(count):
        event_id = int(event_ids[row_index]) if event_ids is not None else None
        energy = optional_float(energies[row_index]) if energies is not None else None
        event_suffix = f"_event_{event_id}" if event_id is not None else ""
        output_path = file_output_dir / (
            f"{hdf5_path.stem}_current_waveform_{row_index + 1:02d}{event_suffix}.png"
        )
        plot_current_waveform(
            current[row_index],
            output_path,
            source,
            hdf5_path.name,
            row_index,
            event_id,
            energy,
            sample_period_ns,
        )
        plots.append(
            {
                "row_index_zero_based": row_index,
                "event_id": event_id,
                "reconstructed_energy_kev": energy,
                "output_png": str(output_path),
            }
        )

    return {
        "input_hdf5": str(hdf5_path),
        "source": source,
        "processing_status": processing_status,
        "status": "OK",
        "waveform_count": waveform_count,
        "plotted_count": count,
        "plots": plots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processed_data/waveform_hdf5"),
        help="directory containing processed waveform HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/waveform_hdf5/current_first10"),
        help="directory receiving one PNG per plotted waveform",
    )
    parser.add_argument(
        "--waveforms-per-file",
        type=int,
        default=DEFAULT_WAVEFORMS_PER_FILE,
        help="number of initial waveforms to plot from each HDF5 file",
    )
    parser.add_argument(
        "--sample-period-ns",
        type=float,
        default=DEFAULT_SAMPLE_PERIOD_NS,
        help="sample period used for the time axis",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/waveform_hdf5/current_first10_manifest.json"),
        help="JSON manifest describing generated plots and skipped files",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.waveforms_per_file <= 0:
        parser.error("--waveforms-per-file must be positive")
    if args.sample_period_ns <= 0:
        parser.error("--sample-period-ns must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    hdf5_files = sorted(args.input_dir.glob("*.h5"))
    if not hdf5_files:
        parser.error(f"no HDF5 files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        plot_file(path, args.output_dir, args.waveforms_per_file, args.sample_period_ns)
        for path in hdf5_files
    ]
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "sample_period_ns": args.sample_period_ns,
        "current_conversion": "-(charge[n + 1] - charge[n - 1]) / 2; endpoints set to zero",
        "input_file_count": len(hdf5_files),
        "plotted_figure_count": sum(len(record["plots"]) for record in records),
        "files": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "input_file_count": len(hdf5_files),
        "plotted_figure_count": manifest["plotted_figure_count"],
        "skipped_file_count": sum(record["status"].startswith("SKIPPED") for record in records),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
