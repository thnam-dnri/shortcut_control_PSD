#!/usr/bin/env python3
"""Create corrected full HDF5 copies without modifying original products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.optimize import least_squares


SOFTWARE_VERSION = "1.0"
ORIGINAL_CALIBRATION_INTERCEPT_KEV = 0.8706343947031012
SUPPORTED_SOURCES = {"ba133", "co60", "cs137", "na22", "th232"}


def normalize_source(value: Any) -> str:
    text = str(value.decode() if isinstance(value, bytes) else value).strip().lower()
    return text.replace("-", "").replace("_", "")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationModel:
    source: str
    model_label: str
    scale: float
    offset_kev: float
    fixed_intercept: bool
    pivot_intercept_kev: float | None
    lines: tuple[dict[str, Any], ...]
    fit_status_counts: dict[str, int]
    assumptions: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    def correct(self, energy: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return (np.asarray(energy, dtype=np.float64) - self.offset_kev) / self.scale


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_peak_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_calibration_models(
    peak_rows: Iterable[dict[str, Any]],
    fixed_intercept_kev: float = ORIGINAL_CALIBRATION_INTERCEPT_KEV,
) -> dict[str, CalibrationModel]:
    """Build one absolute source model from OK line centroids only."""

    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in peak_rows:
        source = normalize_source(row.get("source", ""))
        if source not in SUPPORTED_SOURCES:
            continue
        status = str(row.get("fit_status", ""))
        status_counts[source][status] += 1
        reference = _finite_float(row.get("reference_energy_kev"))
        centroid = _finite_float(row.get("fitted_centroid_kev"))
        name = str(row.get("peak_name", ""))
        if status != "OK" or reference is None or centroid is None or not name:
            continue
        grouped[source][name].append((reference, centroid))

    models: dict[str, CalibrationModel] = {}
    for source, line_values in grouped.items():
        aggregated: list[tuple[str, float, float]] = []
        for name, values in line_values.items():
            references = {reference for reference, _ in values}
            if len(references) != 1:
                continue
            aggregated.append((name, references.pop(), float(np.median([value for _, value in values]))))
        aggregated.sort(key=lambda item: item[1])
        if len(aggregated) >= 3:
            reference = np.asarray([item[1] for item in aggregated], dtype=np.float64)
            observed = np.asarray([item[2] for item in aggregated], dtype=np.float64)
            initial = np.polyfit(reference, observed, 1)
            fit = least_squares(
                lambda parameters: parameters[0] * reference + parameters[1] - observed,
                initial,
                loss="soft_l1",
                f_scale=1.0,
            )
            scale, offset = (float(value) for value in fit.x)
            label = "PRELIMINARY"
            assumptions = (
                "OK line fits only",
                "one source-level affine model; no per-file correction",
                "robust soft_l1 fit across aggregated line medians",
            )
        elif len(aggregated) == 2:
            (name_a, ref_a, obs_a), (name_b, ref_b, obs_b) = aggregated
            if ref_a == ref_b:
                continue
            scale = (obs_b - obs_a) / (ref_b - ref_a)
            offset = obs_a - scale * ref_a
            label = "PRELIMINARY"
            assumptions = (
                "OK line fits only",
                "one source-level affine model; no per-file correction",
                "exact two-point affine fit; goodness-of-fit is not testable",
            )
        elif len(aggregated) == 1 and source == "cs137":
            _, reference, observed = aggregated[0]
            if reference == fixed_intercept_kev:
                continue
            scale = (observed - fixed_intercept_kev) / (reference - fixed_intercept_kev)
            # observed = pivot + scale * (reference - pivot)
            #          = scale * reference + pivot * (1 - scale)
            # Store the effective affine offset so correct() remains
            # (observed - offset) / scale for every model type.
            offset = fixed_intercept_kev * (1.0 - scale)
            label = "CONSTRAINED"
            assumptions = (
                "OK one-line Cs-137 fit only",
                "original calibration intercept held fixed",
                "pivoted multiplicative correction; offset is not independently identified",
            )
        else:
            continue
        if not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0.0:
            continue
        lines = tuple(
            {
                "peak_name": name,
                "reference_energy_kev": reference,
                "observed_centroid_kev": observed,
                "model_centroid_kev": scale * reference + offset,
                "residual_kev": observed - (scale * reference + offset),
                "fit_status": "OK",
                "n_files": len(line_values[name]),
            }
            for name, reference, observed in aggregated
        )
        models[source] = CalibrationModel(
            source=source,
            model_label=label,
            scale=scale,
            offset_kev=offset,
            fixed_intercept=source == "cs137" and len(aggregated) == 1,
            pivot_intercept_kev=(
                fixed_intercept_kev if source == "cs137" and len(aggregated) == 1 else None
            ),
            lines=lines,
            fit_status_counts=dict(sorted(status_counts[source].items())),
            assumptions=assumptions,
        )
    return models


def _copy_bytes_with_hash(source: Path, destination: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copystat(source, destination, follow_symlinks=True)
        for chunk in iter(lambda: source_handle.read(chunk_size), b""):
            digest.update(chunk)
            destination_handle.write(chunk)
    return digest.hexdigest()


def _dataset_chunks(source_dataset: h5py.Dataset, count: int) -> tuple[int]:
    if source_dataset.chunks and len(source_dataset.chunks) == 1:
        return (max(1, int(source_dataset.chunks[0])),)
    return (max(1, min(count if count else 1, 4096)),)


def copy_one(
    source_path: Path,
    output_path: Path,
    model: CalibrationModel | None,
    calibration_csv: Path,
    calibration_json: Path,
    overwrite_partial: bool = False,
) -> dict[str, Any]:
    """Copy one HDF5 atomically and append corrected_energy_kev when possible."""

    if output_path.exists() and not overwrite_partial:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    original_hash = _copy_bytes_with_hash(source_path, partial)
    status = "OK"
    record: dict[str, Any] = {
        "input_hdf5": str(source_path),
        "output_hdf5": str(output_path),
        "source_sha256": original_hash,
        "model_source": model.source if model else None,
        "status": status,
    }
    try:
        with h5py.File(partial, "r+") as handle:
            source_label = normalize_source(handle.attrs.get("source_label", ""))
            processing_status = str(handle.attrs.get("processing_status", ""))
            record["source"] = source_label
            record["processing_status"] = processing_status
            if processing_status not in {"", "OK"}:
                status = "SKIPPED_PROCESSING_STATUS"
                record["status"] = status
            elif model is None:
                status = "SKIPPED_NO_CALIBRATION_MODEL"
                record["status"] = status
            elif "reconstructed_energy_kev" not in handle:
                status = "SKIPPED_MISSING_ENERGY"
                record["status"] = status
            else:
                source_dataset = handle["reconstructed_energy_kev"]
                if source_dataset.ndim != 1:
                    raise ValueError("reconstructed_energy_kev must be one-dimensional")
                original_energy = np.asarray(source_dataset[:], dtype=np.float64)
                corrected = model.correct(original_energy).astype(np.float32)
                if "corrected_energy_kev" in handle:
                    del handle["corrected_energy_kev"]
                dataset = handle.create_dataset(
                    "corrected_energy_kev",
                    data=corrected,
                    dtype="f4",
                    chunks=_dataset_chunks(source_dataset, corrected.size),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                dataset.attrs.update(
                    {
                        "correction_formula": "(reconstructed_energy_kev - offset_kev) / scale",
                        "source_dataset": "reconstructed_energy_kev",
                        "calibration_source": model.source,
                        "calibration_model_label": model.model_label,
                        "calibration_scale": model.scale,
                        "calibration_offset_kev": model.offset_kev,
                        "calibration_fixed_intercept": model.fixed_intercept,
                        "calibration_pivot_intercept_kev": (
                            model.pivot_intercept_kev if model.pivot_intercept_kev is not None else np.nan
                        ),
                        "source_sha256": original_hash,
                        "software_version": SOFTWARE_VERSION,
                        "calibration_input_peak_csv": str(calibration_csv),
                        "calibration_input_results_json": str(calibration_json),
                        "assumptions": json.dumps(model.assumptions),
                    }
                )
                handle.attrs.update(
                    {
                        "corrected_energy_dataset": "corrected_energy_kev",
                        "corrected_energy_calibration_model": model.model_label,
                        "corrected_energy_calibration_scale": model.scale,
                        "corrected_energy_calibration_offset_kev": model.offset_kev,
                        "corrected_energy_source_sha256": original_hash,
                        "corrected_energy_software_version": SOFTWARE_VERSION,
                    }
                )
                record.update(
                    {
                        "input_entries": int(original_energy.size),
                        "finite_corrected_entries": int(np.count_nonzero(np.isfinite(corrected))),
                        "corrected_dataset": "corrected_energy_kev",
                        "scale": model.scale,
                        "offset_kev": model.offset_kev,
                    }
                )
            handle.flush()
        final_hash = sha256_file(partial)
        record["output_sha256"] = final_hash
        partial.replace(output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return record


def _is_safe_output_dir(path: Path) -> bool:
    resolved = path.resolve()
    return resolved.name == "waveform_hdf5_corrected" or resolved.name.startswith("waveform_hdf5_corrected_")


def create_copies(
    input_dir: Path,
    output_dir: Path,
    peak_csv: Path,
    results_json: Path,
    manifest_path: Path,
    overwrite: bool = False,
    min_free_fraction: float = 1.05,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    paths = sorted(input_dir.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 files found under {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty; use --overwrite: {output_dir}")
        if not _is_safe_output_dir(output_dir):
            raise ValueError("--overwrite requires an explicitly named corrected output directory")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = sum(path.stat().st_size for path in paths)
    free = shutil.disk_usage(output_dir).free
    if free < required * min_free_fraction:
        raise OSError(f"insufficient free disk space: need at least {required * min_free_fraction:.0f} bytes, have {free}")
    models = build_calibration_models(load_peak_rows(peak_csv))
    records: list[dict[str, Any]] = []
    for source_path in paths:
        relative = source_path.relative_to(input_dir)
        output_path = output_dir / relative
        with h5py.File(source_path, "r") as handle:
            source = normalize_source(handle.attrs.get("source_label", ""))
        records.append(
            copy_one(
                source_path,
                output_path,
                models.get(source),
                peak_csv,
                results_json,
            )
        )
    manifest = {
        "status": "OK",
        "software_version": SOFTWARE_VERSION,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "calibration_peak_csv": str(peak_csv.resolve()),
        "calibration_results_json": str(results_json.resolve()),
        "models": {source: model.as_json() for source, model in sorted(models.items())},
        "file_count": len(records),
        "records": records,
        "assumptions": [
            "Original HDF5 files are never modified.",
            "Calibration models use OK peak fits only and are source-level, not per-file drift corrections.",
            "Corrected energies are calculated in float64 and stored as float32.",
            "NaN and infinity values are preserved through the affine operation and float32 storage.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("processed_data/waveform_hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data/waveform_hdf5_corrected"))
    parser.add_argument("--peak-csv", type=Path, default=Path("outputs/gain_drift/peak_positions.csv"))
    parser.add_argument("--results-json", type=Path, default=Path("outputs/gain_drift/gain_drift_results.json"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/gain_drift/corrected_hdf5_manifest.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = create_copies(
        args.input_dir,
        args.output_dir,
        args.peak_csv,
        args.results_json,
        args.manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "file_count": manifest["file_count"], "output": manifest["output_directory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
