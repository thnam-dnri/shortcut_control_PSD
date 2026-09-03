#!/usr/bin/env python3
"""Fit one shared exponential threshold shape with an optional low-energy rise.

The fitted model is

    T(E; G) = C_G + A * exp(-max(E - E_0, 0) / tau),

where A and tau are shared by every P/B-improvement target, and only the
target-specific constant C_G changes. With ``--low-energy-power 3``, the
function instead rises cubically from T(0)=0 to its maximum at E_0, then uses
the shared exponential decay above E_0. The input anchor CSV is produced by
fit_th232_threshold_curves_with_uncertainty.py and supplies the 95% bootstrap
intervals used as fit weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHOR_CSV = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409_uncertainty/th232_threshold_anchor_uncertainty.csv"
)
DEFAULT_RETENTION_CSV = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409_uncertainty/th232_2614_retention_diagnostic.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409_hinged_exponential"
)
TARGET_GAINS_PCT = (5, 10, 20, 30, 45)
FIT_ORIGIN_KEV = 238.90303771420758
DEFAULT_HINGE_KEV = 230.0


def relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "target_pb_gain_percent",
        "observed_centroid_kev",
        "target_threshold",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Anchor CSV is missing required fields: {sorted(required)}")
    return rows


def shared_exponential_values(
    energy_kev: np.ndarray | float,
    constants: np.ndarray,
    amplitude: float,
    tau_kev: float,
    target_indices: np.ndarray,
    origin_kev: float = FIT_ORIGIN_KEV,
    low_energy_power: float | None = None,
) -> np.ndarray:
    energy = np.asarray(energy_kev, dtype=np.float64)
    high_energy = constants[target_indices] + amplitude * np.exp(
        -np.maximum(energy - origin_kev, 0.0) / tau_kev
    )
    if low_energy_power is None:
        return high_energy
    if low_energy_power <= 0.0:
        raise ValueError("low_energy_power must be positive")
    peak = constants[target_indices] + amplitude
    low_energy = peak * np.power(
        np.clip(energy, 0.0, None) / origin_kev,
        low_energy_power,
    )
    return np.where(energy <= origin_kev, low_energy, high_energy)


def fit_diagnostics(
    target_pct: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
    parameter_count: int,
) -> dict[str, Any]:
    residual = observed - predicted
    standardized = residual / sigma
    chi2 = float(np.sum(np.square(standardized)))
    return {
        "target_pb_gain_percent": float(target_pct),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "weighted_rmse": float(np.sqrt(np.mean(np.square(standardized)))),
        "mae": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "chi2": chi2,
        "aic": float(chi2 + 2.0 * parameter_count),
        "bic": float(chi2 + parameter_count * np.log(observed.size)),
    }


def fit_shared_exponential(
    rows: list[dict[str, Any]],
    hinge_kev: float = DEFAULT_HINGE_KEV,
    low_energy_power: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, np.ndarray]]:
    target_values = np.asarray(
        [int(float(row["target_pb_gain_percent"])) for row in rows], dtype=np.int64
    )
    target_indices = np.asarray(
        [TARGET_GAINS_PCT.index(int(value)) for value in target_values],
        dtype=np.int64,
    )
    energy = np.asarray(
        [float(row["observed_centroid_kev"]) for row in rows], dtype=np.float64
    )
    observed = np.asarray(
        [float(row["target_threshold"]) for row in rows], dtype=np.float64
    )
    ci_low = np.asarray(
        [float(row["bootstrap_ci_low"]) for row in rows], dtype=np.float64
    )
    ci_high = np.asarray(
        [float(row["bootstrap_ci_high"]) for row in rows], dtype=np.float64
    )
    sigma = np.maximum((ci_high - ci_low) / 2.0, 0.001)
    origin_kev = float(hinge_kev)
    if origin_kev <= 0.0 or origin_kev >= float(np.max(energy)):
        raise ValueError(
            f"Hinge must be positive and below the highest anchor energy; got {origin_kev}"
        )

    def model(
        values: np.ndarray,
        c5: float,
        c10: float,
        c20: float,
        c30: float,
        c45: float,
        amplitude: float,
        tau_kev: float,
    ) -> np.ndarray:
        constants = np.asarray([c5, c10, c20, c30, c45], dtype=np.float64)
        return shared_exponential_values(
            values,
            constants,
            amplitude,
            tau_kev,
            target_indices,
            origin_kev,
            low_energy_power,
        )

    high_energy_values = np.asarray(
        [
            np.mean(observed[(target_values == target) & (energy >= np.percentile(energy, 75))])
            for target in TARGET_GAINS_PCT
        ],
        dtype=np.float64,
    )
    initial_amplitude = 0.15
    initial_tau = 100.0
    initial_constants = high_energy_values - initial_amplitude * np.exp(
        -(
            np.max(energy) - origin_kev
        )
        / initial_tau
    )
    initial_constants = np.clip(initial_constants, 0.0, 1.0)
    parameters, covariance = curve_fit(
        model,
        energy,
        observed,
        p0=np.r_[initial_constants, initial_amplitude, initial_tau],
        sigma=sigma,
        absolute_sigma=True,
        bounds=(
            np.r_[np.zeros(len(TARGET_GAINS_PCT)), 0.0, 10.0],
            np.r_[np.ones(len(TARGET_GAINS_PCT)), 2.0, 10000.0],
        ),
        maxfev=100000,
    )
    predicted = model(energy, *parameters)
    global_diagnostics = fit_diagnostics(
        0,
        observed,
        predicted,
        sigma,
        len(parameters),
    )
    global_diagnostics["target_pb_gain_percent"] = "all"
    global_diagnostics["parameter_count"] = len(parameters)

    per_target_rows: list[dict[str, Any]] = []
    predictions_by_target: dict[int, np.ndarray] = {}
    constants = np.asarray(parameters[: len(TARGET_GAINS_PCT)], dtype=np.float64)
    amplitude = float(parameters[-2])
    tau_kev = float(parameters[-1])
    for target_index, target in enumerate(TARGET_GAINS_PCT):
        mask = target_values == target
        target_predicted = predicted[mask]
        target_observed = observed[mask]
        target_sigma = sigma[mask]
        per_target_rows.append(
            fit_diagnostics(
                target,
                target_observed,
                target_predicted,
                target_sigma,
                len(parameters),
            )
        )
        target_energy = energy[mask]
        predictions_by_target[target] = np.asarray(
            shared_exponential_values(
                target_energy,
                constants,
                amplitude,
                tau_kev,
                np.full(target_energy.shape, target_index, dtype=np.int64),
                origin_kev,
                low_energy_power,
            ),
            dtype=np.float64,
        )

    fit_summary = {
        "origin_kev": origin_kev,
        "hinge_energy_kev": origin_kev,
        "low_energy_power": (
            float(low_energy_power) if low_energy_power is not None else None
        ),
        "constants": {
            str(target): float(constants[index])
            for index, target in enumerate(TARGET_GAINS_PCT)
        },
        "amplitude": amplitude,
        "tau_kev": tau_kev,
        "parameter_vector": [float(value) for value in parameters],
        "parameter_standard_errors": [
            float(value) for value in np.sqrt(np.maximum(np.diag(covariance), 0.0))
        ],
        "global_diagnostics": global_diagnostics,
    }
    return fit_summary, per_target_rows, predictions_by_target


def load_retention(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    return (
        np.asarray([float(row["score_threshold"]) for row in records]),
        np.asarray([float(row["th232_2614_roi_retention"]) for row in records]),
    )


def plot_results(
    output_path: Path,
    rows: list[dict[str, Any]],
    fit_summary: dict[str, Any],
    retention_scores: np.ndarray,
    retention_values: np.ndarray,
) -> None:
    colors = dict(
        zip(
            TARGET_GAINS_PCT,
            plt.cm.viridis(np.linspace(0.1, 0.9, len(TARGET_GAINS_PCT))),
        )
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": (1.2, 1.0)},
        constrained_layout=True,
    )
    threshold_axis, retention_axis = axes
    origin_kev = float(fit_summary["origin_kev"])
    constants = np.asarray(
        [fit_summary["constants"][str(target)] for target in TARGET_GAINS_PCT],
        dtype=np.float64,
    )
    amplitude = float(fit_summary["amplitude"])
    tau_kev = float(fit_summary["tau_kev"])
    low_energy_power = fit_summary.get("low_energy_power")
    low_energy_power = (
        float(low_energy_power) if low_energy_power is not None else None
    )
    anchor_energy = np.asarray(
        [float(row["observed_centroid_kev"]) for row in rows], dtype=np.float64
    )
    plot_min_kev = max(0.0, min(float(np.min(anchor_energy)) - 20.0, origin_kev - 40.0))
    dense_energy = np.linspace(plot_min_kev, 1500.0, 1200)
    for target_index, target in enumerate(TARGET_GAINS_PCT):
        color = colors[target]
        target_rows = [
            row
            for row in rows
            if int(float(row["target_pb_gain_percent"])) == target
        ]
        target_rows.sort(key=lambda row: float(row["observed_centroid_kev"]))
        energy = np.asarray(
            [float(row["observed_centroid_kev"]) for row in target_rows]
        )
        threshold = np.asarray(
            [float(row["target_threshold"]) for row in target_rows]
        )
        lower = np.asarray([float(row["bootstrap_ci_low"]) for row in target_rows])
        upper = np.asarray([float(row["bootstrap_ci_high"]) for row in target_rows])
        threshold_axis.errorbar(
            energy,
            threshold,
            yerr=np.vstack((threshold - lower, upper - threshold)),
            fmt="o",
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.75,
            label="_nolegend_",
            zorder=3,
        )
        fitted = shared_exponential_values(
            dense_energy,
            constants,
            amplitude,
            tau_kev,
            np.full(dense_energy.shape, target_index, dtype=np.int64),
            origin_kev,
            low_energy_power,
        )
        threshold_axis.plot(
            dense_energy,
            np.clip(fitted, 0.0, 1.0),
            color=color,
            linewidth=2.1,
            label=f"+{target}% P/B",
        )
    threshold_axis.axvline(1460.830, color="0.35", linestyle="--", linewidth=1.0)
    threshold_axis.axvline(origin_kev, color="tab:orange", linestyle=":", linewidth=1.0)
    threshold_axis.text(
        origin_kev,
        0.97,
        "hinge / peak",
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        color="tab:orange",
    )
    threshold_axis.set_xlim(plot_min_kev, 1500.0)
    threshold_axis.set_ylim(0.0, 1.0)
    threshold_axis.set_xlabel("Corrected energy (keV)")
    threshold_axis.set_ylabel("DS-CNN score threshold T(E)")
    threshold_axis.set_title(
        (
            "Cubic-rise/exponential-decay threshold shape; target dependence only in C(G)"
            if low_energy_power is not None
            else "Shared exponential threshold shape; target dependence only in C(G)"
        ),
        fontweight="bold",
    )
    formula = (
        rf"T(E;G) = (C(G) + {amplitude:.3f})(E/{origin_kev:.1f})^{low_energy_power:.0f}, E <= {origin_kev:.1f}; "
        rf"C(G) + {amplitude:.3f} exp[-(E - {origin_kev:.1f})/{tau_kev:.1f}], E > {origin_kev:.1f}"
        if low_energy_power is not None
        else rf"T(E;G) = C(G) + {amplitude:.3f} exp[-max(E - {origin_kev:.1f}, 0)/{tau_kev:.1f}]"
    )
    threshold_axis.text(
        0.02,
        0.04,
        formula,
        transform=threshold_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    threshold_axis.grid(alpha=0.25)
    threshold_axis.legend(ncol=3, fontsize=9, loc="upper right")

    retention_axis.plot(
        retention_scores,
        retention_values,
        color="black",
        linewidth=1.8,
        label="2614.5-keV ROI retention",
    )
    for target_index, target in enumerate(TARGET_GAINS_PCT):
        target_rows = [
            row
            for row in rows
            if int(float(row["target_pb_gain_percent"])) == target
        ]
        boundary_row = max(
            target_rows,
            key=lambda row: float(row["observed_centroid_kev"]),
        )
        boundary = float(boundary_row["target_threshold"])
        retention = float(
            retention_values[np.argmin(np.abs(retention_scores - boundary))]
        )
        color = colors[target]
        retention_axis.axvline(boundary, color=color, linestyle="--", linewidth=0.9)
        retention_axis.scatter([boundary], [retention], color=color, s=30, zorder=3)
        retention_axis.annotate(
            f"+{target}%",
            (boundary, retention),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    retention_axis.set_xlim(0.0, 0.90)
    retention_axis.set_ylim(0.0, 1.05)
    retention_axis.set_xlabel("Global score threshold")
    retention_axis.set_ylabel("2614-keV ROI retention")
    retention_axis.set_title(
        "2614.5-keV retention-only diagnostic (not used for P/B fitting)",
        fontweight="bold",
    )
    retention_axis.grid(alpha=0.25)
    retention_axis.legend(fontsize=9, loc="upper right")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def make_report(
    anchor_path: Path,
    output_dir: Path,
    fit_summary: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    per_target_rows: list[dict[str, Any]],
) -> str:
    diagnostics = fit_summary["global_diagnostics"]
    low_energy_power = fit_summary.get("low_energy_power")
    if low_energy_power is not None:
        model_description = (
            f"`T(E;G) = (C(G) + A) (E/{fit_summary['origin_kev']:.3f})^{float(low_energy_power):.0f}` for E <= {fit_summary['origin_kev']:.3f} keV, "
            f"then `C(G) + A exp(-(E - {fit_summary['origin_kev']:.3f}) / tau)` above the peak."
        )
        boundary_description = (
            f"The low-energy branch is a fixed power-{float(low_energy_power):.0f} rise from T(0)=0 to the maximum at {fit_summary['origin_kev']:.3f} keV."
        )
    else:
        model_description = (
            f"`T(E;G) = C(G) + {fit_summary['amplitude']:.6f} exp(-max(E - {fit_summary['origin_kev']:.3f}, 0) / {fit_summary['tau_kev']:.6f})`"
        )
        boundary_description = (
            f"For E <= {fit_summary['origin_kev']:.3f} keV, the threshold is held at the plateau value C(G) + A; above the hinge it decays exponentially."
        )
    lines = [
        "# Shared exponential Th-232 threshold fit",
        "",
        "The model uses one common shape for all P/B targets; only the target-specific constant C(G) changes.",
        "",
        f"- Input anchor uncertainty table: `{relative(anchor_path)}`",
        f"- Plot: `{relative(output_dir / 'th232_threshold_shared_exponential.png')}`",
        f"- P/B domain: {min(float(row['observed_centroid_kev']) for row in anchor_rows):.1f}--{max(float(row['observed_centroid_kev']) for row in anchor_rows):.1f} keV observed centroids.",
        "- 2614.5 keV remains retention-only and is not used in the fit.",
        "- Error bars are the previously calculated 95% local event-bootstrap intervals.",
        "",
        "## Fitted model",
        "",
        model_description,
        "",
        boundary_description,
        "",
        "| Target | C(G) | RMSE | Weighted RMSE |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_target_rows:
        target = int(row["target_pb_gain_percent"])
        lines.append(
            f"| +{target}% | {fit_summary['constants'][str(target)]:.6f} | {row['rmse']:.4f} | {row['weighted_rmse']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Global RMSE: {diagnostics['rmse']:.4f}; weighted RMSE: {diagnostics['weighted_rmse']:.2f}; AIC: {diagnostics['aic']:.2f}; BIC: {diagnostics['bic']:.2f}.",
            "",
            "This is a constrained descriptive fit to the historical Th-232 optimization points. It intentionally cannot reproduce every high-energy local fluctuation or serve as external validation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-csv", type=Path, default=DEFAULT_ANCHOR_CSV)
    parser.add_argument("--retention-csv", type=Path, default=DEFAULT_RETENTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hinge-kev",
        type=float,
        default=DEFAULT_HINGE_KEV,
        help="Fixed low-energy hinge or peak energy in keV (default: 230)",
    )
    parser.add_argument(
        "--low-energy-power",
        type=float,
        default=None,
        help="Use a power-law rise from T(0)=0 below the hinge/peak (for example, 3)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = args.anchor_csv.resolve()
    retention_path = args.retention_csv.resolve()
    rows = load_rows(anchor_path)
    fit_summary, per_target_rows, _ = fit_shared_exponential(
        rows,
        args.hinge_kev,
        args.low_energy_power,
    )
    retention_scores, retention_values = load_retention(retention_path)

    constants_csv = output_dir / "th232_shared_exponential_constants.csv"
    fit_csv = output_dir / "th232_shared_exponential_fit_metrics.csv"
    curve_csv = output_dir / "th232_threshold_shared_exponential_1kev.csv"
    retention_csv = output_dir / "th232_2614_retention_diagnostic.csv"
    plot_path = output_dir / "th232_threshold_shared_exponential.png"
    report_json_path = output_dir / "th232_threshold_shared_exponential_report.json"
    report_path = output_dir / "report.md"

    constants_rows = [
        {
            "target_pb_gain_percent": float(target),
            "constant_C": float(fit_summary["constants"][str(target)]),
            "shared_amplitude": float(fit_summary["amplitude"]),
            "shared_tau_kev": float(fit_summary["tau_kev"]),
            "origin_kev": float(fit_summary["origin_kev"]),
            "low_energy_power": fit_summary["low_energy_power"],
        }
        for target in TARGET_GAINS_PCT
    ]
    fit_rows = [fit_summary["global_diagnostics"], *per_target_rows]
    anchor_energies = np.asarray(
        [float(row["observed_centroid_kev"]) for row in rows], dtype=np.float64
    )
    dense_energy = np.arange(
        float(np.floor(np.min(anchor_energies))),
        float(np.ceil(np.max(anchor_energies))) + 1.0,
        1.0,
    )
    curve_rows = [{"energy_kev": float(value)} for value in dense_energy]
    constants = np.asarray(
        [fit_summary["constants"][str(target)] for target in TARGET_GAINS_PCT],
        dtype=np.float64,
    )
    for target_index, target in enumerate(TARGET_GAINS_PCT):
        values = shared_exponential_values(
            dense_energy,
            constants,
            float(fit_summary["amplitude"]),
            float(fit_summary["tau_kev"]),
            np.full(dense_energy.shape, target_index, dtype=np.int64),
            float(fit_summary["origin_kev"]),
            fit_summary["low_energy_power"],
        )
        for row, value in zip(curve_rows, values):
            row[f"threshold_plus_{target}pct"] = float(value)

    write_csv(constants_csv, constants_rows)
    write_csv(fit_csv, fit_rows)
    write_csv(curve_csv, curve_rows)
    write_csv(
        retention_csv,
        [
            {
                "score_threshold": float(score),
                "th232_2614_roi_retention": float(retention),
            }
            for score, retention in zip(retention_scores, retention_values)
        ],
    )
    plot_results(
        plot_path,
        rows,
        fit_summary,
        retention_scores,
        retention_values,
    )

    report = {
        "schema_version": 1,
        "status": "TH232_SHARED_EXPONENTIAL_THRESHOLD_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_csv": relative(anchor_path),
        "retention_csv_input": relative(retention_path),
        "target_gains_percent": list(TARGET_GAINS_PCT),
        "hinge_energy_kev": float(args.hinge_kev),
        "low_energy_power": fit_summary["low_energy_power"],
        "fit_summary": fit_summary,
        "per_target_fit_metrics": per_target_rows,
        "artifacts": {
            "plot": relative(plot_path),
            "constants_csv": relative(constants_csv),
            "fit_csv": relative(fit_csv),
            "curve_csv": relative(curve_csv),
            "retention_csv": relative(retention_csv),
        },
        "claim_boundary": "One shared exponential shape with target-specific constants is a constrained descriptive fit to historical Th-232 optimization points, not external validation.",
    }
    report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        make_report(anchor_path, output_dir, fit_summary, rows, per_target_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "artifacts": [
                    plot_path.name,
                    constants_csv.name,
                    fit_csv.name,
                    curve_csv.name,
                    retention_csv.name,
                    report_json_path.name,
                    report_path.name,
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
