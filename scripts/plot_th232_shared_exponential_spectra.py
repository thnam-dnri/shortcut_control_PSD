#!/usr/bin/env python3
"""Plot Th-232 spectra after applying the shared exponential thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SCORE_CACHE = (
    PROJECT_ROOT
    / "outputs/experiments/th232_all_ba_ds_cnn_threshold_20260823/th232_all_ba_ds_cnn_scores.h5"
)
DEFAULT_SHARED_FIT_REPORT = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_threshold_curves_20260824_with_209_300_409_hinged_exponential/th232_threshold_shared_exponential_report.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/th232_revised_pb_spectra_20260824_with_209_300_409_hinged_exponential"
)
TARGET_GAINS_PCT = (5, 10, 20, 30, 45)
COLORS = {
    "No cut": "black",
    "+5% P/B": "#1f77b4",
    "+10% P/B": "#2ca02c",
    "+20% P/B": "#9467bd",
    "+30% P/B": "#ff7f0e",
    "+45% P/B": "#d62728",
}

from scripts.evaluate_th232_o2_3p_energy_threshold import (  # noqa: E402
    ENERGY_CENTERS,
    ENERGY_EDGES,
)
from scripts.plot_th232_flat_pb_spectra import (  # noqa: E402
    identify_th232_no_filter_peaks,
    plot_annotated_no_filter_spectrum,
)
from scripts.plot_th232_revised_pb_threshold_curves import (  # noqa: E402
    PB_ANCHORS,
    RETENTION_ONLY_ANCHOR,
    fit_anchor_window,
    json_scalar,
    line_aware_metrics,
    relative,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_label(target_pct: int) -> str:
    return f"+{target_pct}% P/B"


def load_shared_fit(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "TH232_SHARED_EXPONENTIAL_THRESHOLD_COMPLETE":
        raise ValueError(f"Unexpected shared-fit report status: {report.get('status')}")
    fit_summary = report["fit_summary"]
    for target in TARGET_GAINS_PCT:
        if str(target) not in fit_summary["constants"]:
            raise ValueError(f"Shared-fit report lacks +{target}% constant")
    return report


def shared_thresholds(
    energy: np.ndarray,
    fit_summary: dict[str, Any],
    target_pct: int,
) -> np.ndarray:
    origin = float(fit_summary["origin_kev"])
    constant = float(fit_summary["constants"][str(target_pct)])
    amplitude = float(fit_summary["amplitude"])
    tau = float(fit_summary["tau_kev"])
    energy_array = np.asarray(energy, dtype=np.float64)
    high_energy = constant + amplitude * np.exp(
        -np.maximum(energy_array - origin, 0.0) / tau
    )
    low_energy_power = fit_summary.get("low_energy_power")
    if low_energy_power is None:
        threshold = high_energy
    else:
        peak = constant + amplitude
        low_energy = peak * np.power(
            np.clip(energy_array, 0.0, None) / origin,
            float(low_energy_power),
        )
        threshold = np.where(energy_array <= origin, low_energy, high_energy)
    return np.clip(threshold, 0.0, 1.0)


def plot_combined_spectra(
    output_path: Path,
    histograms: dict[str, np.ndarray],
    windows: list[tuple[Any, str]],
    fit_summary: dict[str, Any],
    pb_domain_kev: tuple[float, float],
) -> None:
    names = ["No cut", *[target_label(target) for target in TARGET_GAINS_PCT]]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 11),
        sharex=True,
        constrained_layout=True,
    )
    for name in names:
        color = COLORS[name]
        for axis in axes:
            axis.step(
                ENERGY_CENTERS,
                histograms[name],
                where="mid",
                linewidth=0.95 if name == "No cut" else 0.85,
                color=color,
                alpha=0.95 if name == "No cut" else 0.82,
                label=name,
            )

    for window, mode in windows:
        color = "#c0392b" if mode != "diagnostic_only" else "#555555"
        for axis in axes:
            axis.axvline(
                window.centroid_kev,
                color=color,
                linewidth=0.5,
                linestyle="--",
                alpha=0.35,
            )
    pb_min_kev, pb_max_kev = pb_domain_kev
    for axis in axes:
        axis.axvspan(pb_min_kev, pb_max_kev, color="#4c78a8", alpha=0.025)
        axis.axvline(pb_min_kev, color="#4c78a8", linewidth=0.6, linestyle=":")
        axis.axvline(pb_max_kev, color="#4c78a8", linewidth=0.8, linestyle=":")
        axis.grid(alpha=0.25, which="both")
        axis.set_xlim(0.0, 3000.0)

    axes[0].set_ylabel("Counts / 1 keV")
    amplitude = float(fit_summary["amplitude"])
    tau = float(fit_summary["tau_kev"])
    low_energy_power = fit_summary.get("low_energy_power")
    if low_energy_power is None:
        spectrum_title = "hinged shared-exponential thresholds"
        scale_title = "Log scale — hinged shared function evaluated through the full plotted range"
        formula = rf"T(E;G) = C(G) + {amplitude:.3f} exp[-max(E - {fit_summary['origin_kev']:.1f}, 0)/{tau:.1f}]"
    else:
        spectrum_title = "cubic-rise/exponential-decay thresholds"
        scale_title = "Log scale — cubic-rise/exponential-decay function evaluated through the full plotted range"
        formula = (
            rf"T(E;G) = (C(G) + {amplitude:.3f})(E/{fit_summary['origin_kev']:.1f})^{float(low_energy_power):.0f}, E <= {fit_summary['origin_kev']:.1f}; "
            rf"C(G) + {amplitude:.3f} exp[-(E - {fit_summary['origin_kev']:.1f})/{tau:.1f}], E > {fit_summary['origin_kev']:.1f}"
        )
    axes[0].set_title(
        f"Th-232 spectra with {spectrum_title} "
        "(P/B anchors 209--1460 keV; 2614-keV retention-only)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(ncol=3, fontsize=9, loc="upper right", framealpha=0.9)
    axes[1].set_ylabel("Counts / 1 keV")
    axes[1].set_xlabel("Corrected energy (keV)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.8, top=max(histograms["No cut"]) * 2.0)
    axes[1].set_title(scale_title, fontsize=11)
    axes[0].text(
        0.01,
        0.03,
        formula,
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_individual_spectra(
    output_dir: Path,
    histograms: dict[str, np.ndarray],
    windows: list[tuple[Any, str]],
) -> None:
    for name, histogram in histograms.items():
        color = COLORS[name]
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(15, 10),
            sharex=True,
            constrained_layout=True,
        )
        for axis in axes:
            axis.step(
                ENERGY_CENTERS,
                histogram,
                where="mid",
                linewidth=0.9,
                color=color,
                label=f"Th-232 {name} ({int(histogram.sum()):,} events)",
            )
            for window, _ in windows:
                axis.axvline(
                    window.centroid_kev,
                    color="0.45",
                    linewidth=0.5,
                    linestyle="--",
                    alpha=0.4,
                )
            axis.grid(alpha=0.25, which="both")
            axis.set_xlim(0.0, 3000.0)
            axis.legend(loc="upper right", framealpha=0.9)
        axes[0].set_ylabel("Counts / 1 keV")
        axes[0].set_title(
            f"Th-232 energy spectrum — energy-dependent threshold {name}",
            fontweight="bold",
        )
        axes[1].set_ylabel("Counts / 1 keV")
        axes[1].set_xlabel("Corrected energy (keV)")
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=0.8, top=max(histograms["No cut"]) * 2.0)
        safe_name = (
            name.lower()
            .replace("+", "plus_")
            .replace("%", "")
            .replace("/", "_")
            .replace(" ", "_")
        )
        figure.savefig(
            output_dir / f"th232_shared_exponential_spectrum_{safe_name}.png",
            dpi=200,
        )
        plt.close(figure)


def plot_peak_zooms(
    output_path: Path,
    histograms: dict[str, np.ndarray],
    windows: list[tuple[Any, str]],
) -> None:
    names = ["No cut", *[target_label(target) for target in TARGET_GAINS_PCT]]
    figure, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    flat_axes = list(axes.flat)
    for axis, (window, mode) in zip(flat_axes, windows):
        half_width = max(18.0, 6.0 * window.sigma_kev)
        selected = (ENERGY_CENTERS >= window.centroid_kev - half_width) & (
            ENERGY_CENTERS <= window.centroid_kev + half_width
        )
        for name in names:
            axis.step(
                ENERGY_CENTERS[selected],
                histograms[name][selected],
                where="mid",
                linewidth=0.8,
                color=COLORS[name],
                label=name,
            )
        if mode != "diagnostic_only":
            axis.axvspan(window.roi_low_kev, window.roi_high_kev, color="#984ea3", alpha=0.12)
            axis.axvspan(window.left_low_kev, window.left_high_kev, color="0.5", alpha=0.10)
            axis.axvspan(window.right_low_kev, window.right_high_kev, color="0.5", alpha=0.10)
        axis.set_title(
            f"{window.reference_kev:g} keV ({window.centroid_kev:.2f} observed)",
            fontsize=9,
            fontweight="bold",
        )
        axis.set_xlabel("Energy (keV)", fontsize=8)
        axis.set_ylabel("Counts / 1 keV", fontsize=8)
        axis.grid(alpha=0.25)
    for axis in flat_axes[len(windows):]:
        axis.axis("off")
    flat_axes[0].legend(fontsize=6.5, loc="upper right")
    figure.suptitle(
        "Th-232 peak windows after energy-dependent score thresholds",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--th232-score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument(
        "--shared-fit-report", type=Path, default=DEFAULT_SHARED_FIT_REPORT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.th232_score_cache.resolve()
    fit_report_path = args.shared_fit_report.resolve()
    fit_report = load_shared_fit(fit_report_path)
    fit_summary = fit_report["fit_summary"]

    with h5py.File(cache_path, "r") as handle:
        energy = np.asarray(handle["corrected_energy_kev"], dtype=np.float64)
        scores = np.asarray(handle["score"], dtype=np.float64)
        cache_metadata = {
            str(key): json_scalar(value) for key, value in handle.attrs.items()
        }

    base_histogram = np.histogram(energy, ENERGY_EDGES)[0]
    smoothed = gaussian_filter1d(base_histogram.astype(np.float64), 1.5)
    specs = [*PB_ANCHORS, RETENTION_ONLY_ANCHOR]
    fitted_windows = [
        (spec, fit_anchor_window(base_histogram, spec.reference_kev, smoothed))
        for spec in specs
    ]
    pb_windows = fitted_windows[: len(PB_ANCHORS)]
    retention_window = fitted_windows[-1][1]
    pb_domain_kev = (
        float(min(window.centroid_kev for _, window in pb_windows)),
        float(max(window.centroid_kev for _, window in pb_windows)),
    )

    histograms: dict[str, np.ndarray] = {"No cut": base_histogram}
    threshold_rows: list[dict[str, Any]] = []
    anchor_metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    threshold_grid_rows: list[dict[str, Any]] = []
    threshold_grid_energy = ENERGY_CENTERS.copy()

    base_metrics = {
        spec.reference_kev: line_aware_metrics(
            base_histogram, window, spec.sideband_mode
        )
        for (spec, window) in pb_windows
    }
    base_2614_count = float(
        np.count_nonzero(
            (energy >= retention_window.roi_low_kev)
            & (energy <= retention_window.roi_high_kev)
        )
    )

    for target in TARGET_GAINS_PCT:
        threshold_values = shared_thresholds(energy, fit_summary, target)
        selected_mask = scores >= threshold_values
        histogram = np.histogram(energy[selected_mask], ENERGY_EDGES)[0]
        label = target_label(target)
        histograms[label] = histogram
        threshold_at_grid = shared_thresholds(threshold_grid_energy, fit_summary, target)
        threshold_grid_rows.extend(
            {
                "energy_kev": float(e_value),
                "target_pb_gain_percent": float(target),
                "shared_threshold": float(t_value),
            }
            for e_value, t_value in zip(threshold_grid_energy, threshold_at_grid)
        )
        threshold_rows.append(
            {
                "target_pb_gain_percent": float(target),
                "threshold_at_pb_min_kev": float(pb_domain_kev[0]),
                "threshold_at_pb_min": float(
                    shared_thresholds(np.asarray([pb_domain_kev[0]]), fit_summary, target)[0]
                ),
                "threshold_at_hinge_kev": float(fit_summary["origin_kev"]),
                "threshold_at_hinge": float(
                    shared_thresholds(
                        np.asarray([fit_summary["origin_kev"]]), fit_summary, target
                    )[0]
                ),
                "threshold_at_238_9kev": float(
                    shared_thresholds(np.asarray([238.903]), fit_summary, target)[0]
                ),
                "threshold_at_1460_4kev": float(
                    shared_thresholds(np.asarray([1460.438]), fit_summary, target)[0]
                ),
                "threshold_at_2614_5kev": float(
                    shared_thresholds(np.asarray([2614.533]), fit_summary, target)[0]
                ),
            }
        )
        for spec, window in pb_windows:
            metrics = line_aware_metrics(histogram, window, spec.sideband_mode)
            base = base_metrics[spec.reference_kev]
            raw_gain = metrics["peak_to_background"] / base["peak_to_background"]
            gain = float(raw_gain) if np.isfinite(raw_gain) else None
            retention = metrics["net_peak_counts"] / base["net_peak_counts"]
            anchor_metric_rows.append(
                {
                    "target_pb_gain_percent": float(target),
                    "reference_energy_kev": float(spec.reference_kev),
                    "label": spec.label,
                    "observed_centroid_kev": float(window.centroid_kev),
                    "sideband_mode": spec.sideband_mode,
                    "shared_threshold_at_anchor": float(
                        shared_thresholds(
                            np.asarray([window.centroid_kev]), fit_summary, target
                        )[0]
                    ),
                    "continuous_pb_improvement": gain,
                    "continuous_net_peak_retention": float(retention),
                    "base_peak_to_background": float(base["peak_to_background"]),
                }
            )
        roi_mask = (energy >= retention_window.roi_low_kev) & (
            energy <= retention_window.roi_high_kev
        )
        retention_2614 = (
            float(np.count_nonzero(selected_mask & roi_mask)) / base_2614_count
            if base_2614_count > 0.0
            else float("nan")
        )
        target_metrics = [
            row["continuous_pb_improvement"]
            for row in anchor_metric_rows
            if row["target_pb_gain_percent"] == float(target)
            and row["continuous_pb_improvement"] is not None
            and np.isfinite(row["continuous_pb_improvement"])
        ]
        summary_rows.append(
            {
                "target_pb_gain_percent": float(target),
                "selected_events": int(np.count_nonzero(selected_mask)),
                "selected_fraction": float(np.mean(selected_mask)),
                "2614_roi_retention": retention_2614,
                "minimum_anchor_pb_improvement": (
                    float(min(target_metrics)) if target_metrics else None
                ),
                "maximum_anchor_pb_improvement": (
                    float(max(target_metrics)) if target_metrics else None
                ),
            }
        )

    spectrum_rows: list[dict[str, Any]] = []
    for index, energy_value in enumerate(ENERGY_CENTERS):
        row = {"energy_kev": float(energy_value)}
        for name, histogram in histograms.items():
            safe_name = (
                name.lower().replace("+", "plus_").replace("%", "").replace(" ", "_")
            )
            row[f"counts_{safe_name}"] = int(histogram[index])
        spectrum_rows.append(row)

    windows_for_plot = [(window, spec.sideband_mode) for spec, window in fitted_windows]
    combined_path = output_dir / "th232_shared_exponential_spectra.png"
    peak_zooms_path = output_dir / "th232_shared_exponential_peak_zooms.png"
    plot_combined_spectra(
        combined_path,
        histograms,
        windows_for_plot,
        fit_summary,
        pb_domain_kev,
    )
    plot_peak_zooms(peak_zooms_path, histograms, windows_for_plot)
    plot_individual_spectra(output_dir, histograms, windows_for_plot)
    annotated_peak_rows = identify_th232_no_filter_peaks(base_histogram)
    annotated_path = plot_annotated_no_filter_spectrum(
        output_dir, base_histogram, annotated_peak_rows
    )
    annotated_csv = output_dir / "th232_no_filter_identifiable_peaks.csv"
    write_csv(annotated_csv, annotated_peak_rows)

    spectrum_csv = output_dir / "th232_shared_exponential_spectra_1kev.csv"
    threshold_csv = output_dir / "th232_shared_exponential_thresholds_1kev.csv"
    target_threshold_csv = output_dir / "th232_shared_exponential_thresholds_by_target.csv"
    anchor_metrics_csv = output_dir / "th232_shared_exponential_anchor_metrics.csv"
    write_csv(spectrum_csv, spectrum_rows)
    write_csv(threshold_csv, threshold_grid_rows)
    write_csv(target_threshold_csv, threshold_rows)
    write_csv(anchor_metrics_csv, anchor_metric_rows)

    low_energy_power = fit_summary.get("low_energy_power")
    if low_energy_power is None:
        model_description = (
            f"The spectra use one shared exponential decay shape with a constant threshold below the {fit_summary['origin_kev']:.3f}-keV hinge; only C(G) changes between targets."
        )
        boundary_description = (
            f"- Below the {fit_summary['origin_kev']:.3f}-keV hinge, the threshold is constant; the function is evaluated over 0--3200 keV for plotting."
        )
        claim_boundary = (
            "The hinged shared exponential is fitted to historical Th-232 optimization points from the 209-keV anchor through 1460 keV. "
            "Below the hinge the threshold is held constant by construction; outside the P/B anchor domain the spectrum is an application diagnostic, and 2614.5 keV is reported by retention only."
        )
    else:
        model_description = (
            f"The spectra use a shared power-{float(low_energy_power):.0f} rise from T(0)=0 to {fit_summary['origin_kev']:.3f} keV, followed by a shared exponential decay; only C(G) changes between targets."
        )
        boundary_description = (
            f"- Below the {fit_summary['origin_kev']:.3f}-keV peak, the threshold follows the power-{float(low_energy_power):.0f} rise from T(0)=0; the function is evaluated over 0--3200 keV for plotting."
        )
        claim_boundary = (
            "The cubic-rise/exponential-decay function is fitted to historical Th-232 optimization points from the 209-keV anchor through 1460 keV. "
            "The T(0)=0 low-energy branch is a design constraint, and 2614.5 keV is reported by retention only."
        )

    report = {
        "schema_version": 1,
        "status": "TH232_SHARED_EXPONENTIAL_SPECTRA_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "score_cache": relative(cache_path),
        "score_cache_metadata": cache_metadata,
        "shared_fit_report": relative(fit_report_path),
        "fit_summary": fit_summary,
        "target_gains_percent": list(TARGET_GAINS_PCT),
        "application_domain_kev": [0.0, 3200.0],
        "pb_fit_domain_observed_centroids_kev": list(pb_domain_kev),
        "hinge_energy_kev": float(fit_summary["origin_kev"]),
        "low_energy_power": low_energy_power,
        "retention_only_anchor": {
            "reference_energy_kev": 2614.533,
            "observed_centroid_kev": float(retention_window.centroid_kev),
            "roi_low_kev": float(retention_window.roi_low_kev),
            "roi_high_kev": float(retention_window.roi_high_kev),
            "sideband_used_for_claim": False,
        },
        "threshold_rows": threshold_rows,
        "anchor_metric_rows": anchor_metric_rows,
        "summary_rows": summary_rows,
        "claim_boundary": claim_boundary,
        "artifacts": {
            "combined_spectrum_png": relative(combined_path),
            "peak_zooms_png": relative(peak_zooms_path),
            "spectrum_csv": relative(spectrum_csv),
            "threshold_csv": relative(threshold_csv),
            "target_threshold_csv": relative(target_threshold_csv),
            "anchor_metrics_csv": relative(anchor_metrics_csv),
            "annotated_no_filter_png": relative(annotated_path),
            "annotated_no_filter_csv": relative(annotated_csv),
        },
    }
    report_json_path = output_dir / "th232_shared_exponential_spectra_report.json"
    report_path = output_dir / "report.md"
    report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Th-232 spectra with energy-dependent shared thresholds",
        "",
        model_description,
        "",
        f"- Score cache: `{relative(cache_path)}`",
        f"- Shared-fit report: `{relative(fit_report_path)}`",
        f"- P/B fit domain: {pb_domain_kev[0]:.3f}--{pb_domain_kev[1]:.3f} keV observed centroids.",
        boundary_description,
        "- 2614.5 keV is retention-only; no sideband is used to claim P/B improvement there.",
        "",
        "| Target | Selected events | Total retention | 2614-keV retention | Minimum anchor P/B gain | Maximum anchor P/B gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    def format_gain(value: Any) -> str:
        return "undefined" if value is None or not np.isfinite(value) else f"{value:.4f}x"

    for row in summary_rows:
        lines.append(
            f"| +{row['target_pb_gain_percent']:.0f}% | {row['selected_events']:,} | {row['selected_fraction']:.2%} | {row['2614_roi_retention']:.2%} | {format_gain(row['minimum_anchor_pb_improvement'])} | {format_gain(row['maximum_anchor_pb_improvement'])} |"
        )
    lines.extend(
        [
            "",
            "The anchor metrics are diagnostic checks of how the shared function compares with the requested P/B targets; the shared fit is intentionally constrained and does not exactly pass every empirical anchor.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "artifacts": [
                    combined_path.name,
                    peak_zooms_path.name,
                    spectrum_csv.name,
                    threshold_csv.name,
                    target_threshold_csv.name,
                    anchor_metrics_csv.name,
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
