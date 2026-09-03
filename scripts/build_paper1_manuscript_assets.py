#!/usr/bin/env python3
"""Build publication figures and machine-readable tables for Paper 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER1_LINEAGE_DIR = (
    PROJECT_ROOT / "outputs/experiments/paper1_equal_weight_ds_cnn_20260826"
)
PAPER1_GLOBAL_DIR = PAPER1_LINEAGE_DIR / "th232_global_threshold"
PAPER1_RDL_DIR = PAPER1_LINEAGE_DIR / "th232_peak_relative_detection_limit"
PAPER1_LABEL_DIR = (
    PROJECT_ROOT / "outputs/labels/three_peak_positive_polarity_20260820"
)
PAPER1_EVENT_STORE_DIR = (
    PROJECT_ROOT
    / "processed_data/event_store/architecture_pass_warn_20260815_source_ablation"
)
STRICT_PEAK_PURITY_REPORT = (
    PROJECT_ROOT
    / "outputs/experiments/strict_ds_cnn_reproducibility_20260825"
    / "strict_peak_purity/strict_peak_purity.json"
)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=400)
    plt.close(figure)


def copy_figure_1(output_dir: Path) -> None:
    import shutil
    src_jpg = PROJECT_ROOT / "docs/paper/manuscript/figure_1.jpg"
    if not src_jpg.exists():
        src_jpg = PROJECT_ROOT / "docs/paper/manuscript/Fig_1.jpg"
    if src_jpg.exists():
        dest_jpg = output_dir / "figure_1.jpg"
        if src_jpg != dest_jpg:
            shutil.copy2(src_jpg, dest_jpg)
        try:
            img = Image.open(dest_jpg)
            img.save(output_dir / "figure_1.png", "PNG")
            img.save(output_dir / "figure_1.pdf", "PDF")
        except Exception:
            pass


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.10, 1.10, label, transform=axis.transAxes, fontweight="bold", fontsize=10)


def build_energy_matching(output_dir: Path) -> None:
    train = pd.read_csv(PAPER1_LABEL_DIR / "label_pairs_train.csv")
    validation = pd.read_csv(PAPER1_LABEL_DIR / "label_pairs_validation.csv")
    order = [
        "ba133_356kev",
        "na22_511kev",
        "cs137_662kev",
    ]
    labels = ["$^{133}$Ba 356", "$^{22}$Na 511", "$^{137}$Cs 662"]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.50))
    
    # (a) Matched pair counts per task
    axis = axes[0]
    width = 0.38
    positions = np.arange(len(order))
    train_counts = train["peak_id"].value_counts().reindex(order).to_numpy()
    validation_counts = validation["peak_id"].value_counts().reindex(order).to_numpy()
    axis.bar(
        positions - width / 2,
        train_counts,
        width,
        color="#2C3E50",
        edgecolor="#1A252F",
        linewidth=0.9,
        label="Development",
    )
    axis.bar(
        positions + width / 2,
        validation_counts,
        width,
        color="#EAECEE",
        edgecolor="#2C3E50",
        hatch="///",
        linewidth=0.9,
        label="Held-out",
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Matched pairs")
    axis.set_ylim(0, max(train_counts.max(), validation_counts.max()) * 1.35)
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    panel_label(axis, "(a)")

    # (b) Full Gaussian Cs-137 661.7 keV photopeak, Compton continuum, and matched gate (0.5 keV bins, log scale)
    axis = axes[1]
    part_manifest = load_json(PROJECT_ROOT / "outputs/labels/file_partition_manifest.json")
    cs_val_files = [
        f["hdf5"]
        for f in part_manifest["files"]
        if f.get("partition") == "validation" and f.get("source") == "cs137"
    ]
    co_val_files = [
        f["hdf5"]
        for f in part_manifest["files"]
        if f.get("partition") == "validation" and f.get("source") == "co60"
    ]

    cs_energies = []
    for h5_rel in cs_val_files:
        with h5py.File(PROJECT_ROOT / h5_rel, "r") as h5:
            e = h5["corrected_energy_kev"][:]
            cs_energies.append(e[(e >= 648.0) & (e <= 672.0)])
    cs_energies = np.concatenate(cs_energies)

    co_energies = []
    for h5_rel in co_val_files:
        with h5py.File(PROJECT_ROOT / h5_rel, "r") as h5:
            e = h5["corrected_energy_kev"][:]
            co_energies.append(e[(e >= 648.0) & (e <= 672.0)])
    co_energies = np.concatenate(co_energies)

    chosen = validation[validation["peak_id"] == "cs137_662kev"]
    bins = np.arange(650.0, 670.5, 0.5)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cs_counts, _ = np.histogram(cs_energies, bins=bins)
    co_counts, _ = np.histogram(co_energies, bins=bins)
    pos_counts, _ = np.histogram(chosen["positive_energy_kev"], bins=bins)

    axis.set_yscale("log")

    # Plot full Cs-137 photopeak spectrum (complete Gaussian across 650-670 keV)
    axis.step(bin_centers, cs_counts, where="mid", color=OKABE_ITO["blue"], linewidth=1.4, label=r"$^{137}$Cs full peak")
    # Plot Co-60 Compton continuum baseline
    axis.step(bin_centers, co_counts, where="mid", color=OKABE_ITO["red"], linestyle="--", linewidth=1.1, label=r"$^{60}$Co continuum")

    # Highlight FWHM gating range [659.79, 663.54] keV
    roi_low, roi_high = 659.794, 663.543
    axis.axvspan(roi_low, roi_high, color=OKABE_ITO["sky"], alpha=0.18, label="FWHM Gate")
    axis.axvline(roi_low, color=OKABE_ITO["black"], linestyle=":", linewidth=0.8)
    axis.axvline(roi_high, color=OKABE_ITO["black"], linestyle=":", linewidth=0.8)

    # Highlight matched 1:1 pairs inside the gate sitting directly along the continuum level
    mask = (bin_centers >= roi_low - 0.25) & (bin_centers <= roi_high + 0.25)
    matched_y = pos_counts[mask]
    axis.step(bin_centers[mask], matched_y, where="mid", color=OKABE_ITO["green"], linewidth=1.3, label="Matched 1:1 pairs")
    axis.fill_between(bin_centers[mask], 50, matched_y, step="mid", color=OKABE_ITO["green"], alpha=0.30)

    axis.set_xlabel("Corrected energy (keV)")
    axis.set_ylabel("Events per 0.5 keV")
    axis.set_xlim(650.0, 670.0)
    axis.set_ylim(50, 1e6)
    axis.legend(frameon=False, loc="upper left", fontsize=6.2, ncol=1)
    axis.grid(alpha=0.2, which="both")
    panel_label(axis, "(b)")

    # (c) Energy difference distribution |Delta E|
    axis = axes[2]
    delta = np.abs(validation["positive_energy_kev"] - validation["negative_energy_kev"])
    delta_counts, _ = np.histogram(delta, bins=np.linspace(0, 0.5, 51))
    axis.hist(delta, bins=np.linspace(0, 0.5, 51), color=OKABE_ITO["green"], alpha=0.85)
    axis.axvline(0.5, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"$|\Delta E|$ (keV)")
    axis.set_ylabel("Pairs")
    axis.set_ylim(0, delta_counts.max() * 1.18)
    axis.grid(alpha=0.2)
    panel_label(axis, "(c)")
    
    figure.tight_layout(w_pad=1.4, pad=0.8)
    save_figure(figure, output_dir, "figure_2")


def build_representation_architecture(output_dir: Path) -> None:
    values = np.load(
        PROJECT_ROOT
        / "outputs/compact_cnn_positive_polarity_training_waveforms_20260820/normalized_training_inputs.npz"
    )
    figure = plt.figure(figsize=(7.5, 4.4))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.15), wspace=0.25, hspace=0.36)
    time = (np.arange(750) - 250) * 4.0
    peaks = [
        ("cs137_662kev", r"$^{137}$Cs 662 keV", OKABE_ITO["red"], 1.6, 4.2),
        ("na22_511kev", r"$^{22}$Na 511 keV", OKABE_ITO["green"], 0.0, 0.0),
        ("ba133_356kev", r"$^{133}$Ba 356 keV", OKABE_ITO["blue"], -1.6, -4.2),
    ]

    # Charge Channel Waterfall
    axis_q = figure.add_subplot(grid[0, 0])
    idx_label = int((960.0 / 4.0) + 250)
    for peak, label, color, offset_q, _ in peaks:
        array = values[f"{peak}_charge"]
        mean = array.mean(axis=0) + offset_q
        low = np.percentile(array, 16, axis=0) + offset_q
        high = np.percentile(array, 84, axis=0) + offset_q
        axis_q.plot(time, mean, color=color, linewidth=1.25)
        axis_q.fill_between(time, low, high, color=color, alpha=0.30, linewidth=0)
        axis_q.axhline(offset_q, color=OKABE_ITO["gray"], linestyle=":", linewidth=0.5, alpha=0.5)
        axis_q.text(960, mean[idx_label] + 0.24, label, color=color, fontsize=7.0, ha="right", va="bottom", fontweight="bold")

    axis_q.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    axis_q.set_ylabel("Standardized charge")
    axis_q.set_title("Charge channel (zoomed)")
    axis_q.set_xlim(-500, 1000)
    axis_q.set_ylim(-3.5, 3.8)
    axis_q.grid(alpha=0.15)
    panel_label(axis_q, "(a)")

    # Current Channel Waterfall
    axis_i = figure.add_subplot(grid[1, 0])
    for peak, label, color, _, offset_i in peaks:
        array = values[f"{peak}_current"]
        mean = array.mean(axis=0) + offset_i
        low = np.percentile(array, 16, axis=0) + offset_i
        high = np.percentile(array, 84, axis=0) + offset_i
        axis_i.plot(time, mean, color=color, linewidth=1.25)
        axis_i.fill_between(time, low, high, color=color, alpha=0.30, linewidth=0)
        axis_i.axhline(offset_i, color=OKABE_ITO["gray"], linestyle=":", linewidth=0.5, alpha=0.5)
        axis_i.text(960, offset_i + 0.40, label, color=color, fontsize=7.0, ha="right", va="bottom", fontweight="bold")

    axis_i.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    axis_i.set_xlabel(r"Time relative to charge $t_{10}$ (ns)")
    axis_i.set_ylabel("Standardized current")
    axis_i.set_title("Current channel (zoomed)")
    axis_i.set_xlim(-500, 1000)
    axis_i.set_ylim(-6.2, 10.5)
    axis_i.grid(alpha=0.15)

    # Panel B: Architecture Diagram (Custom user drawing)
    axis_b = figure.add_subplot(grid[:, 1])
    axis_b.axis("off")
    arch_img_path = PROJECT_ROOT / "docs/paper/manuscript/DS-CNN Architecture.png"
    if not arch_img_path.exists():
        arch_img_path = PROJECT_ROOT / "docs/paper/manuscript/figure_3b.png"
    if arch_img_path.exists():
        user_img = Image.open(arch_img_path)
        axis_b.imshow(user_img)
    panel_label(axis_b, "(b)")
    save_figure(figure, output_dir, "figure_3")

    # Also build standalone vector panel (a) for high-resolution LaTeX minipage inclusion
    fig_a, (ax_qa, ax_ia) = plt.subplots(2, 1, figsize=(3.8, 4.4), sharex=True)
    plt.subplots_adjust(hspace=0.28, left=0.18, right=0.96, top=0.92, bottom=0.10)
    for peak, label, color, offset_q, _ in peaks:
        array = values[f"{peak}_charge"]
        mean = array.mean(axis=0) + offset_q
        low = np.percentile(array, 16, axis=0) + offset_q
        high = np.percentile(array, 84, axis=0) + offset_q
        ax_qa.plot(time, mean, color=color, linewidth=1.25)
        ax_qa.fill_between(time, low, high, color=color, alpha=0.30, linewidth=0)
        ax_qa.axhline(offset_q, color=OKABE_ITO["gray"], linestyle=":", linewidth=0.5, alpha=0.5)
        ax_qa.text(960, mean[idx_label] + 0.24, label, color=color, fontsize=7.0, ha="right", va="bottom", fontweight="bold")
    ax_qa.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    ax_qa.set_ylabel("Standardized charge")
    ax_qa.set_title("Charge channel (zoomed)", fontsize=9.5)
    ax_qa.set_xlim(-500, 1000)
    ax_qa.set_ylim(-3.5, 3.8)
    ax_qa.grid(alpha=0.15)
    panel_label(ax_qa, "(a)")

    for peak, label, color, _, offset_i in peaks:
        array = values[f"{peak}_current"]
        mean = array.mean(axis=0) + offset_i
        low = np.percentile(array, 16, axis=0) + offset_i
        high = np.percentile(array, 84, axis=0) + offset_i
        ax_ia.plot(time, mean, color=color, linewidth=1.25)
        ax_ia.fill_between(time, low, high, color=color, alpha=0.30, linewidth=0)
        ax_ia.axhline(offset_i, color=OKABE_ITO["gray"], linestyle=":", linewidth=0.5, alpha=0.5)
        ax_ia.text(960, offset_i + 0.40, label, color=color, fontsize=7.0, ha="right", va="bottom", fontweight="bold")
    ax_ia.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    ax_ia.set_xlabel(r"Time relative to charge $t_{10}$ (ns)")
    ax_ia.set_ylabel("Standardized current")
    ax_ia.set_title("Current channel (zoomed)", fontsize=9.5)
    ax_ia.set_xlim(-500, 1000)
    ax_ia.set_ylim(-6.2, 10.5)
    ax_ia.grid(alpha=0.15)
    save_figure(fig_a, output_dir, "figure_3a")

    # Copy panel (b) standalone assets
    import shutil
    arch_pdf_path = PROJECT_ROOT / "docs/paper/manuscript/figure_3b.pdf"
    if not arch_pdf_path.exists():
        arch_pdf_path = PROJECT_ROOT / "docs/paper/manuscript/DS-CNN Architecture.pdf"
    if arch_pdf_path.exists() and arch_pdf_path != (output_dir / "figure_3b.pdf"):
        shutil.copy2(arch_pdf_path, output_dir / "figure_3b.pdf")
    if arch_img_path.exists() and arch_img_path != (output_dir / "figure_3b.png"):
        shutil.copy2(arch_img_path, output_dir / "figure_3b.png")


def build_shortcut_controls(output_dir: Path) -> None:
    cache_dir = PROJECT_ROOT / "processed_data/relaxed_continuum_roi_ds_cnn_20260822"
    relaxed_meta = np.load(cache_dir / "relaxed_file_validation_metadata.npz")
    strict_meta = np.load(cache_dir / "strict_internal_metadata.npz")
    held_out = np.load(PAPER1_LINEAGE_DIR / "traditional_ae/held_out_ae_scores.npz")

    peaks_info = {
        "ba133_356kev": (355.709, 3.941),
        "na22_511kev": (510.926, 4.447),
        "cs137_662kev": (661.668, 3.749),
    }

    r_labels = relaxed_meta["label"]
    r_peaks = relaxed_meta["peak_id"]
    r_energies = relaxed_meta["energy_kev"]
    r_offsets = np.zeros_like(r_energies)
    for pk, (center, fwhm) in peaks_info.items():
        m = r_peaks == pk
        r_offsets[m] = -np.abs(r_energies[m] - center) / fwhm

    s_labels = strict_meta["label"]
    s_peaks = strict_meta["peak_id"]
    s_energies = strict_meta["energy_kev"]
    s_offsets = np.zeros_like(s_energies)
    for pk, (center, fwhm) in peaks_info.items():
        m = s_peaks == pk
        s_offsets[m] = -np.abs(s_energies[m] - center) / fwhm

    fpr_r_e, tpr_r_e, _ = roc_curve(r_labels, r_offsets)
    auc_r_e = roc_auc_score(r_labels, r_offsets)

    fpr_s_e, tpr_s_e, _ = roc_curve(s_labels, s_offsets)
    auc_s_e = roc_auc_score(s_labels, s_offsets)

    h_labels = held_out["labels"]
    h_cnn = held_out["cnn_scores"]
    fpr_cnn, tpr_cnn, _ = roc_curve(h_labels, h_cnn)
    auc_cnn = roc_auc_score(h_labels, h_cnn)

    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.55, 4.4))

    # Panel (a): 137Cs 662 keV spectral density
    pk_choice = "cs137_662kev"
    center_e, fwhm_e = peaks_info[pk_choice]
    mask_r = r_peaks == pk_choice
    e_r_pos = r_energies[mask_r & (r_labels == 1)]
    e_r_neg = r_energies[mask_r & (r_labels == 0)]
    mask_s = s_peaks == pk_choice
    e_s_neg = s_energies[mask_s & (s_labels == 0)]

    bins = np.linspace(center_e - 1.55 * fwhm_e, center_e + 1.55 * fwhm_e, 35)

    ax1.hist(
        e_r_neg,
        bins=bins,
        density=True,
        color="#EAEDED",
        edgecolor="#BDC3C7",
        linewidth=0.8,
        label=r"Unmatched cont. ($\pm 1.5\,\mathrm{FWHM}$)",
    )
    ax1.hist(
        e_r_pos,
        bins=bins,
        density=True,
        histtype="step",
        color="#1B4F72",
        linewidth=1.6,
        label="Candidate photopeaks",
    )
    ax1.hist(
        e_s_neg,
        bins=bins,
        density=True,
        histtype="step",
        color="#D35400",
        linewidth=1.4,
        linestyle="--",
        label=r"Strict 0.5-keV matched cont.",
    )

    ax1.axvline(center_e, color="#7F8C8D", linestyle=":", linewidth=0.9)

    ax1.set_xlabel(r"Reconstructed energy $E$ (keV)")
    ax1.set_ylabel("Normalized event density")
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.92,
        fontsize=5.7,
        borderpad=0.35,
        handlelength=1.4,
        labelspacing=0.25,
    )
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_xlim(center_e - 1.55 * fwhm_e, center_e + 1.55 * fwhm_e)
    ax1.set_ylim(0, 0.48)
    panel_label(ax1, "(a)")

    # Panel (b): Full Diagnostic ROC Curves
    # Full continuous Input SNR curve
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    mean_val = np.float32(manifest["feature_statistics"]["means"][0])
    std_val = np.float32(manifest["feature_statistics"]["standard_deviations"][0])
    values = np.load(cache_dir / "strict_internal_values.npy", mmap_mode="r")
    charge = np.asarray(values[:, 0], dtype=np.float32) * std_val + mean_val
    noise = np.std(charge[:, :180], axis=1) + 1.0e-6
    snr = np.max(charge, axis=1) / noise
    fpr_snr, tpr_snr, _ = roc_curve(s_labels, snr)
    auc_snr = roc_auc_score(s_labels, snr)

    ax2.plot(
        fpr_r_e,
        tpr_r_e,
        color="#C0392B",
        linewidth=1.5,
        label=f"Unmatched energy ({auc_r_e:.3f})",
    )
    ax2.plot(
        fpr_cnn,
        tpr_cnn,
        color="#2980B9",
        linewidth=1.5,
        label=f"DS-CNN on strict pairs ({auc_cnn:.3f})",
    )
    # Combined strict matched energy and chance floor line
    ax2.plot(
        fpr_s_e,
        tpr_s_e,
        color="#27AE60",
        linewidth=1.3,
        linestyle="--",
        label=f"Strict matched energy ({auc_s_e:.3f})",
    )
    ax2.plot(
        fpr_snr,
        tpr_snr,
        color="#8E44AD",
        linewidth=1.1,
        linestyle=":",
        label=f"Input SNR control ({auc_snr:.3f})",
    )

    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    # Placed strictly in the lower-right empty triangle below the diagonal
    ax2.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.92,
        fontsize=5.7,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.4,
    )
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    panel_label(ax2, "(b)")

    figure.tight_layout()
    save_figure(figure, output_dir, "figure_4")


def metric_by_peak(labels: np.ndarray, peaks: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {peak: roc_auc_score(labels[peaks == peak], score[peaks == peak]) for peak in np.unique(peaks)}


def build_classification(output_dir: Path, tables_dir: Path) -> None:
    score_path = PAPER1_LINEAGE_DIR / "traditional_ae/held_out_ae_scores.npz"
    values = np.load(score_path)
    labels = values["labels"]
    peaks = values["peak_ids"]
    scores = {
        "A/E": values["waveform_ae"],
        "A/shaped energy": values["shaped_energy_ae"],
        "DS-CNN": values["cnn_scores"],
    }
    rows = []
    for name, score in scores.items():
        auc = roc_auc_score(labels, score)
        per_peak = metric_by_peak(labels, peaks, score)
        rows.append({"discriminator": name, "pooled_auroc": auc, **per_peak})
    pd.DataFrame(rows).to_csv(tables_dir / "classification_ae_metrics.csv", index=False)

    figure, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # (a) Taskwise ROC Curves - Print-Separable Color & Stroke
    styles_task = {
        "cs137_662kev": {"ls": "-", "color": "#0072B2", "lw": 1.5, "name": r"DS-CNN 662 keV"},
        "na22_511kev": {"ls": "-.", "color": "#009E73", "lw": 1.3, "name": r"DS-CNN 511 keV"},
        "ba133_356kev": {"ls": "--", "color": "#D55E00", "lw": 1.3, "name": r"DS-CNN 356 keV"},
    }
    for pk in ["cs137_662kev", "na22_511kev", "ba133_356kev"]:
        m = peaks == pk
        fpr, tpr, _ = roc_curve(labels[m], scores["DS-CNN"][m])
        auc = roc_auc_score(labels[m], scores["DS-CNN"][m])
        cfg = styles_task[pk]
        ax_a.plot(fpr, tpr, linestyle=cfg["ls"], color=cfg["color"], linewidth=cfg["lw"], label=f"{cfg['name']} ({auc:.3f})")

    fpr_ae, tpr_ae, _ = roc_curve(labels, scores["A/shaped energy"])
    auc_ae = roc_auc_score(labels, scores["A/shaped energy"])
    ax_a.plot(fpr_ae, tpr_ae, color="#7F8C8D", linestyle=":", linewidth=1.3, label=f"Classical $A/E_{{\\mathrm{{trap}}}}$ ({auc_ae:.3f})")
    ax_a.plot([0, 1], [0, 1], color="#BDC3C7", linestyle=(0, (3, 3)), linewidth=0.8, label="Chance floor (0.500)")

    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    ax_a.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=4.6,
        borderpad=0.2,
        labelspacing=0.15,
        handlelength=1.3,
    )
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    panel_label(ax_a, "(a)")

    # (b) Classical A/E Distribution (137Cs 662 keV) - Color + Pattern Separable
    m_cs = peaks == "cs137_662kev"
    ae_cs_pos = scores["A/shaped energy"][m_cs & (labels == 1)] * 1e3
    ae_cs_neg = scores["A/shaped energy"][m_cs & (labels == 0)] * 1e3
    bins_ae = np.linspace(3, 16, 32)

    ax_b.hist(
        ae_cs_pos,
        bins=bins_ae,
        density=True,
        facecolor="#A9CCE3",
        edgecolor="#1B4F72",
        linewidth=0.9,
        alpha=0.65,
        label=r"Candidate photopeak ($y=1$)",
    )
    ax_b.hist(
        ae_cs_neg,
        bins=bins_ae,
        density=True,
        histtype="step",
        edgecolor="#D35400",
        linewidth=1.3,
        linestyle="--",
        label=r"Matched continuum ($y=0$)",
    )
    ax_b.set_xlabel(r"Classical $A/E_{\mathrm{trap}}$ ($\times 10^{-3}$, $^{137}\mathrm{Cs}$)")
    ax_b.set_ylabel("Probability density")
    ax_b.set_title(r"Classical $A/E_{\mathrm{trap}}$: Overlap ($0.555$)", fontsize=7.5, pad=3)
    ax_b.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=5.0,
        borderpad=0.22,
        labelspacing=0.18,
        handlelength=1.4,
    )
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.set_xlim(3, 16)
    ax_b.set_ylim(0, 0.48)
    panel_label(ax_b, "(b)")

    # (c) DS-CNN Output Score Distribution (137Cs 662 keV) - Color + Pattern Separable
    cnn_pos = scores["DS-CNN"][m_cs & (labels == 1)]
    cnn_neg = scores["DS-CNN"][m_cs & (labels == 0)]
    bins_cnn = np.linspace(0.15, 0.85, 32)

    ax_c.hist(
        cnn_pos,
        bins=bins_cnn,
        density=True,
        facecolor="#A9CCE3",
        edgecolor="#1B4F72",
        linewidth=0.9,
        alpha=0.65,
        label=r"Candidate photopeak ($y=1$)",
    )
    ax_c.hist(
        cnn_neg,
        bins=bins_cnn,
        density=True,
        histtype="step",
        edgecolor="#D35400",
        linewidth=1.3,
        linestyle="--",
        label=r"Matched continuum ($y=0$)",
    )
    ax_c.axvline(0.50, color="#7F8C8D", linestyle=":", linewidth=0.9)

    ax_c.set_xlabel(r"DS-CNN score $s \in [0, 1]$ ($^{137}\mathrm{Cs}$)")
    ax_c.set_ylabel("Probability density")
    ax_c.set_title(r"DS-CNN: Separation ($0.688$)", fontsize=7.5, pad=3)
    ax_c.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=5.0,
        borderpad=0.22,
        labelspacing=0.18,
        handlelength=1.4,
    )
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.set_xlim(0.15, 0.85)
    ax_c.set_ylim(0, 3.4)
    panel_label(ax_c, "(c)")

    figure.tight_layout()
    save_figure(figure, output_dir, "figure_5")


def build_pair_scalar_frame(split: str) -> pd.DataFrame:
    pairs = pd.read_csv(PAPER1_LABEL_DIR / f"label_pairs_{split}.csv")
    pair_delta = np.abs(
        pairs["positive_energy_kev"].to_numpy()
        - pairs["negative_energy_kev"].to_numpy()
    )
    event_parts = []
    for side, label in (("positive", 1), ("negative", 0)):
        event_parts.append(
            pd.DataFrame(
                {
                    "event_hdf5": pairs[f"{side}_hdf5"],
                    "event_source_row": pairs[f"{side}_row"].astype(np.int64),
                    "label": label,
                    "pair_abs_energy_delta_kev": pair_delta,
                }
            )
        )
    events = pd.concat(event_parts, ignore_index=True)
    lookup = pd.read_csv(
        PAPER1_EVENT_STORE_DIR / f"event_lookup_{split}.csv",
        usecols=["source_hdf5", "source_row", "store_index"],
    ).rename(
        columns={
            "source_hdf5": "event_hdf5",
            "source_row": "event_source_row",
        }
    )
    events = events.merge(
        lookup,
        on=["event_hdf5", "event_source_row"],
        how="left",
        validate="many_to_one",
    )
    if events["store_index"].isna().any():
        raise KeyError(f"Event store lookup is incomplete for {split}")

    store_rows = events["store_index"].to_numpy(dtype=np.int64)
    unique_rows, inverse = np.unique(store_rows, return_inverse=True)
    with h5py.File(PAPER1_EVENT_STORE_DIR / f"{split}_events.h5", "r") as store:
        events["event_reconstructed_energy_kev"] = np.asarray(
            store["reconstructed_energy_kev"][unique_rows], dtype=np.float64
        )[inverse]
        events["event_trigger_time_s"] = np.asarray(
            store["trigger_time_s"][unique_rows], dtype=np.float64
        )[inverse]
        noise = np.asarray(
            store["noise_rms_adc"][unique_rows, :5], dtype=np.float64
        ).mean(axis=1)
        events["event_noise_mean_adc"] = noise[inverse]
    return events


def build_domain_audit(output_dir: Path) -> dict[str, float]:
    train = build_pair_scalar_frame("train")
    validation = build_pair_scalar_frame("validation")
    features = [
        ("pair_abs_energy_delta_kev", r"Strict $|\Delta E|$ matching"),
        ("event_noise_mean_adc", "Baseline noise RMS"),
        ("event_trigger_time_s", "Trigger timestamp"),
        ("event_reconstructed_energy_kev", "Naive unstratified energy"),
    ]
    metrics: dict[str, float] = {}
    roc_data = {}
    for feature, name in features:
        train_auc = float(roc_auc_score(train["label"], train[feature]))
        score = validation[feature] if train_auc >= 0.5 else -validation[feature]
        fpr, tpr, _ = roc_curve(validation["label"], score)
        validation_auc = float(roc_auc_score(validation["label"], score))
        metrics[feature] = validation_auc
        roc_data[feature] = {"fpr": fpr, "tpr": tpr, "auc": validation_auc, "name": name}

    # DS-CNN curve
    score_path = PAPER1_LINEAGE_DIR / "traditional_ae/held_out_ae_scores.npz"
    cnn_data = np.load(score_path)
    fpr_cnn, tpr_cnn, _ = roc_curve(cnn_data["labels"], cnn_data["cnn_scores"])
    auc_cnn = roc_auc_score(cnn_data["labels"], cnn_data["cnn_scores"])

    figure, ax = plt.subplots(figsize=(3.55, 2.75))

    ax.plot(fpr_cnn, tpr_cnn, color="#0072B2", linewidth=1.5, label=f"DS-CNN model ({auc_cnn:.3f})")
    ax.plot(
        roc_data["event_reconstructed_energy_kev"]["fpr"],
        roc_data["event_reconstructed_energy_kev"]["tpr"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label=f"Naive unstratified energy ({roc_data['event_reconstructed_energy_kev']['auc']:.3f})",
    )
    ax.plot(
        roc_data["event_trigger_time_s"]["fpr"],
        roc_data["event_trigger_time_s"]["tpr"],
        color="#009E73",
        linestyle="-.",
        linewidth=1.2,
        label=f"Trigger timestamp ({roc_data['event_trigger_time_s']['auc']:.3f})",
    )
    ax.plot(
        roc_data["event_noise_mean_adc"]["fpr"],
        roc_data["event_noise_mean_adc"]["tpr"],
        color="#CC79A7",
        linestyle=":",
        linewidth=1.2,
        label=f"Baseline noise RMS ({roc_data['event_noise_mean_adc']['auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#BDC3C7", linestyle=(0, (3, 3)), linewidth=0.9, label=r"Strict $|\Delta E|$ matching (0.500)")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=5.2,
        borderpad=0.2,
        labelspacing=0.15,
        handlelength=1.3,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    figure.tight_layout()
    save_figure(figure, output_dir, "figure_6")
    return metrics


def build_thorium_spectra(output_dir: Path) -> None:
    score_path = PAPER1_LINEAGE_DIR / "th232/th232_ds_cnn_scores.h5"
    with h5py.File(score_path, "r") as store:
        energy = np.asarray(store["corrected_energy_kev"], dtype=np.float32)
        score = np.asarray(store["score"], dtype=np.float32)
    presets = pd.read_csv(PAPER1_GLOBAL_DIR / "fpga_recommended_threshold_presets.csv").set_index("preset")
    conservative_threshold = float(presets.loc["no_brainer_conservative", "threshold"])
    sweet_threshold = float(presets.loc["sweet_spot", "threshold"])
    thresholds = [
        (0.0, "No cut", OKABE_ITO["blue"], "-", 0.65),
        (conservative_threshold, r"Conservative ($t=0.268$)", OKABE_ITO["green"], "--", 0.75),
        (sweet_threshold, r"Balanced ($t=0.459$)", OKABE_ITO["red"], "-", 0.65),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 3.2), dpi=300)
    full_bins = np.arange(100, 2701, 1.0)
    centers = (full_bins[:-1] + full_bins[1:]) / 2
    for threshold, label, color, linestyle, lw in thresholds:
        selected = score >= threshold
        counts, _ = np.histogram(energy[selected], bins=full_bins)
        axis.step(centers, counts, where="mid", color=color, linestyle=linestyle, linewidth=lw, label=label)
    axis.set_yscale("log")
    axis.set_xlim(100, 2720)
    axis.set_ylim(1e1, 1e6)
    axis.set_xlabel("Corrected energy (keV)")
    axis.set_ylabel("Counts / 1 keV")
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.92, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        ncol=3,
        fontsize=5.8,
        borderpad=0.25,
    )
    axis.grid(True, linestyle="--", alpha=0.3)

    peak_layout = [
        (209.25, r"$^{228}\mathrm{Ac}$ (209)", 7.5e3, 5.0e4, 180, "center"),
        (238.63, r"$^{212}\mathrm{Pb}$ (239)", 3.2e4, 1.6e5, 239, "center"),
        (300.09, r"$^{212}\mathrm{Pb}$ (300)", 6.5e3, 3.2e4, 305, "center"),
        (409.46, r"$^{228}\mathrm{Ac}$ (409)", 4.5e3, 2.2e4, 409, "center"),
        (510.77, r"$^{208}\mathrm{Tl}$/pair (511)", 1.1e4, 6.5e4, 465, "center"),
        (583.19, r"$^{208}\mathrm{Tl}$ (583)", 2.5e4, 1.3e5, 610, "center"),
        (727.33, r"$^{212}\mathrm{Bi}$ (727)", 6.8e3, 3.2e4, 727, "center"),
        (911.20, r"$^{228}\mathrm{Ac}$ (911)", 1.8e4, 1.1e5, 911, "center"),
        (1247.08, r"$^{228}\mathrm{Ac}$ (1247)", 6.5e2, 3.0e3, 1247, "center"),
        (1460.83, r"$^{40}\mathrm{K}$ (1461)", 6.5e2, 6.0e3, 1420, "center"),
        (1592.53, r"DEP (1593)", 1.6e3, 1.8e4, 1630, "center"),
        (2103.53, r"SEP (2104)", 1.4e3, 1.2e4, 2104, "center"),
        (2614.53, r"$^{208}\mathrm{Tl}$ (2615)", 1.1e4, 6.5e4, 2615, "center"),
    ]
    for e_peak, label_txt, y_tip, y_txt, x_txt, ha in peak_layout:
        axis.annotate(
            label_txt,
            xy=(e_peak, y_tip),
            xytext=(x_txt, y_txt),
            fontsize=5.2,
            fontweight="bold",
            color="#1B2631",
            ha=ha,
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color="#5D6D7E",
                lw=0.45,
                shrinkA=1,
                shrinkB=1,
            ),
        )
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_7")


def build_thorium_atlas(output_dir: Path) -> None:
    score_path = PAPER1_LINEAGE_DIR / "th232/th232_ds_cnn_scores.h5"
    with h5py.File(score_path, "r") as store:
        energy = np.asarray(store["corrected_energy_kev"], dtype=np.float32)
        score = np.asarray(store["score"], dtype=np.float32)
    presets = pd.read_csv(PAPER1_GLOBAL_DIR / "fpga_recommended_threshold_presets.csv").set_index("preset")
    conservative_threshold = float(presets.loc["no_brainer_conservative", "threshold"])
    sweet_threshold = float(presets.loc["sweet_spot", "threshold"])
    thresholds = [
        (0.0, "No cut", OKABE_ITO["blue"], "-", 0.70),
        (conservative_threshold, r"Conservative ($t=0.268$)", OKABE_ITO["green"], "--", 0.80),
        (sweet_threshold, r"Balanced ($t=0.459$)", OKABE_ITO["red"], "-", 0.70),
    ]
    metrics = pd.read_csv(PAPER1_GLOBAL_DIR / "per_peak_metrics.csv")
    peaks_12_exact = [
        (209.253, 8, r"$^{228}\mathrm{Ac}$ (209.3 keV)", (1e3, 5e3)),
        (238.632, 10, r"$^{212}\mathrm{Pb}$ (238.6 keV)", (1e3, 3.5e4)),
        (300.087, 10, r"$^{212}\mathrm{Pb}$ (300.1 keV)", (7e2, 5e3)),
        (409.462, 12, r"$^{228}\mathrm{Ac}$ (409.5 keV)", (1e2, 4e3)),
        (510.770, 12, r"$^{208}\mathrm{Tl}$/pair (510.8 keV)", (1e2, 8e3)),
        (583.191, 14, r"$^{208}\mathrm{Tl}$ (583.2 keV)", (1e2, 3e4)),
        (727.330, 14, r"$^{212}\mathrm{Bi}$ (727.3 keV)", (1e2, 2.5e4)),
        (911.204, 16, r"$^{228}\mathrm{Ac}$ (911.2 keV)", (1e1, 3e4)),
        (1247.080, 18, r"$^{228}\mathrm{Ac}$ (1247.1 keV)", (1e1, 5e2)),
        (1460.830, 18, r"$^{40}\mathrm{K}$ (1460.8 keV)", (1e1, 5e2)),
        (2103.533, 20, r"$^{208}\mathrm{Tl}$ SEP (2103.5 keV)", (1e1, 2.5e3)),
        (2614.533, 20, r"$^{208}\mathrm{Tl}$ (2614.5 keV)", (1e0, 3.5e4)),
    ]

    def get_retention(ref_e: float, thresh: float) -> float:
        if thresh == 0:
            return 1.0
        point = "no_brainer_conservative" if np.isclose(thresh, conservative_threshold) else "reoptimized_global"
        sub = metrics[(metrics["operating_point"] == point) & np.isclose(metrics["reference_energy_kev"], ref_e)]
        if len(sub) > 0:
            return max(float(sub.iloc[0]["net_peak_retention_vs_no_cut"]), 1e-6)
        w = (energy >= ref_e - 3.0) & (energy <= ref_e + 3.0)
        n0 = max(np.sum(w), 1)
        n_sel = np.sum(w & (score >= thresh))
        return max(float(n_sel / n0), 1e-6)

    figure, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), dpi=300)
    axes_flat = axes.flatten()
    lines = []
    labels = []
    for idx, (ref_e, hw, title, (ymin, ymax)) in enumerate(peaks_12_exact):
        ax = axes_flat[idx]
        bins = np.arange(ref_e - hw, ref_e + hw + 0.25, 0.25)
        centers = (bins[:-1] + bins[1:]) / 2
        for thresh, label, color, linestyle, lw in thresholds:
            selected = score >= thresh
            counts, _ = np.histogram(energy[selected], bins=bins)
            ret = get_retention(ref_e, thresh)
            scaled_counts = counts / ret
            line = ax.step(centers, scaled_counts, where="mid", color=color, linestyle=linestyle, linewidth=lw, label=label)
            if idx == 0:
                lines.append(line[0])
                labels.append(label)
        ax.set_yscale("log")
        ax.set_xlim(ref_e - hw, ref_e + hw)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=6.2, pad=1.5)
        ax.tick_params(axis="both", which="both", labelsize=5.5, pad=1.5)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")
        letter = chr(ord("a") + idx)
        ax.text(0.05, 0.90, f"({letter})", transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="top")

        # Direct performance annotations per plot
        sub_c = metrics[(metrics["operating_point"] == "no_brainer_conservative") & np.isclose(metrics["reference_energy_kev"], ref_e)]
        sub_s = metrics[(metrics["operating_point"] == "reoptimized_global") & np.isclose(metrics["reference_energy_kev"], ref_e)]

        if len(sub_c) > 0 and len(sub_s) > 0:
            pb_c = sub_c.iloc[0]["pb_improvement_factor_vs_no_cut"]
            ret_c = sub_c.iloc[0]["net_peak_retention_vs_no_cut"]
            pb_s = sub_s.iloc[0]["pb_improvement_factor_vs_no_cut"]
            ret_s = sub_s.iloc[0]["net_peak_retention_vs_no_cut"]
            pb_c_pct = (pb_c - 1.0) * 100.0
            pb_s_pct = (pb_s - 1.0) * 100.0
            anno_text = (
                f"Cons: +{pb_c_pct:.1f}% P/B, {ret_c*100:.1f}% ret\n"
                f"Bal: +{pb_s_pct:.1f}% P/B, {ret_s*100:.1f}% ret"
            )
        else:
            ret_c = get_retention(ref_e, conservative_threshold)
            ret_s = get_retention(ref_e, sweet_threshold)
            anno_text = (
                f"Cons: {ret_c*100:.1f}% ret\n"
                f"Bal: {ret_s*100:.1f}% ret"
            )

        ax.text(
            0.96,
            0.92,
            anno_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.5,
            color="#1C2833",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#D5D8DC", alpha=0.88, lw=0.4),
        )

        if idx % 4 == 0:
            ax.set_ylabel(r"Counts ($N_t = N_0$)", fontsize=5.8, labelpad=2.0)
        if idx >= 8:
            ax.set_xlabel("Energy (keV)", fontsize=5.8, labelpad=2.0)
    figure.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.96,
        fontsize=7.5,
        borderpad=0.3,
        columnspacing=1.8,
    )
    figure.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.98, hspace=0.38, wspace=0.32)
    save_figure(figure, output_dir, "figure_8")


def build_operating_metrics(output_dir: Path, tables_dir: Path) -> None:
    base = PAPER1_GLOBAL_DIR
    scan = pd.read_csv(base / "threshold_scan.csv")
    per_peak = pd.read_csv(base / "per_peak_metrics.csv")
    rdl = pd.read_csv(PAPER1_RDL_DIR / "per_peak_optimal_thresholds.csv")
    presets = pd.read_csv(base / "fpga_recommended_threshold_presets.csv").set_index("preset")
    conservative_threshold = float(presets.loc["no_brainer_conservative", "threshold"])
    sweet_threshold = float(presets.loc["sweet_spot", "threshold"])
    reliable = scan[scan["all_peak_statistics_reliable"]].copy()

    figure = plt.figure(figsize=(7.2, 2.50), dpi=300)
    gs = figure.add_gridspec(1, 2, left=0.08, right=0.96, bottom=0.18, top=0.88, wspace=0.35)

    # Panel (a): Global Pareto scan across geometric-mean curve range [0.20, 0.51]
    axis_a = figure.add_subplot(gs[0, 0])
    l1 = axis_a.plot(
        reliable["threshold"],
        reliable["geometric_mean_pb_improvement"],
        color=OKABE_ITO["blue"],
        linewidth=1.4,
        label=r"Geom.-mean $\mathrm{P/B}$ gain",
        zorder=3,
    )
    axis_a_twin = axis_a.twinx()
    l2 = axis_a_twin.plot(
        reliable["threshold"],
        reliable["minimum_peak_retention"],
        color=OKABE_ITO["orange"],
        linewidth=1.4,
        linestyle="-",
        label=r"Worst retention ($\epsilon_{\min}$)",
        zorder=2,
    )

    # Vertical threshold markers
    axis_a.axvline(conservative_threshold, color=OKABE_ITO["green"], linestyle="--", linewidth=0.9, zorder=4)
    axis_a.axvline(sweet_threshold, color=OKABE_ITO["red"], linestyle="--", linewidth=0.9, zorder=4)

    axis_a.text(
        conservative_threshold - 0.005,
        1.24,
        r"Cons." + "\n" + r"$t=0.268$",
        va="top",
        ha="right",
        fontsize=5.2,
        color=OKABE_ITO["green"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=OKABE_ITO["green"], alpha=0.90, lw=0.45),
    )
    axis_a.text(
        sweet_threshold - 0.005,
        1.09,
        r"Balanced" + "\n" + r"$t=0.459$",
        va="bottom",
        ha="right",
        fontsize=5.2,
        color=OKABE_ITO["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=OKABE_ITO["red"], alpha=0.90, lw=0.45),
    )

    axis_a.set_xlim(0.20, 0.51)
    axis_a.set_ylim(0.98, 1.34)
    axis_a_twin.set_ylim(0.48, 1.08)

    axis_a.set_xlabel("Score threshold ($t$)", fontsize=7.8)
    axis_a.set_ylabel(r"Geometric-mean $\mathrm{P/B}$ gain", color=OKABE_ITO["blue"], fontsize=7.8)
    axis_a_twin.set_ylabel("Worst peak retention", color=OKABE_ITO["orange"], fontsize=7.8)
    axis_a.tick_params(axis="y", labelcolor=OKABE_ITO["blue"], labelsize=6.8)
    axis_a_twin.tick_params(axis="y", labelcolor=OKABE_ITO["orange"], labelsize=6.8)
    axis_a.tick_params(axis="x", labelsize=6.8)
    axis_a.grid(True, linestyle="--", alpha=0.25)

    lines_a = l1 + l2
    labels_a = [l.get_label() for l in lines_a]
    axis_a.legend(
        lines_a,
        labels_a,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=5.5,
        borderpad=0.25,
        labelspacing=0.2,
    )
    panel_label(axis_a, "(a)")

    # Panel (b): Relative detection limit reduction
    axis_b = figure.add_subplot(gs[0, 1])
    improvement = 100 * (1 - rdl["relative_detection_limit"])
    lower = 100 * (1 - rdl["bootstrap_fixed_threshold_relative_detection_limit_ci95_high"])
    upper = 100 * (1 - rdl["bootstrap_fixed_threshold_relative_detection_limit_ci95_low"])
    yerr = np.vstack((improvement - lower, upper - improvement))
    colors = [OKABE_ITO["blue"] if flag else OKABE_ITO["gray"] for flag in rdl["bootstrap_supports_improvement"]]

    axis_b.errorbar(
        rdl["reference_energy_kev"],
        improvement,
        yerr=yerr,
        fmt="none",
        ecolor="#85929E",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.8,
    )
    axis_b.scatter(rdl["reference_energy_kev"], improvement, c=colors, s=26, zorder=4, edgecolor="black", linewidth=0.4)
    axis_b.axhline(0.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8, alpha=0.8)

    axis_b.scatter([], [], c=OKABE_ITO["blue"], s=26, edgecolor="black", linewidth=0.4, label=r"Supported ($p < 0.05$)")
    axis_b.scatter([], [], c=OKABE_ITO["gray"], s=26, edgecolor="black", linewidth=0.4, label="Not significant")

    axis_b.set_xlabel("Peak energy (keV)", fontsize=7.8)
    axis_b.set_ylabel(r"Relative DL reduction $1 - R_t$ (%)", fontsize=7.8)
    axis_b.set_ylim(-1.0, 9.2)
    axis_b.tick_params(axis="both", labelsize=6.8)
    axis_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#D5D8DC",
        framealpha=0.94,
        fontsize=5.5,
        borderpad=0.25,
        labelspacing=0.2,
    )
    axis_b.grid(True, linestyle="--", alpha=0.25)
    panel_label(axis_b, "(b)")

    save_figure(figure, output_dir, "figure_9")
    per_peak.to_csv(tables_dir / "th232_global_per_peak_metrics.csv", index=False)
    rdl.to_csv(tables_dir / "th232_relative_detection_limits.csv", index=False)


def build_tables(tables_dir: Path) -> None:
    train = pd.read_csv(PAPER1_LABEL_DIR / "label_pairs_train.csv")
    validation = pd.read_csv(PAPER1_LABEL_DIR / "label_pairs_validation.csv")
    purity_rows = {
        row["peak_id"]: row for row in load_json(STRICT_PEAK_PURITY_REPORT)["rows"]
    }
    tasks = [
        ("ba133_356kev", "Ba-133 356 keV", "[353.74, 357.68]"),
        ("na22_511kev", "Na-22 511 keV", "[508.70, 513.15]"),
        ("cs137_662kev", "Cs-137 662 keV", "[659.79, 663.54]"),
    ]
    rows = []
    train_vc = train["peak_id"].value_counts()
    val_vc = validation["peak_id"].value_counts()
    for peak_id, task_name, gate_range in tasks:
        n_train = int(train_vc.get(peak_id, 0))
        n_val = int(val_vc.get(peak_id, 0))
        rows.append({
            "photopeak_task": task_name,
            "gate_range_kev": gate_range,
            "estimated_gate_impurity_percent": (
                100.0 * purity_rows[peak_id]["contamination_fraction"]
            ),
            "train_pairs": n_train,
            "valid_pairs": n_val,
            "total_events": 2 * (n_train + n_val),
        })
    rows.append({
        "photopeak_task": "Total Matched Pairs",
        "gate_range_kev": "---",
        "estimated_gate_impurity_percent": "",
        "train_pairs": len(train),
        "valid_pairs": len(validation),
        "total_events": 2 * (len(train) + len(validation)),
    })
    pd.DataFrame(rows).to_csv(tables_dir / "strict_dataset_counts.csv", index=False)

    profile = load_json(
        PROJECT_ROOT / "outputs/architecture_profile/architecture_candidates_20260816/architecture_profile.json"
    )["candidates"]["ds_cnn"]
    pd.DataFrame(
        [
            {
                "input_shape": "2 x 750",
                "parameters": profile["parameter_count"],
                "macs_per_event": profile["macs_per_event"],
                "activation_bytes_sum_fp32": profile["activation_bytes_sum"],
                "peak_single_activation_bytes_fp32": profile["peak_single_activation_bytes"],
            }
        ]
    ).to_csv(tables_dir / "ds_cnn_analytical_profile.csv", index=False)

    presets = pd.read_csv(PAPER1_GLOBAL_DIR / "fpga_recommended_threshold_presets.csv")
    presets.to_csv(tables_dir / "global_operating_points.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/paper/manuscript",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir
    tables_dir = output_dir
    generated_assets = [
        *output_dir.glob("figure_*.pdf"),
        *output_dir.glob("figure_*.png"),
        *output_dir.glob("figure_*.jpg"),
        *output_dir.glob("fig*.pdf"),
        *output_dir.glob("fig*.png"),
        *output_dir.glob("*.csv"),
    ] if output_dir.exists() else []
    if generated_assets and not args.overwrite:
        raise FileExistsError(
            f"Generated manuscript assets already exist in: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    copy_figure_1(figures_dir)
    build_energy_matching(figures_dir)
    build_representation_architecture(figures_dir)
    build_shortcut_controls(figures_dir)
    build_classification(figures_dir, tables_dir)
    scalar_controls = build_domain_audit(figures_dir)
    build_thorium_spectra(figures_dir)
    build_thorium_atlas(figures_dir)
    build_operating_metrics(figures_dir, tables_dir)
    build_tables(tables_dir)
    held_out_report = load_json(PAPER1_LINEAGE_DIR / "held_out/held_out_evaluation.json")
    model_lineage = held_out_report["models"]["ds_cnn"]
    manifest = {
        "status": "PAPER_1_MANUSCRIPT_ASSETS_BUILT",
        "model_lineage": {
            "checkpoint": model_lineage["checkpoint"],
            "checkpoint_sha256": model_lineage["checkpoint_sha256"],
            "selected_peak_weights": model_lineage["selected_peak_weights"],
            "selection_rule": held_out_report["lineage"]["selection_rule"],
            "held_out_scores": held_out_report["score_artifact"],
            "training_labels": (
                PAPER1_LABEL_DIR / "label_pairs_train.csv"
            ).relative_to(PROJECT_ROOT).as_posix(),
            "held_out_labels": (
                PAPER1_LABEL_DIR / "label_pairs_validation.csv"
            ).relative_to(PROJECT_ROOT).as_posix(),
            "strict_gate_impurity_report": STRICT_PEAK_PURITY_REPORT.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "th232_score_cache": (
                PAPER1_LINEAGE_DIR / "th232/th232_ds_cnn_scores.h5"
            ).relative_to(PROJECT_ROOT).as_posix(),
            "global_threshold_results": PAPER1_GLOBAL_DIR.relative_to(PROJECT_ROOT).as_posix(),
            "peak_relative_detection_limit_results": PAPER1_RDL_DIR.relative_to(PROJECT_ROOT).as_posix(),
        },
        "figures": sorted(
            path.name
            for pattern in ("figure_*.pdf", "figure_*.png", "figure_*.jpg")
            for path in output_dir.glob(pattern)
        ),
        "tables": sorted(path.name for path in output_dir.glob("*.csv")),
        "strict_three_peak_scalar_controls": scalar_controls,
        "note": (
            "All selected-model manuscript merits use the same canonical equal-weight "
            "strict three-peak DS-CNN checkpoint. Assets are colocated with paper1.tex."
        ),
    }
    (output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
