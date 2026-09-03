#!/usr/bin/env python3
"""Generate publication-quality diagnostic and evaluation plots for peak specialists.

Generates all 12 plots required by the Peak-Specialist HPGe CNN Experiment Plan:
1. cross_energy_auroc_matrix.png
2. cross_energy_auprc_matrix.png
3. specialist_roc_curves.png
4. specialist_score_distributions.png
5. score_356A_vs_energy.png
6. score_511A_vs_energy.png
7. score_661A_vs_energy.png
8. fusion_comparison.png
9. th232_spectra_before_after.png
10. th232_photopeak_retention_vs_energy.png
11. th232_pb_improvement_vs_energy.png
12. best_specialist_vs_energy.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/peak_specialist_ds_cnn_20260820"

PEAK_IDS = ("ba133_356kev", "na22_511kev", "cs137_662kev")
PEAK_LABELS = {
    "ba133_356kev": "356 keV (Ba-133)",
    "na22_511kev": "511 keV (Na-22)",
    "cs137_662kev": "662 keV (Cs-137)",
}
SPECIALISTS = ("356A", "511A", "661A")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def plot_cross_energy_matrices(exp_dir: Path) -> None:
    csv_path = exp_dir / "cross_energy_transfer_matrix.csv"
    if not csv_path.is_file():
        return
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    auroc_matrix = np.zeros((3, 3))
    auprc_matrix = np.zeros((3, 3))

    for row in rows:
        row_idx = PEAK_IDS.index(row["evaluation_domain"])
        col_idx = SPECIALISTS.index(row["specialist"])
        auroc_matrix[row_idx, col_idx] = float(row["auroc"])
        auprc_matrix[row_idx, col_idx] = float(row["auprc"])

    # 1. AUROC matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(auroc_matrix, cmap="YlGnBu", vmin=0.55, vmax=0.72)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SPECIALISTS)
    ax.set_yticklabels([PEAK_LABELS[p] for p in PEAK_IDS])
    ax.set_xlabel("Trained Specialist")
    ax.set_ylabel("Held-Out Evaluation Domain")
    ax.set_title("Cross-Energy Transfer AUROC Matrix")

    for i in range(3):
        for j in range(3):
            val = auroc_matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    plt.tight_layout()
    plt.savefig(exp_dir / "cross_energy_auroc_matrix.png", dpi=300)
    plt.close()

    # 2. AUPRC matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(auprc_matrix, cmap="YlOrRd", vmin=0.55, vmax=0.72)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SPECIALISTS)
    ax.set_yticklabels([PEAK_LABELS[p] for p in PEAK_IDS])
    ax.set_xlabel("Trained Specialist")
    ax.set_ylabel("Held-Out Evaluation Domain")
    ax.set_title("Cross-Energy Transfer AUPRC Matrix")

    for i in range(3):
        for j in range(3):
            val = auprc_matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUPRC (Average Precision)")
    plt.tight_layout()
    plt.savefig(exp_dir / "cross_energy_auprc_matrix.png", dpi=300)
    plt.close()


def plot_roc_and_distributions(exp_dir: Path) -> None:
    npz_path = exp_dir / "held_out_specialist_scores.npz"
    if not npz_path.is_file():
        return
    data = np.load(npz_path)
    labels = data["true_label"]
    peak_domains = data["peak_domain"]
    energies = data["energy_kev"]

    s_356 = data["score_356A"]
    s_511 = data["score_511A"]
    s_661 = data["score_661A"]
    s_joint = data["score_joint_ds_cnn"]
    s_sel = data["selected_fusion_score"]

    # 3. Specialist ROC curves per domain
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    models_to_plot = [
        ("356A", s_356, "#1f77b4"),
        ("511A", s_511, "#2ca02c"),
        ("661A", s_661, "#d62728"),
        ("Selected Fusion", s_sel, "#9467bd"),
        ("Joint DS-CNN", s_joint, "#333333"),
    ]

    for ax_idx, pid in enumerate(PEAK_IDS):
        ax = axes[ax_idx]
        mask = peak_domains == pid
        y_true = labels[mask]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

        for mname, scores, col in models_to_plot:
            fpr, tpr, _ = roc_curve(y_true, scores[mask])
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, scores[mask])
            ls = "--" if mname == "Joint DS-CNN" else "-"
            lw = 2 if mname in ("Selected Fusion", "Joint DS-CNN") else 1.5
            ax.plot(fpr, tpr, label=f"{mname} ({auc:.3f})", color=col, linestyle=ls, linewidth=lw)

        ax.set_title(PEAK_LABELS[pid])
        ax.set_xlabel("False Positive Rate")
        if ax_idx == 0:
            ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.suptitle("Held-Out ROC Curves by Photopeak Domain", y=1.02)
    plt.tight_layout()
    plt.savefig(exp_dir / "specialist_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Specialist score distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    spec_score_arrays = [("356A", s_356), ("511A", s_511), ("661A", s_661)]

    for row_idx, (spec_name, s_arr) in enumerate(spec_score_arrays):
        for col_idx, pid in enumerate(PEAK_IDS):
            ax = axes[row_idx, col_idx]
            mask = peak_domains == pid
            y_sub = labels[mask]
            s_sub = s_arr[mask]

            pos_sc = s_sub[y_sub == 1]
            neg_sc = s_sub[y_sub == 0]

            ax.hist(pos_sc, bins=40, range=(0, 1), density=True, alpha=0.6, color="blue", label="Photopeak (Pos)")
            ax.hist(neg_sc, bins=40, range=(0, 1), density=True, alpha=0.6, color="red", label="Continuum (Neg)")

            if row_idx == 0:
                ax.set_title(f"Eval: {PEAK_LABELS[pid]}")
            if col_idx == 0:
                ax.set_ylabel(f"{spec_name} Density")
            if row_idx == 2:
                ax.set_xlabel("Classifier Score")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Specialist Score Distributions Across Domains (Held-Out Validation)", y=1.01)
    plt.tight_layout()
    plt.savefig(exp_dir / "specialist_score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5, 6, 7. Score vs Energy for each specialist
    for spec_name, s_arr, fname, col in [
        ("356A", s_356, "score_356A_vs_energy.png", "#1f77b4"),
        ("511A", s_511, "score_511A_vs_energy.png", "#2ca02c"),
        ("661A", s_661, "score_661A_vs_energy.png", "#d62728"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos_e = energies[labels == 1]
        pos_s = s_arr[labels == 1]
        neg_e = energies[labels == 0]
        neg_s = s_arr[labels == 0]

        ax.scatter(pos_e, pos_s, alpha=0.2, s=8, color="blue", label="Photopeak (Label 1)")
        ax.scatter(neg_e, neg_s, alpha=0.2, s=8, color="red", label="Continuum (Label 0)")

        ax.set_xlabel("Reconstructed Energy (keV)")
        ax.set_ylabel(f"Score ({spec_name})")
        ax.set_title(f"Specialist {spec_name} Score vs Energy (Held-Out Validation)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(exp_dir / fname, dpi=300)
        plt.close()


def plot_fusion_comparison(exp_dir: Path) -> None:
    csv_path = exp_dir / "fusion_results.csv"
    if not csv_path.is_file():
        return
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    names = [r["model_or_rule"] for r in rows]
    macro_aurocs = [float(r["held_out_macro_auroc"]) for r in rows]
    worst_aurocs = [float(r["held_out_worst_peak_auroc"]) for r in rows]
    weighted_aurocs = [float(r["held_out_weighted_auroc"]) for r in rows]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, macro_aurocs, width, label="Macro AUROC", color="#1f77b4", alpha=0.9)
    ax.bar(x, worst_aurocs, width, label="Worst-Energy AUROC", color="#d62728", alpha=0.9)
    ax.bar(x + width, weighted_aurocs, width, label="Weighted AUROC", color="#2ca02c", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.50, 0.72)
    ax.set_title("Comparison of Specialists, Fusion Rules, and Joint DS-CNN Baseline (Held-Out)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(exp_dir / "fusion_comparison.png", dpi=300)
    plt.close()


def plot_th232_results(exp_dir: Path) -> None:
    spec_csv = exp_dir / "th232_specialist_results.csv"
    fusion_csv = exp_dir / "th232_fusion_results.csv"
    scores_h5 = exp_dir / "th232_specialist_scores.h5"

    if not spec_csv.is_file() or not fusion_csv.is_file():
        return

    spec_rows: list[dict[str, Any]] = []
    with spec_csv.open(newline="", encoding="utf-8") as stream:
        spec_rows = list(csv.DictReader(stream))

    fusion_rows: list[dict[str, Any]] = []
    with fusion_csv.open(newline="", encoding="utf-8") as stream:
        fusion_rows = list(csv.DictReader(stream))

    all_rows = spec_rows + fusion_rows

    # 9. Th-232 spectra before/after filtering
    if scores_h5.is_file():
        with h5py.File(scores_h5, "r") as h5in:
            energies = h5in["energy_kev"][:]
            s_joint = h5in["score_joint_ds_cnn"][:]
            s_nearest = h5in["score_nearest_expert"][:]

        eval_summary_path = exp_dir / "th232_specialist_evaluation.json"
        if eval_summary_path.is_file():
            th_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            th_catalog = th_summary.get("threshold_catalog", {})

            fig, ax = plt.subplots(figsize=(12, 6))
            bins = np.linspace(100, 1000, 901)

            # Raw spectrum
            ax.hist(energies, bins=bins, histtype="step", color="black", label="No Cut (Raw Th-232)", linewidth=1.2)

            # Joint DS-CNN 90% and 50%
            if "joint_ds_cnn" in th_catalog:
                thr_90 = th_catalog["joint_ds_cnn"].get("90pct", 0.5)
                thr_50 = th_catalog["joint_ds_cnn"].get("50pct", 0.5)
                ax.hist(
                    energies[s_joint >= thr_90],
                    bins=bins,
                    histtype="step",
                    color="blue",
                    label=f"Joint DS-CNN 90% Pass",
                    linewidth=1.0,
                    linestyle="--",
                )
                ax.hist(
                    energies[s_joint >= thr_50],
                    bins=bins,
                    histtype="step",
                    color="deepskyblue",
                    label=f"Joint DS-CNN 50% Pass",
                    linewidth=1.0,
                )

            # Nearest expert 90% and 50%
            if "nearest_energy_expert" in th_catalog:
                thr_90_n = th_catalog["nearest_energy_expert"].get("90pct", 0.5)
                thr_50_n = th_catalog["nearest_energy_expert"].get("50pct", 0.5)
                ax.hist(
                    energies[s_nearest >= thr_90_n],
                    bins=bins,
                    histtype="step",
                    color="red",
                    label=f"Nearest Expert 90% Pass",
                    linewidth=1.0,
                    linestyle="--",
                )
                ax.hist(
                    energies[s_nearest >= thr_50_n],
                    bins=bins,
                    histtype="step",
                    color="magenta",
                    label=f"Nearest Expert 50% Pass",
                    linewidth=1.0,
                )

            ax.set_yscale("log")
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts / 1 keV")
            ax.set_title("Th-232 Energy Spectra Before and After Specialist / Joint Filtering")
            ax.set_xlim(100, 1000)
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(exp_dir / "th232_spectra_before_after.png", dpi=300)
            plt.close()

    # 10. Th-232 photopeak retention vs energy
    # 11. Th-232 P/B improvement vs energy
    # 12. Best specialist vs energy
    models_for_pb = ["joint_ds_cnn", "356A", "511A", "661A", "nearest_energy_expert", "calibrated_mean"]
    cut_to_plot = "90pct"  # Standard high-acceptance operating cut

    fig_ret, ax_ret = plt.subplots(figsize=(9, 5))
    fig_pb, ax_pb = plt.subplots(figsize=(9, 5))

    colors = {
        "joint_ds_cnn": "black",
        "356A": "#1f77b4",
        "511A": "#2ca02c",
        "661A": "#d62728",
        "nearest_energy_expert": "#9467bd",
        "calibrated_mean": "#ff7f0e",
    }

    peak_energies_sorted = sorted({float(r["reference_energy_kev"]) for r in all_rows})

    for mname in models_for_pb:
        m_rows = [r for r in all_rows if r["model_or_rule"] == mname and r["cut_name"] == cut_to_plot]
        if not m_rows:
            continue
        m_rows.sort(key=lambda r: float(r["reference_energy_kev"]))
        e_vals = [float(r["reference_energy_kev"]) for r in m_rows]
        ret_vals = [float(r["peak_retention"]) * 100.0 for r in m_rows]
        pb_gains = [float(r["pb_relative_improvement"]) * 100.0 for r in m_rows]

        ls = "--" if mname == "joint_ds_cnn" else "-o"
        ax_ret.plot(e_vals, ret_vals, ls, label=mname, color=colors.get(mname, "gray"), linewidth=1.5)
        ax_pb.plot(e_vals, pb_gains, ls, label=mname, color=colors.get(mname, "gray"), linewidth=1.5)

    ax_ret.set_xlabel("Photopeak Energy (keV)")
    ax_ret.set_ylabel("Net Photopeak Retention (%)")
    ax_ret.set_title(f"Th-232 Photopeak Retention vs Energy ({cut_to_plot} Operating Cut)")
    ax_ret.legend(loc="lower left")
    plt.tight_layout()
    fig_ret.savefig(exp_dir / "th232_photopeak_retention_vs_energy.png", dpi=300)
    plt.close(fig_ret)

    ax_pb.set_xlabel("Photopeak Energy (keV)")
    ax_pb.set_ylabel("P/B Relative Improvement (%)")
    ax_pb.set_title(f"Th-232 Peak-to-Background Improvement vs Energy ({cut_to_plot} Operating Cut)")
    ax_pb.legend(loc="upper left")
    plt.tight_layout()
    fig_pb.savefig(exp_dir / "th232_pb_improvement_vs_energy.png", dpi=300)
    plt.close(fig_pb)

    # 12. Best specialist vs energy
    fig_best, ax_best = plt.subplots(figsize=(9, 5))
    for p_energy in peak_energies_sorted:
        p_rows = [
            r
            for r in spec_rows
            if float(r["reference_energy_kev"]) == p_energy
            and r["cut_name"] == cut_to_plot
            and r["model_or_rule"] in SPECIALISTS
        ]
        if p_rows:
            best_r = max(p_rows, key=lambda r: float(r["pb_relative_improvement"]))
            ax_best.scatter(
                p_energy,
                float(best_r["pb_relative_improvement"]) * 100.0,
                color=colors.get(best_r["model_or_rule"], "gray"),
                s=100,
                zorder=5,
            )
            ax_best.annotate(
                f"{best_r['model_or_rule']}\n(+{float(best_r['pb_relative_improvement'])*100.0:.1f}%)",
                (p_energy, float(best_r["pb_relative_improvement"]) * 100.0),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

    # Add legend handles for specialists
    for sname in SPECIALISTS:
        ax_best.scatter([], [], color=colors[sname], s=80, label=f"Best: {sname}")

    ax_best.set_xlabel("Photopeak Energy (keV)")
    ax_best.set_ylabel("P/B Relative Improvement (%)")
    ax_best.set_title(f"Best Performing Specialist vs Energy on Th-232 ({cut_to_plot} Cut)")
    ax_best.legend(loc="upper left")
    plt.tight_layout()
    fig_best.savefig(exp_dir / "best_specialist_vs_energy.png", dpi=300)
    plt.close(fig_best)


def main() -> int:
    setup_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args()

    print(f"Generating diagnostic and evaluation plots in {args.experiment_dir}...")
    plot_cross_energy_matrices(args.experiment_dir)
    plot_roc_and_distributions(args.experiment_dir)
    plot_fusion_comparison(args.experiment_dir)
    plot_th232_results(args.experiment_dir)
    print("All plots generated successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
