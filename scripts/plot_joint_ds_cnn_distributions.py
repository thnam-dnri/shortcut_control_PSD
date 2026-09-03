"""Plots positive vs negative score distributions for the Joint 3-peak DS-CNN."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "outputs/experiments/peak_specialist_ds_cnn_20260820"
SCORES_NPZ = EXP_DIR / "held_out_specialist_scores.npz"
OUTPUT_PLOT = EXP_DIR / "joint_ds_cnn_score_distributions.png"
ARTIFACT_DIR = Path("/home/adminministrator/.gemini/antigravity-cli/brain/61943258-80d3-4cd0-a202-107518e80860")


def main() -> None:
    data = np.load(SCORES_NPZ)
    scores = data["score_joint_ds_cnn"]
    labels = data["true_label"]
    peaks = data["peak_domain"]

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    # Calculate statistics
    print("=== Global Joint DS-CNN Score Distribution Statistics ===")
    print(f"Total events: {len(scores):,} (Positive: {len(pos_scores):,}, Negative: {len(neg_scores):,})")
    print(f"Positive: Mean = {np.mean(pos_scores):.4f}, Std = {np.std(pos_scores):.4f}, Median = {np.median(pos_scores):.4f}")
    print(f"Negative: Mean = {np.mean(neg_scores):.4f}, Std = {np.std(neg_scores):.4f}, Median = {np.median(neg_scores):.4f}")
    
    ks_res = stats.ks_2samp(pos_scores, neg_scores)
    print(f"Kolmogorov-Smirnov stat: {ks_res.statistic:.4f} (p-val: {ks_res.pvalue:.2e})")
    
    cohen_d = (np.mean(pos_scores) - np.mean(neg_scores)) / np.sqrt((np.var(pos_scores) + np.var(neg_scores)) / 2.0)
    print(f"Cohen's d: {cohen_d:.4f}")

    # Set up 2x2 multi-panel figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    bins = np.linspace(0.0, 1.0, 60)

    # Panel (a): Overall Global Distribution
    ax0 = axes[0, 0]
    ax0.hist(neg_scores, bins=bins, density=True, alpha=0.55, color="#d95f02", label=f"Negative / Continuum (N={len(neg_scores):,})", edgecolor="none")
    ax0.hist(pos_scores, bins=bins, density=True, alpha=0.55, color="#1b9e77", label=f"Positive / Photopeak (N={len(pos_scores):,})", edgecolor="none")
    
    # Kernel density estimates
    kde_bins = np.linspace(0.0, 1.0, 200)
    kde_neg = stats.gaussian_kde(neg_scores)
    kde_pos = stats.gaussian_kde(pos_scores)
    ax0.plot(kde_bins, kde_neg(kde_bins), color="#a63603", linewidth=2.0)
    ax0.plot(kde_bins, kde_pos(kde_bins), color="#006d2c", linewidth=2.0)

    ax0.axvline(np.mean(neg_scores), color="#d95f02", linestyle="--", linewidth=1.5, label=f"Neg Mean: {np.mean(neg_scores):.3f}")
    ax0.axvline(np.mean(pos_scores), color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Pos Mean: {np.mean(pos_scores):.3f}")

    ax0.set_title("Overall Held-Out Score Distribution (Macro AUROC: 0.6621)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Joint DS-CNN Sigmoid Output Score", fontsize=11)
    ax0.set_ylabel("Probability Density", fontsize=11)
    ax0.legend(loc="upper right", frameon=True, fontsize=9)
    ax0.set_xlim(0.0, 1.0)
    ax0.text(0.03, 0.92, f"KS Stat: {ks_res.statistic:.3f}\nCohen's d: {cohen_d:.3f}", transform=ax0.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panels for each peak
    peak_configs = [
        ("ba133_356kev", "Ba-133 356 keV", axes[0, 1], 0.6310),
        ("na22_511kev", "Na-22 511 keV", axes[1, 0], 0.6760),
        ("cs137_662kev", "Cs-137 662 keV", axes[1, 1], 0.6792),
    ]

    for p_key, p_title, ax, auroc in peak_configs:
        p_mask = peaks == p_key
        p_pos = scores[p_mask & pos_mask]
        p_neg = scores[p_mask & neg_mask]

        ax.hist(p_neg, bins=bins, density=True, alpha=0.55, color="#d95f02", label=f"Negative (N={len(p_neg):,})", edgecolor="none")
        ax.hist(p_pos, bins=bins, density=True, alpha=0.55, color="#1b9e77", label=f"Positive (N={len(p_pos):,})", edgecolor="none")

        kde_p_neg = stats.gaussian_kde(p_neg)
        kde_p_pos = stats.gaussian_kde(p_pos)
        ax.plot(kde_bins, kde_p_neg(kde_bins), color="#a63603", linewidth=2.0)
        ax.plot(kde_bins, kde_p_pos(kde_bins), color="#006d2c", linewidth=2.0)

        ax.axvline(np.mean(p_neg), color="#d95f02", linestyle="--", linewidth=1.5, label=f"Neg Mean: {np.mean(p_neg):.3f}")
        ax.axvline(np.mean(p_pos), color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Pos Mean: {np.mean(p_pos):.3f}")

        p_ks = stats.ks_2samp(p_pos, p_neg).statistic
        p_d = (np.mean(p_pos) - np.mean(p_neg)) / np.sqrt((np.var(p_pos) + np.var(p_neg)) / 2.0)

        ax.set_title(f"{p_title} (AUROC: {auroc:.4f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Joint DS-CNN Sigmoid Output Score", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.text(0.03, 0.92, f"KS Stat: {p_ks:.3f}\nCohen's d: {p_d:.3f}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        print(f"\n--- {p_title} ---")
        print(f"Positive: Mean = {np.mean(p_pos):.4f}, Std = {np.std(p_pos):.4f}, Median = {np.median(p_pos):.4f}")
        print(f"Negative: Mean = {np.mean(p_neg):.4f}, Std = {np.std(p_neg):.4f}, Median = {np.median(p_neg):.4f}")
        print(f"KS Stat: {p_ks:.4f}, Cohen's d: {p_d:.4f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"\nSaved score distribution plot to {OUTPUT_PLOT}")

    # Copy to artifact directory for display
    if ARTIFACT_DIR.exists():
        import shutil
        dest = ARTIFACT_DIR / "joint_ds_cnn_score_distributions.png"
        shutil.copy(OUTPUT_PLOT, dest)
        print(f"Copied to artifact directory: {dest}")


if __name__ == "__main__":
    main()
