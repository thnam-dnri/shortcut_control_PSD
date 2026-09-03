#!/usr/bin/env python3
"""Plot aggregated energy spectra for Ba-133, Co-60, Cs-137, Na-22, and Th-232."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SOURCES = ["ba133", "co60", "cs137", "na22", "th232"]

SOURCE_INFO = {
    "ba133": {
        "name": "Ba-133",
        "color": "#1f77b4",
        "peaks": [(81.0, "81.0 keV"), (276.4, "276.4"), (302.9, "302.9"), (356.0, "356.0"), (383.8, "383.8")],
        "max_e": 600.0,
    },
    "co60": {
        "name": "Co-60",
        "color": "#d62728",
        "peaks": [(1173.2, "1173.2 keV"), (1332.5, "1332.5 keV")],
        "max_e": 1600.0,
    },
    "cs137": {
        "name": "Cs-137",
        "color": "#2ca02c",
        "peaks": [(661.7, "661.7 keV")],
        "max_e": 900.0,
    },
    "na22": {
        "name": "Na-22",
        "color": "#ff7f0e",
        "peaks": [(511.0, "511.0 keV (annih.)"), (1274.5, "1274.5 keV")],
        "max_e": 1500.0,
    },
    "th232": {
        "name": "Th-232 Chain",
        "color": "#9467bd",
        "peaks": [
            (238.6, "238.6 keV (Pb-212)"),
            (583.2, "583.2 keV (Tl-208)"),
            (911.2, "911.2 keV (Ac-228)"),
            (969.0, "969.0 keV (Ac-228)"),
            (2614.5, "2614.5 keV (Tl-208)"),
        ],
        "max_e": 3000.0,
    },
}


def load_source_energies(hdf5_dir: Path, source: str) -> np.ndarray:
    energies: list[np.ndarray] = []
    # Search both top-level and subdirectories (e.g. th232_evaluation_20260813)
    files = sorted(hdf5_dir.rglob(f"{source}_*.h5"))
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as h5:
                if "corrected_energy_kev" in h5:
                    e = h5["corrected_energy_kev"][:]
                elif "reconstructed_energy_kev" in h5:
                    e = h5["reconstructed_energy_kev"][:]
                else:
                    continue
                valid = np.isfinite(e) & (e > 0)
                energies.append(e[valid])
        except Exception as err:
            print(f"Warning: failed reading {file_path.name}: {err}")
    if not energies:
        return np.array([], dtype=np.float32)
    return np.concatenate(energies)


def plot_single_source_spectrum(
    energy: np.ndarray,
    src: str,
    output_dir: Path,
    bin_width_kev: float = 1.0,
) -> Path:
    info = SOURCE_INFO[src]
    bins = np.arange(0.0, info["max_e"] + bin_width_kev, bin_width_kev)
    counts, edges = np.histogram(energy, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Save CSV
    csv_path = output_dir / f"{src}_spectrum_{int(bin_width_kev)}kev.csv"
    np.savetxt(
        csv_path,
        np.column_stack([centers, counts]),
        delimiter=",",
        header="energy_kev,counts",
        comments="",
        fmt="%.2f,%d",
    )

    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # Linear scale
    ax_lin.plot(centers, counts, color=info["color"], lw=1.2, label=f"{info['name']} ({len(energy):,} events)")
    ax_lin.set_title(f"{info['name']} Energy Spectrum — Linear Scale ({len(energy):,} events)", fontsize=13, fontweight="bold")
    ax_lin.set_xlabel("Reconstructed Energy (keV)", fontsize=11)
    ax_lin.set_ylabel(f"Counts / {bin_width_kev:g} keV", fontsize=11)
    ax_lin.set_xlim(0, info["max_e"])
    max_c = np.max(counts[centers > 50]) if np.any(centers > 50) else np.max(counts)
    ax_lin.set_ylim(0, max_c * 1.25)
    ax_lin.grid(True, alpha=0.3, linestyle="--")

    for peak_e, label in info["peaks"]:
        if peak_e <= info["max_e"]:
            near_mask = (centers >= peak_e - 6) & (centers <= peak_e + 6)
            peak_count = np.max(counts[near_mask]) if np.any(near_mask) else 0
            ax_lin.axvline(peak_e, color="black", linestyle=":", alpha=0.5, lw=1)
            ax_lin.annotate(
                label,
                xy=(peak_e, peak_count),
                xytext=(peak_e, peak_count + max_c * 0.08),
                arrowprops=dict(facecolor="black", shrink=0.05, width=0.8, headwidth=4),
                fontsize=8.5,
                ha="center",
                fontweight="bold",
            )
    ax_lin.legend(loc="upper right", framealpha=0.9)

    # Log scale
    ax_log.semilogy(centers, np.maximum(counts, 0.5), color=info["color"], lw=1.2, label=f"{info['name']}")
    ax_log.set_title(f"{info['name']} Energy Spectrum — Logarithmic Scale", fontsize=13, fontweight="bold")
    ax_log.set_xlabel("Reconstructed Energy (keV)", fontsize=11)
    ax_log.set_ylabel(f"Counts / {bin_width_kev:g} keV", fontsize=11)
    ax_log.set_xlim(0, info["max_e"])
    ax_log.set_ylim(bottom=1)
    ax_log.grid(True, alpha=0.3, which="both", linestyle="--")

    for peak_e, label in info["peaks"]:
        if peak_e <= info["max_e"]:
            ax_log.axvline(peak_e, color="black", linestyle=":", alpha=0.5, lw=1)
            near_mask = (centers >= peak_e - 6) & (centers <= peak_e + 6)
            peak_count = np.max(counts[near_mask]) if np.any(near_mask) else 1
            ax_log.annotate(
                label,
                xy=(peak_e, peak_count),
                xytext=(peak_e, peak_count * 2.5),
                arrowprops=dict(facecolor="black", shrink=0.05, width=0.8, headwidth=4),
                fontsize=8.5,
                ha="center",
                fontweight="bold",
            )
    ax_log.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = output_dir / f"{src}_energy_spectrum.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=Path("processed_data/waveform_hdf5_corrected"),
        help="Directory containing corrected HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spectra"),
        help="Output directory for spectrum plots and CSVs",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Specific source to plot (ba133, co60, cs137, na22, th232) or all if omitted",
    )
    parser.add_argument(
        "--bin-width-kev",
        type=float,
        default=1.0,
        help="Energy bin width in keV",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources_to_plot = [args.source] if args.source else SOURCES
    for src in sources_to_plot:
        energy = load_source_energies(args.hdf5_dir, src)
        if energy.size == 0:
            print(f"No energy data found for {src}")
            continue
        print(f"Loaded {len(energy):,} events for {src}")
        plot_single_source_spectrum(energy, src, args.output_dir, args.bin_width_kev)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
