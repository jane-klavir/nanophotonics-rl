
"""
Generate Mie spectra for gold spheres across a sweep of radii.

Mirrors the miepython 04_gold.py example. Produces a 3-panel figure:
  (1) n(λ) and k(λ) of gold (material — shared across all radii)
  (2) Bulk absorption depth λ/(4π k) (material — shared across all radii)
  (3) σ_sca(λ) and σ_abs(λ), one curve per radius (Mie — sweep)

Also writes the dataset to disk in the unified schema (X_scalar, X_material, Y_*).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nano_mie.dataset import build_dataset_from_samples, save_dataset
from nano_mie.materials import gold_nk


RADII_NM = [20.0, 40.0, 80.0, 100.0, 150.0, 200.0]
WAVELENGTHS_NM = np.linspace(300.0, 900.0, 301)
N_MEDIUM = 1.0
OUTPUT_PATH = Path("data/processed/mie_gold_v1.npz")
FIGURE_PATH = Path("outputs/figures/gold_radius_sweep.png")


def main() -> None:
    n_au, k_au = gold_nk(WAVELENGTHS_NM)

    samples = [
        {"radius_nm": r, "n": n_au, "k": k_au, "material_id": "gold"}
        for r in RADII_NM
    ]

    dataset = build_dataset_from_samples(
        samples=samples,
        wavelengths_nm=WAVELENGTHS_NM,
        n_medium=N_MEDIUM,
    )
    save_dataset(dataset, OUTPUT_PATH)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1, 3],
        hspace=0.32, wspace=0.30,
    )
    ax_nk = fig.add_subplot(gs[0, 0])
    ax_depth = fig.add_subplot(gs[0, 1])
    ax_xs = fig.add_subplot(gs[1, :])

    # Top-left: dispersion n(λ), k(λ) on shared x with twin y-axes
    ax_nk.plot(WAVELENGTHS_NM, n_au, color="tab:blue", label="n(λ)")
    ax_nk.set_ylabel("n", color="tab:blue")
    ax_nk.tick_params(axis="y", labelcolor="tab:blue")
    ax_nk.set_xlabel("Wavelength (nm)")
    ax_nk.grid(True, alpha=0.4)
    ax_nk.set_title("Gold dispersion n(λ), k(λ)", fontsize=11)
    ax_k = ax_nk.twinx()
    ax_k.plot(WAVELENGTHS_NM, k_au, color="tab:red", label="k(λ)")
    ax_k.set_ylabel("k", color="tab:red")
    ax_k.tick_params(axis="y", labelcolor="tab:red")

    # Top-right: bulk absorption depth λ / (4π k)
    eps = 1e-12
    absorption_depth_nm = WAVELENGTHS_NM / (4.0 * np.pi * k_au + eps)
    ax_depth.plot(WAVELENGTHS_NM, absorption_depth_nm, color="tab:purple")
    ax_depth.set_ylabel("Absorption depth (nm)")
    ax_depth.set_xlabel("Wavelength (nm)")
    ax_depth.grid(True, alpha=0.4)
    ax_depth.set_title("Bulk gold absorption depth λ/(4π k)", fontsize=11)

    # Bottom (full width): σ_sca and σ_abs across radii
    radius_palette = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#d62728",  # red
        "#9467bd",  # purple
        "#17becf",  # cyan
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]
    radius_handles = []
    for i, r in enumerate(RADII_NM):
        color = radius_palette[i % len(radius_palette)]
        line, = ax_xs.plot(
            WAVELENGTHS_NM, dataset["Y_sigma_sca"][i],
            color=color, linestyle="-", linewidth=2.0,
            label=f"r = {r:.0f} nm",
        )
        ax_xs.plot(
            WAVELENGTHS_NM, dataset["Y_sigma_abs"][i],
            color=color, linestyle="--", linewidth=1.5, alpha=0.9,
        )
        radius_handles.append(line)

    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="σ_sca"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="σ_abs"),
    ]

    ax_xs.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_xs.set_ylabel("Cross section (nm²)", fontsize=12)
    ax_xs.set_title("Mie cross-sections of gold spheres — radius sweep", fontsize=12)
    ax_xs.tick_params(labelsize=10)
    ax_xs.grid(True, alpha=0.4)
    ax_xs.set_xlim(WAVELENGTHS_NM.min(), WAVELENGTHS_NM.max())

    leg_radius = ax_xs.legend(
        handles=radius_handles, loc="upper right",
        title="Radius", fontsize=10, title_fontsize=10, frameon=True,
    )
    ax_xs.add_artist(leg_radius)
    ax_xs.legend(
        handles=style_handles, loc="upper left",
        title="Channel", fontsize=10, title_fontsize=10, frameon=True,
    )

    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")

    print(f"Saved gold dataset to: {OUTPUT_PATH}")
    print(f"Saved radius-sweep figure to: {FIGURE_PATH}")
    print(f"X_scalar shape: {dataset['X_scalar'].shape}")
    print(f"X_material shape: {dataset['X_material'].shape}")
    print(f"Y_sigma_sca shape: {dataset['Y_sigma_sca'].shape}")


if __name__ == "__main__":
    main()
