"""
Generate Mie spectra for Johnson & Christy gold spheres across a sweep of radii.

Mirrors the miepython 04_gold.py example, but as a dataset entry conforming to
the unified schema (X_scalar, X_material, Y_*) and a side-by-side plot of
σ_sca(λ) and σ_abs(λ) for all radii.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nano_mie.dataset import build_dataset_from_samples, save_dataset
from nano_mie.materials import gold_jc_nk


RADII_NM = [20.0, 40.0, 60.0, 80.0, 100.0, 125.0]
WAVELENGTHS_NM = np.linspace(300.0, 900.0, 301)
N_MEDIUM = 1.0
OUTPUT_PATH = Path("data/processed/mie_gold_v1.npz")
FIGURE_PATH = Path("outputs/figures/gold_radius_sweep.png")


def main() -> None:
    n_au, k_au = gold_jc_nk(WAVELENGTHS_NM)

    samples = [
        {"radius_nm": r, "n": n_au, "k": k_au, "material_id": "gold_JC"}
        for r in RADII_NM
    ]

    dataset = build_dataset_from_samples(
        samples=samples,
        wavelengths_nm=WAVELENGTHS_NM,
        n_medium=N_MEDIUM,
    )
    save_dataset(dataset, OUTPUT_PATH)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for i, r in enumerate(RADII_NM):
        axes[0].plot(WAVELENGTHS_NM, dataset["Y_sigma_sca"][i], label=f"r={r:.0f} nm")
        axes[1].plot(WAVELENGTHS_NM, dataset["Y_sigma_abs"][i], label=f"r={r:.0f} nm")

    axes[0].set_ylabel("σ_sca (nm²)")
    axes[0].set_title("Gold (Johnson & Christy) — radius sweep")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("σ_abs (nm²)")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)

    print(f"Saved gold dataset to: {OUTPUT_PATH}")
    print(f"Saved radius-sweep figure to: {FIGURE_PATH}")
    print(f"X_scalar shape: {dataset['X_scalar'].shape}")
    print(f"X_material shape: {dataset['X_material'].shape}")
    print(f"Y_sigma_sca shape: {dataset['Y_sigma_sca'].shape}")


if __name__ == "__main__":
    main()
