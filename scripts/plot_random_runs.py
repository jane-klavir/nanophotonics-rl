"""
Pick N random runs from the materials dataset and plot their σ_sca and σ_abs
spectra on two stacked panels, colour-coded by material.

Default dataset:  data/processed/mie_materials_v1.npz
Default output:   outputs/figures/random_runs.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


MATERIAL_COLORS = {
    "Ag":   "#1f77b4",  # blue
    "Al":   "#2ca02c",  # green
    "Au":   "#ff7f0e",  # orange
    "Cu":   "#d62728",  # red
    "GaAs": "#9467bd",  # purple
    "Ge":   "#17becf",  # cyan
    "Si":   "#8c564b",  # brown
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=Path("data/processed/mie_materials_v1.npz"))
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--radius-min-nm", type=float, default=None,
                   help="Restrict picks to runs with radius >= this (nm)")
    p.add_argument("--radius-max-nm", type=float, default=None,
                   help="Restrict picks to runs with radius <= this (nm)")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/figures/random_runs.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    wl = data["wavelengths_nm"]
    radius_nm = data["radius_nm"]
    material_id = data["material_id"]
    sigma_sca = data["sigma_sca"]
    sigma_abs = data["sigma_abs"]

    candidate_mask = np.ones(radius_nm.size, dtype=bool)
    if args.radius_min_nm is not None:
        candidate_mask &= radius_nm >= args.radius_min_nm
    if args.radius_max_nm is not None:
        candidate_mask &= radius_nm <= args.radius_max_nm
    candidates = np.where(candidate_mask)[0]
    if candidates.size == 0:
        raise SystemExit("No runs match the requested radius range.")

    rng = np.random.default_rng(args.seed)
    n_runs = min(args.n_runs, candidates.size)
    indices = rng.choice(candidates, size=n_runs, replace=False)
    # Sort by material then radius so the legend reads cleanly
    indices = sorted(indices, key=lambda i: (str(material_id[i]), float(radius_nm[i])))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_sca, ax_abs = axes

    used_materials: set[str] = set()
    for i in indices:
        mat = str(material_id[i])
        r = float(radius_nm[i])
        color = MATERIAL_COLORS.get(mat, "#666666")
        label = f"{mat}, r={r:.0f} nm"
        ax_sca.plot(wl, sigma_sca[i], color=color, linewidth=1.8, label=label)
        ax_abs.plot(wl, sigma_abs[i], color=color, linewidth=1.8, label=label)
        used_materials.add(mat)

    ax_sca.set_ylabel("σ_sca (nm²)", fontsize=11)
    ax_sca.set_title(f"{n_runs} random runs from {args.dataset.name}", fontsize=12)
    ax_sca.grid(True, alpha=0.4)
    ax_sca.legend(fontsize=8, loc="best", ncol=2)

    ax_abs.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_abs.set_ylabel("σ_abs (nm²)", fontsize=11)
    ax_abs.grid(True, alpha=0.4)
    ax_abs.set_xlim(wl.min(), wl.max())

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")

    print(f"Plotted {n_runs} runs (seed={args.seed}) — materials: {sorted(used_materials)}")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
