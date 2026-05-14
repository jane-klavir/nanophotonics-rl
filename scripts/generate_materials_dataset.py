"""
Generate a Mie spectra dataset across every material in the Materials/ folder.

For each material:
  * load tabulated n(λ), k(λ) from Materials/<name>.csv
  * interpolate onto a shared wavelength grid (300–900 nm, 10 nm step → 61 pts)
  * sample N radii uniformly from [50, 1000] nm
  * compute σ_sca(λ), σ_abs(λ), σ_ext(λ) via Mie

Saves a single compressed .npz with:
  wavelengths_nm:  (W,)                shared λ grid (nm)
  material_names:  (M,)                e.g. ['Ag','Al','Au',...]
  materials_n:     (M, W)              n(λ) for each material on the grid
  materials_k:     (M, W)              k(λ) for each material on the grid
  radius_nm:       (N_total,)          per-run radius
  n_medium:        (N_total,)          per-run medium index
  material_id:     (N_total,)          per-run material name (string)
  geometry:        (N_total,)          per-run geometry tag, e.g. 'sphere'
  sigma_sca:       (N_total, W)        scattering cross-section vs λ (nm²)
  sigma_abs:       (N_total, W)        absorption cross-section vs λ (nm²)
  sigma_ext:       (N_total, W)        extinction = sigma_sca + sigma_abs (nm²)
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from nano_mie.materials import material_nk, available_materials
from nano_mie.simulator import compute_mie_spectrum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples-per-material", type=int, default=100)
    p.add_argument("--radius-min-nm", type=float, default=50.0)
    p.add_argument("--radius-max-nm", type=float, default=1000.0)
    p.add_argument("--wavelength-min-nm", type=float, default=300.0)
    p.add_argument("--wavelength-max-nm", type=float, default=900.0)
    p.add_argument("--wavelength-step-nm", type=float, default=10.0)
    p.add_argument("--n-medium", type=float, default=1.0, help="Air = 1.0")
    p.add_argument("--geometry", type=str, default="sphere",
                   help="Geometry tag stored per run (currently only 'sphere').")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mie_materials_v1.npz"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_steps = int(round(
        (args.wavelength_max_nm - args.wavelength_min_nm) / args.wavelength_step_nm
    )) + 1
    wavelengths_nm = np.linspace(args.wavelength_min_nm, args.wavelength_max_nm, n_steps)
    W = wavelengths_nm.size

    material_names = available_materials()
    M = len(material_names)
    if M == 0:
        raise RuntimeError("No .csv files found in Materials/")

    materials_n = np.empty((M, W), dtype=float)
    materials_k = np.empty((M, W), dtype=float)
    for i, name in enumerate(material_names):
        materials_n[i], materials_k[i] = material_nk(name, wavelengths_nm)

    rng = np.random.default_rng(args.seed)
    N_per = args.n_samples_per_material
    N_total = N_per * M

    radius_nm = np.empty(N_total, dtype=float)
    n_medium = np.full(N_total, args.n_medium, dtype=float)
    material_id = np.empty(N_total, dtype=object)
    geometry = np.full(N_total, args.geometry, dtype=object)
    sigma_sca = np.empty((N_total, W), dtype=float)
    sigma_abs = np.empty((N_total, W), dtype=float)
    sigma_ext = np.empty((N_total, W), dtype=float)

    for m_idx, name in enumerate(material_names):
        radii = rng.uniform(args.radius_min_nm, args.radius_max_nm, size=N_per)
        n_lam = materials_n[m_idx]
        k_lam = materials_k[m_idx]
        for j, r in enumerate(tqdm(radii, desc=name, leave=False)):
            idx = m_idx * N_per + j
            spec = compute_mie_spectrum(
                radius_nm=float(r),
                n=n_lam,
                k=k_lam,
                wavelengths_nm=wavelengths_nm,
                n_medium=args.n_medium,
            )
            radius_nm[idx] = float(r)
            material_id[idx] = name
            sigma_sca[idx] = spec["sigma_sca"]
            sigma_abs[idx] = spec["sigma_abs"]
            sigma_ext[idx] = spec["sigma_ext"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wavelengths_nm=wavelengths_nm,
        material_names=np.array(material_names),
        materials_n=materials_n,
        materials_k=materials_k,
        radius_nm=radius_nm,
        n_medium=n_medium,
        material_id=material_id.astype(str),
        geometry=geometry.astype(str),
        sigma_sca=sigma_sca,
        sigma_abs=sigma_abs,
        sigma_ext=sigma_ext,
    )

    print(f"\nSaved dataset to: {args.output}")
    print(f"  wavelengths_nm: {wavelengths_nm.shape} (range {wavelengths_nm[0]:.0f}–{wavelengths_nm[-1]:.0f} nm)")
    print(f"  material_names: {list(material_names)}")
    print(f"  materials_n / materials_k: {materials_n.shape}")
    print(f"  per-run arrays: {N_total} runs × {W} wavelengths")
    print(f"  radii sampled uniformly in [{args.radius_min_nm}, {args.radius_max_nm}] nm "
          f"with seed {args.seed}")
    print(f"  geometry: '{args.geometry}'")


if __name__ == "__main__":
    main()
