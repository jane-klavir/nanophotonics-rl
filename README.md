# Nanophotonics RL

This project explores ML/RL methods for nanophotonic design.

The first step is to use `miepython` to generate optical spectra for spherical nanoparticles. We vary physical parameters such as particle radius, refractive index, absorption coefficient, and wavelength, then compute scattering, extinction, and absorption spectra.

The long-term goal is to build a surrogate model that predicts optical spectra from geometry/material parameters, then use optimization or reinforcement learning to search for parameters that produce a desired target response.

## Project Plan

1. Generate synthetic spectra using Mie theory with `miepython`.
2. Save the data in an ML-ready format.
3. Train surrogate models such as Gaussian Process Regression, PCA + regression, or neural networks.
4. Adapt the same pipeline to COMSOL-generated data.
5. Use the surrogate model inside an optimization/RL loop for inverse design.

## Setup

First, create and activate a Python environment using either `venv`, `conda`, or another environment manager.

Then, from the project root, install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the dataset generation script:
```bash
python scripts/generate_dataset.py
```

Plot a random generated spectrum:
```bash
python scripts/plot_random_spectrum.py
```

Generate Mie spectra for gold spheres across a sweep of radii:
```bash
python scripts/generate_gold_spectra.py
```
This writes `data/processed/mie_gold_v1.npz` and saves a σ_sca / σ_abs vs. λ figure to `outputs/figures/gold_radius_sweep.png`.

## Dataset format

Datasets are saved as `.npz` files with a unified schema that supports both
synthetic constant-`(n, k)` samples and tabulated material data with
wavelength-dependent `n(λ), k(λ)`:

| Key | Shape | Description |
| --- | --- | --- |
| `wavelengths_nm` | `(W,)` | Shared wavelength grid (nm) |
| `X_scalar` | `(N, 2)` | Per-sample scalars: `[radius_nm, n_medium]` |
| `X_material` | `(N, W, 2)` | Per-sample `[n(λ), k(λ)]` (constant samples are broadcast across `W`) |
| `material_id` | `(N,)` | String tag, e.g. `"synthetic_const"`, `"gold"` |
| `Y_qext`, `Y_qsca`, `Y_qabs`, `Y_qback`, `Y_g` | `(N, W)` | Mie efficiencies and asymmetry parameter |
| `Y_sigma_ext`, `Y_sigma_sca`, `Y_sigma_abs` | `(N, W)` | Cross-sections (`q · π r²`) |
| `Y_log_sigma_*_over_geo` | `(N, W)` | `log(q + ε)` — useful for training targets |

The same schema is used regardless of whether `n, k` come from a synthetic grid
sweep or a tabulated reference material — constant-in-λ samples simply have
`X_material[i, :, 0]` and `X_material[i, :, 1]` set to constant arrays.

## Materials

`nano_mie.materials` exposes tabulated complex refractive indices, interpolated
onto an arbitrary wavelength grid. Currently included:

- `gold` — tabulated gold `n(λ), k(λ)` (the same table shipped with `miepython`'s
  `04_gold.py` example, converted to nm).