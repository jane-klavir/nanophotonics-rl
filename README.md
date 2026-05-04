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