import numpy as np
from tqdm import tqdm

from nano_mie.config import MieDatasetConfig
from nano_mie.simulator import compute_mie_spectrum


SPECTRUM_KEYS = (
    "qext", "qsca", "qabs", "qback", "g",
    "sigma_ext", "sigma_sca", "sigma_abs",
    "log_sigma_ext_over_geo", "log_sigma_sca_over_geo", "log_sigma_abs_over_geo",
)


def _empty_y(n_samples: int, n_wavelengths: int) -> dict[str, np.ndarray]:
    return {key: np.empty((n_samples, n_wavelengths), dtype=float) for key in SPECTRUM_KEYS}


def build_dataset_from_samples(
    samples: list[dict],
    wavelengths_nm: np.ndarray,
    n_medium: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Generic dataset builder. Each `sample` is a dict with:
      - radius_nm: float
      - n: scalar or (W,) array
      - k: scalar or (W,) array
      - material_id: str
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    W = wavelengths_nm.size
    N = len(samples)

    X_scalar = np.empty((N, 2), dtype=float)
    X_material = np.empty((N, W, 2), dtype=float)
    material_id = np.empty(N, dtype=object)

    Y = _empty_y(N, W)

    for i, sample in enumerate(tqdm(samples)):
        spec = compute_mie_spectrum(
            radius_nm=sample["radius_nm"],
            n=sample["n"],
            k=sample["k"],
            wavelengths_nm=wavelengths_nm,
            n_medium=n_medium,
        )
        X_scalar[i] = [sample["radius_nm"], n_medium]
        X_material[i, :, 0] = spec["n_lambda"]
        X_material[i, :, 1] = spec["k_lambda"]
        material_id[i] = sample.get("material_id", "synthetic_const")

        for key in SPECTRUM_KEYS:
            Y[key][i] = spec[key]

    return {
        "wavelengths_nm": wavelengths_nm,
        "X_scalar": X_scalar,
        "X_material": X_material,
        "material_id": material_id.astype(str),
        **{f"Y_{k}": v for k, v in Y.items()},
    }


def generate_dataset(config: MieDatasetConfig) -> dict[str, np.ndarray]:
    """
    Synthetic grid sweep over (radius, n, k) with constant-in-λ material.
    """
    samples = [
        {"radius_nm": r, "n": n, "k": k, "material_id": "synthetic_const"}
        for r in config.radii_nm
        for n in config.n_values
        for k in config.k_values
    ]
    return build_dataset_from_samples(
        samples=samples,
        wavelengths_nm=config.wavelengths_nm,
        n_medium=config.n_medium,
    )


def save_dataset(dataset: dict[str, np.ndarray], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)


def load_dataset(path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
