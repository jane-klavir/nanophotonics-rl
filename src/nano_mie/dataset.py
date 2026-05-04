import numpy as np
from tqdm import tqdm

from nano_mie.config import MieDatasetConfig
from nano_mie.simulator import compute_mie_spectrum


def generate_dataset(config: MieDatasetConfig) -> dict[str, np.ndarray]:
    """
    Generate an ML-ready Mie dataset.

    Inputs:
        X[:, 0] = radius_nm
        X[:, 1] = n
        X[:, 2] = k

    Outputs:
        - Y_qext, Y_qsca, Y_qabs, Y_qback, Y_g
        - Y_sigma_ext, Y_sigma_sca, Y_sigma_abs
        - Y_log_sigma_ext_over_geo, Y_log_sigma_sca_over_geo, Y_log_sigma_abs_over_geo

        each with shape (n_samples, n_wavelengths)
    """
    wavelengths_nm = config.wavelengths_nm

    parameter_grid = [
        (radius_nm, n, k)
        for radius_nm in config.radii_nm
        for n in config.n_values
        for k in config.k_values
    ]

    n_samples = len(parameter_grid)
    n_wavelengths = len(wavelengths_nm)

    X = np.empty((n_samples, 3), dtype=float)

    Y_qext = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_qsca = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_qabs = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_qback = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_g = np.empty((n_samples, n_wavelengths), dtype=float)

    Y_sigma_ext = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_sigma_sca = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_sigma_abs = np.empty((n_samples, n_wavelengths), dtype=float)

    Y_log_sigma_ext_over_geo = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_log_sigma_sca_over_geo = np.empty((n_samples, n_wavelengths), dtype=float)
    Y_log_sigma_abs_over_geo = np.empty((n_samples, n_wavelengths), dtype=float)

    for idx, (radius_nm, n, k) in enumerate(tqdm(parameter_grid)):
        spectrum = compute_mie_spectrum(
            radius_nm=radius_nm,
            n=n,
            k=k,
            wavelengths_nm=wavelengths_nm,
        )

        X[idx] = np.array([radius_nm, n, k])

        Y_qext[idx] = spectrum["qext"]
        Y_qsca[idx] = spectrum["qsca"]
        Y_qabs[idx] = spectrum["qabs"]
        Y_qback[idx] = spectrum["qback"]
        Y_g[idx] = spectrum["g"]

        Y_sigma_ext[idx] = spectrum["sigma_ext"]
        Y_sigma_sca[idx] = spectrum["sigma_sca"]
        Y_sigma_abs[idx] = spectrum["sigma_abs"]

        Y_log_sigma_ext_over_geo[idx] = spectrum["log_sigma_ext_over_geo"]
        Y_log_sigma_sca_over_geo[idx] = spectrum["log_sigma_sca_over_geo"]
        Y_log_sigma_abs_over_geo[idx] = spectrum["log_sigma_abs_over_geo"]

    return {
        "wavelengths_nm": wavelengths_nm,
        "X": X,
        "Y_qext": Y_qext,
        "Y_qsca": Y_qsca,
        "Y_qabs": Y_qabs,
        "Y_qback": Y_qback,
        "Y_g": Y_g,
        "Y_sigma_ext": Y_sigma_ext,
        "Y_sigma_sca": Y_sigma_sca,
        "Y_sigma_abs": Y_sigma_abs,
        "Y_log_sigma_ext_over_geo": Y_log_sigma_ext_over_geo,
        "Y_log_sigma_sca_over_geo": Y_log_sigma_sca_over_geo,
        "Y_log_sigma_abs_over_geo": Y_log_sigma_abs_over_geo,
    }


def save_dataset(dataset: dict[str, np.ndarray], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)


def load_dataset(path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}