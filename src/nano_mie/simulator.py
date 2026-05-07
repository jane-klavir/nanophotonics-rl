import numpy as np
import miepython as mie


def _broadcast_to_lambda(value, n_wavelengths: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_wavelengths, float(arr))
    if arr.shape != (n_wavelengths,):
        raise ValueError(
            f"Expected scalar or array of shape ({n_wavelengths},), got {arr.shape}"
        )
    return arr


def compute_mie_spectrum(
    radius_nm: float,
    n,
    k,
    wavelengths_nm: np.ndarray,
    n_medium: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Compute Mie scattering spectra for a homogeneous sphere.

    `n` and `k` may be scalars (constant in wavelength) or arrays of shape
    (n_wavelengths,) for tabulated material data — see miepython's 04_gold.py.
    miepython convention: m = n - i*k.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    W = wavelengths_nm.size

    n_arr = _broadcast_to_lambda(n, W)
    k_arr = _broadcast_to_lambda(k, W)

    m_particle = n_arr - 1j * k_arr
    m = m_particle / n_medium
    x = 2.0 * np.pi * radius_nm * n_medium / wavelengths_nm

    qext, qsca, qback, g = mie.efficiencies_mx(m, x)
    qabs = qext - qsca

    area_geo = np.pi * radius_nm**2
    sigma_ext = qext * area_geo
    sigma_sca = qsca * area_geo
    sigma_abs = qabs * area_geo

    eps = 1e-12
    log_sigma_ext_over_geo = np.log(sigma_ext / area_geo + eps)
    log_sigma_sca_over_geo = np.log(sigma_sca / area_geo + eps)
    log_sigma_abs_over_geo = np.log(sigma_abs / area_geo + eps)

    return {
        "qext": qext,
        "qsca": qsca,
        "qabs": qabs,
        "qback": qback,
        "g": g,
        "sigma_ext": sigma_ext,
        "sigma_sca": sigma_sca,
        "sigma_abs": sigma_abs,
        "log_sigma_ext_over_geo": log_sigma_ext_over_geo,
        "log_sigma_sca_over_geo": log_sigma_sca_over_geo,
        "log_sigma_abs_over_geo": log_sigma_abs_over_geo,
        "n_lambda": n_arr,
        "k_lambda": k_arr,
    }
