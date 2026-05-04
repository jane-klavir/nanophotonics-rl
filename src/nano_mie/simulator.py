import numpy as np
import miepython as mie


def compute_mie_spectrum(
    radius_nm: float,
    n: float,
    k: float,
    wavelengths_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute Mie scattering spectra for a homogeneous sphere.

    Parameters
    ----------
    radius_nm:
        Sphere radius in nm.
    n:
        Real part of refractive index.
    k:
        Absorption coefficient. miepython convention uses m = n - i k.
    wavelengths_nm:
        Wavelength grid in nm.

    Returns
    -------
    dict containing qext, qsca, qabs, qback, and g spectra.
    """
    diameter_nm = 2.0 * radius_nm
    m = n - 1j * k

    qext = np.empty_like(wavelengths_nm, dtype=float)
    qsca = np.empty_like(wavelengths_nm, dtype=float)
    qback = np.empty_like(wavelengths_nm, dtype=float)
    g = np.empty_like(wavelengths_nm, dtype=float)

    for i, wavelength_nm in enumerate(wavelengths_nm):
        qext_i, qsca_i, qback_i, g_i = mie.efficiencies(
            m,
            diameter_nm,
            wavelength_nm,
        )

        qext[i] = qext_i
        qsca[i] = qsca_i
        qback[i] = qback_i
        g[i] = g_i

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
    }