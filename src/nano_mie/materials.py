import numpy as np


GOLD_JC_LAMBDA_NM = np.array(
    [
        187.9, 191.6, 195.3, 199.3, 203.3, 207.3, 211.9, 216.4, 221.4, 226.2,
        231.3, 237.1, 242.6, 249.0, 255.1, 261.6, 268.9, 276.1, 284.4, 292.4,
        300.9, 310.7, 320.4, 331.5, 342.5, 354.2, 367.9, 381.5, 397.4, 413.3,
        430.5, 450.9, 471.4, 495.9, 520.9, 548.6, 582.1, 616.8, 659.5, 704.5,
        756.0, 821.1, 892.0, 984.0, 1088.0, 1216.0, 1393.0, 1610.0, 1937.0,
    ]
)

GOLD_JC_N = np.array(
    [
        1.28, 1.32, 1.34, 1.33, 1.33, 1.30, 1.30, 1.30, 1.30, 1.31,
        1.30, 1.32, 1.32, 1.33, 1.33, 1.35, 1.38, 1.43, 1.47, 1.49,
        1.53, 1.53, 1.54, 1.48, 1.48, 1.50, 1.48, 1.46, 1.47, 1.46,
        1.45, 1.38, 1.31, 1.04, 0.62, 0.43, 0.29, 0.21, 0.14, 0.13,
        0.14, 0.16, 0.17, 0.22, 0.27, 0.35, 0.43, 0.56, 0.92,
    ]
)

GOLD_JC_K = np.array(
    [
        1.188, 1.203, 1.226, 1.251, 1.277, 1.304, 1.350, 1.387, 1.427, 1.460,
        1.497, 1.536, 1.577, 1.631, 1.688, 1.749, 1.803, 1.847, 1.869, 1.878,
        1.889, 1.893, 1.898, 1.883, 1.871, 1.866, 1.895, 1.933, 1.952, 1.958,
        1.948, 1.914, 1.849, 1.833, 2.081, 2.455, 2.863, 3.272, 3.697, 4.103,
        4.542, 5.083, 5.663, 6.350, 7.150, 8.145, 9.519, 11.21, 13.78,
    ]
)


def gold_jc_nk(wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Johnson & Christy gold complex refractive index, interpolated onto the
    requested wavelength grid (nm). Source table: refractiveindex.info / Au / Johnson,
    as shipped with miepython's 04_gold.py example.

    Wavelengths outside the tabulated range [187.9, 1937] nm are clipped to the
    endpoint values via np.interp's default behavior.
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    n_arr = np.interp(wl, GOLD_JC_LAMBDA_NM, GOLD_JC_N)
    k_arr = np.interp(wl, GOLD_JC_LAMBDA_NM, GOLD_JC_K)
    return n_arr, k_arr


MATERIALS = {
    "gold_JC": gold_jc_nk,
}
