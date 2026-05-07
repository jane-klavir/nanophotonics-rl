from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _fmt_nk(n: np.ndarray, k: np.ndarray) -> str:
    n = np.asarray(n)
    k = np.asarray(k)
    if n.size == 1 or np.allclose(n, n.flat[0]):
        n_label = f"n={float(n.flat[0]):.2f}"
    else:
        n_label = f"n(λ)∈[{n.min():.2f},{n.max():.2f}]"
    if k.size == 1 or np.allclose(k, k.flat[0]):
        k_label = f"k={float(k.flat[0]):.2f}"
    else:
        k_label = f"k(λ)∈[{k.min():.2f},{k.max():.2f}]"
    return f"{n_label}, {k_label}"


def plot_spectrum(
    wavelengths_nm: np.ndarray,
    qext: np.ndarray,
    qsca: np.ndarray,
    qabs: np.ndarray,
    radius_nm: float,
    n,
    k,
    material_id: str | None = None,
    output_path: Path | None = None,
) -> None:
    plt.figure(figsize=(7, 4))

    plt.plot(wavelengths_nm, qext, label="Qext")
    plt.plot(wavelengths_nm, qsca, label="Qsca")
    plt.plot(wavelengths_nm, qabs, label="Qabs")

    nk_label = _fmt_nk(np.asarray(n), np.asarray(k))
    mat_prefix = f"{material_id} | " if material_id else ""
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.title(f"{mat_prefix}r={float(radius_nm):.1f} nm, {nk_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()
