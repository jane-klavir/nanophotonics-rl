from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(
    wavelengths_nm: np.ndarray,
    qext: np.ndarray,
    qsca: np.ndarray,
    qabs: np.ndarray,
    radius_nm: float,
    n: float,
    k: float,
    output_path: Path | None = None,
) -> None:
    plt.figure(figsize=(7, 4))

    plt.plot(wavelengths_nm, qext, label="Qext")
    plt.plot(wavelengths_nm, qsca, label="Qsca")
    plt.plot(wavelengths_nm, qabs, label="Qabs")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.title(f"r={radius_nm:.1f} nm, n={n:.2f}, k={k:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()