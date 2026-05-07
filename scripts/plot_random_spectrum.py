from pathlib import Path
import numpy as np

from nano_mie.dataset import load_dataset
from nano_mie.plotting import plot_spectrum


def main() -> None:
    dataset_path = Path("data/processed/mie_dataset_v1.npz")
    dataset = load_dataset(dataset_path)

    wavelengths_nm = dataset["wavelengths_nm"]
    X_scalar = dataset["X_scalar"]
    X_material = dataset["X_material"]
    material_id = dataset["material_id"]

    rng = np.random.default_rng(seed=0)
    idx = int(rng.integers(0, len(X_scalar)))

    radius_nm = X_scalar[idx, 0]
    n_lambda = X_material[idx, :, 0]
    k_lambda = X_material[idx, :, 1]

    plot_spectrum(
        wavelengths_nm=wavelengths_nm,
        qext=dataset["Y_qext"][idx],
        qsca=dataset["Y_qsca"][idx],
        qabs=dataset["Y_qabs"][idx],
        radius_nm=radius_nm,
        n=n_lambda,
        k=k_lambda,
        material_id=str(material_id[idx]),
        output_path=Path("outputs/figures/random_spectrum.png"),
    )


if __name__ == "__main__":
    main()
