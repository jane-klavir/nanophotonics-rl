from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class MieDatasetConfig:
    wavelength_min_nm: float = 300.0
    wavelength_max_nm: float = 900.0
    n_wavelengths: int = 301

    radius_min_nm: float = 20.0
    radius_max_nm: float = 125.0
    n_radii: int = 80

    n_min: float = 1.3
    n_max: float = 4.0
    n_refractive_indices: int = 40

    k_min: float = 0.0
    k_max: float = 1.0
    n_absorption_values: int = 25

    n_medium: float = 1.0

    output_dir: Path = Path("data/processed")
    output_filename: str = "mie_dataset_v1.npz"

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return np.linspace(
            self.wavelength_min_nm,
            self.wavelength_max_nm,
            self.n_wavelengths,
        )

    @property
    def radii_nm(self) -> np.ndarray:
        return np.linspace(
            self.radius_min_nm,
            self.radius_max_nm,
            self.n_radii,
        )

    @property
    def n_values(self) -> np.ndarray:
        return np.linspace(
            self.n_min,
            self.n_max,
            self.n_refractive_indices,
        )

    @property
    def k_values(self) -> np.ndarray:
        return np.linspace(
            self.k_min,
            self.k_max,
            self.n_absorption_values,
        )

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename