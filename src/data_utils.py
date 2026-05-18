from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class MieData:
    wavelengths_nm: np.ndarray
    material_id: np.ndarray
    radius_nm: np.ndarray
    sigma_sca: np.ndarray
    sigma_abs: np.ndarray


def load_mie_data(path: str | Path, verbose: bool = True) -> MieData:
    """Load precomputed Mie simulation data from an .npz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = np.load(path, allow_pickle=False)
    required = ["wavelengths_nm", "material_id", "radius_nm", "sigma_sca", "sigma_abs"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    mie_data = MieData(
        wavelengths_nm=data["wavelengths_nm"],
        material_id=data["material_id"],
        radius_nm=data["radius_nm"],
        sigma_sca=data["sigma_sca"],
        sigma_abs=data["sigma_abs"],
    )

    if verbose:
        print(f"Loaded: {path}")
        print(f"Wavelengths : {mie_data.wavelengths_nm.shape}")
        print(f"Combos      : {mie_data.material_id.shape}")
        print(f"Radii       : {mie_data.radius_nm.shape}")
        print(f"sigma_sca   : {mie_data.sigma_sca.shape}")
        print(f"sigma_abs   : {mie_data.sigma_abs.shape}")

    return mie_data
