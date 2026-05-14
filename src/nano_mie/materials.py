from pathlib import Path
import numpy as np


MATERIALS_DIR = Path(__file__).resolve().parents[2] / "Materials"


def load_material_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a refractiveindex.info-style CSV with two stacked tables — `wl,n` and
    `wl,k` separated by a blank line — and wavelengths in micrometers.

    Returns four arrays: (wl_n_um, n_values, wl_k_um, k_values). The n and k
    tables may have different wavelength samples; we don't merge them here, the
    caller interpolates onto a target grid.
    """
    text = Path(csv_path).read_text().splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(current)

    if len(blocks) != 2:
        raise ValueError(
            f"Expected 2 blocks (wl,n and wl,k) in {csv_path}, got {len(blocks)}"
        )

    def parse_block(block: list[str]) -> tuple[np.ndarray, np.ndarray]:
        rows = [row.split(",") for row in block[1:]]
        wl_um = np.array([float(r[0]) for r in rows], dtype=float)
        val = np.array([float(r[1]) for r in rows], dtype=float)
        return wl_um, val

    wl_n_um, n_vals = parse_block(blocks[0])
    wl_k_um, k_vals = parse_block(blocks[1])
    return wl_n_um, n_vals, wl_k_um, k_vals


def material_nk(name: str, wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Load material `name` from `<repo>/Materials/<name>.csv` and linearly
    interpolate n(λ), k(λ) onto the supplied wavelength grid (nm).

    Wavelengths outside the tabulated range are clamped to endpoint values
    (np.interp's default behavior).
    """
    csv_path = MATERIALS_DIR / f"{name}.csv"
    wl_n_um, n_vals, wl_k_um, k_vals = load_material_csv(csv_path)
    wl_target_um = np.asarray(wavelengths_nm, dtype=float) / 1000.0
    n_arr = np.interp(wl_target_um, wl_n_um, n_vals)
    k_arr = np.interp(wl_target_um, wl_k_um, k_vals)
    return n_arr, k_arr


def available_materials() -> list[str]:
    """Names of all CSV materials in the `Materials/` folder, sorted."""
    return sorted(p.stem for p in MATERIALS_DIR.glob("*.csv"))


def gold_nk(wavelengths_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat shim — loads Au.csv via material_nk."""
    return material_nk("Au", wavelengths_nm)


MATERIALS = {"gold": gold_nk}
