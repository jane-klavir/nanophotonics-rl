from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .data_utils import MieData


@dataclass
class ErrorScales:
    wl_scale: float
    sca_sigma_scale: float
    abs_sigma_scale: float


@dataclass
class ErrorWeights:
    peak_wl: float = 0.80
    sca_sig: float = 0.10
    abs_sig: float = 0.10

    def validate(self) -> None:
        total = self.peak_wl + self.sca_sig + self.abs_sig
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1. Current sum: {total}")


def build_peak_dataframe(
    data: MieData,
    peak_mode: int = 1,
    prominence_frac: float = 0.14,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    wl = data.wavelengths_nm

    for i in range(len(data.radius_nm)):
        y_sca = data.sigma_sca[i]
        y_abs = data.sigma_abs[i]
        anchor_signal = y_sca if peak_mode == 1 else y_abs

        local_peaks, _ = find_peaks(
            anchor_signal,
            prominence=prominence_frac * anchor_signal.max(),
        )
        global_peak = int(np.argmax(anchor_signal))
        all_peaks = np.unique(np.append(local_peaks, global_peak))

        for p in all_peaks:
            rows.append({
                "combo_idx": i,
                "material": str(data.material_id[i]),
                "radius_nm": float(data.radius_nm[i]),
                "peak_lambda_nm": float(wl[p]),
                "sigma_sca_nm2": float(y_sca[p]),
                "sigma_abs_nm2": float(y_abs[p]),
                "is_global": bool(p == global_peak),
            })

    peak_df = pd.DataFrame(rows)

    if verbose:
        anchor_name = "scattering" if peak_mode == 1 else "absorption"
        print(f"Mode          : PEAK_MODE={peak_mode} ({anchor_name}-anchored)")
        print(f"Total peaks   : {len(peak_df)}")
        print(f"Unique combos : {peak_df['combo_idx'].nunique()}")

    return peak_df


def build_combo_lookup(
    peak_df: pd.DataFrame,
    global_only: int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, ErrorScales]:
    lookup_src = peak_df[peak_df["is_global"]].copy() if global_only else peak_df.copy()

    scales = ErrorScales(
        wl_scale=float(lookup_src["peak_lambda_nm"].max() - lookup_src["peak_lambda_nm"].min()),
        sca_sigma_scale=float(lookup_src["sigma_sca_nm2"].max()),
        abs_sigma_scale=float(lookup_src["sigma_abs_nm2"].max()),
    )

    combo_lookup = (
        lookup_src
        .groupby(["material", "radius_nm"], as_index=False)
        .agg(
            peak_nm=("peak_lambda_nm", "first"),
            sca_sigma_nm2=("sigma_sca_nm2", "first"),
            abs_sigma_nm2=("sigma_abs_nm2", "first"),
            is_global=("is_global", "first"),
        )
    )

    if verbose:
        label = "global-only" if global_only else "all-peaks"
        print(f"Lookup source : {label} ({len(lookup_src)} rows)")
        print(f"combo_lookup  : {len(combo_lookup)} rows | materials: {combo_lookup['material'].nunique()}")

    return combo_lookup, scales
