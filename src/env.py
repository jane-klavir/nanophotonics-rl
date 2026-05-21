import random
import numpy as np
import pandas as pd
from .peak_utils import ErrorScales, ErrorWeights


class MieEnv:
    N_ACTIONS = 5
    STATE_DIM = 9
    STOP_BONUS = 3.0
    STOP_PENALTY = -0.2
    STOP_THRESHOLD = 0.10
    MAX_STEPS = 150
    RADIUS_STEP = 1
    REWARD_SCALE = 10.0
    STEP_PENALTY = 0.01
    LAMBDA_STOP_ERR = 10.0

    def __init__(
        self,
        combo_lookup: pd.DataFrame,
        peak_mode: int,
        global_only: int,
        scales: ErrorScales,
        weights: ErrorWeights | None = None,
    ):
        self.lookup = combo_lookup.copy()
        self.peak_mode = float(peak_mode)
        self.global_only = float(global_only)
        self.scales = scales
        self.weights = weights if weights is not None else ErrorWeights()
        self.weights.validate()

        self.mat_list = sorted(self.lookup["material"].unique())
        self.n_mats = len(self.mat_list)

        self.wl_min = self.lookup["peak_nm"].min()
        self.wl_max = self.lookup["peak_nm"].max()
        self.sca_sig_min = self.lookup["sca_sigma_nm2"].min()
        self.sca_sig_max = self.lookup["sca_sigma_nm2"].max()
        self.abs_sig_min = self.lookup["abs_sigma_nm2"].min()
        self.abs_sig_max = self.lookup["abs_sigma_nm2"].max()

        self._build_index()

    def _build_index(self) -> None:
        self.mat_radii = {}
        self.peak_table = {}

        for mat in self.mat_list:
            sub = self.lookup[self.lookup["material"] == mat].sort_values("radius_nm")
            self.mat_radii[mat] = sub["radius_nm"].tolist()
            for r_idx, row in enumerate(sub.itertuples()):
                self.peak_table[(mat, r_idx)] = (
                    row.peak_nm,
                    row.sca_sigma_nm2,
                    row.abs_sigma_nm2,
                    row.is_global,
                )

    @staticmethod
    def _norm(val: float, lo: float, hi: float) -> float:
        return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _compute_error(self, mat: str, r_idx: int) -> float:
        peak_wl, sca_sig, abs_sig, _ = self.peak_table[(mat, r_idx)]
        e_wl = abs(peak_wl - self.target_peak_wl) / (self.scales.wl_scale + 1e-9)
        e_sca = abs(sca_sig - self.target_sca_sig) / (self.scales.sca_sigma_scale + 1e-9)
        e_abs = abs(abs_sig - self.target_abs_sig) / (self.scales.abs_sigma_scale + 1e-9)
        return self.weights.peak_wl * e_wl + self.weights.sca_sig * e_sca + self.weights.abs_sig * e_abs

    def _get_state(self) -> np.ndarray:
        peak_wl, sca_sig, abs_sig, _ = self.peak_table[(self.mat, self.r_idx)]
        return np.array([
            self._norm(self.r_idx, 0, len(self.mat_radii[self.mat]) - 1),
            self._norm(self.mat_idx, 0, self.n_mats - 1),
            self._norm(peak_wl, self.wl_min, self.wl_max),
            self._norm(sca_sig, self.sca_sig_min, self.sca_sig_max),
            self._norm(abs_sig, self.abs_sig_min, self.abs_sig_max),
            self._norm(self.target_peak_wl, self.wl_min, self.wl_max),
            self._norm(self.target_sca_sig, self.sca_sig_min, self.sca_sig_max),
            self._norm(self.target_abs_sig, self.abs_sig_min, self.abs_sig_max),
            float(np.clip(self.current_error, 0.0, 1.0)),
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        row = self.lookup.sample(1).iloc[0]
        self.target_peak_wl = float(row["peak_nm"])
        self.target_sca_sig = float(row["sca_sigma_nm2"])
        self.target_abs_sig = float(row["abs_sigma_nm2"])

        self.mat_idx = random.randrange(self.n_mats)
        self.mat = self.mat_list[self.mat_idx]
        self.r_idx = random.randrange(len(self.mat_radii[self.mat]))
        self.steps = 0

        self.current_error = self._compute_error(self.mat, self.r_idx)
        return self._get_state()

    def set_target(self, target_wl: float, target_sca: float, target_abs: float) -> np.ndarray:
        self.target_peak_wl = float(target_wl)
        self.target_sca_sig = float(target_sca)
        self.target_abs_sig = float(target_abs)
        self.current_error = self._compute_error(self.mat, self.r_idx)
        return self._get_state()

    def step(self, action: int):
        self.steps += 1
        old_error = self.current_error
        n_radii = len(self.mat_radii[self.mat])

        if action == 0:
            self.r_idx = max(0, self.r_idx - self.RADIUS_STEP)
        elif action == 1:
            self.r_idx = min(n_radii - 1, self.r_idx + self.RADIUS_STEP)
        elif action == 2:
            self.mat_idx = (self.mat_idx - 1) % self.n_mats
            self.mat = self.mat_list[self.mat_idx]
            self.r_idx = min(self.r_idx, len(self.mat_radii[self.mat]) - 1)
        elif action == 3:
            self.mat_idx = (self.mat_idx + 1) % self.n_mats
            self.mat = self.mat_list[self.mat_idx]
            self.r_idx = min(self.r_idx, len(self.mat_radii[self.mat]) - 1)
        elif action == 4:
            wl_err = abs(self.peak_table[(self.mat, self.r_idx)][0] - self.target_peak_wl)
            bonus = self.STOP_BONUS if wl_err < self.LAMBDA_STOP_ERR and old_error < self.STOP_THRESHOLD else self.STOP_PENALTY
            return self._get_state(), bonus, True, self._info()
        else:
            raise ValueError(f"Unknown action: {action}")

        self.current_error = self._compute_error(self.mat, self.r_idx)
        reward = (old_error - self.current_error) * self.REWARD_SCALE - self.STEP_PENALTY
        done = self.steps >= self.MAX_STEPS
        return self._get_state(), reward, done, self._info()

    def _info(self) -> dict:
        peak_wl, sca_sig, abs_sig, is_global = self.peak_table[(self.mat, self.r_idx)]
        return {
            "material": self.mat,
            "radius_nm": self.mat_radii[self.mat][self.r_idx],
            "peak_wl": peak_wl,
            "sca_sig": sca_sig,
            "abs_sig": abs_sig,
            "is_global": is_global,
            "target_wl": self.target_peak_wl,
            "target_sca": self.target_sca_sig,
            "target_abs": self.target_abs_sig,
            "wl_err": abs(peak_wl - self.target_peak_wl),
            "sca_sig_err": abs(sca_sig - self.target_sca_sig),
            "abs_sig_err": abs(abs_sig - self.target_abs_sig),
            "combined_error": self.current_error,
            "steps": self.steps,
        }
