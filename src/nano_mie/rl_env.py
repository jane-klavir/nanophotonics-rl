import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NanoparticleEnv(gym.Env):
    """
    Gymnasium environment for nanophotonics inverse design.
    The agent modifies the radius and material of a nanoparticle
    to match a target optical response (absorption and scattering at a specific wavelength).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path: str = "data/processed/mie_materials_v1.npz", 
                 max_steps: int = 50,
                 radius_step_nm: float = 5.0,
                 success_threshold: float = 1e-3,
                 w_abs: float = 0.5,
                 w_sca: float = 0.5):
        super().__init__()
        
        self.max_steps = max_steps
        self.radius_step_nm = radius_step_nm
        self.success_threshold = success_threshold
        self.w_abs = w_abs
        self.w_sca = w_sca
        
        # Load dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        data = np.load(data_path)
        self.wavelengths_nm = data["wavelengths_nm"]
        self.material_names = data["material_names"]
        self.radius_nm = data["radius_nm"]
        self.material_id = data["material_id"]
        
        self.sigma_abs = data["sigma_abs"]
        self.sigma_sca = data["sigma_sca"]
        
        self.unique_materials = np.unique(self.material_id)
        self.num_materials = len(self.unique_materials)
        self.num_wavelengths = len(self.wavelengths_nm)
        self.num_samples = len(self.radius_nm)
        
        # Normalization constants
        self.min_rad = np.min(self.radius_nm)
        self.max_rad = np.max(self.radius_nm)
        self.max_abs = np.max(self.sigma_abs) + 1e-8
        self.max_sca = np.max(self.sigma_sca) + 1e-8
        self.max_wv = np.max(self.wavelengths_nm)
        
        # Pre-compute sorted row indices for each material (mapping internal index to dataset rows)
        self.mat_to_rows = {}
        for i, m_id in enumerate(self.unique_materials):
            m_rows = np.where(self.material_id == m_id)[0]
            m_radii = self.radius_nm[m_rows]
            sorted_idx = np.argsort(m_radii)
            self.mat_to_rows[i] = m_rows[sorted_idx]
            
        # Actions: 0: dec radius, 1: inc radius, 2: prev mat, 3: next mat
        self.action_space = spaces.Discrete(4)
        
        # Obs space dimension: 
        # 1 (wv) + 2 (target_abs, target_sca) + 1 (curr_rad) + num_materials (one-hot) + 2 (curr_abs, curr_sca) + 1 (error)
        obs_dim = 7 + self.num_materials
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self._current_step = 0
        self._target_row = 0
        self._target_wv_idx = 0
        self._curr_rad = 0.0
        self._curr_mat = 0
        self._curr_rad_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sample target
        self._target_row = self.np_random.integers(0, self.num_samples)
        self._target_wv_idx = self.np_random.integers(0, self.num_wavelengths)
        
        self.target_rad = self.radius_nm[self._target_row]
        self.target_mat = self.material_id[self._target_row]
        self.target_wv = self.wavelengths_nm[self._target_wv_idx]
        self.target_abs = self.sigma_abs[self._target_row, self._target_wv_idx]
        self.target_sca = self.sigma_sca[self._target_row, self._target_wv_idx]

        # Init state (randomly select from available discrete radii)
        self._curr_mat = self.np_random.integers(0, self.num_materials)
        self._curr_rad_idx = self.np_random.integers(0, len(self.mat_to_rows[self._curr_mat]))
        
        curr_row = self.mat_to_rows[self._curr_mat][self._curr_rad_idx]
        self._curr_rad = self.radius_nm[curr_row]
        
        self._current_step = 0
        self._prev_error = self._compute_error(curr_row)
        
        return self._get_obs(), {}

    def step(self, action):
        self._current_step += 1
        
        old_rad = self._curr_rad
        
        # Apply action
        if action == 0:
            self._curr_rad_idx = max(0, self._curr_rad_idx - 1)
        elif action == 1:
            self._curr_rad_idx = min(len(self.mat_to_rows[self._curr_mat]) - 1, self._curr_rad_idx + 1)
        elif action == 2:
            self._curr_mat = (self._curr_mat - 1) % self.num_materials
            # snap to closest radius
            valid_radii = self.radius_nm[self.mat_to_rows[self._curr_mat]]
            self._curr_rad_idx = int(np.argmin(np.abs(valid_radii - old_rad)))
        elif action == 3:
            self._curr_mat = (self._curr_mat + 1) % self.num_materials
            # snap to closest radius
            valid_radii = self.radius_nm[self.mat_to_rows[self._curr_mat]]
            self._curr_rad_idx = int(np.argmin(np.abs(valid_radii - old_rad)))

        curr_row = self.mat_to_rows[self._curr_mat][self._curr_rad_idx]
        self._curr_rad = self.radius_nm[curr_row]

        curr_error = self._compute_error(curr_row)
        
        terminated = False
        truncated = self._current_step >= self.max_steps
        
        improvement_reward = 10.0 * (self._prev_error - curr_error)
        reward = improvement_reward - 0.01  # improvement minus step penalty
        
        terminal_bonus = 0.0
        success = (curr_error < self.success_threshold)
        
        if success:
            terminated = True
            terminal_bonus = 2.0
            
        reward += terminal_bonus
            
        old_error = self._prev_error
        self._prev_error = curr_error
        
        info = {
            "old_error": float(old_error),
            "new_error": float(curr_error),
            "improvement_reward": float(improvement_reward),
            "terminal_bonus": float(terminal_bonus),
            "success": bool(success),
            "stopped": False,
            "radius_nm": float(self._curr_rad),
            "material": int(self._curr_mat),
            "steps": int(self._current_step),
            "lookup_row": int(curr_row)
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_error(self, row):
        abs_val, sca_val = self._lookup_spectrum(row)
        
        norm_t_abs = self.target_abs / self.max_abs
        norm_t_sca = self.target_sca / self.max_sca
        norm_c_abs = abs_val / self.max_abs
        norm_c_sca = sca_val / self.max_sca
        
        err = self.w_abs * (norm_c_abs - norm_t_abs)**2 + self.w_sca * (norm_c_sca - norm_t_sca)**2
        return float(err)

    def _lookup_spectrum(self, row):
        return self.sigma_abs[row, self._target_wv_idx], self.sigma_sca[row, self._target_wv_idx]

    def _get_obs(self):
        curr_row = self.mat_to_rows[self._curr_mat][self._curr_rad_idx]
        curr_abs, curr_sca = self._lookup_spectrum(curr_row)
        curr_err = self._compute_error(curr_row)
        
        mat_onehot = np.zeros(self.num_materials, dtype=np.float32)
        mat_onehot[self._curr_mat] = 1.0
        
        obs = np.array([
            self.target_wv / self.max_wv,
            self.target_abs / self.max_abs,
            self.target_sca / self.max_sca,
            (self._curr_rad - self.min_rad) / (self.max_rad - self.min_rad),
            *mat_onehot,
            curr_abs / self.max_abs,
            curr_sca / self.max_sca,
            min(curr_err, 1.0)
        ], dtype=np.float32)
        
        return obs
