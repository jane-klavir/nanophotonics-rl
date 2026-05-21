import numpy as np
import pandas as pd
import torch
from .env import MieEnv
from .peak_utils import ErrorScales, ErrorWeights


def query(
    net,
    env: MieEnv,
    target_wl: float,
    target_sca: float,
    target_abs: float,
    n_trials: int = 200,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    device = torch.device(device or next(net.parameters()).device)
    net.eval()
    results = []

    for _ in range(n_trials):
        env.reset()
        state = env.set_target(target_wl, target_sca, target_abs)

        for _ in range(MieEnv.MAX_STEPS):
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(net(t).argmax(dim=1).item())
            state, _, done, info = env.step(action)
            if done:
                break

        results.append(info)

    df = pd.DataFrame(results).sort_values("combined_error").reset_index(drop=True)

    if verbose:
        anchor_label = "Sca" if int(env.peak_mode) == 1 else "Abs"
        gl_label = "global-only" if int(env.global_only) == 1 else "all-peaks"
        print(f"=== Query: target λ={target_wl} nm | σ_sca={target_sca:.0f} | σ_abs={target_abs:.0f} ===")
        print(f"    Mode: PEAK_MODE={int(env.peak_mode)} ({anchor_label}-anchored) | GLOBAL_ONLY={int(env.global_only)} ({gl_label})")
        print(f"\n--- Top 5 results ({n_trials} trials) ---")
        cols = ["material", "radius_nm", "peak_wl", "sca_sig", "abs_sig", "wl_err", "sca_sig_err", "abs_sig_err", "combined_error", "steps"]
        print(df[cols].head(5).to_string(index=False))
        solved = (df["combined_error"] < MieEnv.STOP_THRESHOLD).mean() * 100
        print(f"\nSolved (<{MieEnv.STOP_THRESHOLD}): {solved:.1f}%")
        print(f"Mean combined error : {df['combined_error'].mean():.4f}")
        print(f"Best combined error : {df['combined_error'].min():.4f}")
        print(f"Mean steps          : {df['steps'].mean():.1f}")

    return df


def brute_force_search(
    combo_lookup: pd.DataFrame,
    target_wl: float,
    target_sca: float,
    target_abs: float,
    scales: ErrorScales,
    weights: ErrorWeights | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Exact table search baseline using the same weighted error as the environment."""
    weights = weights if weights is not None else ErrorWeights()
    weights.validate()

    df = combo_lookup.copy()
    df["lambda_diff"] = df["peak_nm"] - target_wl
    df["sigma_sca_diff"] = df["sca_sigma_nm2"] - target_sca
    df["sigma_abs_diff"] = df["abs_sigma_nm2"] - target_abs

    df["lambda_sq_error"] = df["lambda_diff"] ** 2
    df["sigma_sca_sq_error"] = df["sigma_sca_diff"] ** 2
    df["sigma_abs_sq_error"] = df["sigma_abs_diff"] ** 2

    df["combined_error"] = (
        weights.peak_wl * np.abs(df["peak_nm"] - target_wl) / (scales.wl_scale + 1e-9)
        + weights.sca_sig * np.abs(df["sca_sigma_nm2"] - target_sca) / (scales.sca_sigma_scale + 1e-9)
        + weights.abs_sig * np.abs(df["abs_sigma_nm2"] - target_abs) / (scales.abs_sigma_scale + 1e-9)
    )

    df = df.sort_values("combined_error").reset_index(drop=True)
    cols = [
        "material", "radius_nm",
        "peak_nm", "lambda_diff", "lambda_sq_error",
        "sca_sigma_nm2", "sigma_sca_diff", "sigma_sca_sq_error",
        "abs_sigma_nm2", "sigma_abs_diff", "sigma_abs_sq_error",
        "combined_error", "is_global",
    ]
    return df[cols].head(top_k)
