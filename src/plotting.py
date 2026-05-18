import numpy as np
import matplotlib.pyplot as plt


def smooth(x, w: int = 20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training_curves(history: dict, stop_threshold: float = 0.10, smooth_window: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(smooth(history["episode_rewards"], smooth_window))
    axes[0].set(xlabel="Episode", ylabel="Total reward", title="Reward per episode (smoothed)")
    axes[0].grid(True)

    axes[1].plot(smooth(history["episode_errors"], smooth_window))
    axes[1].axhline(stop_threshold, ls="--", label=f"Stop threshold ({stop_threshold:.2f})")
    axes[1].set(xlabel="Episode", ylabel="Final combined error", title="Combined error")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    return fig, axes


def plot_material_curves(data, material_name: str = "Au", step: int = 20):
    mask = data.material_id == material_name
    selected_radii = data.radius_nm[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx in np.argsort(selected_radii)[::step]:
        axes[0].plot(data.wavelengths_nm, data.sigma_sca[mask][idx], label=f"{selected_radii[idx]:.0f} nm")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="σ_sca (nm²)", title=f"Scattering — {material_name}")
    axes[0].grid(True)
    axes[0].legend(fontsize=7)

    for idx in np.argsort(selected_radii)[::step]:
        axes[1].plot(data.wavelengths_nm, data.sigma_abs[mask][idx], label=f"{selected_radii[idx]:.0f} nm")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="σ_abs (nm²)", title=f"Absorption — {material_name}")
    axes[1].grid(True)
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    return fig, axes


def plot_peak_summary(peak_df, peak_mode: int = 1):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    peak_label = "Sca" if peak_mode == 1 else "Abs"

    for mat in peak_df["material"].unique():
        sub = peak_df[peak_df["material"] == mat]
        axes[0].scatter(sub["radius_nm"], sub["peak_lambda_nm"], s=8, label=mat, alpha=0.7)
    axes[0].set(xlabel="Radius (nm)", ylabel=f"{peak_label} Peak λ (nm)", title=f"{peak_label} Peaks by Material")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    global_df = peak_df[peak_df["is_global"]]
    for mat in global_df["material"].unique():
        sub = global_df[global_df["material"] == mat]
        axes[1].scatter(sub["radius_nm"], sub["peak_lambda_nm"], s=8, label=mat, alpha=0.7)
    axes[1].set(xlabel="Radius (nm)", ylabel=f"{peak_label} Peak λ (nm)", title=f"Global {peak_label} Peaks by Material")
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    axes[2].scatter(peak_df["sigma_sca_nm2"], peak_df["sigma_abs_nm2"], s=5, alpha=0.3)
    axes[2].set(xlabel="σ_sca at anchor λ (nm²)", ylabel="σ_abs at anchor λ (nm²)", title="σ_sca vs σ_abs")
    axes[2].grid(True)

    plt.tight_layout()
    return fig, axes
