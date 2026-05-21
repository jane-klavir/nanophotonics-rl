from __future__ import annotations

from pathlib import Path
import html
import json
import base64
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mie_materials_v1.npz"
NOTEBOOK_PATH = ROOT / "scripts" / "rl_mie_dqn_mode_select_v4.1.ipynb"
OUT = ROOT / "outputs" / "poster"
POSTER_V4 = ROOT / "outputs" / "poster_v4"
TRAINING_HISTORY_PATH = POSTER_V4 / "tables" / "training_history.csv"

MATERIAL_COLORS = {
    "Ag": "#3b82f6",
    "Al": "#6b7280",
    "Au": "#d6a11d",
    "Cu": "#b45309",
    "GaAs": "#8b5cf6",
    "Ge": "#ef4444",
    "Si": "#059669",
}

TRAINING_LOG = [
    (100, -0.18, 0.3648),
    (200, -0.06, 0.3729),
    (300, 1.32, 0.2592),
    (400, 1.60, 0.2346),
    (500, 2.38, 0.1671),
    (600, 2.79, 0.1719),
    (700, 2.94, 0.1741),
    (800, 2.08, 0.2111),
    (900, 1.95, 0.1994),
    (1000, 2.67, 0.1632),
    (1100, 2.71, 0.1340),
    (1200, 3.64, 0.1010),
    (1300, 4.21, 0.1301),
    (1400, 3.30, 0.1261),
    (1500, 3.59, 0.1254),
    (1600, 3.60, 0.1130),
    (1700, 4.05, 0.0819),
]

QUERY_RESULTS = [
    {
        "label": "648 nm target",
        "target_wl": 648.0,
        "target_sca": 771_369.0,
        "target_abs": 500_000.0,
        "solved_pct": 89.5,
        "mean_error": 0.0748,
        "best_error": 0.044984,
        "best_design": "Cu, r=374.7 nm",
        "best_peak": 630.0,
    },
    {
        "label": "700 nm target",
        "target_wl": 700.0,
        "target_sca": 900_000.0,
        "target_abs": 400_000.0,
        "solved_pct": 44.5,
        "mean_error": 0.1341,
        "best_error": 0.013965,
        "best_design": "Si, r=223.7 nm",
        "best_peak": 700.0,
    },
]


def scale(vals, lo, hi, out_lo, out_hi):
    vals = np.asarray(vals, dtype=float)
    return out_lo + (vals - lo) / (hi - lo + 1e-12) * (out_hi - out_lo)


def polyline(xs, ys, color, width=3):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def axis_labels(x, y, text, size=13, anchor="middle", rotate=None):
    rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="#111827"{rot}>{html.escape(text)}</text>'


def write_svg(path: Path, width: int, height: int, body: str):
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f"{body}</svg>\n",
        encoding="utf-8",
    )


def load_global_lookup():
    d = np.load(DATA_PATH, allow_pickle=False)
    wl = d["wavelengths_nm"]
    material = d["material_id"].astype(str)
    radius = d["radius_nm"]
    sca = d["sigma_sca"]
    abs_ = d["sigma_abs"]
    p = np.argmax(sca, axis=1)
    return {
        "wl": wl,
        "material": material,
        "radius": radius,
        "peak_nm": wl[p],
        "sca_peak": sca[np.arange(len(radius)), p],
        "abs_at_peak": abs_[np.arange(len(radius)), p],
    }


def brute_force_rows(lookup):
    peak_nm = lookup["peak_nm"]
    sca = lookup["sca_peak"]
    abs_ = lookup["abs_at_peak"]
    wl_scale = peak_nm.max() - peak_nm.min()
    sca_scale = sca.max()
    abs_scale = abs_.max()
    rows = []
    for q in QUERY_RESULTS:
        err = (
            0.80 * np.abs(peak_nm - q["target_wl"]) / (wl_scale + 1e-9)
            + 0.10 * np.abs(sca - q["target_sca"]) / (sca_scale + 1e-9)
            + 0.10 * np.abs(abs_ - q["target_abs"]) / (abs_scale + 1e-9)
        )
        i = int(np.argmin(err))
        rows.append({
            "target": q["label"],
            "material": lookup["material"][i],
            "radius_nm": float(lookup["radius"][i]),
            "peak_nm": float(peak_nm[i]),
            "sca_nm2": float(sca[i]),
            "abs_nm2": float(abs_[i]),
            "combined_error": float(err[i]),
        })
    return rows


def make_landscape(lookup):
    width, height = 920, 300
    left, right, top, bottom = 74, 24, 16, 50
    x = scale(lookup["radius"], 50, 1000, left, width - right)
    y = scale(lookup["peak_nm"], 300, 900, height - bottom, top)
    body = [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    for tick in [50, 250, 500, 750, 1000]:
        tx = scale([tick], 50, 1000, left, width - right)[0]
        body.append(f'<line x1="{tx:.1f}" y1="{height-bottom}" x2="{tx:.1f}" y2="{height-bottom+6}" stroke="#111827"/>')
        body.append(axis_labels(tx, height - bottom + 18, str(tick), 10))
    for tick in [300, 500, 700, 900]:
        ty = scale([tick], 300, 900, height - bottom, top)[0]
        body.append(f'<line x1="{left-6}" y1="{ty:.1f}" x2="{left}" y2="{ty:.1f}" stroke="#111827"/>')
        body.append(axis_labels(left - 12, ty + 4, str(tick), 10, anchor="end"))
    body.append(axis_labels(width / 2, height - 10, "Radius (nm)", 12))
    body.append(axis_labels(18, height / 2, "Peak wavelength (nm)", 12, rotate=-90))
    for mat in sorted(set(lookup["material"])):
        idx = np.where(lookup["material"] == mat)[0]
        color = MATERIAL_COLORS.get(mat, "#111827")
        for j in idx:
            body.append(f'<circle cx="{x[j]:.1f}" cy="{y[j]:.1f}" r="3.2" fill="{color}" opacity="0.72"/>')
    lx, ly = width - 150, 28
    legend_w = 92
    legend_h = 22 * len(set(lookup["material"])) + 12
    body.append(
        f'<rect x="{lx - 15}" y="{ly - 18}" width="{legend_w}" height="{legend_h}" '
        'fill="white" fill-opacity="0.82" stroke="#d1d5db" stroke-width="1"/>'
    )
    for n, mat in enumerate(sorted(set(lookup["material"]))):
        color = MATERIAL_COLORS.get(mat, "#111827")
        body.append(f'<circle cx="{lx}" cy="{ly+n*22}" r="5" fill="{color}"/>')
        body.append(axis_labels(lx + 14, ly + 4 + n * 22, mat, 11, anchor="start"))
    write_svg(OUT / "mie_peak_landscape.svg", width, height, "\n".join(body))


def make_training_curve():
    width, height = 880, 300
    left, right, top, bottom = 66, 22, 16, 48
    ep, reward, error = load_training_series()
    x = scale(ep, ep.min(), ep.max(), left, width - right)
    y_err = scale(error, 0, 0.40, height - bottom, top)
    y_reward = scale(reward, -1, 6, height - bottom, top)
    threshold_y = scale([0.10], 0, 0.40, height - bottom, top)[0]
    body = [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width-right}" y2="{threshold_y:.1f}" stroke="#059669" stroke-dasharray="7 6"/>',
        axis_labels(width - right - 4, threshold_y - 6, "threshold 0.10", 11, anchor="end"),
        polyline(x, y_err, "#dc2626", 3),
        polyline(x, y_reward, "#2563eb", 3),
        axis_labels(width / 2, height - 9, "Episode", 12),
        axis_labels(18, height / 2, "100-episode avg.", 12, rotate=-90),
    ]
    tick_candidates = [100, 500, 1000, 1500, 1700]
    ticks = [t for t in tick_candidates if ep.min() <= t <= ep.max()]
    if int(ep.max()) not in ticks:
        ticks.append(int(ep.max()))
    for tick in ticks:
        tx = scale([tick], ep.min(), ep.max(), left, width - right)[0]
        body.append(f'<line x1="{tx:.1f}" y1="{height-bottom}" x2="{tx:.1f}" y2="{height-bottom+6}" stroke="#111827"/>')
        body.append(axis_labels(tx, height - bottom + 18, str(tick), 10))
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        ty = scale([tick], 0, 0.40, height - bottom, top)[0]
        body.append(f'<line x1="{left-6}" y1="{ty:.1f}" x2="{left}" y2="{ty:.1f}" stroke="#111827"/>')
        body.append(axis_labels(left - 12, ty + 4, f"{tick:.1f}", 10, anchor="end"))
    body.append('<rect x="626" y="25" width="210" height="48" fill="white" stroke="#d1d5db"/>')
    body.append('<line x1="642" y1="43" x2="684" y2="43" stroke="#dc2626" stroke-width="3"/>')
    body.append(axis_labels(694, 47, "combined error", 11, anchor="start"))
    body.append('<line x1="642" y1="61" x2="684" y2="61" stroke="#2563eb" stroke-width="3"/>')
    body.append(axis_labels(694, 65, "reward", 11, anchor="start"))
    write_svg(OUT / "mie_rl_training_curve.svg", width, height, "\n".join(body))


def load_training_series():
    if TRAINING_HISTORY_PATH.exists():
        history = np.genfromtxt(TRAINING_HISTORY_PATH, delimiter=",", names=True)
        episodes = np.asarray(history["episode"], dtype=int)
        rewards = np.asarray(history["reward"], dtype=float)
        errors = np.asarray(history["combined_error"], dtype=float)
        checkpoints = list(range(100, int(episodes.max()) + 1, 100))
        if checkpoints[-1] != int(episodes.max()):
            checkpoints.append(int(episodes.max()))
        avg_rewards = []
        avg_errors = []
        for checkpoint in checkpoints:
            end = np.searchsorted(episodes, checkpoint, side="right")
            start = max(0, end - 100)
            avg_rewards.append(float(np.mean(rewards[start:end])))
            avg_errors.append(float(np.mean(errors[start:end])))
        return np.asarray(checkpoints), np.asarray(avg_rewards), np.asarray(avg_errors)

    ep = np.array([r[0] for r in TRAINING_LOG])
    reward = np.array([r[1] for r in TRAINING_LOG])
    error = np.array([r[2] for r in TRAINING_LOG])
    return ep, reward, error


def make_query_summary(brute_rows):
    width, height = 900, 270
    left = 58
    body = []
    groups_x = [245, 635]
    bar_w = 62
    max_h = 145
    base = 172
    for gx, q, bf in zip(groups_x, QUERY_RESULTS, brute_rows):
        solved_h = max_h * q["solved_pct"] / 100.0
        mean_h = max_h * min(q["mean_error"] / 0.15, 1.0)
        best_h = max_h * min(q["best_error"] / 0.15, 1.0)
        bf_h = max_h * min(bf["combined_error"] / 0.15, 1.0)
        bars = [
            ("Solved %", solved_h, "#059669", f"{q['solved_pct']:.1f}%"),
            ("Mean err", mean_h, "#2563eb", f"{q['mean_error']:.3f}"),
            ("Best err", best_h, "#dc2626", f"{q['best_error']:.3f}"),
            ("BF err", bf_h, "#6b7280", f"{bf['combined_error']:.3f}"),
        ]
        x0 = gx - 1.8 * bar_w
        for i, (lab, h, color, val) in enumerate(bars):
            x = x0 + i * (bar_w + 18)
            body.append(f'<rect x="{x:.1f}" y="{base-h:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" opacity="0.90"/>')
            body.append(axis_labels(x + bar_w / 2, base - h - 6, val, 10))
            body.append(axis_labels(x + bar_w / 2, base + 16, lab, 9))
        body.append(axis_labels(gx, 218, q["label"], 13))
        body.append(axis_labels(gx, 238, f"DQN: {q['best_design']}, peak {q['best_peak']:.0f} nm", 10))
        body.append(axis_labels(gx, 254, f"BF: {bf['material']}, r={bf['radius_nm']:.1f} nm, peak {bf['peak_nm']:.0f} nm", 10))
    body.append(f'<line x1="{left}" y1="{base}" x2="{width-40}" y2="{base}" stroke="#111827"/>')
    body.append(axis_labels(22, 96, "Scaled value", 11, rotate=-90))
    write_svg(OUT / "mie_query_summary.svg", width, height, "\n".join(body))


def extract_notebook_pngs():
    figure_map = {
        "au_spectra.png": "notebook_au_spectra.png",
        "peak_summary.png": "notebook_peak_summary.png",
        "training_curves.png": "notebook_training_curves.png",
    }
    if all((POSTER_V4 / "figures" / src).exists() for src in figure_map):
        for src, dst in figure_map.items():
            shutil.copy2(POSTER_V4 / "figures" / src, OUT / dst)
        return

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    names = ["notebook_au_spectra.png", "notebook_peak_summary.png", "notebook_training_curves.png"]
    idx = 0
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png and idx < len(names):
                (OUT / names[idx]).write_bytes(base64.b64decode(png))
                idx += 1


def write_results_md(brute_rows):
    lines = [
        "# Mie RL Poster Results",
        "",
        "Dataset: 700 analytical Mie simulations; 7 materials x 100 radii; wavelengths 300-900 nm in 10 nm steps; homogeneous spheres in air.",
        "RL setup: dueling Double DQN, 9D normalized state, 5 actions (radius down/up, material previous/next, stop), scattering-anchored global peaks, weighted error = 0.80 peak wavelength + 0.10 scattering amplitude + 0.10 absorption amplitude.",
        "",
        "| Query | DQN solved | DQN mean error | DQN best error | DQN best design | Brute-force best error | Brute-force design |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for q, bf in zip(QUERY_RESULTS, brute_rows):
        lines.append(
            f"| {q['label']} | {q['solved_pct']:.1f}% | {q['mean_error']:.4f} | {q['best_error']:.4f} | "
            f"{q['best_design']} | {bf['combined_error']:.4f} | {bf['material']}, r={bf['radius_nm']:.1f} nm |"
        )
    (OUT / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = load_global_lookup()
    brute_rows = brute_force_rows(lookup)
    make_landscape(lookup)
    make_training_curve()
    make_query_summary(brute_rows)
    extract_notebook_pngs()
    write_results_md(brute_rows)
    print(f"Wrote poster assets to {OUT}")


if __name__ == "__main__":
    main()
