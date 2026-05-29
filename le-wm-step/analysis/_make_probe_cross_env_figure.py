"""Cross-env summary figure for Experiment 3 (per-episode probe across 4 envs)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCES = [
    ("cube",    "analysis_out/probe_progress_cube_v2/summary.json"),
    ("pusht",   "analysis_out/probe_progress_pusht/summary.json"),
    ("reacher", "analysis_out/probe_progress_reacher/summary.json"),
    ("tworoom", "analysis_out/probe_progress_tworoom/summary.json"),
]
FEATURES = [
    "step_idx",
    "(sin θ, cos θ)",
    "z_prog (8-d)",
    "random-2d (control)",
]
COLORS = {
    "step_idx":           "tab:grey",
    "(sin θ, cos θ)":     "tab:red",
    "z_prog (8-d)":       "tab:blue",
    "random-2d (control)": "lightgrey",
}


def main():
    # Resolve repo-relative paths so this works regardless of where the
    # script is run from.
    base = Path(__file__).resolve().parent.parent
    data = {}
    for env, path in SOURCES:
        s = json.load(open(base / path))
        pe = s["results_per_episode"]
        data[env] = {f: pe[f]["r2_values"] for f in FEATURES if f in pe}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                              gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Panel A: grouped bar — mean per-episode R² per feature × env, with error bars
    ax = axes[0]
    n_env = len(SOURCES); n_feat = len(FEATURES)
    width = 0.20
    x = np.arange(n_env)
    for i, f in enumerate(FEATURES):
        means = [np.mean(data[env].get(f, [])) for env, _ in SOURCES]
        stds = [np.std(data[env].get(f, []))   for env, _ in SOURCES]
        bars = ax.bar(x + (i - 1.5) * width, means, width=width, yerr=stds,
                       color=COLORS[f], edgecolor="black", linewidth=0.4,
                       capsize=3, label=f)
        for j, m in enumerate(means):
            ax.text(x[j] + (i - 1.5) * width, m + 0.025 if m > 0 else m - 0.05,
                    f"{m:.2f}", ha="center", va="bottom" if m > 0 else "top",
                    fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([env for env, _ in SOURCES])
    ax.set_ylabel("mean per-episode R²")
    ax.set_title("Within-episode probe — task-progress R² across feature × env")
    ax.set_ylim(-0.4, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: per-env box+strip of z_prog and (sin θ, cos θ) — show distribution shape
    ax = axes[1]
    positions = []
    box_data = []
    box_colors = []
    labels = []
    for j, (env, _) in enumerate(SOURCES):
        for k, f in enumerate(["(sin θ, cos θ)", "z_prog (8-d)"]):
            positions.append(3 * j + (k + 1))
            box_data.append(data[env].get(f, []))
            box_colors.append(COLORS[f])
            labels.append(f"{env}\n{'(sin θ,cos θ)' if k==0 else 'z_prog'}")
    bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.5))
    for box, c in zip(bp["boxes"], box_colors):
        box.set(facecolor=c, alpha=0.55, edgecolor="black")
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, box_data):
        jit = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), pos) + jit, vals, s=10, c="black",
                   alpha=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylim(-0.6, 1.05)
    ax.set_ylabel("per-episode R² (LOO-CV)")
    ax.set_title("Per-episode distribution: (sin θ, cos θ) vs z_prog across envs")
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle("Experiment 3 — within-episode probe across envs (n=40 episodes each)",
                 fontsize=12)
    fig.tight_layout()
    out = base / "analysis_out" / "probe_cross_env_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
