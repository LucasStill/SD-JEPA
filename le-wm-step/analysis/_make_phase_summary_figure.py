"""Standalone figure-maker for the phase-alignment n=40 result.

Reads ``analysis_out/phase_align_cube_n40/summary.json`` and emits a single
two-panel figure:

* Panel A — Scatter of per-episode AUROC: x = z-MSE, y = |Δθ|, one point per
  episode for each tolerance. Diagonal y=x marks parity. Points above the
  diagonal = θ wins for that episode. The win-rate (39/40 at tol=1, etc.)
  is annotated per panel.
* Panel B — Box+swarm of the per-episode AUROC distributions side-by-side at
  each tolerance.

Run from le-wm-step/:
    python analysis/_make_phase_summary_figure.py \
        --in  analysis_out/phase_align_cube_n40/summary.json \
        --out analysis_out/phase_align_cube_n40/phase_alignment_summary.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    s = json.load(open(args.in_path))
    pa = s["phase_alignment"]
    tols = [str(t) for t in pa["tolerances"]]
    per_ep = pa["per_episode"]

    z_at = {t: [] for t in tols}
    d_at = {t: [] for t in tols}
    for ep, rec in per_ep.items():
        for t in tols:
            apt = rec["auroc_per_tol"][t]
            z_at[t].append(apt["auroc_zmse"])
            d_at[t].append(apt["auroc_dtheta"])

    n_eps = len(per_ep)
    n_tols = len(tols)

    fig = plt.figure(figsize=(4.0 * n_tols + 4.5, 4.4))
    gs = fig.add_gridspec(1, n_tols + 1, width_ratios=[1.0] * n_tols + [1.3])

    # Panel A: scatter per tolerance
    for i, t in enumerate(tols):
        ax = fig.add_subplot(gs[0, i])
        z = np.array(z_at[t])
        d = np.array(d_at[t])
        wins = int((d > z).sum())

        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", zorder=1)
        ax.scatter(z, d, s=28, c="tab:blue", alpha=0.75, edgecolor="white",
                   linewidth=0.5, zorder=2)
        # mean cross
        ax.scatter([z.mean()], [d.mean()], marker="x", c="black", s=80,
                   linewidth=2.0, zorder=3, label=f"mean ({z.mean():.2f}, {d.mean():.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("AUROC z-MSE")
        if i == 0:
            ax.set_ylabel(r"AUROC $|\Delta\theta|$")
        ax.set_title(f"tolerance ±{t} step\n"
                     fr"$|\Delta\theta|$ wins on {wins}/{n_eps} episodes",
                     fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.legend(fontsize=7, loc="upper left")

    # Panel B: box+strip per tolerance (paired view)
    ax = fig.add_subplot(gs[0, n_tols])
    positions = []
    data = []
    labels = []
    colors = []
    for i, t in enumerate(tols):
        positions.extend([3 * i + 1, 3 * i + 2])
        data.extend([z_at[t], d_at[t]])
        labels.extend([f"z-MSE\n±{t}", f"|Δθ|\n±{t}"])
        colors.extend(["tab:orange", "tab:blue"])

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.5))
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, alpha=0.55, edgecolor="black")
    # strip
    for pos, vals in zip(positions, data):
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.2
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10, c="black",
                   alpha=0.4)

    ax.axhline(0.5, color="grey", ls="--", lw=0.6)
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("per-episode AUROC")
    ax.set_title(f"AUROC distributions by metric × tolerance  (n={n_eps} episodes)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, lw=0.4)

    fig.suptitle(f"Phase-event alignment — cube · {pa['column']}  "
                 f"({n_eps} episodes, {pa['n_total_events']} transition events)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    pdf_out = args.out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"wrote {args.out} and {pdf_out}")


if __name__ == "__main__":
    main()
