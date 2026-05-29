"""Summary figure for the regime-change CPD experiment.

Reads ``analysis_out/regime_change_cube_n40/summary.json`` and emits a single
two-panel figure:

* Panel A — Per-tolerance F1 bar chart with the three metrics side-by-side.
  Pooled F1 across 40 episodes / 160 events.
* Panel B — Box+strip of per-episode F1 distributions at tol±2 (the one
  paper-relevant tolerance), three metrics side-by-side.
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
    tols = [str(t) for t in s["tolerances"]]
    metrics = ["dtheta_dt", "zmse", "dtheta_pred"]
    pretty = {
        "dtheta_dt":   r"$|\mathrm{d}\theta/\mathrm{d}t|$ (observed)",
        "zmse":        "z-MSE",
        "dtheta_pred": r"$|\Delta\theta|$ (pred err)",
    }
    colors = {"dtheta_dt": "tab:red", "zmse": "tab:blue", "dtheta_pred": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel A: pooled F1 grouped by tolerance, 3 bars per group
    ax = axes[0]
    width = 0.27
    x = np.arange(len(tols))
    for i, m in enumerate(metrics):
        f1s = [s["pooled"][m][t]["f1"] for t in tols]
        bars = ax.bar(x + (i - 1) * width, f1s, width=width,
                      color=colors[m], label=pretty[m], edgecolor="black",
                      linewidth=0.5)
        # annotate values
        for j, v in enumerate(f1s):
            ax.text(x[j] + (i - 1) * width, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"tolerance ±{t}" for t in tols])
    ax.set_ylabel("pooled F1 (40 episodes / 160 events)")
    ax.set_title("Change-point F1 vs ground-truth gripper-contact transitions")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: per-episode F1 distributions at tol=2 (or middle tolerance)
    mid = tols[len(tols) // 2]
    ax = axes[1]
    data = []
    labels = []
    box_colors = []
    for m in metrics:
        f1s = [rec["per_metric"][m]["per_tol"][mid]["f1"]
               for rec in s["per_episode"].values()]
        data.append(np.asarray(f1s))
        labels.append(pretty[m])
        box_colors.append(colors[m])

    bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.5))
    for box, c in zip(bp["boxes"], box_colors):
        box.set(facecolor=c, alpha=0.55, edgecolor="black")
    rng = np.random.default_rng(0)
    for pos, vals in zip([1, 2, 3], data):
        jit = (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), pos) + jit, vals, s=12, c="black",
                   alpha=0.45)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([l.replace(" (pred err)", "") for l in labels],
                       fontsize=9, rotation=12, ha="right")
    ax.set_ylabel(f"per-episode F1 (tol±{mid})")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Per-episode F1 spread at tol±{mid}")
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle(f"Experiment 2 — Regime-change CPD on cube  ({len(s['per_episode'])} eps)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    pdf_out = args.out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"wrote {args.out} and {pdf_out}")


if __name__ == "__main__":
    main()
