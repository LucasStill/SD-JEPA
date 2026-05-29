"""Experiment 3 — sequence-level information density in θ.

Hypothesis: θ packs *task-progress* information densely. A linear probe with
**(sin θ, cos θ)** as a 2-d input should achieve R² close to a probe with
the full 192-d latent z, indicating that the progression-subspace
phase coordinate captures most of the task-relevant variance.

For each held-out cube episode we:

1. Encode every observed frame → z_t (192-d).
2. Compute θ_t = atan2(z_prog[1], z_prog[0]) and the 2-d wrap-safe
   expansion (sin θ_t, cos θ_t).
3. Fetch the **ground-truth task-progress quantity** for cube:
   y_t = ‖privileged_block_0_pos − privileged_target_block_pos‖_2
   (cube-to-target Euclidean distance — a clean monotone-ish measure
   of task progress).

We then evaluate **linear regression** y ~ X on a leave-episodes-out
train/test split for each of these feature representations:

  * step_idx  (1-d clock baseline)
  * (sin θ, cos θ)     (2-d)
  * z_prog              (k_prog-d; here 8)
  * z_cont              ((D - k_prog)-d; here 184)
  * full z              (D-d; here 192)
  * random-2-d projection of z (2-d control: same dim as the θ expansion)

Report **R²**, **R² / dim**, and **MAE** on the held-out episodes. The cleanest
"θ packs info densely" signature is: (sin θ, cos θ) achieving R² close to
full z while using 100× fewer dims than full z, and far above the random-2-d
projection of z (showing it's the *specific* 2-d that matters, not just any
2-d slice).

Usage
-----
    python analysis/probe_progress.py \\
        --ckpt $STABLEWM_HOME/checkpoints/A2_JZ_kprog2/jepa_step_A2_kprog8_cube_seed42_535787_epoch_10_object.ckpt \\
        --data cube_single_expert --cache-dir $STABLEWM_HOME/datasets \\
        --episodes <list> \\
        --target-col-a privileged_block_0_pos \\
        --target-col-b privileged_target_block_pos \\
        --out analysis_out/probe_progress_cube/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_LEWM_DIR = _THIS_DIR.parent
if str(_LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(_LEWM_DIR))

from surprise_compare import (  # type: ignore  # noqa: E402
    _build_img_transform, _episode_rows, _collect_strided, _canonical_env,
    encode, project_theta,
)


def gather_target(dataset, obs_rows, col_a: str, col_b: str | None) -> np.ndarray:
    """Return a (T_obs,) target array.

    If ``col_b`` is None: use a scalar/1-d signal directly (e.g.
    ``distance_to_target``).
    If ``col_b`` is given: compute ``‖col_a − col_b‖_2`` per step
    (e.g. cube-to-target Euclidean distance from privileged columns).
    """
    A = np.stack([np.asarray(dataset[int(r)][col_a]).reshape(-1) for r in obs_rows])
    if col_b is None:
        return A.flatten().astype(float)
    B = np.stack([np.asarray(dataset[int(r)][col_b]).reshape(-1) for r in obs_rows])
    return np.linalg.norm(A - B, axis=1).astype(float)


# Pusht canonical goal pose: block centered at (256, 256) image-pixel coords
# (matches the env's default goal in stable_worldmodel.envs.pusht).
_PUSHT_GOAL_XY = np.array([256.0, 256.0])


def gather_env_target(dataset, obs_rows, env: str) -> tuple[np.ndarray, str]:
    """Env-specific natural task-progress target. Returns (target, pretty_name)."""
    from surprise_compare import _canonical_env  # local import to keep module light
    env = _canonical_env(env)
    if env == "cube":
        A = np.stack([np.asarray(dataset[int(r)]["privileged_block_0_pos"]).reshape(-1)
                      for r in obs_rows])
        B = np.stack([np.asarray(dataset[int(r)]["privileged_target_block_pos"]).reshape(-1)
                      for r in obs_rows])
        return np.linalg.norm(A - B, axis=1).astype(float), "cube–target distance (m)"
    if env == "reacher":
        A = np.stack([np.asarray(dataset[int(r)]["finger_pos"]).reshape(-1)
                      for r in obs_rows])
        B = np.stack([np.asarray(dataset[int(r)]["target_pos"]).reshape(-1)
                      for r in obs_rows])
        return np.linalg.norm(A - B, axis=1).astype(float), "ee–target distance (m)"
    if env == "tworoom":
        v = np.stack([np.asarray(dataset[int(r)]["distance_to_target"]).reshape(-1)
                      for r in obs_rows]).flatten().astype(float)
        return v, "agent–target distance (px)"
    if env == "pusht":
        # state = (agent_x, agent_y, block_x, block_y, block_angle, ...)
        s = np.stack([np.asarray(dataset[int(r)]["state"]).reshape(-1)
                      for r in obs_rows])
        block = s[:, 2:4]
        return (np.linalg.norm(block - _PUSHT_GOAL_XY, axis=1).astype(float),
                "block–goal distance (px)")
    raise ValueError(f"No env-target preset for env={env!r}")


def split_train_test_eps(eps: list[int], rng: np.random.Generator,
                          test_frac: float = 0.25) -> tuple[set, set]:
    n = len(eps); n_test = max(1, int(round(test_frac * n)))
    perm = rng.permutation(eps)
    return set(perm[n_test:].tolist()), set(perm[:n_test].tolist())


def fit_probe(X_train, y_train, X_test, y_test):
    """Linear regression with closed-form OLS. Returns (R2, MAE)."""
    from sklearn.linear_model import Ridge
    # Tiny ridge for numerical stability with high-D inputs and few samples
    reg = Ridge(alpha=1e-3, fit_intercept=True).fit(X_train, y_train)
    pred = reg.predict(X_test)
    ss_res = np.sum((y_test - pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.abs(y_test - pred).mean())
    return float(r2), mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--target-col-a", default=None,
                   help="Primary column. With --target-col-b, output is ‖a−b‖. "
                        "Without, the column itself is taken as the target.")
    p.add_argument("--target-col-b", default=None,
                   help="If given, target = ‖col_a − col_b‖.")
    p.add_argument("--target-name", default=None,
                   help="Pretty name for the target (default derived from cols)")
    p.add_argument("--env-target", action="store_true",
                   help="Use the env-specific natural task-progress target "
                        "(cube=block-target dist, reacher=ee-target dist, "
                        "tworoom=distance_to_target column, pusht=‖block−(256,256)‖). "
                        "Overrides --target-col-a/b.")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--n-bootstrap", type=int, default=20,
                   help="Re-shuffle train/test split N times to get error bars")
    p.add_argument("--target-relative", action="store_true",
                   help="Use the episode-normalised progress (y0−y_t)/(max abs diff) "
                        "instead of the raw target. Removes per-episode scale/offset.")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.env_target:
        target_name = "(env-default; computed per env)"
    elif args.target_col_a is None:
        raise SystemExit("Provide either --env-target or --target-col-a")
    else:
        target_name = args.target_name or (
            f"‖{args.target_col_a} − {args.target_col_b}‖"
            if args.target_col_b else args.target_col_a
        )

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    model.eval()
    k_prog = int(getattr(model, "k_prog", 0))
    print(f"  k_prog = {k_prog}")

    print(f"Loading dataset: {args.data}")
    import stable_worldmodel as swm
    from hdf5_dataset import HDF5Dataset
    cache_dir = args.cache_dir or swm.data.utils.get_cache_dir()
    dataset = HDF5Dataset(args.data, cache_dir=Path(cache_dir), keys_to_cache=[])
    img_transform = _build_img_transform(args.img_size)

    # Per-episode features + targets
    per_ep = {}
    for ep_id in args.episodes:
        rows = _episode_rows(dataset, ep_id)
        if len(rows) < args.frameskip * 4:
            continue
        obs_rows, pixels, _, _ = _collect_strided(
            dataset, rows, img_transform, frameskip=args.frameskip)
        emb = encode(model, pixels, args.device)             # (T, D)
        D = emb.shape[1]
        z = emb.numpy()
        if k_prog > 0:
            P = model.P.detach().cpu().numpy()
            Q = model.Q.detach().cpu().numpy()
            z_prog = z @ P
            z_cont = z @ Q
        else:
            z_prog = np.zeros((z.shape[0], 0))
            z_cont = z
        theta = np.arctan2(z_prog[:, 1], z_prog[:, 0]) if k_prog >= 2 else np.zeros(z.shape[0])
        sincos = np.stack([np.sin(theta), np.cos(theta)], axis=1)
        T = z.shape[0]
        step_idx = np.arange(T, dtype=float).reshape(-1, 1)
        if args.env_target:
            target, target_name = gather_env_target(dataset, obs_rows, args.data)
        else:
            target = gather_target(dataset, obs_rows, args.target_col_a, args.target_col_b)

        # Episode-relative features: anchor θ and z to the starting frame so
        # that the per-episode sign ambiguity (atan2 unwrap depends on the
        # starting angle) is removed. Δθ_t is the unwrapped phase advance
        # since t=0.
        theta_un = np.unwrap(theta)
        delta_theta = (theta_un - theta_un[0]).reshape(-1, 1)
        delta_z_prog = z_prog - z_prog[0:1]
        # Episode-normalised target: progress = (y_0 − y_t) / max(|y_0 − y_min|, ε)
        # — 0 at t=0, grows toward 1 as the cube approaches the goal.
        y0 = float(target[0])
        denom = max(abs(y0 - float(target.min())), 1e-3)
        progress = (y0 - target) / denom

        per_ep[ep_id] = dict(
            z=z, z_prog=z_prog, z_cont=z_cont, sincos=sincos,
            theta=theta.reshape(-1, 1), step_idx=step_idx, y=target,
            delta_theta=delta_theta,
            delta_z_prog=delta_z_prog,
            progress=progress,
        )
        print(f"  ep {ep_id}: T_obs={T}, target range [{target.min():.3f}, {target.max():.3f}]")

    eps_list = list(per_ep.keys())
    print(f"\nLoaded {len(eps_list)} episodes; running probe with "
          f"{args.n_bootstrap} bootstraps (test_frac={args.test_frac})")

    # Build the feature dictionary (key -> per-episode feature array)
    feature_specs = {
        "step_idx":              ("step_idx",  1),
        "θ (1-d, raw)":          ("theta",     1),
        "Δθ (1-d, from start)":  ("delta_theta", 1),
        "(sin θ, cos θ)":        ("sincos",    2),
        f"z_prog ({k_prog}-d)":  ("z_prog",    k_prog),
        f"Δz_prog ({k_prog}-d, from start)": ("delta_z_prog", k_prog),
        f"z_cont ({D - k_prog}-d)": ("z_cont",  D - k_prog),
        f"z (full, {D}-d)":      ("z",         D),
        "random-2d (control)":   ("z_random2d", 2),
    }

    # Pre-build the random 2-d projection of z (deterministic seed across bootstraps)
    rng_proj = np.random.default_rng(123)
    proj_2d = rng_proj.normal(size=(D, 2))
    proj_2d /= np.linalg.norm(proj_2d, axis=0, keepdims=True) + 1e-12
    for ep in per_ep.values():
        ep["z_random2d"] = ep["z"] @ proj_2d

    # Pick the target series (raw distance vs episode-normalised progress)
    target_key = "progress" if args.target_relative else "y"
    if args.target_relative:
        target_name = "progress = (y₀−y_t)/|max diff|  (per-episode normalised)"

    # Bootstrap loop
    results = {fname: dict(r2=[], mae=[]) for fname in feature_specs}
    rng = np.random.default_rng(0)
    for b in range(args.n_bootstrap):
        train_eps, test_eps = split_train_test_eps(eps_list, rng, test_frac=args.test_frac)
        for fname, (key, dim) in feature_specs.items():
            X_tr = np.concatenate([per_ep[e][key]        for e in train_eps], axis=0)
            y_tr = np.concatenate([per_ep[e][target_key] for e in train_eps], axis=0)
            X_te = np.concatenate([per_ep[e][key]        for e in test_eps],  axis=0)
            y_te = np.concatenate([per_ep[e][target_key] for e in test_eps],  axis=0)
            r2, mae = fit_probe(X_tr, y_tr, X_te, y_te)
            results[fname]["r2"].append(r2)
            results[fname]["mae"].append(mae)

    # ── Per-episode probe (within-episode progress) ─────────────────────────
    # The pooled probe averages across episodes; the per-episode probe asks
    # "within a single rollout, does this feature explain variance in
    # progress beyond what time alone explains?" — which is the right
    # question for a phase coordinate that may be sign-flipped across episodes.
    print("\nPer-episode probe (within-episode regression, leave-one-out CV on time):")

    def loo_r2(X, y):
        """Leave-one-step-out R² for a 1d/low-d regression within a single ep.
        With ~30-40 steps and ≤8 feature dims, this is well-conditioned."""
        from sklearn.linear_model import Ridge
        if len(y) < 4 or X.shape[1] >= len(y) - 1:
            return float("nan")
        preds = np.zeros_like(y)
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            r = Ridge(alpha=1e-3, fit_intercept=True).fit(X[mask], y[mask])
            preds[i] = r.predict(X[i:i+1])[0]
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    # Only low-d features can be probed within-episode (T~40 steps)
    low_d_features = ["step_idx", "θ (1-d, raw)", "Δθ (1-d, from start)",
                       "(sin θ, cos θ)", f"z_prog ({k_prog}-d)",
                       f"Δz_prog ({k_prog}-d, from start)",
                       "random-2d (control)"]
    per_ep_r2: dict[str, list[float]] = {f: [] for f in low_d_features}
    for ep_id, ep in per_ep.items():
        y = ep[target_key]
        for fname in low_d_features:
            key = feature_specs[fname][0]
            X = ep[key]
            r2 = loo_r2(X, y)
            if not np.isnan(r2):
                per_ep_r2[fname].append(r2)

    print(f"  {'feature':<28s}  {'mean R²':<10s}  {'median':<10s}  {'fraction R²>0':<10s}")
    per_ep_summary = {}
    for fname in low_d_features:
        arr = np.asarray(per_ep_r2[fname])
        if len(arr) == 0:
            continue
        m = float(arr.mean()); md = float(np.median(arr)); pos = float((arr > 0).mean())
        per_ep_summary[fname] = {
            "mean_r2": m, "median_r2": md, "n_eps": int(len(arr)),
            "frac_positive": pos,
            "r2_values": arr.tolist(),
        }
        print(f"  {fname:<28s}  {m:+.3f}     {md:+.3f}     {pos:.3f}")

    print(f"\n{'feature':<28s}  {'dim':<6s}  {'R² mean ± std':<18s}  {'MAE mean ± std':<18s}  R²/dim")
    summary_table = []
    for fname, (key, dim) in feature_specs.items():
        r2s = np.asarray(results[fname]["r2"])
        maes = np.asarray(results[fname]["mae"])
        r2_per_dim = r2s.mean() / max(dim, 1)
        print(f"  {fname:<28s}  {dim:<6d}  "
              f"{r2s.mean():.3f} ± {r2s.std():.3f}    "
              f"{maes.mean():.4f} ± {maes.std():.4f}  "
              f"{r2_per_dim:+.4f}")
        summary_table.append({
            "feature": fname, "dim": dim,
            "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
            "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
            "r2_per_dim": float(r2_per_dim),
        })

    # ── Visualisation ────────────────────────────────────────────────────────
    # Two panels:
    #  (A) pooled probe — one bar per feature, R² with bootstrap error
    #  (B) per-episode probe — boxplot of LOO R² per feature (low-d only)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                              gridspec_kw={"width_ratios": [1.0, 1.0]})

    ax = axes[0]
    names = [r["feature"] for r in summary_table]
    r2_means = [r["r2_mean"] for r in summary_table]
    r2_stds = [r["r2_std"] for r in summary_table]
    dims = [r["dim"] for r in summary_table]
    palette = {
        "step_idx": "tab:grey",
        "θ (1-d, raw)": "tab:purple",
        "Δθ (1-d, from start)": "tab:purple",
        "(sin θ, cos θ)": "tab:red",
        "random-2d (control)": "lightgrey",
    }
    colors_a = [palette.get(n, "tab:olive" if "z_prog" in n else
                ("tab:blue" if n.startswith("z (") else "tab:cyan"))
                for n in names]
    ax.bar(range(len(names)), r2_means, yerr=r2_stds, color=colors_a,
            edgecolor="black", linewidth=0.5, capsize=4)
    for i, (v, d) in enumerate(zip(r2_means, dims)):
        # clip the label to keep it readable when R² is very negative
        v_show = max(v, -0.4)
        ax.text(i, v_show + 0.025, f"{v:.2f}\n(d={d})", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"pooled test R²")
    ax.set_title(f"Pooled probe (test_frac=0.25, {args.n_bootstrap} bootstraps)\n"
                 f"target = {target_name}", fontsize=10)
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    # Panel B: per-episode LOO R² distribution
    ax = axes[1]
    pe_names = [n for n in low_d_features if n in per_ep_summary]
    pe_data = [per_ep_summary[n]["r2_values"] for n in pe_names]
    bp = ax.boxplot(pe_data, positions=range(len(pe_names)), widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.5))
    pe_colors = [palette.get(n, "tab:olive" if "z_prog" in n else "tab:cyan")
                 for n in pe_names]
    for box, c in zip(bp["boxes"], pe_colors):
        box.set(facecolor=c, alpha=0.55, edgecolor="black")
    rng_jit = np.random.default_rng(0)
    for pos, vals in zip(range(len(pe_names)), pe_data):
        jit = (rng_jit.random(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), pos) + jit, vals, s=10, c="black",
                   alpha=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(pe_names)))
    ax.set_xticklabels(pe_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("per-episode LOO R²")
    ax.set_title(f"Per-episode probe — does this feature predict {target_name}\n"
                 f"within a single trajectory?  (n={len(eps_list)} eps)",
                 fontsize=10)
    ax.set_ylim(-1.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle(f"Experiment 3 — task-progress probe on {_canonical_env(args.data)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "probe_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "ckpt": str(args.ckpt),
        "env": args.data,
        "k_prog": k_prog,
        "embed_dim": int(D),
        "n_episodes": len(eps_list),
        "n_bootstraps": args.n_bootstrap,
        "target_col_a": args.target_col_a,
        "target_col_b": args.target_col_b,
        "target_name": target_name,
        "target_relative": bool(args.target_relative),
        "results_pooled": summary_table,
        "results_per_episode": per_ep_summary,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
