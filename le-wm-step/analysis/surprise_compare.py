"""Compare angular-drift surprise (Δθ) against latent-MSE surprise on JEPA-STEP.

For each held-out episode we:

1. Encode every frame through the trained encoder + projector → ``z_{0..T}``.
2. Encode actions via ``model.action_encoder`` → ``a_emb_{0..T-1}``.
3. Teacher-force the predictor at every step ``t``: feed the last ``H`` observed
   embeddings and actions, take the last output as ``pred_z_{t+1}``.
4. Compute, per step:
     * ``zmse_t   = || pred_z_{t+1} − obs_z_{t+1} ||^2``
     * ``dtheta_t = wrap( θ(pred) − θ(obs) )``  with θ = atan2(z_prog[1], z_prog[0]).

5. *(Optional, for OOD)* Build a corrupted episode by replacing a ``[t1, t2]``
   segment of the action sequence with actions sampled from a different episode
   (or swapped uniformly at random). Recompute the two surprise series and
   benchmark ``zmse`` vs ``|dtheta|`` for **per-step anomaly localization**
   via AUROC.

Usage
-----

Qualitative trajectory heatmap (single episode):
    python analysis/surprise_compare.py \
        --ckpt <ckpt> --data cube_single_expert \
        --cache-dir $STABLEWM_HOME/datasets \
        --episodes 500 \
        --traj-heatmap \
        --out analysis_out/surprise_cube_ep500/

Quantitative AUROC (multi-episode, action-corruption OOD):
    python analysis/surprise_compare.py \
        --ckpt <ckpt> --data reacher \
        --cache-dir $STABLEWM_HOME/datasets \
        --episodes 0 100 500 1000 2000 \
        --auroc --corrupt-fraction 0.25 \
        --out analysis_out/surprise_reacher_auroc/
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


# ---------------------------------------------------------------------------
# Helpers (mirrors latent_compass.py — duplicated here so the script is self-contained)
# ---------------------------------------------------------------------------

def _build_img_transform(img_size: int):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def _episode_rows(dataset, episode_id: int) -> np.ndarray:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = dataset.get_col_data(col)
    step = dataset.get_col_data("step_idx")
    rows = np.nonzero(ep == episode_id)[0]
    return rows[np.argsort(step[rows], kind="stable")]


def _collect_pixels(dataset, rows, img_transform) -> torch.Tensor:
    frames = []
    for r in rows:
        f = dataset[int(r)]["pixels"]
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        f = f.squeeze(0)
        frames.append(img_transform(f))
    return torch.stack(frames, dim=0)


def _raw_action(dataset, r) -> torch.Tensor:
    a = dataset[int(r)]["action"]
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    return torch.as_tensor(a, dtype=torch.float32).squeeze(0)


def _collect_strided(dataset, rows, img_transform, frameskip: int = 5):
    """Subsample frames at ``frameskip`` and stack ``frameskip`` consecutive raw
    actions per observed step (matches the train-time action_block convention).

    Returns:
        obs_rows  — np.ndarray of length T_obs (the row indices we kept)
        pixels    — (T_obs, C, H, W) image tensor
        actions   — (T_obs, frameskip * raw_action_dim) action-block tensor
        raw_actions — (T_obs, frameskip, raw_action_dim) — preserved for OOD swaps
    """
    n = len(rows)
    # We need ``frameskip`` raw actions starting at each observed row, so the
    # last admissible observed index is ``n - frameskip``. Equivalently: keep
    # the observed steps where we have a full 5-action block ahead.
    observed_idx = list(range(0, n - frameskip + 1, frameskip))
    obs_rows = rows[observed_idx]

    pixels = []
    for r in obs_rows:
        f = dataset[int(r)]["pixels"]
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        pixels.append(img_transform(f.squeeze(0)))
    pixels = torch.stack(pixels, dim=0)

    raw_blocks = []
    flat_blocks = []
    for i in observed_idx:
        block = torch.stack([_raw_action(dataset, rows[i + k]) for k in range(frameskip)], dim=0)
        # block: (frameskip, A_raw)
        raw_blocks.append(block)
        flat_blocks.append(block.reshape(-1))
    raw_actions = torch.stack(raw_blocks, dim=0)  # (T_obs, frameskip, A_raw)
    actions = torch.stack(flat_blocks, dim=0)     # (T_obs, frameskip * A_raw)
    return obs_rows, pixels, actions, raw_actions


def _canonical_env(name: str) -> str:
    n = name.lower()
    if n.startswith("pusht"): return "pusht"
    if n.startswith("cube"):  return "cube"
    if n.startswith("reacher"): return "reacher"
    if n.startswith("tworoom"): return "tworoom"
    return n


def _extract_xy(dataset, rows, env: str):
    env = _canonical_env(env)
    cols = dataset.column_names
    if env == "pusht" and "state" in cols:
        s = np.stack([dataset[int(r)]["state"] for r in rows]).squeeze()
        if s.ndim == 1: s = s[None, :]
        return s[:, :2]
    if env == "tworoom" and "proprio" in cols:
        p = np.stack([dataset[int(r)]["proprio"] for r in rows]).squeeze()
        if p.ndim == 1: p = p[None, :]
        return p[:, :2]
    if env == "reacher" and "finger_pos" in cols:
        f = np.stack([dataset[int(r)]["finger_pos"] for r in rows]).squeeze()
        if f.ndim == 1: f = f[None, :]
        return f[:, :2]
    if env == "cube" and "privileged_block_0_pos" in cols:
        b = np.stack([dataset[int(r)]["privileged_block_0_pos"] for r in rows]).squeeze()
        if b.ndim == 1: b = b[None, :]
        return b[:, :2]
    return None


# ---------------------------------------------------------------------------
# Encoding & teacher-forced prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(model, pixels, device, chunk=32):
    model.eval().to(device)
    embs = []
    for s in range(0, pixels.size(0), chunk):
        x = pixels[s:s+chunk].to(device)
        out = model.encoder(x, interpolate_pos_encoding=True)
        embs.append(model.projector(out.last_hidden_state[:, 0]).cpu())
    return torch.cat(embs, dim=0)  # (T, D)


@torch.no_grad()
def encode_actions(model, actions, device):
    """``actions`` has shape (T, A); the action_encoder expects (B, T, A)."""
    a = actions.unsqueeze(0).to(device)  # (1, T, A)
    return model.action_encoder(a).squeeze(0).cpu()  # (T, A_emb)


@torch.no_grad()
def teacher_force_predict(model, emb, act_emb, history_size: int = 3):
    """Per-step teacher-forced predictions.

    Convention: at index ``t``, the predictor takes the last ``H`` observed
    ``(z_{t-H+1..t}, a_{t-H+1..t})`` and outputs prediction of ``z_{t+1}``.
    Returns a (T - H, D) tensor where ``preds[i]`` predicts ``z_{H + i}``.
    """
    T, D = emb.shape
    H = history_size
    preds = []
    emb_b = emb.unsqueeze(0)  # (1, T, D)
    act_b = act_emb.unsqueeze(0)  # (1, T, A_emb)
    for t in range(H - 1, T - 1):
        emb_h = emb_b[:, t - H + 1 : t + 1].to(next(model.parameters()).device)
        act_h = act_b[:, t - H + 1 : t + 1].to(next(model.parameters()).device)
        p = model.predict(emb_h, act_h)[:, -1, :]  # (1, D)
        preds.append(p.cpu().squeeze(0))
    return torch.stack(preds, dim=0)  # (T - H, D)


def project_theta(model, z: torch.Tensor) -> np.ndarray:
    """Return θ = atan2(z_prog[1], z_prog[0]) for a (..., D) tensor."""
    P = model.P.detach().cpu()
    zp = (z @ P).numpy()  # (..., k)
    return np.arctan2(zp[..., 1], zp[..., 0])


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap to (-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Action corruption (for OOD/AUROC)
# ---------------------------------------------------------------------------

def corrupt_actions(actions: torch.Tensor, t1: int, t2: int, mode: str = "shuffle",
                     other_actions: torch.Tensor | None = None,
                     rng: np.random.Generator | None = None) -> torch.Tensor:
    """Return a corrupted copy of ``actions`` with the [t1:t2] slice replaced."""
    rng = rng or np.random.default_rng(0)
    out = actions.clone()
    seg = t2 - t1
    if mode == "shuffle":
        idx = np.arange(actions.shape[0])
        rng.shuffle(idx)
        out[t1:t2] = actions[idx[:seg]]
    elif mode == "reverse":
        out[t1:t2] = actions[t1:t2].flip(0)
    elif mode == "other_episode" and other_actions is not None:
        n_other = other_actions.shape[0]
        if n_other >= seg:
            start = rng.integers(0, n_other - seg + 1)
            out[t1:t2] = other_actions[start:start + seg]
    elif mode == "random":
        # Sample from the range of observed actions to keep magnitudes plausible
        amin, amax = actions.min(0).values, actions.max(0).values
        u = torch.rand(seg, *actions.shape[1:])
        out[t1:t2] = amin + u * (amax - amin)
    return out


# ---------------------------------------------------------------------------
# Surprise computation
# ---------------------------------------------------------------------------

def compute_surprise(model, emb: torch.Tensor, act_emb: torch.Tensor,
                      history_size: int = 3) -> dict:
    """Return per-step ``zmse`` and ``|dtheta|`` series."""
    T, D = emb.shape
    H = history_size
    preds = teacher_force_predict(model, emb, act_emb, H)  # (T-H, D)
    obs_targets = emb[H:]  # (T-H, D)

    # z-MSE per step (sum over feature dim; same units as the planner's split cost)
    zmse = ((preds - obs_targets) ** 2).sum(dim=-1).numpy()  # (T-H,)

    # Δθ per step
    th_obs = project_theta(model, obs_targets)
    th_pred = project_theta(model, preds)
    dtheta = np.abs(wrap_pi(th_obs - th_pred))  # (T-H,) in [0, π]

    return {
        "zmse": zmse,
        "dtheta": dtheta,
        "history_size": H,
        "first_step": H,  # zmse[0] corresponds to obs step H
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_traj_heatmap(xy: np.ndarray, zmse: np.ndarray, dtheta: np.ndarray,
                       env: str, ep_id: int, first_step: int, out: Path):
    """Side-by-side: agent path colored by zmse vs by |Δθ|.

    The two metrics are plotted on the same trajectory but with their own
    color scales, so spikes in one but not the other are visually obvious.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    H = first_step
    xy_ts = xy[H:H + len(zmse)]  # (T-H, 2) — aligned with zmse/dtheta

    for ax, val, name, cmap in [
        (axes[0], zmse,   "z-MSE surprise",          "magma"),
        (axes[1], dtheta, "|Δθ| surprise (rad)",     "magma"),
    ]:
        ax.plot(xy_ts[:, 0], xy_ts[:, 1], color="lightgrey", lw=0.6, alpha=0.7, zorder=1)
        sc = ax.scatter(xy_ts[:, 0], xy_ts[:, 1], c=val, cmap=cmap, s=22,
                        edgecolor="white", linewidth=0.4, zorder=2)
        # mark start/end
        ax.scatter(xy_ts[0:1, 0], xy_ts[0:1, 1], marker="s", s=80,
                   facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
        ax.scatter(xy_ts[-1:, 0], xy_ts[-1:, 1], marker="*", s=120,
                   facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        env_canon = _canonical_env(env)
        ax_labels = {
            "pusht":   ("agent_x (px)",        "agent_y (px)"),
            "reacher": ("end-effector x (m)",  "end-effector y (m)"),
            "tworoom": ("agent_x (px)",        "agent_y (px)"),
            "cube":    ("cube_x (m)",          "cube_y (m)"),
        }.get(env_canon, ("x", "y"))
        ax.set_xlabel(ax_labels[0]); ax.set_ylabel(ax_labels[1])
        ax.set_title(name)
        fig.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(f"Surprise comparison on {_canonical_env(env)} ep {ep_id} "
                 f"(per-step teacher-forced prediction error)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_phase_event_overlay(zmse: np.ndarray, dtheta: np.ndarray,
                                first_step: int, env: str, ep_id: int,
                                event_steps: np.ndarray, signal: np.ndarray,
                                signal_name: str, out: Path):
    """Both metrics overlaid with vertical lines at ground-truth event timestamps."""
    fig, ax1 = plt.subplots(figsize=(10, 4.0))
    t = np.arange(first_step, first_step + len(zmse))

    z_n = (zmse - zmse.min()) / (zmse.max() - zmse.min() + 1e-12)
    d_n = (dtheta - dtheta.min()) / (dtheta.max() - dtheta.min() + 1e-12)

    ax1.plot(t, z_n, color="tab:blue", lw=1.4, label="z-MSE (norm.)")
    ax1.plot(t, d_n, color="tab:red",  lw=1.4, label="|Δθ| (norm.)")
    for e in event_steps:
        ax1.axvline(e, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_xlabel("step t (observed)")
    ax1.set_ylabel("normalised surprise")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"{_canonical_env(env)} ep {ep_id} — surprise vs ground-truth events "
                  f"({signal_name}; dashed lines = transitions)")

    ax2 = ax1.twinx()
    sig_t = np.arange(len(signal))
    ax2.plot(sig_t, signal, color="tab:green", lw=1.0, alpha=0.5,
             label=signal_name, drawstyle="steps-post")
    ax2.set_ylabel(signal_name, color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_surprise_timeseries(zmse: np.ndarray, dtheta: np.ndarray,
                              first_step: int, env: str, ep_id: int,
                              out: Path, corrupt_range: tuple | None = None):
    """Time series of both metrics, normalized to [0, 1] for visual comparison."""
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    t = np.arange(first_step, first_step + len(zmse))

    z_n = (zmse - zmse.min()) / (zmse.max() - zmse.min() + 1e-12)
    d_n = (dtheta - dtheta.min()) / (dtheta.max() - dtheta.min() + 1e-12)

    ax1.plot(t, z_n, color="tab:blue",  lw=1.4, label="z-MSE (norm.)")
    ax1.plot(t, d_n, color="tab:red",   lw=1.4, label="|Δθ| (norm.)")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("normalised surprise")
    if corrupt_range is not None:
        ax1.axvspan(corrupt_range[0], corrupt_range[1], color="orange",
                    alpha=0.18, label="corrupted segment")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"Per-step surprise on {_canonical_env(env)} ep {ep_id}")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# AUROC (per-step localization)
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard AUROC: P(score(positive) > score(negative))."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def detect_phase_events(signal: np.ndarray, kind: str = "binary",
                          threshold: float = 0.5) -> np.ndarray:
    """Return the list of step indices ``t`` (1 ≤ t < T) where a phase event
    occurred, defined as a transition in ``signal``.

    For ``kind='binary'``, threshold the signal and detect any 0↔1 flip.
    For ``kind='threshold'`` (continuous), detect crossings of ``threshold``.
    """
    s = np.asarray(signal).flatten().astype(float)
    if kind == "binary":
        b = (s > threshold).astype(int)
        return np.nonzero(np.diff(b) != 0)[0] + 1
    if kind == "threshold":
        above = (s > threshold).astype(int)
        return np.nonzero(np.diff(above) != 0)[0] + 1
    raise ValueError(kind)


def labels_from_events(n_steps: int, event_steps: np.ndarray,
                        tolerance: int = 1) -> np.ndarray:
    """Per-step binary labels: 1 if within ``tolerance`` of any event."""
    out = np.zeros(n_steps, dtype=int)
    for e in event_steps:
        lo = max(0, e - tolerance)
        hi = min(n_steps, e + tolerance + 1)
        out[lo:hi] = 1
    return out


def gather_phase_signal(dataset, obs_rows, col: str) -> np.ndarray:
    """Load ``col`` for each observed row, returning a (T_obs,) array."""
    vals = []
    for r in obs_rows:
        v = dataset[int(r)][col]
        v = np.asarray(v).flatten()
        # column may be (1,) or (1, K); take first scalar
        vals.append(float(v.flat[0]))
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--history-size", type=int, default=3,
                   help="Predictor's history window. Default matches train-time.")
    p.add_argument("--frameskip", type=int, default=5,
                   help="Stride between observed frames; raw actions are stacked into blocks of this size to match train-time action_block")
    p.add_argument("--traj-heatmap", action="store_true",
                   help="For each episode, emit zmse vs |Δθ| side-by-side trajectory heatmap")
    p.add_argument("--time-series", action="store_true",
                   help="For each episode, also emit normalised time-series plots")
    p.add_argument("--auroc", action="store_true",
                   help="Run action-corruption OOD experiment and report AUROC")
    p.add_argument("--corrupt-fraction", type=float, default=0.25,
                   help="Fraction of each episode to corrupt for AUROC")
    p.add_argument("--corrupt-mode", default="shuffle",
                   choices=["shuffle", "reverse", "other_episode", "random"])
    p.add_argument("--phase-event-col", default=None,
                   help="Dataset column whose transitions define ground-truth phase "
                        "events (e.g. 'proprio_gripper_contact' for cube). Enables "
                        "the phase-alignment experiment.")
    p.add_argument("--phase-event-kind", default="binary",
                   choices=["binary", "threshold"],
                   help="How to interpret the phase column (default: binary 0/1 flip)")
    p.add_argument("--phase-event-threshold", type=float, default=0.5)
    p.add_argument("--phase-event-tolerance", type=int, nargs="+", default=[1],
                   help="±k steps around each event count as 'event-aligned' positives. "
                        "Multiple values: report AUROC at each tolerance.")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    model.eval()
    print(f"  k_prog = {getattr(model, 'k_prog', 0)}")

    print(f"Loading dataset: {args.data}")
    import stable_worldmodel as swm
    from hdf5_dataset import HDF5Dataset
    cache_dir = args.cache_dir or swm.data.utils.get_cache_dir()
    dataset = HDF5Dataset(args.data, cache_dir=Path(cache_dir), keys_to_cache=[])
    img_transform = _build_img_transform(args.img_size)

    summary: dict = {
        "ckpt": str(args.ckpt),
        "env": args.data,
        "k_prog": int(getattr(model, "k_prog", 0)),
        "history_size": args.history_size,
        "per_episode": {},
        "auroc": {},
    }

    H = args.history_size
    rng = np.random.default_rng(0)

    # Cache embeddings for all episodes (reused for AUROC if needed)
    cache: dict[int, dict] = {}
    for ep_id in args.episodes:
        rows = _episode_rows(dataset, ep_id)
        T_raw = len(rows)
        if T_raw < args.frameskip * (H + 2):
            print(f"  ep {ep_id}: too short ({T_raw}), skipping")
            continue

        obs_rows, pixels, actions, raw_actions = _collect_strided(
            dataset, rows, img_transform, frameskip=args.frameskip
        )
        T = len(obs_rows)
        print(f"  ep {ep_id}: T_raw={T_raw}, T_obs={T} (stride={args.frameskip}), "
              f"action_block dim={actions.shape[-1]}")

        emb = encode(model, pixels, args.device)
        a_emb = encode_actions(model, actions, args.device)
        # xy at observed timestamps, for the trajectory heatmap
        xy_full = _extract_xy(dataset, rows, args.data)
        xy = xy_full[::args.frameskip][:T] if xy_full is not None else None

        s_clean = compute_surprise(model, emb, a_emb, history_size=H)

        cache[ep_id] = {
            "rows": rows, "obs_rows": obs_rows,
            "actions": actions, "raw_actions": raw_actions,
            "emb": emb, "a_emb": a_emb,
            "xy": xy, "T": T, "s_clean": s_clean,
        }

        ep_record = {
            "T": int(T),
            "zmse_mean":   float(s_clean["zmse"].mean()),
            "dtheta_mean": float(s_clean["dtheta"].mean()),
        }

        if args.traj_heatmap and xy is not None:
            plot_traj_heatmap(xy, s_clean["zmse"], s_clean["dtheta"],
                               args.data, ep_id, s_clean["first_step"],
                               out_dir / f"surprise_traj_ep{ep_id}.png")
            ep_record["heatmap"] = f"surprise_traj_ep{ep_id}.png"

        if args.time_series:
            plot_surprise_timeseries(s_clean["zmse"], s_clean["dtheta"],
                                       s_clean["first_step"], args.data, ep_id,
                                       out_dir / f"surprise_ts_ep{ep_id}.png")
            ep_record["timeseries"] = f"surprise_ts_ep{ep_id}.png"

        summary["per_episode"][str(ep_id)] = ep_record

    # ----- AUROC (action-corruption OOD) -----
    if args.auroc:
        print("\nRunning AUROC experiment (action corruption)...")
        rows_zmse_scores, rows_dtheta_scores, rows_labels = [], [], []
        per_ep_auc = {}
        for ep_id, c in cache.items():
            T, emb, a_emb, actions = c["T"], c["emb"], c["a_emb"], c["actions"]
            seg = max(H + 1, int(round(args.corrupt_fraction * T)))
            seg = min(seg, T - H - 1)
            t1 = rng.integers(H, T - seg)
            t2 = t1 + seg

            # Pick a *different* episode's actions for "other_episode"
            other = None
            if args.corrupt_mode == "other_episode" and len(cache) > 1:
                other_id = next(eid for eid in cache if eid != ep_id)
                other = cache[other_id]["actions"]

            corrupted_actions = corrupt_actions(actions, t1, t2,
                                                 mode=args.corrupt_mode,
                                                 other_actions=other, rng=rng)
            a_emb_corr = encode_actions(model, corrupted_actions, args.device)
            s_corr = compute_surprise(model, emb, a_emb_corr, history_size=H)

            # Per-step labels: 1 if the *target* step (t in [H, T-1]) falls in
            # the corrupted window. The action a_t was corrupted, so its
            # effect surfaces as a poor prediction of z_{t+1}; we tag that
            # as the anomalous step.
            n = len(s_corr["zmse"])
            steps = np.arange(H, H + n)  # observed-step indices
            labels = ((steps >= t1) & (steps < t2)).astype(int)

            zauc = auroc(s_corr["zmse"],   labels)
            tauc = auroc(s_corr["dtheta"], labels)
            per_ep_auc[str(ep_id)] = {
                "corrupt_window": [int(t1), int(t2)],
                "n_anomalous": int(labels.sum()),
                "auroc_zmse":   zauc,
                "auroc_dtheta": tauc,
            }

            # Time-series figure with the corrupted window highlighted
            plot_surprise_timeseries(s_corr["zmse"], s_corr["dtheta"],
                                       s_corr["first_step"], args.data, ep_id,
                                       out_dir / f"surprise_ts_corr_ep{ep_id}.png",
                                       corrupt_range=(t1, t2))

            rows_zmse_scores.append(s_corr["zmse"])
            rows_dtheta_scores.append(s_corr["dtheta"])
            rows_labels.append(labels)

        # Pooled AUROC across all episodes
        z_pool = np.concatenate(rows_zmse_scores)
        t_pool = np.concatenate(rows_dtheta_scores)
        l_pool = np.concatenate(rows_labels)

        # Combined score: z-MSE and |Δθ| live on incompatible scales, so we
        # rank-normalize each before summing. This is the classic "does adding
        # a second metric improve detection" test: if the combined AUROC >
        # max(individual AUROCs), the second metric carries complementary info.
        from scipy.stats import rankdata
        z_rank = rankdata(z_pool) / len(z_pool)
        t_rank = rankdata(t_pool) / len(t_pool)
        combined_pool = z_rank + t_rank

        summary["auroc"] = {
            "mode": args.corrupt_mode,
            "corrupt_fraction": args.corrupt_fraction,
            "per_episode": per_ep_auc,
            "pooled_auroc_zmse":     auroc(z_pool, l_pool),
            "pooled_auroc_dtheta":   auroc(t_pool, l_pool),
            "pooled_auroc_combined": auroc(combined_pool, l_pool),
        }
        print(f"  pooled AUROC  z-MSE       : {summary['auroc']['pooled_auroc_zmse']:.3f}")
        print(f"  pooled AUROC  |Δθ|        : {summary['auroc']['pooled_auroc_dtheta']:.3f}")
        print(f"  pooled AUROC  combined    : {summary['auroc']['pooled_auroc_combined']:.3f}  "
              f"(rank-sum of z-MSE + |Δθ|)")

    # ----- Phase-event alignment (semantic ground-truth) -----
    if args.phase_event_col is not None:
        tols = args.phase_event_tolerance
        print(f"\nPhase-event alignment using column '{args.phase_event_col}' "
              f"(kind={args.phase_event_kind}, tols={tols})")
        per_ep_phase: dict[str, dict] = {}

        # Collect per-episode score series + per-tolerance label series
        zmse_series, dtheta_series = [], []
        labels_per_tol = {tol: [] for tol in tols}
        n_events_total = 0

        from scipy.stats import rankdata

        for ep_id, c in cache.items():
            sig = gather_phase_signal(dataset, c["obs_rows"], args.phase_event_col)
            events_obs = detect_phase_events(
                sig, kind=args.phase_event_kind, threshold=args.phase_event_threshold
            )
            n_score = len(c["s_clean"]["zmse"])
            first = c["s_clean"]["first_step"]
            events_in_range = events_obs[
                (events_obs >= first) & (events_obs < first + n_score)
            ]
            event_score_idx = events_in_range - first
            n_events_total += len(events_in_range)

            zmse, dtheta = c["s_clean"]["zmse"], c["s_clean"]["dtheta"]

            ep_record = {"n_events": int(len(events_in_range)), "auroc_per_tol": {}}
            for tol in tols:
                lab = labels_from_events(n_score, event_score_idx, tolerance=tol)
                ep_record["auroc_per_tol"][str(tol)] = {
                    "n_pos": int(lab.sum()),
                    "auroc_zmse":   auroc(zmse,   lab),
                    "auroc_dtheta": auroc(dtheta, lab),
                }
                labels_per_tol[tol].append(lab)

            per_ep_phase[str(ep_id)] = ep_record
            zmse_series.append(zmse)
            dtheta_series.append(dtheta)

            # Only emit per-episode plots for the first 8 episodes (file count cap)
            if len(per_ep_phase) <= 8:
                plot_phase_event_overlay(
                    zmse, dtheta, first, args.data, ep_id,
                    event_steps=events_in_range, signal=sig,
                    signal_name=args.phase_event_col,
                    out=out_dir / f"phase_overlay_ep{ep_id}.png",
                )

        # Pooled across all episodes, per tolerance
        z_pool = np.concatenate(zmse_series)
        t_pool = np.concatenate(dtheta_series)
        z_rank = rankdata(z_pool) / len(z_pool)
        t_rank = rankdata(t_pool) / len(t_pool)
        comb_pool = z_rank + t_rank

        pooled = {}
        for tol in tols:
            l_pool = np.concatenate(labels_per_tol[tol])
            pooled[str(tol)] = {
                "n_pos": int(l_pool.sum()),
                "n_total": int(len(l_pool)),
                "auroc_zmse":     auroc(z_pool, l_pool),
                "auroc_dtheta":   auroc(t_pool, l_pool),
                "auroc_combined": auroc(comb_pool, l_pool),
            }
            print(f"  tol={tol}: AUROC  z-MSE={pooled[str(tol)]['auroc_zmse']:.3f}  "
                  f"|Δθ|={pooled[str(tol)]['auroc_dtheta']:.3f}  "
                  f"combined={pooled[str(tol)]['auroc_combined']:.3f}  "
                  f"(n_pos={pooled[str(tol)]['n_pos']}/{pooled[str(tol)]['n_total']})")

        # Per-episode head-to-head: how often does |Δθ| beat z-MSE?
        head_to_head = {}
        for tol in tols:
            wins_dtheta = sum(
                1 for ep, rec in per_ep_phase.items()
                if rec["auroc_per_tol"][str(tol)]["auroc_dtheta"]
                   > rec["auroc_per_tol"][str(tol)]["auroc_zmse"]
            )
            head_to_head[str(tol)] = {"theta_wins": wins_dtheta,
                                       "n_eps": len(per_ep_phase)}
            print(f"  tol={tol}: |Δθ| beats z-MSE on {wins_dtheta} / "
                  f"{len(per_ep_phase)} episodes")

        summary["phase_alignment"] = {
            "column": args.phase_event_col,
            "kind": args.phase_event_kind,
            "threshold": args.phase_event_threshold,
            "tolerances": tols,
            "n_episodes": len(per_ep_phase),
            "n_total_events": n_events_total,
            "pooled": pooled,
            "head_to_head": head_to_head,
            "per_episode": per_ep_phase,
        }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
