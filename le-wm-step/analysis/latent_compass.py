"""Latent compass interpretability suite for SD-JEPA checkpoints.

Produces three families of figures from a trained ``*_object.ckpt``:

1.  ``tsne_zprog.png``  / ``tsne_zcont.png``  — t-SNE projections of the
    progression and content subspaces over many episodes. Expectation:
    z^prog mixes across episodes (shared progression manifold), z^cont
    separates into per-episode clusters (episode-specific content).

2.  ``state_traj_theta.png`` — the trajectory-overlay heatmap. For each
    held-out episode, plot the agent's path in raw state space (Push-T:
    agent (x,y); Reacher: end-effector (x,y) computed from qpos; Tworoom:
    agent (x,y) from proprio; OGB-Cube: cube (x,y) from qpos), with each
    point colour-coded by theta_t. Reveals whether the compass coordinate
    advances coherently across the env's physical space.

3.  ``state_grid_theta.png`` — (Push-T only, optional, --grid) a state-
    space scan of theta over a regular grid of agent positions while
    holding the block fixed at its mean dataset position. Heatmap of
    theta(x_agent, y_agent). Slow but produces the cleanest visual.

CPU-friendly by default. Tested on Push-T, Reacher, Tworoom, OGB-Cube.

Usage
-----
    python analysis/latent_compass.py \
        --ckpt  ~/.stable_worldmodel/checkpoints/<run>/<ckpt>_object.ckpt \
        --data  pusht \
        --episodes 0 1000 5000 8000 12000 15000 18000 \
        --out   analysis_out/<run>_compass/

    # add the slow grid-scan figure on Push-T:
    python analysis/latent_compass.py ... --grid

Environment-specific state extraction is handled by a small dispatch
table at the top of ``_extract_xy``; if your dataset stores positions
under non-standard column names, edit there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
#  Project import — pickled JEPA references modules in le-wm-step/
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_LEWM_DIR = _THIS_DIR.parent
if str(_LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(_LEWM_DIR))


# ---------------------------------------------------------------------------
#  Image / dataset helpers (mirror eval-time preprocessing exactly)
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
        sample = dataset[int(r)]
        f = sample["pixels"]
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f)
        f = f.squeeze(0)            # (1, C, H, W) → (C, H, W)
        frames.append(img_transform(f))
    return torch.stack(frames, dim=0)


# ---------------------------------------------------------------------------
#  State extraction per environment
# ---------------------------------------------------------------------------

def _canonical_env(name: str) -> str:
    """Map a swm dataset name (e.g. ``pusht_expert_train``, ``cube_single_expert``)
    onto the canonical env key used by ``_extract_xy``."""
    n = name.lower()
    if n.startswith("pusht"):
        return "pusht"
    if n.startswith("cube"):
        return "cube"
    if n.startswith("reacher"):
        return "reacher"
    if n.startswith("tworoom"):
        return "tworoom"
    return n


def _extract_xy(dataset, rows: np.ndarray, env: str) -> np.ndarray | None:
    """Return a (T, 2) array of (x, y) coordinates suitable for the
    trajectory-overlay heatmap, or None if not extractable for this env.

    The exact convention follows the environment's stored state vector:
        - pusht:    state = (agent_x, agent_y, block_x, block_y, block_angle).
                    Use agent_(x,y) as the trajectory.
        - tworoom:  proprio = (agent_x, agent_y).
        - reacher:  qpos = (theta1, theta2, target_x, target_y) — we use the
                    forward-kinematics end-effector (x, y) computed from
                    the two joint angles with link lengths (0.12, 0.12)
                    (DM-Control reacher default).
        - cube:     qpos contains (cube_x, cube_y, cube_z, ...). We use the
                    cube's (x, y) as the trajectory anchor.
    """
    env = _canonical_env(env)
    cols = dataset.column_names
    if env == "pusht" and "state" in cols:
        s = np.stack([dataset[int(r)]["state"] for r in rows])
        # state may be (T, 1, 5) or (T, 5)
        s = np.array(s).squeeze()
        if s.ndim == 1:
            s = s[None, :]
        return s[:, :2]
    if env == "tworoom" and "proprio" in cols:
        p = np.stack([dataset[int(r)]["proprio"] for r in rows]).squeeze()
        if p.ndim == 1:
            p = p[None, :]
        return p[:, :2]
    if env == "reacher" and "qpos" in cols:
        q = np.stack([dataset[int(r)]["qpos"] for r in rows]).squeeze()
        if q.ndim == 1:
            q = q[None, :]
        # DM-Control reacher: link lengths roughly L1=L2=0.12. Compute
        # end-effector (x, y) from (theta1, theta2).
        L1, L2 = 0.12, 0.12
        t1, t2 = q[:, 0], q[:, 1]
        x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
        y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
        return np.stack([x, y], axis=1)
    if env == "cube" and "privileged_block_0_pos" in cols:
        # OGBCube dataset stores the cube's 3D position directly under
        # ``privileged_block_0_pos`` — use the (x, y) projection.
        b = np.stack([dataset[int(r)]["privileged_block_0_pos"] for r in rows]).squeeze()
        if b.ndim == 1:
            b = b[None, :]
        return b[:, :2]
    return None


# ---------------------------------------------------------------------------
#  Task-progress proxies (for the θ correlation panel)
# ---------------------------------------------------------------------------

# Pusht canonical goal pose: block centered at (256,256) in image-coords,
# angle π/4 rad. Confirmed empirically by the last frame of expert episodes
# (block lands at ~(251–256, 250–260) and angle ~0.78 rad).
_PUSHT_GOAL_XY = np.array([256.0, 256.0])
_PUSHT_GOAL_ANGLE = float(np.pi / 4.0)


def _angle_diff(a, b):
    """|wrap(a - b)| ∈ [0, π]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d)


def _compute_proxies(env: str, dataset, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict of (T,) per-frame quantities that θ_t might track.

    Always includes ``step_idx`` (the elapsed-time clock) as a baseline. Anything
    that beats step_idx in correlation with θ is the interesting story —
    that's θ tracking task content rather than just elapsed time.
    """
    env = _canonical_env(env)
    T = len(rows)
    out: dict[str, np.ndarray] = {"step_idx": np.arange(T, dtype=float)}

    if env == "pusht":
        states = np.stack([dataset[int(r)]["state"] for r in rows]).squeeze()
        if states.ndim == 1:
            states = states[None, :]
        agent = states[:, :2]
        block = states[:, 2:4]
        angle = states[:, 4]

        agent_speed = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(agent, axis=0), axis=1)]
        )
        out["agent_block_dist"] = np.linalg.norm(agent - block, axis=1)
        out["block_goal_dist"] = np.linalg.norm(block - _PUSHT_GOAL_XY, axis=1)
        out["block_angle_err"] = _angle_diff(angle, _PUSHT_GOAL_ANGLE)
        out["agent_speed"] = agent_speed

    elif env == "reacher":
        # The h5 stores finger_pos (end-effector) and target_pos directly,
        # so we use those rather than recomputing forward kinematics from qpos.
        finger = np.stack([dataset[int(r)]["finger_pos"] for r in rows]).squeeze()
        target = np.stack([dataset[int(r)]["target_pos"] for r in rows]).squeeze()
        if finger.ndim == 1:
            finger = finger[None, :]
        if target.ndim == 1:
            target = target[None, :]
        out["ee_target_dist"] = np.linalg.norm(finger - target, axis=1)
        if "qvel" in dataset.column_names:
            qvel = np.stack([dataset[int(r)]["qvel"] for r in rows]).squeeze()
            if qvel.ndim == 1:
                qvel = qvel[None, :]
            out["joint_speed"] = np.linalg.norm(qvel, axis=1)
        if "score" in dataset.column_names:
            score = np.asarray(
                np.stack([dataset[int(r)]["score"] for r in rows])
            ).squeeze().astype(float)
            if not np.all(np.isnan(score)):
                out["score"] = score

    elif env == "tworoom":
        proprio = np.stack([dataset[int(r)]["proprio"] for r in rows]).squeeze()
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        agent = proprio[:, :2]
        out["agent_speed"] = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(agent, axis=0), axis=1)]
        )
        cols = dataset.column_names
        if "distance_to_target" in cols:
            d = np.stack([dataset[int(r)]["distance_to_target"] for r in rows]).squeeze()
            out["agent_target_dist"] = np.atleast_1d(d).astype(float)
        elif "pos_target" in cols:
            tgt = np.stack([dataset[int(r)]["pos_target"] for r in rows]).squeeze()
            if tgt.ndim == 1:
                tgt = tgt[None, :]
            out["agent_target_dist"] = np.linalg.norm(agent - tgt[:, :2], axis=1)

    elif env == "cube":
        cols = dataset.column_names
        # The OGBCube dataset stores privileged scene state directly; use
        # those rather than guessing offsets in qpos.
        if "privileged_block_0_pos" in cols and "privileged_target_block_pos" in cols:
            block = np.stack(
                [dataset[int(r)]["privileged_block_0_pos"] for r in rows]
            ).squeeze()
            tgt = np.stack(
                [dataset[int(r)]["privileged_target_block_pos"] for r in rows]
            ).squeeze()
            if block.ndim == 1: block = block[None, :]
            if tgt.ndim == 1: tgt = tgt[None, :]
            out["block_target_dist"] = np.linalg.norm(block - tgt, axis=1)
        if "proprio_effector_pos" in cols and "privileged_block_0_pos" in cols:
            eff = np.stack(
                [dataset[int(r)]["proprio_effector_pos"] for r in rows]
            ).squeeze()
            block = np.stack(
                [dataset[int(r)]["privileged_block_0_pos"] for r in rows]
            ).squeeze()
            if eff.ndim == 1: eff = eff[None, :]
            if block.ndim == 1: block = block[None, :]
            out["effector_block_dist"] = np.linalg.norm(eff - block, axis=1)
        if "proprio_gripper_contact" in cols:
            c = np.stack(
                [dataset[int(r)]["proprio_gripper_contact"] for r in rows]
            ).squeeze()
            out["gripper_contact"] = np.atleast_1d(c).astype(float)
        if "proprio_gripper_opening" in cols:
            o = np.stack(
                [dataset[int(r)]["proprio_gripper_opening"] for r in rows]
            ).squeeze()
            out["gripper_opening"] = np.atleast_1d(o).astype(float)
        if "privileged_block_0_yaw" in cols:
            y = np.stack(
                [dataset[int(r)]["privileged_block_0_yaw"] for r in rows]
            ).squeeze()
            out["block_yaw"] = np.atleast_1d(y).astype(float)

    return out


# ---------------------------------------------------------------------------
#  Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(model, pixels: torch.Tensor, device: str, chunk: int = 32) -> torch.Tensor:
    model.eval().to(device)
    embs = []
    for s in range(0, pixels.size(0), chunk):
        x = pixels[s:s + chunk].to(device)
        out = model.encoder(x, interpolate_pos_encoding=True)
        embs.append(model.projector(out.last_hidden_state[:, 0]).cpu())
    return torch.cat(embs, dim=0)


def split_pq(model, emb: torch.Tensor):
    if not hasattr(model, "k_prog") or model.k_prog == 0:
        return None, emb
    P = model.P.detach().cpu()
    Q = model.Q.detach().cpu()
    return emb @ P, emb @ Q


def theta_of(zp: np.ndarray) -> np.ndarray:
    """Unwrapped angular coordinate of the first 2 dims of z_prog."""
    th = np.arctan2(zp[:, 1], zp[:, 0])
    return np.unwrap(th)


# ---------------------------------------------------------------------------
#  Plotters
# ---------------------------------------------------------------------------

def _save(fig, out: Path):
    """Save the figure to ``out`` and also next to it as a vector PDF."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tsne(z_by_ep: dict, kind: str, out: Path, n_components: int = 2):
    """t-SNE projection coloured by episode, marker size by time-in-episode.

    ``n_components`` may be 2 (default) or 3; 3D version saves a single static
    view (azim=-60, elev=20) sized to roughly match the 2D plot.
    """
    from sklearn.manifold import TSNE

    all_z = np.concatenate(list(z_by_ep.values()), axis=0)
    if all_z.shape[1] < 2:
        return
    n = all_z.shape[0]
    perplexity = max(5, min(30, n // 4))
    print(f"  [{kind}] t-SNE-{n_components}D on {n} points, perplexity={perplexity} ...")
    proj = TSNE(n_components=n_components, perplexity=perplexity, init="pca",
                random_state=0).fit_transform(all_z)

    cmap = plt.cm.tab10

    if n_components == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        cursor = 0
        for i, (ep_id, z) in enumerate(z_by_ep.items()):
            T = z.shape[0]
            seg = proj[cursor:cursor + T]
            cursor += T
            sizes = np.linspace(8, 28, T)
            ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2], s=sizes,
                       c=[cmap(i % 10)], alpha=0.7, label=f"ep {ep_id}",
                       edgecolor="white", linewidth=0.3)
            ax.scatter(seg[0, 0], seg[0, 1], seg[0, 2], marker="s", s=80,
                       facecolor="none", edgecolor=cmap(i % 10), linewidth=1.2)
            ax.scatter(seg[-1, 0], seg[-1, 1], seg[-1, 2], marker="*", s=140,
                       facecolor=cmap(i % 10), edgecolor="black", linewidth=0.8)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")
        ax.set_title(f"3D t-SNE of $z^{{\\mathrm{{{kind}}}}}$ across {len(z_by_ep)} episodes")
        ax.legend(fontsize=7, loc="upper right", ncol=2 if len(z_by_ep) > 4 else 1)
        ax.view_init(elev=22, azim=-58)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    cursor = 0
    for i, (ep_id, z) in enumerate(z_by_ep.items()):
        T = z.shape[0]
        seg = proj[cursor:cursor + T]
        cursor += T
        sizes = np.linspace(8, 28, T)
        ax.scatter(seg[:, 0], seg[:, 1], s=sizes, c=[cmap(i % 10)],
                   alpha=0.7, label=f"ep {ep_id}", edgecolor="white",
                   linewidth=0.4)
        ax.scatter(seg[0:1, 0], seg[0:1, 1], marker="s", s=80,
                   facecolor="none", edgecolor=cmap(i % 10), linewidth=1.2)
        ax.scatter(seg[-1:, 0], seg[-1:, 1], marker="*", s=140,
                   facecolor=cmap(i % 10), edgecolor="black", linewidth=0.8)

    ax.set_title(f"t-SNE of $z^{{\\mathrm{{{kind}}}}}$ across {len(z_by_ep)} held-out episodes\n"
                 f"(squares: episode start, stars: episode end; bigger marker = later in episode)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="best", ncol=2 if len(z_by_ep) > 4 else 1)
    _save(fig, out)


def plot_state_trajectory(xy_by_ep: dict, theta_by_ep: dict, env: str, out: Path):
    """Each episode's (x, y) path coloured by theta_t. One subplot per episode
    if many; otherwise overlaid in a single panel."""
    eps = list(xy_by_ep.keys())
    if not eps:
        return
    n = len(eps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows),
                             squeeze=False)

    # Common theta range across all episodes for consistent colour
    all_theta = np.concatenate(list(theta_by_ep.values()))
    vmin, vmax = float(all_theta.min()), float(all_theta.max())

    for i, ep_id in enumerate(eps):
        ax = axes[i // cols][i % cols]
        xy = xy_by_ep[ep_id]
        th = theta_by_ep[ep_id]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=th, cmap="viridis",
                        s=18, vmin=vmin, vmax=vmax, edgecolor="white",
                        linewidth=0.4)
        # Trace the path lightly underneath
        ax.plot(xy[:, 0], xy[:, 1], color="grey", lw=0.5, alpha=0.4, zorder=0)
        # mark start/end
        ax.scatter(xy[0:1, 0], xy[0:1, 1], marker="s", s=64,
                   facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
        ax.scatter(xy[-1:, 0], xy[-1:, 1], marker="*", s=120,
                   facecolor="white", edgecolor="black", linewidth=1.0, zorder=3)
        ax.set_title(f"episode {ep_id}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")

    # hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_axis_off()

    fig.suptitle(f"$\\theta_t$ along the agent's trajectory in {_canonical_env(env)}-state space\n"
                 f"(squares: start, stars: end)",
                 fontsize=11)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7,
                        label=r"unwrapped $\theta_t$ (rad)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _raw_frames(dataset, rows: np.ndarray) -> np.ndarray:
    """Return raw (T, H, W, C) uint8 frames for plotting (no normalization)."""
    out = []
    for r in rows:
        f = dataset[int(r)]["pixels"]
        if isinstance(f, torch.Tensor):
            f = f.numpy()
        f = np.asarray(f).squeeze()  # (C,H,W) or (H,W,C)
        if f.shape[0] in (1, 3) and f.ndim == 3:
            f = np.transpose(f, (1, 2, 0))  # CHW → HWC
        out.append(f)
    return np.stack(out, axis=0)


def plot_frame_strip(dataset, rows: np.ndarray, theta: np.ndarray,
                      xy: np.ndarray | None, env: str, out: Path,
                      n_frames: int = 8, ep_id: int | None = None):
    """Frame strip + θ-colored agent trajectory side-by-side.

    Top row: ``n_frames`` evenly-spaced frames from the episode. Each frame is
    annotated with its θ value. Bottom row (full width): the agent's path in
    state space colored by θ_t, with the sampled-frame positions ringed.
    """
    import matplotlib.gridspec as gridspec

    T = len(rows)
    if T < n_frames:
        n_frames = T
    sample_idx = np.linspace(0, T - 1, n_frames).round().astype(int)

    frames = _raw_frames(dataset, rows[sample_idx])

    fig = plt.figure(figsize=(2.0 * n_frames, 5.5))
    gs = gridspec.GridSpec(2, n_frames, figure=fig,
                            height_ratios=[1.0, 1.4], hspace=0.35, wspace=0.08)

    th_min, th_max = float(theta.min()), float(theta.max())
    cmap = plt.cm.viridis

    # Top: frames
    for j, (frame, idx) in enumerate(zip(frames, sample_idx)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        # color the title to match the θ at that step
        c = cmap((theta[idx] - th_min) / (th_max - th_min + 1e-9))
        ax.set_title(f"t={idx}\nθ={theta[idx]:+.2f}", fontsize=9,
                     color=c, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(2.0)

    # Bottom: trajectory (or a θ-vs-time fallback if xy is unavailable)
    ax_traj = fig.add_subplot(gs[1, :])
    if xy is not None and len(xy) == T:
        sc = ax_traj.scatter(xy[:, 0], xy[:, 1], c=theta, cmap="viridis",
                             s=18, vmin=th_min, vmax=th_max,
                             edgecolor="white", linewidth=0.4, zorder=2)
        ax_traj.plot(xy[:, 0], xy[:, 1], color="grey", lw=0.5, alpha=0.4, zorder=1)
        # ring the sampled frame positions
        ax_traj.scatter(xy[sample_idx, 0], xy[sample_idx, 1],
                        s=120, facecolor="none", edgecolor="black",
                        linewidth=1.5, zorder=3)
        for j, idx in enumerate(sample_idx):
            ax_traj.annotate(str(idx), (xy[idx, 0], xy[idx, 1]),
                             textcoords="offset points", xytext=(6, 6),
                             fontsize=7, alpha=0.7)
        ax_traj.scatter(xy[0:1, 0], xy[0:1, 1], marker="s", s=80,
                        facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
        ax_traj.scatter(xy[-1:, 0], xy[-1:, 1], marker="*", s=140,
                        facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
        ax_traj.set_aspect("equal", adjustable="datalim")
        env_canon = _canonical_env(env)
        # Per-env axis label (the source of (x, y) the trajectory was plotted from)
        ax_labels = {
            "pusht":   ("agent_x (px)",        "agent_y (px)"),
            "reacher": ("end-effector x (m)",  "end-effector y (m)"),
            "tworoom": ("agent_x (px)",        "agent_y (px)"),
            "cube":    ("cube_x (m)",          "cube_y (m)"),
        }.get(env_canon, ("x", "y"))
        ax_traj.set_xlabel(ax_labels[0])
        ax_traj.set_ylabel(ax_labels[1])
        ax_traj.set_title(f"Agent trajectory in {env_canon}-state space, "
                          f"colored by θ_t  (rings: sampled frames)",
                          fontsize=10)
        cbar = fig.colorbar(sc, ax=ax_traj, shrink=0.8,
                            label=r"unwrapped $\theta_t$ (rad)")
    else:
        ax_traj.plot(theta, "o-", ms=3, lw=0.8)
        for idx in sample_idx:
            ax_traj.axvline(idx, color="black", alpha=0.2, lw=0.5)
        ax_traj.set_title(r"$\theta_t$ over time (state-space xy unavailable for this env)",
                          fontsize=10)
        ax_traj.set_xlabel("step t"); ax_traj.set_ylabel(r"$\theta$ (rad)")

    label_ep = f"episode {ep_id}" if ep_id is not None else "episode (n/a)"
    fig.suptitle(f"Frame strip + θ trajectory  —  {_canonical_env(env)}, "
                 f"{label_ep} (T={T})",
                 fontsize=11, y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_correlations(theta_by_ep: dict, proxies_by_ep: dict[int, dict],
                       env: str, out: Path):
    """Heatmap of per-episode Spearman ρ(θ_t, proxy_t).

    Rows: episodes; columns: proxies (always includes step_idx as baseline).
    Values: |ρ|, signed-ρ shown as text. Highlight rows where some proxy
    other than step_idx achieves higher |ρ| — the interesting case.
    """
    from scipy.stats import spearmanr

    # Build a stable column order: step_idx first, then the rest sorted
    all_proxy_names: list[str] = []
    for d in proxies_by_ep.values():
        for k in d.keys():
            if k not in all_proxy_names:
                all_proxy_names.append(k)
    if not all_proxy_names:
        return None
    cols = ["step_idx"] + sorted([c for c in all_proxy_names if c != "step_idx"])

    eps = [e for e in theta_by_ep if e in proxies_by_ep]
    if not eps:
        return None

    rho = np.full((len(eps), len(cols)), np.nan)
    for i, ep in enumerate(eps):
        th = theta_by_ep[ep]
        for j, name in enumerate(cols):
            ser = proxies_by_ep[ep].get(name)
            if ser is None or len(ser) != len(th):
                continue
            r, _ = spearmanr(th, ser)
            rho[i, j] = r

    # Heatmap. Use symmetric colormap centered at 0 (negative = anti-correlated).
    fig, ax = plt.subplots(figsize=(1.2 + 1.4 * len(cols), 0.45 * len(eps) + 1.6))
    im = ax.imshow(rho, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(eps))); ax.set_yticklabels([f"ep {e}" for e in eps])
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Spearman ρ(θ, proxy)")
    ax.set_title(f"Per-episode Spearman ρ between θ_t and candidate proxies  ({_canonical_env(env)})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return rho, cols, eps


@torch.no_grad()
def plot_state_grid(model, dataset, env: str, device: str, img_size: int,
                    out: Path, grid_n: int = 32):
    """Push-T only: sweep the agent over a regular (x, y) grid while the
    block is held at its mean dataset position, and read theta(x, y)."""
    if _canonical_env(env) != "pusht":
        print(f"  [grid] skipping; only implemented for pusht (got {env})")
        return
    if not hasattr(dataset, "_pusht_render"):
        # We can't synthesise frames at arbitrary states without the env;
        # this requires the simulator. Skip gracefully with a note.
        print("  [grid] skipping: needs the Push-T simulator to render synthetic"
              " states. Run via the env directly outside this script if needed.")
        return


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to *_object.ckpt")
    p.add_argument("--data", required=True,
                   help="Dataset name (pusht | tworoom | reacher | cube)")
    p.add_argument("--episodes", nargs="+", type=int, default=[0, 1000, 5000, 8000],
                   help="Episode IDs to analyse")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--grid", action="store_true",
                   help="Add the slow state-grid heatmap (pusht only)")
    p.add_argument("--frame-strip-episode", type=int, nargs="+", default=None,
                   help="If set (one or more episode IDs), build a frame-strip+θ-trajectory figure for each")
    p.add_argument("--frame-strip-n", type=int, default=8,
                   help="Number of frames in the strip (default 8)")
    p.add_argument("--correlations", action="store_true",
                   help="Compute Spearman ρ between θ_t and candidate task-progress proxies "
                        "(step_idx, agent–block dist, block–goal dist, etc.) and plot a heatmap")
    p.add_argument("--tsne-3d", action="store_true",
                   help="Also emit 3D t-SNE projections (tsne_zprog_3d.png, tsne_zcont_3d.png)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    print(f"  k_prog = {getattr(model, 'k_prog', 0)}")

    print(f"Loading dataset: {args.data}")
    import stable_worldmodel as swm
    from hdf5_dataset import HDF5Dataset
    cache_dir = args.cache_dir or swm.data.utils.get_cache_dir()
    dataset = HDF5Dataset(args.data, cache_dir=Path(cache_dir), keys_to_cache=[])

    img_transform = _build_img_transform(args.img_size)

    # 1. Encode each episode → split into z_prog, z_cont, theta
    zprog_by_ep, zcont_by_ep, theta_by_ep, xy_by_ep = {}, {}, {}, {}
    rows_by_ep: dict[int, np.ndarray] = {}
    proxies_by_ep: dict[int, dict[str, np.ndarray]] = {}
    for ep_id in args.episodes:
        rows = _episode_rows(dataset, ep_id)
        if len(rows) < 5:
            print(f"  episode {ep_id}: too short ({len(rows)} steps), skipping")
            continue
        print(f"  episode {ep_id}: encoding {len(rows)} frames ...")
        rows_by_ep[ep_id] = rows
        pixels = _collect_pixels(dataset, rows, img_transform)
        emb = encode(model, pixels, args.device).numpy()
        zp, zc = split_pq(model, torch.from_numpy(emb))
        if zp is not None:
            zp_np = zp.numpy()
            zprog_by_ep[ep_id] = zp_np
            theta_by_ep[ep_id] = theta_of(zp_np)
        zcont_by_ep[ep_id] = (zc.numpy() if zc is not None else emb)

        xy = _extract_xy(dataset, rows, args.data)
        if xy is not None:
            xy_by_ep[ep_id] = xy

        fs_eps = args.frame_strip_episode or []
        if args.correlations or ep_id in fs_eps:
            proxies_by_ep[ep_id] = _compute_proxies(args.data, dataset, rows)

    # 2. t-SNE on z_prog and z_cont
    if zprog_by_ep:
        plot_tsne(zprog_by_ep, "prog", out_dir / "tsne_zprog.png")
    plot_tsne(zcont_by_ep, "cont", out_dir / "tsne_zcont.png")
    if args.tsne_3d:
        if zprog_by_ep:
            plot_tsne(zprog_by_ep, "prog", out_dir / "tsne_zprog_3d.png", n_components=3)
        plot_tsne(zcont_by_ep, "cont", out_dir / "tsne_zcont_3d.png", n_components=3)

    # 3. State-trajectory heatmap coloured by theta
    if zprog_by_ep and xy_by_ep:
        common = {ep: (xy_by_ep[ep], theta_by_ep[ep])
                  for ep in xy_by_ep if ep in theta_by_ep}
        if common:
            xy_b = {ep: v[0] for ep, v in common.items()}
            th_b = {ep: v[1] for ep, v in common.items()}
            plot_state_trajectory(xy_b, th_b, args.data,
                                  out_dir / "state_traj_theta.png")
    else:
        print("  [state_traj] skipping: missing z_prog or extractable (x,y)")

    # 4. (optional) frame strip + θ-trajectory side-by-side panel(s)
    for fs_ep in (args.frame_strip_episode or []):
        if fs_ep in theta_by_ep and fs_ep in rows_by_ep:
            plot_frame_strip(
                dataset, rows_by_ep[fs_ep], theta_by_ep[fs_ep],
                xy_by_ep.get(fs_ep), args.data,
                out_dir / f"frame_strip_ep{fs_ep}.png",
                n_frames=args.frame_strip_n,
                ep_id=fs_ep,
            )
            print(f"  frame_strip_ep{fs_ep}.png   (frames + θ-colored trajectory)")

    # 5. (optional) θ vs task-progress proxies — Spearman heatmap
    rho_table = None
    if args.correlations and proxies_by_ep:
        rho_table = plot_correlations(
            theta_by_ep, proxies_by_ep, args.data,
            out_dir / "theta_correlations.png",
        )
        print(f"  theta_correlations.png  (Spearman ρ heatmap)")

    # 6. (optional) state-grid scan
    if args.grid:
        plot_state_grid(model, dataset, args.data, args.device, args.img_size,
                        out_dir / "state_grid_theta.png")

    # 5. Persist a small summary
    summary = {
        "env": args.data,
        "ckpt": str(args.ckpt),
        "k_prog": getattr(model, "k_prog", 0),
        "episodes": list(zprog_by_ep.keys() or zcont_by_ep.keys()),
        "n_frames_per_episode": {
            ep: int(z.shape[0]) for ep, z in (zprog_by_ep or zcont_by_ep).items()
        },
        "theta_range_per_episode_rad": {
            ep: [float(t.min()), float(t.max())] for ep, t in theta_by_ep.items()
        },
    }
    if rho_table is not None:
        rho, cols, eps = rho_table
        summary["theta_proxy_spearman"] = {
            "proxies": cols,
            "per_episode": {
                str(eps[i]): {cols[j]: (None if np.isnan(rho[i, j]) else float(rho[i, j]))
                              for j in range(len(cols))}
                for i in range(len(eps))
            },
            "mean_abs_rho_per_proxy": {
                cols[j]: float(np.nanmean(np.abs(rho[:, j]))) for j in range(len(cols))
            },
        }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone → {out_dir}/")
    print(f"  tsne_zprog.png         (cross-episode mixing of z^prog)")
    print(f"  tsne_zcont.png         (per-episode clusters of z^cont)")
    if zprog_by_ep and xy_by_ep:
        print(f"  state_traj_theta.png   (compass on the agent's path)")


if __name__ == "__main__":
    main()
