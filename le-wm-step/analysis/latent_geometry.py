"""Post-hoc latent-geometry analysis for JEPA-STEP checkpoints.

Given a trained ``*_object.ckpt`` (pickled ``JEPA`` module) and an HDF5 rollout
dataset, this script walks held-out episodes frame-by-frame through the
encoder + projector, projects each frame onto the canonical split ``(P, Q)``
and writes a small bundle of diagnostic plots to disk:

1.  ``prog_xy.png``          — first two progression coords per episode
                                (the plane on which θ is measured).
2.  ``theta_over_time.png``  — unwrapped θ_t trajectories.
3.  ``r_over_time.png``      — radial norm r_t of the progression subspace.
4.  ``content_straight.png`` — per-frame ``1 - cos(Δz_{t+1}, Δz_t)`` on the
                                content subspace — the Hénaff-style
                                straightness diagnostic.
5.  ``summary.json``         — scalar stats: mean straightness, θ coverage,
                                r spread, prototype alignments.

The script is intentionally CPU-friendly (``--device cpu`` by default) so it
can run on a laptop against a checkpoint copied off the cluster. It works
equally well on an A0 baseline checkpoint (``k_prog == 0``) — in that case
we skip the progression-subspace plots and only emit the content straightness
diagnostic + a PCA fallback on the full latent.

Usage
-----
::

    python analysis/latent_geometry.py \
        --ckpt  ~/.cache/stable_worldmodel/<run_id>/lewm_step_epoch_100_object.ckpt \
        --data  pusht \
        --episodes 0 1 2 3 \
        --out   analysis_out/

The dataset name is whatever ``swm.data.HDF5Dataset`` accepts; on a machine
that trained on Push-T this is ``pusht``. If you passed ``cache_dir`` at
training time, mirror it with ``--cache-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Import guard — mirror the project's working directory for pickled modules
# ---------------------------------------------------------------------------
#
# ``torch.save(model, ...)`` pickles references to the *module path* of each
# class. The trained JEPA was built from ``le-wm-step/jepa.py``, so when we
# unpickle we need ``le-wm-step/`` on ``sys.path``. We add it here so this
# script works whether it's launched from the repo root or from any cwd.

_THIS_DIR = Path(__file__).resolve().parent
_LEWM_DIR = _THIS_DIR.parent
if str(_LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(_LEWM_DIR))


# ---------------------------------------------------------------------------
#  Data helpers
# ---------------------------------------------------------------------------


def _build_img_transform(img_size: int):
    """Replicates the eval-time image preprocessing so latents are comparable
    to those the world model saw at training time."""

    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    # Dataset returns (1,3,H,W) uint8 tensors — skip ToImage, handle in collect.
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def _episode_slice(dataset, episode_id: int) -> np.ndarray:
    """Return sorted row indices belonging to ``episode_id``."""
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = dataset.get_col_data(col)
    step = dataset.get_col_data("step_idx")
    rows = np.nonzero(ep == episode_id)[0]
    # Stable sort by step so we walk the episode in temporal order.
    order = np.argsort(step[rows], kind="stable")
    return rows[order]


def _collect_episode_pixels(dataset, rows: np.ndarray, img_transform) -> torch.Tensor:
    """Load one episode's pixel tensor ``(T, C, H, W)``."""
    pixels = []
    for r in rows:
        sample = dataset[int(r)]
        frame = sample["pixels"]
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        # Dataset returns (1, C, H, W) — squeeze leading batch dim.
        frame = frame.squeeze(0)  # → (C, H, W)
        frame = img_transform(frame)
        pixels.append(frame)
    return torch.stack(pixels, dim=0)


# ---------------------------------------------------------------------------
#  Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_episode(model, pixels: torch.Tensor, device: str, chunk: int = 32):
    """Forward an episode through ``encoder + projector`` and return the full
    latent tensor ``(T, D)``. Batches frames in chunks of ``chunk`` to stay
    well within typical CPU memory."""

    model.eval()
    model.to(device)
    pixels = pixels.to(device)

    embs = []
    for start in range(0, pixels.size(0), chunk):
        chunk_pixels = pixels[start : start + chunk]
        output = model.encoder(chunk_pixels, interpolate_pos_encoding=True)
        cls = output.last_hidden_state[:, 0]
        embs.append(model.projector(cls).cpu())
    return torch.cat(embs, dim=0)  # (T, D)


def split_prog_cont(model, emb: torch.Tensor):
    """Project ``(T, D)`` onto ``(emb_prog, emb_cont)`` using the model's
    canonical injections. Returns ``(None, emb)`` when ``k_prog == 0``."""

    if not hasattr(model, "k_prog") or model.k_prog == 0:
        return None, emb

    P = model.P.detach().cpu()
    Q = model.Q.detach().cpu()
    return emb @ P, emb @ Q


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def _theta_unwrap(z_prog: np.ndarray) -> np.ndarray:
    theta = np.arctan2(z_prog[:, 1], z_prog[:, 0])
    return np.unwrap(theta)


def _radius(z_prog: np.ndarray) -> np.ndarray:
    return np.linalg.norm(z_prog, axis=-1)


def _straightness(z: np.ndarray) -> np.ndarray:
    """Per-frame ``1 - cos(Δz_{t+1}, Δz_t)``. Returns length ``T - 2``."""
    d = np.diff(z, axis=0)
    d1 = d[:-1]
    d2 = d[1:]
    num = (d1 * d2).sum(-1)
    den = np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1) + 1e-8
    return 1.0 - num / den


def _plot_pca_xy(z_by_ep, title: str, out: Path):
    """PCA-project the full latent z (T, D) to 2D and plot episode trajectories."""
    from sklearn.decomposition import PCA

    all_z = np.concatenate(list(z_by_ep.values()), axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)

    fig, ax = plt.subplots(figsize=(5, 5))
    for ep_id, z in z_by_ep.items():
        z2 = pca.transform(z)
        ax.plot(z2[:, 0], z2[:, 1], marker="o", ms=2, lw=0.8, label=f"ep {ep_id}")
        ax.scatter([z2[0, 0]], [z2[0, 1]], color="k", marker="s", s=24, zorder=3)
        ax.scatter([z2[-1, 0]], [z2[-1, 1]], color="k", marker="*", s=48, zorder=3)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_aspect("equal")
    var = pca.explained_variance_ratio_
    ax.set_title(f"{title}\n(PC1 {var[0]:.1%}, PC2 {var[1]:.1%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return pca.explained_variance_ratio_[:2].tolist()


def _plot_prog_xy(prog_by_ep, prototypes, out: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for ep_id, z in prog_by_ep.items():
        ax.plot(z[:, 0], z[:, 1], marker="o", ms=2, lw=0.8, label=f"ep {ep_id}")
        ax.scatter([z[0, 0]], [z[0, 1]], color="k", marker="s", s=24, zorder=3)
        ax.scatter([z[-1, 0]], [z[-1, 1]], color="k", marker="*", s=48, zorder=3)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title("Progression subspace (first two coords)")
    ax.set_xlabel("z_prog[0]")
    ax.set_ylabel("z_prog[1]")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fit_and_project(all_z: np.ndarray, method: str, perplexity: float = 30.0):
    """Fit a 2D projection on concatenated latents.

    For PCA we keep .transform(). For t-SNE we fit_transform once on the
    full set and return the joint embedding (caller splits per-episode).
    On high-dim z (z_cont, z_full) t-SNE is preceded by PCA(50) for speed
    and stability — sklearn's recommended default.
    """
    from sklearn.decomposition import PCA

    if method == "pca":
        pca = PCA(n_components=2).fit(all_z)
        return ("pca_transformer", pca, pca.explained_variance_ratio_[:2].tolist())

    if method == "tsne":
        from sklearn.manifold import TSNE

        z = all_z
        if z.shape[1] > 50:
            z = PCA(n_components=50).fit_transform(z)
        # perplexity must be < n_samples
        perp = min(perplexity, max(5, (len(z) - 1) // 3))
        emb = TSNE(
            n_components=2, perplexity=perp, init="pca",
            learning_rate="auto", random_state=0,
        ).fit_transform(z)
        return ("tsne_joint", emb, None)

    raise ValueError(f"unknown method {method!r}")


def _plot_trajectory_grid(
    by_ep_dict: dict[str, dict[int, np.ndarray]],
    out_path: Path,
    method: str = "pca",
):
    """3-panel side-by-side plot of trajectories in (z_prog, z_cont, z_full).

    Points colored by step index within episode (viridis); lines connect
    consecutive steps; start = square, end = star. Different episodes
    distinguished by line style only (color is reserved for time).
    """
    panels = [k for k in ("z_prog", "z_cont", "z_full") if k in by_ep_dict]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    pca_var_per_panel: dict[str, list[float]] = {}

    for ax, name in zip(axes, panels):
        by_ep = by_ep_dict[name]
        # Build episode->slice index so we can split a joint embedding back out
        keys = list(by_ep.keys())
        lengths = [by_ep[k].shape[0] for k in keys]
        all_z = np.concatenate([by_ep[k] for k in keys], axis=0)

        kind, transformer_or_emb, var = _fit_and_project(all_z, method)
        if kind == "pca_transformer":
            pca_var_per_panel[name] = var
            split2d = [transformer_or_emb.transform(by_ep[k]) for k in keys]
        else:  # tsne_joint — already transformed, just split
            offsets = np.cumsum([0] + lengths)
            split2d = [
                transformer_or_emb[offsets[i] : offsets[i + 1]]
                for i in range(len(keys))
            ]

        # Plot each episode trajectory
        for k, z2 in zip(keys, split2d):
            T = len(z2)
            colors = np.linspace(0, 1, T)
            ax.plot(z2[:, 0], z2[:, 1], color="lightgrey", lw=0.6, zorder=1)
            sc = ax.scatter(
                z2[:, 0], z2[:, 1],
                c=colors, cmap="viridis", s=14, zorder=2,
            )
            ax.scatter(z2[0, 0], z2[0, 1], color="k", marker="s", s=30, zorder=3)
            ax.scatter(z2[-1, 0], z2[-1, 1], color="k", marker="*", s=60, zorder=3)

        title = name
        if name in pca_var_per_panel:
            v = pca_var_per_panel[name]
            title += f"  (PC1 {v[0]:.0%}, PC2 {v[1]:.0%})"
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()}-1")
        ax.set_ylabel(f"{method.upper()}-2")
        ax.set_aspect("equal", "datalim")

    fig.suptitle(f"Latent trajectories — {method.upper()}  (color = step, ▪ start, ★ end)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pca_var_per_panel if method == "pca" else None


def _plot_time_series(series_by_ep, title: str, ylabel: str, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for ep_id, y in series_by_ep.items():
        ax.plot(y, lw=1.0, label=f"ep {ep_id}")
    ax.set_title(title)
    ax.set_xlabel("step t")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def _load_model(ckpt: Path, device: str):
    """``torch.load`` the pickled JEPA module. ``weights_only=False`` is
    required because the ckpt holds the full module graph, not just weights."""
    return torch.load(ckpt, map_location=device, weights_only=False)


def _resolve_dataset(name: str, cache_dir: str | None, keys_to_cache: Iterable[str]):
    import stable_worldmodel as swm
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from hdf5_dataset import HDF5Dataset

    cache = Path(cache_dir) if cache_dir else Path(swm.data.utils.get_cache_dir())
    return HDF5Dataset(
        name, keys_to_cache=list(keys_to_cache), cache_dir=cache
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to <...>_object.ckpt from training.")
    parser.add_argument("--data", type=str, required=True,
                        help="swm HDF5Dataset name (e.g. 'pusht').")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override swm cache dir (matches training).")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Episode ids to analyse (held-out recommended).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk", type=int, default=32,
                        help="Forward-batch size for encoding.")
    parser.add_argument("--out", type=Path, default=Path("analysis_out"),
                        help="Output directory for plots + summary.json.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.ckpt, device=args.device)
    img_transform = _build_img_transform(args.img_size)

    dataset = _resolve_dataset(
        args.data,
        cache_dir=args.cache_dir,
        keys_to_cache=["action"],  # pixels loaded lazily per-frame to avoid OOM
    )

    prog_by_ep: dict[int, np.ndarray] = {}
    prog_full_by_ep: dict[int, np.ndarray] = {}  # full k-d prog (not just first 2)
    cont_by_ep: dict[int, np.ndarray] = {}
    theta_by_ep: dict[int, np.ndarray] = {}
    r_by_ep: dict[int, np.ndarray] = {}
    straight_cont_by_ep: dict[int, np.ndarray] = {}
    fullz_by_ep: dict[int, np.ndarray] = {}
    straight_fullz_by_ep: dict[int, np.ndarray] = {}

    for ep_id in args.episodes:
        rows = _episode_slice(dataset, ep_id)
        if len(rows) < 3:
            print(f"[skip] episode {ep_id}: only {len(rows)} frames")
            continue

        pixels = _collect_episode_pixels(dataset, rows, img_transform)
        emb = encode_episode(model, pixels, args.device, chunk=args.chunk)
        emb_prog, emb_cont = split_prog_cont(model, emb)

        z_full = emb.numpy()
        fullz_by_ep[ep_id] = z_full
        straight_fullz_by_ep[ep_id] = _straightness(z_full)

        if emb_prog is not None:
            zp = emb_prog.numpy()
            prog_by_ep[ep_id] = zp[:, :2]
            prog_full_by_ep[ep_id] = zp
            theta_by_ep[ep_id] = _theta_unwrap(zp)
            r_by_ep[ep_id] = _radius(zp)

        z_cont = emb_cont.numpy()
        cont_by_ep[ep_id] = z_cont
        straight_cont_by_ep[ep_id] = _straightness(z_cont)

    # ---------------- plots ----------------
    if prog_by_ep:
        _plot_prog_xy(prog_by_ep, None, args.out / "prog_xy.png")
        _plot_time_series(
            theta_by_ep,
            title="Progression angle θ_t (unwrapped)",
            ylabel="θ (rad)",
            out=args.out / "theta_over_time.png",
        )
        _plot_time_series(
            r_by_ep,
            title="Progression radius r_t",
            ylabel="‖z_prog‖",
            out=args.out / "r_over_time.png",
        )

    _plot_time_series(
        straight_cont_by_ep,
        title="Content-subspace local curvature  1 − cos(Δz_{t+1}, Δz_t)",
        ylabel="1 − cos",
        out=args.out / "content_straight.png",
    )

    # full-z diagnostics
    pca_var = None
    if fullz_by_ep:
        pca_var = _plot_pca_xy(
            fullz_by_ep,
            title="Full latent z (PCA)",
            out=args.out / "fullz_pca.png",
        )
        _plot_time_series(
            straight_fullz_by_ep,
            title="Full-z local curvature  1 − cos(Δz_{t+1}, Δz_t)",
            ylabel="1 − cos",
            out=args.out / "fullz_straight.png",
        )

    # 3-panel trajectory grid (z_prog / z_cont / z_full) in PCA and t-SNE
    by_ep_dict: dict[str, dict[int, np.ndarray]] = {}
    if prog_full_by_ep:
        by_ep_dict["z_prog"] = prog_full_by_ep
    if cont_by_ep:
        by_ep_dict["z_cont"] = cont_by_ep
    if fullz_by_ep:
        by_ep_dict["z_full"] = fullz_by_ep

    grid_pca_var = None
    if by_ep_dict:
        grid_pca_var = _plot_trajectory_grid(
            by_ep_dict, args.out / "trajectory_pca.png", method="pca"
        )
        _plot_trajectory_grid(
            by_ep_dict, args.out / "trajectory_tsne.png", method="tsne"
        )

    # ---------------- summary ----------------
    summary = {
        "ckpt": str(args.ckpt),
        "episodes": list(args.episodes),
        "k_prog": int(getattr(model, "k_prog", 0)),
        "use_polar": bool(getattr(model, "use_polar", False)),
        "pca_var_pc1_pc2": pca_var,
        "per_episode": {},
    }
    for ep_id in args.episodes:
        if ep_id not in straight_cont_by_ep:
            continue
        rec = {
            "mean_content_curvature": float(straight_cont_by_ep[ep_id].mean()),
            "mean_fullz_curvature": float(straight_fullz_by_ep[ep_id].mean()),
        }
        if ep_id in theta_by_ep:
            th = theta_by_ep[ep_id]
            rec["theta_span_rad"] = float(th.max() - th.min())
            rec["r_mean"] = float(r_by_ep[ep_id].mean())
            rec["r_std"] = float(r_by_ep[ep_id].std())
        summary["per_episode"][str(ep_id)] = rec

    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote analysis bundle to {args.out.resolve()}")


if __name__ == "__main__":
    main()
