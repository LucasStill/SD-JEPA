"""Latent surprise / VoE on the progression coordinate.

For a trained JEPA-STEP checkpoint with k_prog >= 2, this script applies a
controlled perturbation to a held-out episode and measures the resulting
discontinuity in the progression angle theta_t. Three perturbation modes:

  * splice    — at step N, replace the rest of the sequence with frames from
                a different episode. Default mode.
  * teleport  — at step N, insert a single frame from a different episode,
                then continue with the original. Tests recovery dynamics.
  * reverse   — reverse a window of frames. Tests directionality without
                changing the set of scenes presented.

Outputs:
  * Console: surprise magnitude at the perturbation point + sigma z-score
    relative to the clean trajectory's |Δθ_t| baseline.
  * PNG: 2-panel figure (theta_t curves + |Δθ_t| with perturbation marked).
  * GIF (optional): per-frame side-by-side of pixel observation + theta
    trajectory with current-step marker. Requires imageio + imageio-ffmpeg
    for mp4 export, or just imageio for gif.

Usage::

    python analysis/surprise_voe.py \\
        --ckpt $STABLEWM_HOME/checkpoints/A2_pusht_seed3072/A2_pusht_seed3072_epoch_10_object.ckpt \\
        --data pusht_expert_train \\
        --mode splice \\
        --episode-a 0 --episode-b 1000 \\
        --perturb-step 50 \\
        --out-dir analysis_out/surprise/ \\
        --gif

    # Single-frame teleport (insert ep-1000's step-50 into ep-0):
    python analysis/surprise_voe.py ... --mode teleport --episode-a 0 --episode-b 1000 \\
        --perturb-step 50

    # Reverse a 10-frame window of ep-0 starting at step 40:
    python analysis/surprise_voe.py ... --mode reverse --episode-a 0 \\
        --perturb-step 40 --window 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_LEWM_DIR = _THIS_DIR.parent
if str(_LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(_LEWM_DIR))

from analysis.latent_geometry import (  # noqa: E402
    _build_img_transform,
    _collect_episode_pixels,
    _episode_slice,
    encode_episode,
    split_prog_cont,
)


# ---------------------------------------------------------------------------
#  Surprise metrics
# ---------------------------------------------------------------------------


def theta_of(zp: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(zp[:, 1], zp[:, 0]))


def angular_surprise(theta: np.ndarray) -> np.ndarray:
    """|Δθ_t| at each step. Length T-1."""
    return np.abs(np.diff(theta))


def latent_surprise(z: np.ndarray) -> np.ndarray:
    """||z_t - z_{t-1}|| at each step. Length T-1."""
    return np.linalg.norm(np.diff(z, axis=0), axis=1)


# ---------------------------------------------------------------------------
#  Perturbation modes
# ---------------------------------------------------------------------------


def mode_splice(pixels_a: torch.Tensor, pixels_b: torch.Tensor,
                step: int) -> torch.Tensor:
    """Episode A's first ``step`` frames + episode B's frames from ``step``."""
    L = min(len(pixels_a), len(pixels_b))
    step = min(step, L - 1)
    return torch.cat([pixels_a[:step], pixels_b[step:L]], dim=0)


def mode_teleport(pixels_a: torch.Tensor, pixels_b: torch.Tensor,
                  step: int) -> torch.Tensor:
    """Insert a single frame from B at ``step`` of A."""
    return torch.cat(
        [pixels_a[:step], pixels_b[step : step + 1], pixels_a[step + 1 :]],
        dim=0,
    )


def mode_reverse(pixels_a: torch.Tensor, _: torch.Tensor, step: int,
                 window: int = 10) -> torch.Tensor:
    """Reverse frames [step, step+window) within A."""
    end = min(step + window, len(pixels_a))
    return torch.cat(
        [pixels_a[:step], pixels_a[step:end].flip(0), pixels_a[end:]], dim=0
    )


def mode_teleport_continue(pixels_a: torch.Tensor, pixels_b: torch.Tensor,
                           step: int) -> torch.Tensor:
    """Teleport-and-continue: A[:step] + B[step:end].

    Pixel-content-wise identical to ``splice``, but rendered with extra
    GIF emphasis at the teleport boundary so the discontinuity is visually
    obvious. The intent is a clearer narrative for a viewer: ``agent
    walks the original episode, gets teleported, and now lives in a
    different episode's trajectory''.
    """
    L = min(len(pixels_a), len(pixels_b))
    step = min(step, L - 1)
    return torch.cat([pixels_a[:step], pixels_b[step:L]], dim=0)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def make_static_figure(theta_clean, theta_perturb, sup_clean, sup_perturb,
                       perturb_step, out_path, mode_label):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(theta_clean, label="clean", color="tab:blue", lw=1.5)
    axes[0].plot(theta_perturb, label=f"perturbed ({mode_label})",
                 color="tab:red", lw=1.5)
    axes[0].axvline(perturb_step, color="black", ls="--", lw=1,
                    label=f"perturbation @ {perturb_step}")
    axes[0].set_ylabel(r"$\theta_t$ (rad)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("Progression angle")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sup_clean, label="clean", color="tab:blue", lw=1.5)
    axes[1].plot(sup_perturb, label="perturbed", color="tab:red", lw=1.5)
    axes[1].axvline(perturb_step - 1, color="black", ls="--", lw=1)
    axes[1].set_ylabel(r"$|\Delta \theta_t|$  (angular surprise)")
    axes[1].set_xlabel("step $t$")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].set_title("Latent surprise (angular)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_gif(pixels_perturb, theta_perturb, perturb_step, out_path,
             fps: int = 10, emphasize_steps: int = 3, mode_label: str = ""):
    """Render a per-step animation with visual emphasis at the perturbation.

    - The pixel frame gets a red border + "TELEPORT" overlay text for
      ``emphasize_steps`` frames starting at ``perturb_step``.
    - The right-panel theta trajectory shows the perturbation marker line.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("imageio not installed; skipping GIF. "
                  "pip install imageio imageio-ffmpeg")
            return
    from matplotlib.patches import Rectangle

    frames_out = []
    T = len(pixels_perturb)
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # left: pixel frame (un-normalised for display)
        img = pixels_perturb[t]
        if isinstance(img, torch.Tensor):
            arr = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.numpy()
        else:
            arr = np.asarray(img)
        # min-max stretch for visibility
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        axes[0].imshow(arr)
        axes[0].axis("off")

        in_emphasis_window = perturb_step <= t < perturb_step + emphasize_steps
        if in_emphasis_window:
            # red border via overlaid rectangle (axes coords)
            h, w = arr.shape[:2]
            border_rect = Rectangle(
                (-0.5, -0.5), w, h, linewidth=8, edgecolor="red",
                facecolor="none", clip_on=False,
            )
            axes[0].add_patch(border_rect)
            # large "TELEPORT" overlay
            axes[0].text(
                0.5, 0.92, "TELEPORT",
                transform=axes[0].transAxes,
                fontsize=20, fontweight="bold", color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="red", alpha=0.85, edgecolor="darkred"),
            )
            title_suffix = " (perturbation)"
        elif t >= perturb_step:
            title_suffix = " (post-perturbation, new trajectory)"
        else:
            title_suffix = ""
        axes[0].set_title(f"frame {t}{title_suffix}", fontsize=11)

        # right: theta trajectory
        # before perturbation: blue (clean baseline path)
        # after perturbation: red (continuing on new trajectory)
        if t < perturb_step:
            axes[1].plot(theta_perturb[: t + 1], color="tab:blue", lw=1.4,
                         label="trajectory")
        else:
            axes[1].plot(theta_perturb[: perturb_step], color="tab:blue", lw=1.4)
            axes[1].plot(range(perturb_step - 1, t + 1),
                         theta_perturb[perturb_step - 1: t + 1],
                         color="tab:red", lw=1.4, label="post-teleport")
        axes[1].plot(t, theta_perturb[t], "o", color="black", ms=8)
        axes[1].axvline(perturb_step, color="black", ls="--", lw=1, alpha=0.5,
                        label=f"perturb @ {perturb_step}")
        axes[1].set_xlim(0, T - 1)
        pad = 0.5
        axes[1].set_ylim(theta_perturb.min() - pad, theta_perturb.max() + pad)
        axes[1].set_xlabel("step $t$")
        axes[1].set_ylabel(r"$\theta_t$  (mental localisation)")
        title_extra = f"  [{mode_label}]" if mode_label else ""
        axes[1].set_title(f"Progression angle{title_extra}", fontsize=11)
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best", fontsize=8)

        plt.tight_layout()
        fig.canvas.draw()
        try:
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        except AttributeError:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames_out.append(image)
        plt.close(fig)

    imageio.mimsave(out_path, frames_out, fps=fps, loop=0)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--mode",
                        choices=["splice", "teleport", "reverse",
                                 "teleport_continue"],
                        default="splice",
                        help="Perturbation mode. ``teleport_continue`` is "
                             "splice with extra GIF emphasis (red border + "
                             "TELEPORT overlay) at the perturbation step.")
    parser.add_argument("--episode-a", type=int, default=0,
                        help="Base episode (the 'clean' control + perturbed start).")
    parser.add_argument("--episode-b", type=int, default=1000,
                        help="Donor episode for splice/teleport (ignored in reverse).")
    parser.add_argument("--perturb-step", type=int, default=50)
    parser.add_argument("--window", type=int, default=10,
                        help="Window size for reverse mode (frames to reverse).")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_out/surprise"))
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()

    cache = (
        args.cache_dir
        or os.environ.get("STABLEWM_HOME")
        or str(Path.home() / ".stable_worldmodel")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache_dir = {cache}")
    print(f"ckpt      = {args.ckpt}")
    print(f"mode      = {args.mode}")

    import stable_worldmodel as swm

    model = torch.load(args.ckpt, map_location=args.device, weights_only=False).eval()
    k_prog = int(getattr(model, "k_prog", 0))
    if k_prog < 2:
        print(f"⚠️ ckpt has k_prog={k_prog} (<2). theta is undefined; aborting.")
        return

    img_tf = _build_img_transform(224)
    ds = swm.data.HDF5Dataset(args.data, keys_to_cache=["action"], cache_dir=cache)

    # ------------ encode clean episode A ------------
    rows_a = _episode_slice(ds, args.episode_a)
    pixels_a = _collect_episode_pixels(ds, rows_a, img_tf)
    emb_a = encode_episode(model, pixels_a, args.device, chunk=args.chunk)
    zp_a, _ = split_prog_cont(model, emb_a)
    theta_a = theta_of(zp_a.numpy())

    # ------------ build perturbed sequence ------------
    perturb_step = args.perturb_step

    if args.mode == "splice":
        rows_b = _episode_slice(ds, args.episode_b)
        pixels_b = _collect_episode_pixels(ds, rows_b, img_tf)
        pixels_perturb = mode_splice(pixels_a, pixels_b, perturb_step)
    elif args.mode == "teleport":
        rows_b = _episode_slice(ds, args.episode_b)
        pixels_b = _collect_episode_pixels(ds, rows_b, img_tf)
        if perturb_step >= min(len(pixels_a), len(pixels_b)):
            print(f"perturb_step {perturb_step} too late; clamping.")
            perturb_step = min(len(pixels_a), len(pixels_b)) - 2
        pixels_perturb = mode_teleport(pixels_a, pixels_b, perturb_step)
    elif args.mode == "teleport_continue":
        rows_b = _episode_slice(ds, args.episode_b)
        pixels_b = _collect_episode_pixels(ds, rows_b, img_tf)
        pixels_perturb = mode_teleport_continue(pixels_a, pixels_b,
                                                perturb_step)
    elif args.mode == "reverse":
        pixels_perturb = mode_reverse(pixels_a, pixels_a, perturb_step,
                                      window=args.window)
    else:
        raise ValueError(args.mode)

    # ------------ encode perturbed sequence ------------
    emb_perturb = encode_episode(model, pixels_perturb, args.device,
                                 chunk=args.chunk)
    zp_perturb, _ = split_prog_cont(model, emb_perturb)
    theta_perturb = theta_of(zp_perturb.numpy())

    # ------------ surprise metrics ------------
    sup_clean = angular_surprise(theta_a)
    sup_perturb = angular_surprise(theta_perturb)

    L = min(len(theta_a), len(theta_perturb))
    theta_a = theta_a[:L]
    theta_perturb = theta_perturb[:L]
    sup_clean = sup_clean[: L - 1]
    sup_perturb = sup_perturb[: L - 1]

    # ------------ report ------------
    idx = perturb_step - 1
    if 0 <= idx < len(sup_perturb):
        spike = sup_perturb[idx]
        baseline_mean = sup_clean.mean()
        baseline_std = sup_clean.std() + 1e-8
        z = (spike - baseline_mean) / baseline_std
        print(f"\n=== Surprise report (mode={args.mode}, "
              f"perturb @ step {perturb_step}) ===")
        print(f"  clean    |Δθ| at step {idx}      = {sup_clean[idx]:.3f}")
        print(f"  perturbed |Δθ| at step {idx}     = {spike:.3f}")
        print(f"  ratio                            = {spike / max(sup_clean[idx], 1e-6):.1f}x")
        print(f"  baseline mean |Δθ| (clean)       = {baseline_mean:.3f}")
        print(f"  baseline std  |Δθ| (clean)       = {baseline_std:.3f}")
        print(f"  spike z-score                    = {z:.1f}σ")

    # ------------ static figure ------------
    tag = f"{args.mode}_ep{args.episode_a}"
    if args.mode in ("splice", "teleport"):
        tag += f"_to_ep{args.episode_b}"
    if args.mode == "reverse":
        tag += f"_w{args.window}"
    tag += f"_at{perturb_step}"
    static_path = args.out_dir / f"{tag}.png"
    make_static_figure(theta_a, theta_perturb, sup_clean, sup_perturb,
                       perturb_step, static_path, args.mode)
    print(f"\nStatic figure: {static_path}")

    # ------------ optional GIF ------------
    if args.gif:
        gif_path = args.out_dir / f"{tag}.gif"
        print(f"Rendering GIF (this can be slow)...")
        # Only the new ``teleport_continue`` mode gets the dramatic red-border
        # + TELEPORT overlay treatment; the other modes use a single-frame
        # emphasis (still visible, less obtrusive).
        emphasize = 3 if args.mode == "teleport_continue" else 1
        make_gif(pixels_perturb, theta_perturb, perturb_step, gif_path,
                 emphasize_steps=emphasize, mode_label=args.mode)
        print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
