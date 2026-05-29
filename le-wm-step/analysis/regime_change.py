"""Experiment 2 — regime-change segmentation.

Hypothesis: at semantic phase events, θ exhibits a *regime change* (different
mode of evolution before vs after) while z-MSE just spikes and returns to
baseline. So change-point detection on θ-velocity should produce changepoints
**aligned with ground-truth phase events**, whereas change-point detection on
z-MSE should either fail to detect regime shifts or produce noisier
(false-positive-heavy) results.

We reuse ``surprise_compare``'s teacher-forced predictor pipeline to compute,
per cube held-out episode:
  * ``zmse_t``                — full-latent MSE between predicted and observed z.
  * ``dtheta_t``              — wrapped angular distance between θ(pred) and θ(obs).
  * ``theta_t``               — θ from observed z (its evolution profile).

Change-point detection (`ruptures.Pelt` with ℓ²/RBF cost) is run on:
  (a) |dθ/dt| (the rate of phase rotation in the *observed* trajectory),
  (b) z-MSE (the per-step prediction surprise),
  (c) |Δθ| (the per-step phase surprise).

We score detected change-points against ground-truth transitions in
``proprio_gripper_contact`` with a tolerance window: TP = detected within
±tol of a ground-truth event, FP = detected far from any event, FN =
ground-truth event with no nearby detection. Precision/recall/F1 reported
per metric, pooled across episodes, and per tolerance.

Usage:
    python analysis/regime_change.py \\
        --ckpt $STABLEWM_HOME/checkpoints/A2_JZ_kprog2/jepa_step_A2_kprog8_cube_seed42_535787_epoch_10_object.ckpt \\
        --data cube_single_expert --cache-dir $STABLEWM_HOME/datasets \\
        --episodes <list> \\
        --phase-event-col proprio_gripper_contact \\
        --tolerance 1 2 3 \\
        --out analysis_out/regime_change_cube/
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

# Reuse helpers from surprise_compare so we encode and predict identically.
from surprise_compare import (  # type: ignore  # noqa: E402
    _build_img_transform, _episode_rows, _collect_strided, _canonical_env,
    encode, encode_actions, compute_surprise, project_theta,
    detect_phase_events, gather_phase_signal,
)


def cpd_detect(series: np.ndarray, n_bkps: int, model: str = "l2"):
    """Detect ``n_bkps`` change-points in ``series`` using PELT (or BinSeg).

    Returns a list of indices (excluding the trivial endpoint).
    """
    import ruptures as rpt
    if len(series) < (n_bkps + 1) * 3:
        return []
    # BinSeg with fixed n_bkps is the most robust when we know K.
    algo = rpt.Binseg(model=model).fit(series.reshape(-1, 1))
    bkps = algo.predict(n_bkps=n_bkps)
    # ruptures returns endpoints including T; drop the trailing one.
    return [int(b) for b in bkps[:-1]]


def f1_against_truth(detected: list[int], truth: np.ndarray, tolerance: int):
    """Match each detected change-point to a ground-truth event greedily.

    Each truth event can match at most one detection, and vice versa.
    Returns (precision, recall, f1, tp, fp, fn).
    """
    matched_truth = set()
    tp = 0
    for d in detected:
        # find closest unmatched truth within tolerance
        best, best_dist = None, 10**9
        for i, t in enumerate(truth):
            if i in matched_truth:
                continue
            dist = abs(int(t) - int(d))
            if dist <= tolerance and dist < best_dist:
                best, best_dist = i, dist
        if best is not None:
            matched_truth.add(best)
            tp += 1
    fp = len(detected) - tp
    fn = len(truth) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def plot_per_episode_cpd(theta: np.ndarray, dtheta_dt: np.ndarray,
                          zmse: np.ndarray, dtheta_pred: np.ndarray,
                          first_step: int,
                          truth: np.ndarray, detected: dict[str, list[int]],
                          env: str, ep_id: int, out: Path):
    """Multi-panel time-series with detected change-points and ground truth."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    t = np.arange(len(theta))

    axes[0].plot(t, theta, color="tab:purple", lw=1.0)
    axes[0].set_ylabel(r"$\theta_t$ (rad)")
    axes[0].set_title(f"{_canonical_env(env)} ep {ep_id} — change-point detection")

    t1 = np.arange(1, len(theta))
    axes[1].plot(t1, dtheta_dt, color="tab:red", lw=1.2)
    axes[1].set_ylabel(r"$|\mathrm{d}\theta/\mathrm{d}t|$")

    t_score = np.arange(first_step, first_step + len(zmse))
    axes[2].plot(t_score, zmse, color="tab:blue", lw=1.2)
    axes[2].set_ylabel("z-MSE")

    axes[3].plot(t_score, dtheta_pred, color="tab:orange", lw=1.2)
    axes[3].set_ylabel(r"$|\Delta\theta|$ (pred err)")
    axes[3].set_xlabel("step t (observed)")

    # mark truth + detections on every panel
    colors = {"dtheta_dt": "tab:red", "zmse": "tab:blue", "dtheta_pred": "tab:orange"}
    for ax in axes:
        for tr in truth:
            ax.axvline(tr, color="black", lw=0.7, ls="--", alpha=0.55)
    for name, idx_panel in [("dtheta_dt", 1), ("zmse", 2), ("dtheta_pred", 3)]:
        for d in detected.get(name, []):
            axes[idx_panel].axvline(d, color=colors[name], lw=1.5, alpha=0.55)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--phase-event-col", required=True,
                   help="Ground-truth phase column (e.g. proprio_gripper_contact)")
    p.add_argument("--phase-event-kind", default="binary",
                   choices=["binary", "threshold"])
    p.add_argument("--phase-event-threshold", type=float, default=0.5)
    p.add_argument("--tolerance", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--cpd-model", default="l2", choices=["l2", "rbf"])
    p.add_argument("--max-plots", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    model.eval()

    print(f"Loading dataset: {args.data}")
    import stable_worldmodel as swm
    from hdf5_dataset import HDF5Dataset
    cache_dir = args.cache_dir or swm.data.utils.get_cache_dir()
    dataset = HDF5Dataset(args.data, cache_dir=Path(cache_dir), keys_to_cache=[])
    img_transform = _build_img_transform(args.img_size)

    summary: dict = {
        "ckpt": str(args.ckpt),
        "env": args.data,
        "phase_event_col": args.phase_event_col,
        "history_size": args.history_size,
        "frameskip": args.frameskip,
        "tolerances": args.tolerance,
        "cpd_model": args.cpd_model,
        "per_episode": {},
        "pooled": {},
    }

    H = args.history_size
    pooled = {
        m: {tol: dict(tp=0, fp=0, fn=0, n_truth=0)
            for tol in args.tolerance}
        for m in ("dtheta_dt", "zmse", "dtheta_pred")
    }

    n_plots = 0
    for ep_id in args.episodes:
        rows = _episode_rows(dataset, ep_id)
        T_raw = len(rows)
        if T_raw < args.frameskip * (H + 2):
            continue

        obs_rows, pixels, actions, _ = _collect_strided(
            dataset, rows, img_transform, frameskip=args.frameskip)
        T = len(obs_rows)

        emb = encode(model, pixels, args.device)
        a_emb = encode_actions(model, actions, args.device)
        s = compute_surprise(model, emb, a_emb, history_size=H)

        # Observed θ trajectory and its first-difference
        theta_obs = project_theta(model, emb)            # (T,)
        # wrap to [-π, π] for diff; use unwrap for natural gradient
        theta_un = np.unwrap(theta_obs)
        dtheta_dt = np.abs(np.diff(theta_un))             # (T-1,)

        # Ground-truth events (in observed-step indices)
        sig = gather_phase_signal(dataset, obs_rows, args.phase_event_col)
        truth = detect_phase_events(
            sig, kind=args.phase_event_kind, threshold=args.phase_event_threshold
        )
        if len(truth) == 0:
            continue
        n_bkps = len(truth)

        # Run CPD on each metric, asking for n_bkps change-points each
        det_dtheta_dt   = cpd_detect(dtheta_dt,    n_bkps=n_bkps, model=args.cpd_model)
        det_zmse        = cpd_detect(s["zmse"],    n_bkps=n_bkps, model=args.cpd_model)
        det_dtheta_pred = cpd_detect(s["dtheta"],  n_bkps=n_bkps, model=args.cpd_model)
        # Align all detection indices into the *observed-step* coordinate frame
        # dtheta_dt index i corresponds to obs-step i+1
        det_dtheta_dt   = [d + 1 for d in det_dtheta_dt]
        # zmse / dtheta_pred indices start at first_step
        det_zmse        = [d + s["first_step"] for d in det_zmse]
        det_dtheta_pred = [d + s["first_step"] for d in det_dtheta_pred]

        ep_record = {"n_truth": int(len(truth)), "per_metric": {}}
        for name, det in [("dtheta_dt",   det_dtheta_dt),
                          ("zmse",        det_zmse),
                          ("dtheta_pred", det_dtheta_pred)]:
            metric_rec = {"n_detected": int(len(det)), "per_tol": {}}
            for tol in args.tolerance:
                prec, rec, f1, tp, fp, fn = f1_against_truth(det, truth, tolerance=tol)
                metric_rec["per_tol"][str(tol)] = {
                    "precision": prec, "recall": rec, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn,
                }
                pooled[name][tol]["tp"] += tp
                pooled[name][tol]["fp"] += fp
                pooled[name][tol]["fn"] += fn
                pooled[name][tol]["n_truth"] += len(truth)
            ep_record["per_metric"][name] = metric_rec
        summary["per_episode"][str(ep_id)] = ep_record

        if n_plots < args.max_plots:
            plot_per_episode_cpd(
                theta_obs, dtheta_dt, s["zmse"], s["dtheta"],
                first_step=s["first_step"],
                truth=truth,
                detected={
                    "dtheta_dt": det_dtheta_dt,
                    "zmse": det_zmse,
                    "dtheta_pred": det_dtheta_pred,
                },
                env=args.data, ep_id=ep_id,
                out=out_dir / f"cpd_overlay_ep{ep_id}.png",
            )
            n_plots += 1

    # Pooled F1 per metric per tolerance
    print("\nPooled CPD scores:")
    print(f"  {'metric':<15s}  {'tol':<5s}  {'prec':<6s}  {'rec':<6s}  {'F1':<6s}  TP/FP/FN")
    for name in ("dtheta_dt", "zmse", "dtheta_pred"):
        for tol in args.tolerance:
            d = pooled[name][tol]
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            summary["pooled"].setdefault(name, {})[str(tol)] = {
                "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
                "n_truth": d["n_truth"],
            }
            print(f"  {name:<15s}  ±{tol:<3d}  {prec:.3f}  {rec:.3f}  {f1:.3f}  {tp}/{fp}/{fn}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone → {out_dir}/")


if __name__ == "__main__":
    main()
