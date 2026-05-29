"""Diagnostic C — does theta encode time, scene state, or both?

For each episode in --episode-ids, compute the unwrapped progression angle
``theta_t = atan2(z^p[t][1], z^p[t][0])`` and report Pearson correlation
of theta with:

  - ``step_idx`` (raw timestep)
  - ``step_idx / episode_length`` (normalised progression — 0 at start, 1 at end)
  - each component of the dataset's ``state`` column (e.g. block xy,
    agent xy, block angle for Push-T)

If theta correlates strongly with step_idx and weakly with everything
else, z^p is acting as a 2-D clock. If theta correlates with one or more
state components in addition to (or instead of) step_idx, the
progression coordinate is genuinely scene-aware. The most informative
case is when theta correlates with a state quantity that *itself* does
NOT trivially correlate with step_idx — e.g. block-to-target distance,
which can regress when the agent has to reposition.

We also report theta vs. step correlation per-episode (not just pooled),
so a non-monotonic episode (low correlation) stands out clearly.

Usage::

    python analysis/diagnostic_c_theta_vs_state.py \\
        --ckpt $STABLEWM_HOME/checkpoints/A2_pusht_seed3072/A2_pusht_seed3072_epoch_10_object.ckpt \\
        --data pusht_expert_train \\
        --episode-ids 0 1000 5000 8000 12000 15000 18000 \\
        --state-col state
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="swm dataset name (e.g. 'pusht_expert_train').")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--episode-ids", type=int, nargs="+",
                        default=[0, 1000, 5000, 8000, 12000, 15000, 18000],
                        help="Episode IDs to probe (default: 7 diverse Push-T eps).")
    parser.add_argument("--state-col", type=str, default="state",
                        help="Dataset column containing scene state (default 'state'). "
                             "Set to 'proprio' for proprioceptive features.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()

    cache = (
        args.cache_dir
        or os.environ.get("STABLEWM_HOME")
        or str(Path.home() / ".stable_worldmodel")
    )
    print(f"cache_dir = {cache}")
    print(f"ckpt      = {args.ckpt}")
    print(f"state col = {args.state_col}")

    import stable_worldmodel as swm

    model = torch.load(args.ckpt, map_location=args.device, weights_only=False).eval()
    k_prog = int(getattr(model, "k_prog", 0))
    if k_prog < 2:
        print(f"⚠️  This ckpt has k_prog={k_prog} (<2). theta is undefined; aborting.")
        return

    img_tf = _build_img_transform(224)
    keys = ["action"]
    if args.state_col not in keys:
        keys.append(args.state_col)
    ds = swm.data.HDF5Dataset(args.data, keys_to_cache=keys, cache_dir=cache)

    if args.state_col not in ds.column_names:
        print(f"⚠️  Column '{args.state_col}' not in dataset. Available columns:")
        print("   ", list(ds.column_names))
        print("Pooled correlations will only include step_idx and norm progress.")
        have_state = False
    else:
        have_state = True

    pooled_theta = []
    pooled_step = []
    pooled_norm = []
    pooled_state = []
    per_ep_corr_step = {}

    for ep_id in args.episode_ids:
        rows = _episode_slice(ds, ep_id)
        if len(rows) < 4:
            print(f"  [skip] ep {ep_id}: only {len(rows)} steps")
            continue

        pixels = _collect_episode_pixels(ds, rows, img_tf)
        emb = encode_episode(model, pixels, args.device, chunk=args.chunk)
        zp, _ = split_prog_cont(model, emb)
        zp_np = zp.numpy()
        theta = np.unwrap(np.arctan2(zp_np[:, 1], zp_np[:, 0]))

        steps = np.arange(len(rows), dtype=np.float64)
        norm_progress = steps / max(len(rows) - 1, 1)

        per_ep_corr_step[ep_id] = safe_corr(theta, steps)

        pooled_theta.append(theta)
        pooled_step.append(steps)
        pooled_norm.append(norm_progress)

        if have_state:
            state_rows = []
            for r in rows:
                s = ds[int(r)][args.state_col]
                if hasattr(s, "numpy"):
                    s = s.numpy()
                state_rows.append(np.asarray(s).reshape(-1))
            state = np.stack(state_rows)
            pooled_state.append(state)

    pooled_theta = np.concatenate(pooled_theta)
    pooled_step = np.concatenate(pooled_step)
    pooled_norm = np.concatenate(pooled_norm)
    if have_state:
        pooled_state = np.concatenate(pooled_state, axis=0)

    # --- per-episode theta vs step correlation ---
    print("\n=== Per-episode corr(theta, step_idx) — low value flags non-monotone θ ===")
    for ep_id, c in per_ep_corr_step.items():
        flag = "← non-monotonic!" if abs(c) < 0.7 else ""
        print(f"  ep {ep_id:>5}  corr = {c:+.3f}  {flag}")

    # --- pooled correlations ---
    print("\n=== Pooled (across all episodes) ===")
    print(f"  corr(theta, step_idx)            = {safe_corr(pooled_theta, pooled_step):+.3f}")
    print(f"  corr(theta, step / ep_length)    = {safe_corr(pooled_theta, pooled_norm):+.3f}")

    if have_state:
        print(f"  state[*] is shape {pooled_state.shape}")
        for i in range(pooled_state.shape[1]):
            c = safe_corr(pooled_theta, pooled_state[:, i])
            # also check whether state[i] itself correlates strongly with step
            c_state_step = safe_corr(pooled_state[:, i], pooled_step)
            residual_marker = ""
            if abs(c) > 0.4 and abs(c_state_step) < 0.7:
                residual_marker = " ← informative beyond time"
            print(f"  corr(theta, state[{i:>2}])           = {c:+.3f}   "
                  f"(corr(state[{i}], step)={c_state_step:+.3f}){residual_marker}")

    print("\nInterpretation guide:")
    print("  - Strong corr(theta, step) AND no informative state corr → 2D clock")
    print("  - Strong corr(theta, step) AND informative state corr   → time + scene info")
    print("  - Per-episode corr(theta, step) <0.7 for some eps       → genuine regression")
    print("    (most consistent with the POMDP-style partial-observability story)")


if __name__ == "__main__":
    main()
