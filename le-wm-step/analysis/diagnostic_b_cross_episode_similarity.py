"""Diagnostic B — cross-episode cosine similarity at fixed step indices.

If the subspace split is doing its job, then for any fixed step index ``t``:
- ``z^p[t]`` should be HIGH cos similarity across episodes (shared progression)
- ``z^c[t]`` should be LOW cos similarity (episode-specific content)
- full ``z[t]``  should be LOW-to-MEDIUM (mostly content, some progression)

If z^p AND full z are both ~1.0, the encoder has collapsed (mode collapse).
If z^p AND full z are both ~0.0, the prog plane has no shared geometry (the
clean prog_xy plot was misleading).

Usage::

    python analysis/diagnostic_b_cross_episode_similarity.py \\
        --ckpt $STABLEWM_HOME/checkpoints/<run>/<run>_epoch_10_object.ckpt \\
        --data pusht_expert_train \\
        --episodes 8 \\
        --steps 5 25 50
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


def avg_cos(arr: np.ndarray) -> float:
    """Mean pairwise cosine similarity across rows (excluding self-pairs)."""
    n = arr.shape[0]
    if n < 2:
        return float("nan")
    arr_n = arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8)
    sims = arr_n @ arr_n.T
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="swm dataset name (e.g. 'pusht_expert_train').")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=8,
                        help="Number of episodes to compare (default 8). Used "
                             "only if --episode-ids is not provided.")
    parser.add_argument("--episode-ids", type=int, nargs="+", default=None,
                        help="Explicit episode IDs to compare (overrides "
                             "--episodes). Use to sample diverse episodes "
                             "rather than consecutive ones.")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 25, 50],
                        help="Step indices to probe (clipped to episode length).")
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

    import stable_worldmodel as swm

    model = torch.load(args.ckpt, map_location=args.device, weights_only=False).eval()
    k_prog = int(getattr(model, "k_prog", 0))
    print(f"model k_prog = {k_prog}")

    img_tf = _build_img_transform(224)
    ds = swm.data.HDF5Dataset(args.data, keys_to_cache=["action"], cache_dir=cache)

    # collect z, z^p, z^c at each requested step from each episode
    ep_id_list = args.episode_ids if args.episode_ids else list(range(args.episodes))
    print(f"Episode IDs probed: {ep_id_list}")
    embs_per_ep = []
    for ep_id in ep_id_list:
        rows = _episode_slice(ds, ep_id)
        if len(rows) < max(args.steps) + 1:
            print(f"  [skip] ep {ep_id} only has {len(rows)} steps")
            continue
        pixels = _collect_episode_pixels(ds, rows, img_tf)
        emb = encode_episode(model, pixels, args.device, chunk=args.chunk)
        zp, zc = split_prog_cont(model, emb)
        embs_per_ep.append({
            "ep_id": ep_id,
            "emb": emb.numpy(),
            "z_prog": zp.numpy() if zp is not None else None,
            "z_cont": zc.numpy() if k_prog > 0 else None,
        })

    if not embs_per_ep:
        print("No usable episodes — increase --episodes or check your dataset.")
        return

    n_eps = len(embs_per_ep)
    print(f"\nUsing {n_eps} episodes for the comparison.\n")

    print(f"{'step':>6} | {'cos(z[t])':>10} | {'cos(z^p[t])':>12} | {'cos(z^c[t])':>12}")
    print("-" * 50)

    rows = []
    for step in args.steps:
        full_z_at_step = np.stack([e["emb"][step] for e in embs_per_ep])
        cos_full = avg_cos(full_z_at_step)
        if k_prog > 0:
            zp_at_step = np.stack([e["z_prog"][step] for e in embs_per_ep])
            zc_at_step = np.stack([e["z_cont"][step] for e in embs_per_ep])
            cos_zp = avg_cos(zp_at_step)
            cos_zc = avg_cos(zc_at_step)
        else:
            cos_zp = float("nan")
            cos_zc = float("nan")
        print(f"{step:>6d} | {cos_full:>10.3f} | {cos_zp:>12.3f} | {cos_zc:>12.3f}")
        rows.append((step, cos_full, cos_zp, cos_zc))

    # heuristic verdict
    print("\nInterpretation:")
    if k_prog == 0:
        median_full = float(np.median([r[1] for r in rows]))
        if median_full > 0.95:
            print(f"  Full-z cos = {median_full:.3f} — possible mode collapse on full latent.")
        elif median_full > 0.7:
            print(f"  Full-z cos = {median_full:.3f} — substantial cross-episode similarity.")
        else:
            print(f"  Full-z cos = {median_full:.3f} — low cross-episode similarity (healthy).")
    else:
        zp_med = float(np.median([r[2] for r in rows]))
        zc_med = float(np.median([r[3] for r in rows]))
        full_med = float(np.median([r[1] for r in rows]))
        print(f"  z^p median cos = {zp_med:.3f}")
        print(f"  z^c median cos = {zc_med:.3f}")
        print(f"  full-z median cos = {full_med:.3f}")
        if zp_med > 0.7 and zc_med < 0.5:
            print("  ✓ BEST CASE: prog subspace is shared, content subspace is episode-specific.")
            print("    Subspace split is doing exactly what we designed it for.")
        elif zp_med > 0.7 and zc_med > 0.7:
            print("  ⚠ Both prog and content high — possible encoder mode collapse.")
        elif zp_med < 0.3:
            print("  ⚠ Prog subspace shows little cross-episode similarity.")
            print("    The 'all episodes line up' visual was likely misleading.")
        else:
            print("  Mixed signal; sanity-check by visualizing prog_xy and re-reading the plot.")


if __name__ == "__main__":
    main()
