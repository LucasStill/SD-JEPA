"""Diagnostic A — episode length distribution for a swm dataset.

Quick check: are all episodes the same length, or is there real variation?
This disambiguates Hypothesis 1 (genuine progression) from Hypothesis 2
(same-length episodes mechanically force latent alignment).

Usage::

    python analysis/diagnostic_a_episode_lengths.py --data pusht
    python analysis/diagnostic_a_episode_lengths.py --data tworoom
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True,
                        help="swm dataset name (e.g. 'pusht_expert_train', 'tworoom').")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override; defaults to $STABLEWM_HOME or ~/.stable_worldmodel.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit to first N episodes (default: all).")
    args = parser.parse_args()

    import stable_worldmodel as swm

    cache = (
        args.cache_dir
        or os.environ.get("STABLEWM_HOME")
        or str(Path.home() / ".stable_worldmodel")
    )
    print(f"cache_dir = {cache}")

    ds = swm.data.HDF5Dataset(
        args.data, keys_to_cache=["action"], cache_dir=cache
    )

    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep = ds.get_col_data(col)
    unique_eps = np.unique(ep)
    if args.max_episodes is not None:
        unique_eps = unique_eps[: args.max_episodes]

    lengths = np.array([(ep == e).sum() for e in unique_eps])

    print(f"\nepisodes      : {len(unique_eps)}")
    print(f"length min    : {lengths.min()}")
    print(f"length max    : {lengths.max()}")
    print(f"length mean   : {lengths.mean():.1f}")
    print(f"length std    : {lengths.std():.1f}")
    print(f"length median : {np.median(lengths):.1f}")
    print(f"unique len cnt: {len(np.unique(lengths))}")
    print(f"first 20 eps  : {lengths[:20].tolist()}")

    if len(np.unique(lengths)) == 1:
        print("\n⚠️  All episodes have identical length — Hypothesis 2 (alignment-by-construction) is plausible.")
        print("   Run Diagnostic B (z^p[t] cross-episode similarity) to disambiguate.")
    elif lengths.std() < 5:
        print("\n⚠️  Episode lengths are tightly clustered — partial Hypothesis-2 risk.")
        print("   Variation is real but small; Diagnostic B still recommended.")
    else:
        print("\n✓  Episode lengths vary meaningfully — Hypothesis 2 unlikely to fully explain alignment.")
        print("   The cleanness in the prog plot is more likely a genuine shared geometry.")


if __name__ == "__main__":
    main()
