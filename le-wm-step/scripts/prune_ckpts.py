#!/usr/bin/env python3
"""Prune per-epoch checkpoint files in a directory tree.

For every run directory found under ``root``, keep only:

- the ``*_epoch_<K>_object.ckpt`` milestone file (default K=10), and
- the ``*_epoch_<max>_object.ckpt`` most-recent file.

Checkpoints within a single directory are grouped by the ``<prefix>`` part of
``<prefix>_epoch_<N>_object.ckpt``. This matters because directories like
``scenario3_sensor_w1_H50_S5/`` often hold multiple runs with distinct
prefixes (``..._v6`` vs ``..._v6p2``) that should be pruned independently.

Usage:
    python scripts/prune_ckpts.py $STABLEWM_HOME              # dry-run
    python scripts/prune_ckpts.py $STABLEWM_HOME --delete     # actually remove
    python scripts/prune_ckpts.py $STABLEWM_HOME --keep-epoch 5   # different milestone
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

# matches ``<prefix>_epoch_<N>_object.ckpt``
PATTERN = re.compile(r"^(.*)_epoch_(\d+)_object\.ckpt$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Directory to walk recursively.")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Actually remove the selected files. Without this flag we only print.",
    )
    ap.add_argument(
        "--keep-epoch",
        type=int,
        default=10,
        help="Milestone epoch to keep alongside the max epoch (default: 10).",
    )
    args = ap.parse_args()

    # {(parent_dir, prefix): [(epoch, path), ...]}
    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)
    for p in args.root.rglob("*_epoch_*_object.ckpt"):
        m = PATTERN.match(p.name)
        if not m:
            continue
        prefix, epoch = m.group(1), int(m.group(2))
        groups[(p.parent, prefix)].append((epoch, p))

    total_bytes = 0
    total_files = 0
    for (parent, prefix), files in sorted(groups.items()):
        files.sort(key=lambda x: x[0])
        max_epoch = files[-1][0]
        keep_epochs = {max_epoch, args.keep_epoch}
        deleters = [(e, p) for e, p in files if e not in keep_epochs]
        keepers = [(e, p) for e, p in files if e in keep_epochs]
        if not deleters:
            continue
        print(f"\n[{parent}]  prefix={prefix!r}")
        for e, p in keepers:
            print(f"  KEEP                 epoch={e:3d}  {p.name}")
        for e, p in deleters:
            size = p.stat().st_size
            total_bytes += size
            total_files += 1
            verb = "DELETE" if args.delete else "WOULD DELETE"
            print(f"  {verb:<20s} epoch={e:3d}  {p.name}  ({size / 2**20:.1f} MB)")
            if args.delete:
                p.unlink()

    verb = "freed" if args.delete else "would free"
    print(
        f"\n{verb} {total_bytes / 2**30:.2f} GiB across {total_files} files"
        f" ({total_files} inodes)"
    )


if __name__ == "__main__":
    main()
