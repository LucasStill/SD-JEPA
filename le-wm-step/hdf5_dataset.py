"""HDF5Dataset compat shim.

stable-worldmodel >= 0.0.6 dropped HDF5Dataset from its public API and
switched to Lance format.  The on-disk training data for this project is still
the original .h5 files (e.g. ~/.stable_worldmodel/pusht_expert_train.h5), so
we carry the last known-good implementation here rather than re-downloading
or converting the dataset.

Ported verbatim from the stable_worldmodel version bundled with le-wm
(commit pinned in le-wm's uv.lock).  The only structural dependency is
stable_worldmodel.data.Dataset, whose interface has not changed in 0.0.6.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — registers HDF5 compression codecs
import numpy as np
import torch

from stable_worldmodel.data import Dataset
from stable_worldmodel.data.utils import get_cache_dir


class HDF5Dataset(Dataset):
    """Dataset loading from a single HDF5 file (SWMR mode for safe reads).

    Path resolution:
        ``Path(cache_dir or get_cache_dir(), f'{name}.h5')``

    This matches the layout produced by the original stable_worldmodel
    downloader (files placed directly under STABLEWM_HOME, *not* in a
    datasets/ sub-directory).
    """

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.h5_path = Path(cache_dir or get_cache_dir(), f"{name}.h5")
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, "r") as f:
            lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ("ep_len", "ep_offset")
            ]
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start = self.offsets[ep_idx] + start
        g_end = self.offsets[ep_idx] + end
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != "action":
                data = data[:: self.frameskip]
            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        self._open()
        if isinstance(source, str):
            source = [k for k in self.h5_file.keys() if re.match(source, k)]
        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1
