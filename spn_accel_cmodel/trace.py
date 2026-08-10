from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Tuple


NeighborOffset = Tuple[float, float]


class OffsetProvider(Protocol):
    height: int
    width: int
    neighbors: int

    def delta(self, y: int, x: int, k: int) -> NeighborOffset:
        ...


@dataclass(frozen=True)
class ArrayOffsetProvider:
    """Adapter for recorded model offsets normalized to [H,W,K,2]."""

    offsets: Any
    height: int
    width: int
    neighbors: int

    def delta(self, y: int, x: int, k: int) -> NeighborOffset:
        value = self.offsets[y, x, k]
        return float(value[0]), float(value[1])

    @staticmethod
    def from_numpy(offsets: Any, neighbors: int = 8, center_index: int = 4) -> "ArrayOffsetProvider":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("numpy is required to load recorded offset traces") from exc

        arr = np.asarray(offsets)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        # PyTorch/NLSPN channel-first: [2K,H,W] or [2(K+1),H,W].
        if arr.ndim == 3 and arr.shape[0] in {2 * neighbors, 2 * (neighbors + 1)}:
            channels, h, w = arr.shape
            points = channels // 2
            arr = arr.reshape(points, 2, h, w).transpose(2, 3, 0, 1)
        # Channel-last flattened: [H,W,2K] / [H,W,2(K+1)].
        elif arr.ndim == 3 and arr.shape[-1] in {2 * neighbors, 2 * (neighbors + 1)}:
            h, w, channels = arr.shape
            arr = arr.reshape(h, w, channels // 2, 2)
        # Explicit kernel-first: [K,2,H,W] / [K+1,2,H,W].
        elif arr.ndim == 4 and arr.shape[1] == 2 and arr.shape[0] in {neighbors, neighbors + 1}:
            arr = arr.transpose(2, 3, 0, 1)
        # Already [H,W,K,2].
        elif not (arr.ndim == 4 and arr.shape[-1] == 2 and arr.shape[-2] in {neighbors, neighbors + 1}):
            raise ValueError(
                "unsupported offset shape; expected [1,2K,H,W], [2K,H,W], "
                "[H,W,2K], [K,2,H,W], or [H,W,K,2]"
            )

        if arr.shape[2] == neighbors + 1:
            arr = np.delete(arr, center_index, axis=2)
        if arr.shape[2:] != (neighbors, 2):
            raise ValueError(f"normalized offset shape is invalid: {arr.shape}")
        return ArrayOffsetProvider(
            offsets=np.ascontiguousarray(arr, dtype=np.float32),
            height=int(arr.shape[0]),
            width=int(arr.shape[1]),
            neighbors=neighbors,
        )

    @staticmethod
    def from_npz(
        path: str | Path,
        key: str = "offset",
        neighbors: int = 8,
        center_index: int = 4,
    ) -> "ArrayOffsetProvider":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required to load NPZ traces") from exc
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{key!r} not found in {path}; keys={data.files}")
            arr = data[key]
        return ArrayOffsetProvider.from_numpy(arr, neighbors=neighbors, center_index=center_index)
