from __future__ import annotations

import math
from typing import Any

from .spn_engine import DEFAULT_NEIGHBORS
from .trace import ArrayOffsetProvider, OffsetProvider


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for functional SPN validation") from exc
    return np


def normalize_state(state: Any):
    np = _numpy()
    arr = np.asarray(state, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"state must reduce to [H,W], got {arr.shape}")
    return arr


def normalize_affinity(affinity: Any, height: int, width: int, neighbors: int = 8, center_index: int = 4):
    """Normalize affinity to [H,W,K+1], including the center coefficient."""
    np = _numpy()
    arr = np.asarray(affinity, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {neighbors, neighbors + 1}:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[:2] != (height, width):
        raise ValueError(f"affinity shape is incompatible with H,W={height,width}: {arr.shape}")
    if arr.shape[2] == neighbors:
        center = 1.0 - arr.sum(axis=2, keepdims=True)
        arr = np.concatenate((arr[:, :, :center_index], center, arr[:, :, center_index:]), axis=2)
    if arr.shape[2] != neighbors + 1:
        raise ValueError(f"affinity needs K or K+1 channels, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def bilinear_sample_zero(state: Any, y: float, x: float) -> float:
    """grid_sample-style bilinear sample in absolute coordinates with zero padding.

    CompletionFormer's fallback converts absolute coordinates to normalized
    align_corners=True coordinates before grid_sample, so this direct absolute
    implementation is equivalent for a scalar depth plane.
    """
    h, w = state.shape
    y0 = math.floor(y)
    x0 = math.floor(x)
    dy = y - y0
    dx = x - x0

    def at(yy: int, xx: int) -> float:
        if yy < 0 or yy >= h or xx < 0 or xx >= w:
            return 0.0
        return float(state[yy, xx])

    v00 = at(y0, x0)
    v01 = at(y0, x0 + 1)
    v10 = at(y0 + 1, x0)
    v11 = at(y0 + 1, x0 + 1)
    top = v00 + dx * (v01 - v00)
    bottom = v10 + dx * (v11 - v10)
    return top + dy * (bottom - top)


def propagate_once(
    state: Any,
    offsets: OffsetProvider | Any,
    affinity: Any,
    neighbors: int = 8,
    center_index: int = 4,
):
    np = _numpy()
    src = normalize_state(state)
    h, w = src.shape
    provider = offsets if hasattr(offsets, "delta") else ArrayOffsetProvider.from_numpy(offsets, neighbors, center_index)
    if provider.height != h or provider.width != w or provider.neighbors != neighbors:
        raise ValueError("offset provider dimensions do not match state")
    aff = normalize_affinity(affinity, h, w, neighbors, center_index)
    dst = np.zeros_like(src, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            value = float(aff[y, x, center_index]) * float(src[y, x])
            aidx = 0
            for k, (base_dy, base_dx) in enumerate(DEFAULT_NEIGHBORS[:neighbors]):
                ddy, ddx = provider.delta(y, x, k)
                sampled = bilinear_sample_zero(src, y + base_dy + ddy, x + base_dx + ddx)
                if aidx == center_index:
                    aidx += 1
                value += float(aff[y, x, aidx]) * sampled
                aidx += 1
            dst[y, x] = value
    return dst


def propagate(
    state: Any,
    offsets: OffsetProvider | Any,
    affinity: Any,
    steps: int,
    neighbors: int = 8,
    center_index: int = 4,
):
    out = normalize_state(state).copy()
    for _ in range(steps):
        out = propagate_once(out, offsets, affinity, neighbors, center_index)
    return out


def error_metrics(actual: Any, golden: Any) -> dict[str, float]:
    np = _numpy()
    a = np.asarray(actual, dtype=np.float64)
    g = np.asarray(golden, dtype=np.float64)
    if a.shape != g.shape:
        raise ValueError(f"shape mismatch: actual={a.shape}, golden={g.shape}")
    diff = a - g
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }
