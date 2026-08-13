"""Independent Torch restatements of released SPN propagation code.

This file deliberately does not import ``spn_accel_cmodel.torch_functional``.
The functions are test oracles, not reusable production helpers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


GRID8_XY = (
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
)


def _sample_absolute_xy(
    state: torch.Tensor,
    offsets_xy: torch.Tensor,
    *,
    align_corners: bool,
    mode: str = "bilinear",
) -> torch.Tensor:
    b, c, h, w = state.shape
    _, k, _, _, _ = offsets_xy.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=state.device),
        torch.arange(w, dtype=torch.float32, device=state.device),
        indexing="ij",
    )
    sx = xx.view(1, 1, h, w) + offsets_xy[:, :, 0]
    sy = yy.view(1, 1, h, w) + offsets_xy[:, :, 1]
    if align_corners:
        gx = torch.zeros_like(sx) if w == 1 else 2.0 * sx / (w - 1) - 1.0
        gy = torch.zeros_like(sy) if h == 1 else 2.0 * sy / (h - 1) - 1.0
    else:
        gx = 2.0 * (sx + 0.5) / w - 1.0
        gy = 2.0 * (sy + 0.5) / h - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(b * k, h, w, 2)
    source = state[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
    return F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    ).reshape(b, k, c, h, w)


def _cspn_shifted_channels(value: torch.Tensor) -> torch.Tensor:
    paddings = (
        (0, 2, 0, 2),
        (1, 1, 0, 2),
        (2, 0, 0, 2),
        (0, 2, 1, 1),
        (2, 0, 1, 1),
        (0, 2, 2, 0),
        (1, 1, 2, 0),
        (2, 0, 2, 0),
    )
    if value.shape[1] == 1:
        channels = [value] * 8
    else:
        channels = list(torch.chunk(value, 8, dim=1))
    return torch.stack(
        [F.pad(channel, padding) for channel, padding in zip(channels, paddings)],
        dim=1,
    )


def reference_cspn(
    initial: torch.Tensor,
    guidance: torch.Tensor,
    iterations: int,
    sparse_depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    gate = _cspn_shifted_channels(guidance)
    gate = gate / torch.sum(torch.abs(gate), dim=1, keepdim=True)
    gate_sum = torch.sum(gate, dim=1)[:, :, 1:-1, 1:-1]
    state = initial
    outputs = []
    mask = sparse_depth.sign() if sparse_depth is not None else None
    for _ in range(iterations):
        shifted_state = _cspn_shifted_channels(state)
        neighbor = torch.sum(gate * shifted_state, dim=1)[:, :, 1:-1, 1:-1]
        state = neighbor + (1.0 - gate_sum) * initial
        if mask is not None:
            state = (1.0 - mask) * state + mask * initial
        outputs.append(state)
    return state, outputs


def reference_nlspn(
    initial: torch.Tensor,
    raw_affinity: torch.Tensor,
    residual_yx: torch.Tensor,
    iterations: int,
    *,
    mode: str,
    confidence: torch.Tensor | None = None,
    sparse_depth: torch.Tensor | None = None,
    preserve_input: bool = False,
    temperature: float = 1.0,
    gamma: float = 0.5,
    legacy_confidence_offsets: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    b, _, h, w = initial.shape
    base = torch.tensor(GRID8_XY, dtype=torch.float32, device=initial.device)
    base = base.view(1, 8, 2, 1, 1).expand(b, -1, -1, h, w)
    residual_xy = residual_yx[:, :, [1, 0]]
    total_xy = base + residual_xy

    affinity = raw_affinity
    if mode in {"TC", "TGASS"}:
        scale = 8.0 if mode == "TC" else gamma * 8.0 + 1.0e-8
        affinity = torch.tanh(affinity / temperature) / scale

    if confidence is not None:
        confidence_offsets = total_xy if legacy_confidence_offsets else residual_xy
        sampled_confidence = _sample_absolute_xy(
            confidence,
            confidence_offsets.detach(),
            align_corners=True,
        )[:, :, 0]
        affinity = affinity * sampled_confidence

    denom = torch.sum(torch.abs(affinity), dim=1, keepdim=True) + 1.0e-4
    if mode in {"ASS", "TGASS"}:
        denom = torch.where(denom < 1.0, torch.ones_like(denom), denom)
    if mode in {"AS", "ASS", "TGASS"}:
        affinity = affinity / denom
    center = 1.0 - torch.sum(affinity, dim=1, keepdim=True)

    state = initial
    outputs = []
    mask = None
    if preserve_input:
        if sparse_depth is None:
            raise ValueError("preserve_input requires sparse_depth")
        mask = (sparse_depth > 0.0).to(dtype=torch.float32)
    for _ in range(iterations):
        if mask is not None:
            state = (1.0 - mask) * state + mask * sparse_depth
        samples = _sample_absolute_xy(state, total_xy, align_corners=True)
        state = torch.sum(samples * affinity[:, :, None], dim=1) + center * state
        outputs.append(state)
    return state, outputs, affinity, center
