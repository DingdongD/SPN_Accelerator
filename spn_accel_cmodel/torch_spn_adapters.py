"""Compile author-specific SPN metadata into canonical propagation plans."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .torch_spn_core import as_nchw
from .torch_spn_types import (
    CanonicalSPNPlan,
    GRID8_XY,
    ReductionMode,
    SPNConfig,
    SPNInputs,
    SparseFusionMode,
)


def _neighbor_map(
    value: torch.Tensor,
    current: torch.Tensor,
    neighbors: int,
    name: str,
) -> torch.Tensor:
    tensor = value.to(device=current.device, dtype=torch.float32)
    b, _, h, w = current.shape
    if tensor.ndim == 1 and tensor.shape[0] == neighbors:
        tensor = tensor.view(1, neighbors, 1, 1)
    elif tensor.ndim == 2 and tensor.shape[1] == neighbors:
        tensor = tensor.view(tensor.shape[0], neighbors, 1, 1)
    elif tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [K], [B,K], or [B,K,H,W], got "
            f"{tuple(value.shape)}"
        )
    if tensor.shape[1] != neighbors:
        raise ValueError(f"{name} expected {neighbors} channels, got {tensor.shape[1]}")
    try:
        return tensor.expand(b, neighbors, h, w)
    except RuntimeError as exc:
        raise ValueError(f"{name} cannot broadcast to [B,K,H,W]") from exc


def _base_offsets(
    values_xy: tuple[tuple[float, float], ...],
    current: torch.Tensor,
) -> torch.Tensor:
    b, _, h, w = current.shape
    return (
        torch.tensor(values_xy, device=current.device, dtype=torch.float32)
        .view(1, len(values_xy), 2, 1, 1)
        .expand(b, len(values_xy), 2, h, w)
    )


def _make_plan(
    *,
    config: SPNConfig,
    current: torch.Tensor,
    offsets_xy: torch.Tensor,
    neighbor_affinity: torch.Tensor,
    current_affinity: torch.Tensor,
    initial_affinity: torch.Tensor,
    pre_gate: torch.Tensor | None = None,
    pre_value: torch.Tensor | None = None,
    post_gate: torch.Tensor | None = None,
    post_value: torch.Tensor | None = None,
    reduction_mode: ReductionMode = ReductionMode.TORCH_SUM,
    reduction_groups: tuple[tuple[int, int], ...] | None = None,
    group_scale: torch.Tensor | None = None,
) -> CanonicalSPNPlan:
    b, c, h, w = current.shape
    if offsets_xy.ndim == 5:
        offsets_xy = offsets_xy.unsqueeze(1)
    if neighbor_affinity.ndim == 5:
        neighbor_affinity = neighbor_affinity.unsqueeze(1)
    if current_affinity.ndim == 4:
        current_affinity = current_affinity.unsqueeze(1)
    if initial_affinity.ndim == 4:
        initial_affinity = initial_affinity.unsqueeze(1)
    k = offsets_xy.shape[2]
    ca = neighbor_affinity.shape[3]
    zero_gate = torch.zeros((b, 1, 1, h, w), device=current.device)
    zero_value = torch.zeros((b, 1, c, h, w), device=current.device)
    groups = reduction_groups or ((0, k),)
    if group_scale is None:
        group_scale = torch.ones(
            (b, 1, len(groups), ca, h, w),
            device=current.device,
        )
    return CanonicalSPNPlan(
        iterations=config.iterations,
        offsets_xy=offsets_xy,
        neighbor_affinity=neighbor_affinity,
        current_affinity=current_affinity,
        initial_affinity=initial_affinity,
        pre_fusion_gate=zero_gate if pre_gate is None else pre_gate,
        pre_fusion_value=zero_value if pre_value is None else pre_value,
        post_fusion_gate=zero_gate if post_gate is None else post_gate,
        post_fusion_value=zero_value if post_value is None else post_value,
        sampling_mode=config.sampling_mode,
        padding_mode=config.padding_mode,
        align_corners=config.align_corners,
        reduction_mode=reduction_mode,
        reduction_groups=groups,
        group_scale=group_scale,
    )


def _shift_cspn_source_channels(value: torch.Tensor) -> torch.Tensor:
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
    channels = list(torch.chunk(value, 8, dim=1))
    return torch.stack(
        [F.pad(channel, padding) for channel, padding in zip(channels, paddings)],
        dim=1,
    )


def compile_cspn_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    guidance = _neighbor_map(inputs.affinity, current, 8, "affinity")
    shifted = _shift_cspn_source_channels(guidance)
    normalized = shifted / torch.sum(torch.abs(shifted), dim=1, keepdim=True)
    neighbor = normalized[:, :, :, 1:-1, 1:-1].unsqueeze(1)
    initial_affinity = 1.0 - torch.sum(neighbor, dim=2)
    current_affinity = torch.zeros_like(initial_affinity)
    target_offsets = tuple((-x, -y) for x, y in GRID8_XY)
    offsets = _base_offsets(target_offsets, current).unsqueeze(1)

    post_gate = None
    post_value = None
    if config.sparse_fusion is SparseFusionMode.CSPN_CODE_POST:
        if inputs.sparse_depth is None:
            raise ValueError("CSPN code post-mask requires sparse_depth")
        sparse = as_nchw(
            inputs.sparse_depth,
            "sparse_depth",
            device=current.device,
        )
        if sparse.shape != current.shape:
            raise ValueError("sparse_depth must have the same shape as current")
        post_gate = sparse.sign().unsqueeze(1)
        post_value = initial.unsqueeze(1)

    return _make_plan(
        config=config,
        current=current,
        offsets_xy=offsets,
        neighbor_affinity=neighbor,
        current_affinity=current_affinity,
        initial_affinity=initial_affinity,
        post_gate=post_gate,
        post_value=post_value,
        reduction_mode=ReductionMode.TORCH_SUM,
    )
