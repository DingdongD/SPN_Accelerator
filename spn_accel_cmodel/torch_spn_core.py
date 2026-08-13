"""Profile-agnostic Torch FP32 SPN propagation core."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .torch_spn_types import (
    CanonicalSPNPlan,
    PaddingMode,
    ReductionMode,
    SPNTrace,
    SamplingMode,
)


def as_nchw(
    value: torch.Tensor,
    name: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"{name} must be NCHW, got {tuple(value.shape)}")
    return value.to(device=device or value.device, dtype=torch.float32)


def _normalized_coordinate(
    coordinate: torch.Tensor,
    size: int,
    align_corners: bool,
) -> torch.Tensor:
    if size <= 1 and align_corners:
        return torch.zeros_like(coordinate)
    if align_corners:
        return 2.0 * coordinate / float(size - 1) - 1.0
    return 2.0 * (coordinate + 0.5) / float(size) - 1.0


def _integer_sample(
    state: torch.Tensor,
    offsets_xy: torch.Tensor,
    padding_mode: PaddingMode,
) -> torch.Tensor:
    b, c, h, w = state.shape
    k = offsets_xy.shape[1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=state.device),
        torch.arange(w, device=state.device),
        indexing="ij",
    )
    sample_x = xx.view(1, 1, h, w) + offsets_xy[:, :, 0].to(torch.int64)
    sample_y = yy.view(1, 1, h, w) + offsets_xy[:, :, 1].to(torch.int64)
    valid = (
        (sample_x >= 0)
        & (sample_x < w)
        & (sample_y >= 0)
        & (sample_y < h)
    )
    linear = sample_y.clamp(0, h - 1) * w + sample_x.clamp(0, w - 1)
    source = state.reshape(b, 1, c, h * w).expand(b, k, c, h * w)
    index = linear.unsqueeze(2).expand(b, k, c, h, w).reshape(b, k, c, h * w)
    sampled = torch.gather(source, 3, index).reshape(b, k, c, h, w)
    if padding_mode is PaddingMode.ZEROS:
        sampled = sampled * valid.unsqueeze(2)
    return sampled


def sample_neighbors(
    state: torch.Tensor,
    offsets_xy: torch.Tensor,
    sampling_mode: SamplingMode,
    padding_mode: PaddingMode,
    *,
    align_corners: bool,
) -> torch.Tensor:
    if state.ndim != 4:
        raise ValueError("state must have shape [B,C,H,W]")
    if offsets_xy.ndim != 5 or offsets_xy.shape[2] != 2:
        raise ValueError("offsets must have shape [B,K,2,H,W]")
    b, c, h, w = state.shape
    if offsets_xy.shape[0] != b or offsets_xy.shape[3:] != (h, w):
        raise ValueError("offsets batch/spatial shape must match state")
    offsets_xy = offsets_xy.to(device=state.device, dtype=torch.float32)
    if sampling_mode is SamplingMode.INTEGER:
        return _integer_sample(state, offsets_xy, padding_mode)

    k = offsets_xy.shape[1]
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=state.dtype, device=state.device),
        torch.arange(w, dtype=state.dtype, device=state.device),
        indexing="ij",
    )
    sample_x = xx.view(1, 1, h, w) + offsets_xy[:, :, 0]
    sample_y = yy.view(1, 1, h, w) + offsets_xy[:, :, 1]
    grid_align_corners = align_corners and h > 1 and w > 1
    grid_x = _normalized_coordinate(sample_x, w, grid_align_corners)
    grid_y = _normalized_coordinate(sample_y, h, grid_align_corners)
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(b * k, h, w, 2)
    source = state[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)
    torch_padding = "zeros" if padding_mode is PaddingMode.ZEROS else "border"
    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=torch_padding,
        align_corners=grid_align_corners,
    )
    return sampled.reshape(b, k, c, h, w)


def _validate_time(name: str, tensor: torch.Tensor, iterations: int) -> None:
    if tensor.shape[1] not in {1, iterations}:
        raise ValueError(
            f"{name} time dimension must be 1 or {iterations}, got {tensor.shape[1]}"
        )


def validate_plan(
    current: torch.Tensor,
    initial: torch.Tensor,
    plan: CanonicalSPNPlan,
) -> None:
    if current.ndim != 4 or initial.shape != current.shape:
        raise ValueError("initial and current must have identical NCHW shapes")
    if current.dtype is not torch.float32 or initial.dtype is not torch.float32:
        raise ValueError("current and initial must be FP32")
    if current.device != initial.device:
        raise ValueError("current and initial must use the same device")
    b, c, h, w = current.shape
    tensors = {
        "offsets_xy": plan.offsets_xy,
        "neighbor_affinity": plan.neighbor_affinity,
        "current_affinity": plan.current_affinity,
        "initial_affinity": plan.initial_affinity,
        "pre_fusion_gate": plan.pre_fusion_gate,
        "pre_fusion_value": plan.pre_fusion_value,
        "post_fusion_gate": plan.post_fusion_gate,
        "post_fusion_value": plan.post_fusion_value,
        "group_scale": plan.group_scale,
    }
    for name, tensor in tensors.items():
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be FP32")
        if tensor.device != current.device:
            raise ValueError(f"{name} must use current device")
        _validate_time(name, tensor, plan.iterations)

    if plan.iterations < 1:
        raise ValueError("plan iterations must be at least one")
    if plan.offsets_xy.ndim != 6 or plan.offsets_xy.shape[3] != 2:
        raise ValueError("offsets_xy must have shape [B,S,K,2,H,W]")
    if plan.offsets_xy.shape[0] != b or plan.offsets_xy.shape[4:] != (h, w):
        raise ValueError("offsets_xy batch/spatial shape must match current")
    k = plan.offsets_xy.shape[2]
    if plan.neighbor_affinity.ndim != 6:
        raise ValueError("neighbor_affinity must have shape [B,S,K,Ca,H,W]")
    if (
        plan.neighbor_affinity.shape[0] != b
        or plan.neighbor_affinity.shape[2] != k
        or plan.neighbor_affinity.shape[4:] != (h, w)
        or plan.neighbor_affinity.shape[3] not in {1, c}
    ):
        raise ValueError("neighbor_affinity shape is incompatible with current")
    ca = plan.neighbor_affinity.shape[3]
    for name, tensor in (
        ("current_affinity", plan.current_affinity),
        ("initial_affinity", plan.initial_affinity),
    ):
        if tensor.ndim != 5 or tensor.shape[0] != b or tensor.shape[2:] != (ca, h, w):
            raise ValueError(f"{name} must have shape [B,S,Ca,H,W]")
    for name, tensor, channels in (
        ("pre_fusion_gate", plan.pre_fusion_gate, 1),
        ("post_fusion_gate", plan.post_fusion_gate, 1),
        ("pre_fusion_value", plan.pre_fusion_value, c),
        ("post_fusion_value", plan.post_fusion_value, c),
    ):
        if tensor.ndim != 5 or tensor.shape[0] != b or tensor.shape[2:] != (channels, h, w):
            raise ValueError(f"{name} has incompatible shape")
    groups = plan.reduction_groups
    if not groups or groups[0][0] != 0 or groups[-1][1] != k:
        raise ValueError("reduction groups must cover every neighbor")
    for index, (start, stop) in enumerate(groups):
        if start >= stop or (index and start != groups[index - 1][1]):
            raise ValueError("reduction groups must be ordered and non-overlapping")
    if plan.reduction_mode is not ReductionMode.GROUPED and groups != ((0, k),):
        raise ValueError("non-grouped reduction requires one complete group")
    if (
        plan.group_scale.ndim != 6
        or plan.group_scale.shape[0] != b
        or plan.group_scale.shape[2:] != (len(groups), ca, h, w)
    ):
        raise ValueError("group_scale must have shape [B,S,G,Ca,H,W]")


def _step(tensor: torch.Tensor, iteration: int) -> torch.Tensor:
    return tensor[:, 0 if tensor.shape[1] == 1 else iteration]


def _blend(
    base: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - gate) * base + gate * value


def reduce_neighbors(
    weighted: torch.Tensor,
    group_scale: torch.Tensor,
    groups: tuple[tuple[int, int], ...],
    mode: ReductionMode,
) -> torch.Tensor:
    if mode is ReductionMode.SEQUENTIAL:
        result = torch.zeros_like(weighted[:, 0])
        for neighbor in range(weighted.shape[1]):
            result = result + weighted[:, neighbor]
        return result * group_scale[:, 0]
    if mode is ReductionMode.TORCH_SUM:
        return torch.sum(weighted, dim=1) * group_scale[:, 0]
    result = torch.zeros_like(weighted[:, 0])
    for group, (start, stop) in enumerate(groups):
        reduced = torch.sum(weighted[:, start:stop], dim=1)
        result = result + reduced * group_scale[:, group]
    return result


def _effective_neighbor_affinity(
    affinity: torch.Tensor,
    scale: torch.Tensor,
    groups: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    effective = torch.empty_like(affinity)
    for group, (start, stop) in enumerate(groups):
        effective[:, start:stop] = affinity[:, start:stop] * scale[:, group : group + 1]
    return effective


def propagate_canonical(
    current: torch.Tensor,
    initial: torch.Tensor,
    plan: CanonicalSPNPlan,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, SPNTrace]:
    validate_plan(current, initial, plan)
    state = current
    trace = SPNTrace()
    for iteration in range(plan.iterations):
        pre_gate = _step(plan.pre_fusion_gate, iteration)
        pre_value = _step(plan.pre_fusion_value, iteration)
        propagated = _blend(state, pre_value, pre_gate)
        offsets = _step(plan.offsets_xy, iteration)
        samples = sample_neighbors(
            propagated,
            offsets,
            plan.sampling_mode,
            plan.padding_mode,
            align_corners=plan.align_corners,
        )
        affinity = _step(plan.neighbor_affinity, iteration)
        scale = _step(plan.group_scale, iteration)
        candidate = reduce_neighbors(
            samples * affinity,
            scale,
            plan.reduction_groups,
            plan.reduction_mode,
        )
        current_affinity = _step(plan.current_affinity, iteration)
        initial_affinity = _step(plan.initial_affinity, iteration)
        candidate = candidate + current_affinity * propagated
        candidate = candidate + initial_affinity * initial
        post_gate = _step(plan.post_fusion_gate, iteration)
        state = _blend(
            candidate,
            _step(plan.post_fusion_value, iteration),
            post_gate,
        )
        trace.pre_fused_states.append(propagated)
        trace.candidates.append(candidate)
        trace.outputs.append(state)
        trace.offsets.append(offsets)
        trace.neighbor_affinities.append(
            _effective_neighbor_affinity(affinity, scale, plan.reduction_groups)
        )
        trace.current_affinities.append(current_affinity)
        trace.initial_affinities.append(initial_affinity)
        trace.pre_fusion_gates.append(pre_gate)
        trace.post_fusion_gates.append(post_gate)
        trace.group_scales.append(scale)
    return (state, trace) if return_trace else state
