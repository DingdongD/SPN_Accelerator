"""Compile author-specific SPN metadata into canonical propagation plans."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .torch_spn_core import as_nchw, sample_neighbors
from .torch_spn_types import (
    AffinityLayout,
    AffinityMode,
    AnchorMode,
    CanonicalSPNPlan,
    GRID8_XY,
    NeighborConfidenceMode,
    NeighborMode,
    NormalizationMode,
    OffsetMode,
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


def _static_offsets(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.offsets is None:
        raise ValueError("offset propagation requires offsets")
    offsets = inputs.offsets.to(device=current.device, dtype=torch.float32)
    if offsets.ndim == 4 and offsets.shape[1] in {
        2 * config.num_neighbors,
        2 * (config.num_neighbors + 1),
    }:
        points = offsets.shape[1] // 2
        offsets = offsets.reshape(
            offsets.shape[0],
            points,
            2,
            offsets.shape[2],
            offsets.shape[3],
        )
        if points == config.num_neighbors + 1:
            center = config.num_neighbors // 2
            offsets = torch.cat((offsets[:, :center], offsets[:, center + 1 :]), dim=1)
    expected = (
        current.shape[0],
        config.num_neighbors,
        2,
        current.shape[2],
        current.shape[3],
    )
    if tuple(offsets.shape) != expected:
        raise ValueError(f"offsets must have shape {expected}, got {tuple(offsets.shape)}")
    residual_xy = offsets[:, :, [1, 0]] if config.offset_mode is OffsetMode.RESIDUAL_YX else offsets
    if config.offset_mode is OffsetMode.ABSOLUTE_XY:
        return residual_xy, residual_xy
    if not config.base_offsets_xy:
        raise ValueError("residual offsets require base_offsets_xy")
    return residual_xy, _base_offsets(config.base_offsets_xy, current) + residual_xy


def _sparse_and_mask(
    inputs: SPNInputs,
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.sparse_depth is None:
        raise ValueError("configured sparse fusion requires sparse_depth")
    sparse = as_nchw(inputs.sparse_depth, "sparse_depth", device=current.device)
    if sparse.shape != current.shape:
        raise ValueError("sparse_depth must have the same shape as current")
    if inputs.sparse_mask is None:
        mask = (sparse > 0.0).to(dtype=torch.float32)
    else:
        raw_mask = as_nchw(inputs.sparse_mask, "sparse_mask", device=current.device)
        if raw_mask.shape != current.shape:
            raise ValueError("sparse_mask must have the same shape as current")
        mask = (raw_mask > 0.0).to(dtype=torch.float32)
    return sparse, mask


def _compile_nlspn_like_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
    *,
    temperature: float,
) -> CanonicalSPNPlan:
    del initial
    affinity = _neighbor_map(inputs.affinity, current, config.num_neighbors, "affinity")
    residual_xy, total_xy = _static_offsets(config, inputs, current)
    if config.normalization in {NormalizationMode.TC, NormalizationMode.TGASS}:
        scale = (
            float(config.num_neighbors)
            if config.normalization is NormalizationMode.TC
            else config.affinity_gamma * float(config.num_neighbors) + 1.0e-8
        )
        affinity = torch.tanh(affinity / temperature) / scale
    if config.neighbor_confidence is NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR:
        if inputs.confidence is None:
            raise ValueError("neighbor confidence sampling requires confidence")
        confidence = as_nchw(inputs.confidence, "confidence", device=current.device)
        if confidence.shape != (current.shape[0], 1, current.shape[2], current.shape[3]):
            raise ValueError("confidence must have shape [B,1,H,W]")
        confidence_offsets = total_xy if config.legacy_confidence_offsets else residual_xy
        sampled_confidence = sample_neighbors(
            confidence,
            confidence_offsets.detach(),
            config.sampling_mode,
            config.padding_mode,
            align_corners=config.align_corners,
        )[:, :, 0]
        affinity = affinity * sampled_confidence
    denominator = torch.sum(torch.abs(affinity), dim=1, keepdim=True) + config.eps
    if config.normalization in {NormalizationMode.ASS, NormalizationMode.TGASS}:
        denominator = torch.where(
            denominator < 1.0,
            torch.ones_like(denominator),
            denominator,
        )
    if config.normalization in {
        NormalizationMode.AS,
        NormalizationMode.ASS,
        NormalizationMode.TGASS,
    }:
        affinity = affinity / denominator
    current_affinity = 1.0 - torch.sum(affinity, dim=1, keepdim=True)
    initial_affinity = torch.zeros_like(current_affinity)
    pre_gate = None
    pre_value = None
    if config.sparse_fusion is SparseFusionMode.HARD_PRE:
        sparse, mask = _sparse_and_mask(inputs, current)
        pre_gate = mask.unsqueeze(1)
        pre_value = sparse.unsqueeze(1)
    return _make_plan(
        config=config,
        current=current,
        offsets_xy=total_xy,
        neighbor_affinity=affinity.unsqueeze(2),
        current_affinity=current_affinity,
        initial_affinity=initial_affinity,
        pre_gate=pre_gate,
        pre_value=pre_value,
        reduction_mode=ReductionMode.TORCH_SUM,
    )


def compile_nlspn_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    return _compile_nlspn_like_plan(
        config,
        inputs,
        current,
        initial,
        temperature=1.0,
    )


def compile_completionformer_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    return _compile_nlspn_like_plan(
        config,
        inputs,
        current,
        initial,
        temperature=100.0,
    )


def compile_dyspn_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    del initial
    logits = inputs.affinity.to(device=current.device, dtype=torch.float32)
    expected_affinity = (
        current.shape[0],
        config.iterations,
        config.num_neighbors,
        current.shape[2],
        current.shape[3],
    )
    if tuple(logits.shape) != expected_affinity:
        raise ValueError(
            f"per-iteration affinity must have shape {expected_affinity}, "
            f"got {tuple(logits.shape)}"
        )
    if inputs.offsets is None:
        raise ValueError("DySPN requires per-iteration offsets")
    residual_yx = inputs.offsets.to(device=current.device, dtype=torch.float32)
    expected_offsets = (
        current.shape[0],
        config.iterations,
        config.num_neighbors,
        2,
        current.shape[2],
        current.shape[3],
    )
    if tuple(residual_yx.shape) != expected_offsets:
        raise ValueError(
            f"DySPN offsets must have shape {expected_offsets}, "
            f"got {tuple(residual_yx.shape)}"
        )
    residual_xy = residual_yx[:, :, :, [1, 0]]
    base = torch.tensor(
        config.base_offsets_xy,
        device=current.device,
        dtype=torch.float32,
    ).view(1, 1, config.num_neighbors, 2, 1, 1)
    offsets_xy = base + residual_xy
    affinity = torch.softmax(logits, dim=2).unsqueeze(3)

    if inputs.sparse_depth is None:
        raise ValueError("DySPN soft-post fusion requires sparse_depth")
    sparse = as_nchw(inputs.sparse_depth, "sparse_depth", device=current.device)
    if sparse.shape != current.shape:
        raise ValueError("sparse_depth must have the same shape as current")
    if inputs.confidence is None:
        raise ValueError("DySPN soft-post fusion requires confidence logits")
    confidence_logits = as_nchw(
        inputs.confidence,
        "confidence",
        device=current.device,
    )
    if confidence_logits.shape != current.shape:
        raise ValueError("DySPN confidence must have the same shape as current")
    post_gate = (torch.sigmoid(confidence_logits) * sparse.sign()).unsqueeze(1)
    post_value = sparse.unsqueeze(1)
    zeros = torch.zeros(
        (current.shape[0], 1, 1, current.shape[2], current.shape[3]),
        device=current.device,
    )
    return _make_plan(
        config=config,
        current=current,
        offsets_xy=offsets_xy,
        neighbor_affinity=affinity,
        current_affinity=zeros,
        initial_affinity=zeros,
        post_gate=post_gate,
        post_value=post_value,
        reduction_mode=ReductionMode.SEQUENTIAL,
    )


def _edge_offsets_xy(kernel: int) -> tuple[tuple[float, float], ...]:
    radius = kernel // 2
    return tuple(
        (float(radius - column), float(radius - row))
        for row in range(kernel)
        for column in range(kernel)
        if row in {0, kernel - 1} or column in {0, kernel - 1}
    )


def _shift_channels_to_target(
    value: torch.Tensor,
    offsets_xy: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    _, _, height, width = value.shape
    radius = max(int(max(abs(x), abs(y))) for x, y in offsets_xy)
    padded = F.pad(value, (radius, radius, radius, radius))
    shifted = []
    for channel, (offset_x, offset_y) in enumerate(offsets_xy):
        start_x = radius + int(offset_x)
        start_y = radius + int(offset_y)
        shifted.append(
            padded[
                :,
                channel : channel + 1,
                start_y : start_y + height,
                start_x : start_x + width,
            ]
        )
    return torch.cat(shifted, dim=1)


def _edge_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
    edge_indices = tuple(
        row * kernel + column
        for row in range(kernel)
        for column in range(kernel)
        if row in {0, kernel - 1} or column in {0, kernel - 1}
    )
    if value.shape[1] != len(edge_indices):
        raise ValueError(
            f"kernel {kernel} edge sum expects {len(edge_indices)} channels, "
            f"got {value.shape[1]}"
        )
    weight = torch.zeros(
        (1, len(edge_indices), kernel, kernel),
        device=value.device,
        dtype=torch.float32,
    )
    for channel, edge_index in enumerate(edge_indices):
        weight[
            0,
            channel,
            -(edge_index // kernel) - 1,
            (-edge_index) % kernel - 1,
        ] = 1.0
    return F.conv2d(value, weight, padding=kernel // 2)


def compile_dyspn_nlpm_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    del initial
    guidance = _neighbor_map(inputs.affinity, current, 48, "affinity")
    if inputs.attention is None:
        raise ValueError("DySPN NLPM requires per-iteration attention")
    attention_logits = inputs.attention.to(device=current.device, dtype=torch.float32)
    expected_attention = (
        current.shape[0],
        config.iterations,
        4,
        current.shape[2],
        current.shape[3],
    )
    if tuple(attention_logits.shape) != expected_attention:
        raise ValueError(
            f"DySPN NLPM attention must have shape {expected_attention}, "
            f"got {tuple(attention_logits.shape)}"
        )
    attention = torch.sigmoid(attention_logits)
    sparse, _ = _sparse_and_mask(inputs, current)
    if inputs.confidence is None:
        raise ValueError("DySPN NLPM requires sparse confidence")
    confidence = as_nchw(inputs.confidence, "confidence", device=current.device)
    if confidence.shape != current.shape:
        raise ValueError("DySPN NLPM confidence must have the same shape as current")

    batch, _, height, width = current.shape
    ones = torch.ones((batch, 1, height, width), device=current.device)
    abs_sums = torch.cat(
        (
            _edge_sum(torch.abs(guidance[:, 0:8]), 3),
            _edge_sum(torch.abs(guidance[:, 8:24]), 5),
            _edge_sum(torch.abs(guidance[:, 24:48]), 7),
            ones,
        ),
        dim=1,
    )
    signed_sums = torch.cat(
        (
            _edge_sum(guidance[:, 0:8], 3),
            _edge_sum(guidance[:, 8:24], 5),
            _edge_sum(guidance[:, 24:48], 7),
            ones,
        ),
        dim=1,
    )
    denominator = (
        torch.sum(attention * abs_sums[:, None], dim=2, keepdim=True)
        + config.eps
    )
    initial_numerator = denominator - torch.sum(
        attention * signed_sums[:, None],
        dim=2,
        keepdim=True,
    )
    groups = ((0, 8), (8, 24), (24, 48))
    group_scale = attention[:, :, 0:3].unsqueeze(3) / denominator.unsqueeze(3)
    current_affinity = attention[:, :, 3:4] / denominator
    initial_affinity = initial_numerator / denominator
    offsets_values = _edge_offsets_xy(3) + _edge_offsets_xy(5) + _edge_offsets_xy(7)
    offsets = _base_offsets(offsets_values, current).unsqueeze(1)
    shifted_guidance = _shift_channels_to_target(guidance, offsets_values)
    neighbor_affinity = shifted_guidance.unsqueeze(1).unsqueeze(3)
    post_gate = (confidence * sparse.sign()).unsqueeze(1)
    post_value = sparse.unsqueeze(1)
    return _make_plan(
        config=config,
        current=current,
        offsets_xy=offsets,
        neighbor_affinity=neighbor_affinity,
        current_affinity=current_affinity,
        initial_affinity=initial_affinity,
        post_gate=post_gate,
        post_value=post_value,
        reduction_mode=ReductionMode.GROUPED,
        reduction_groups=groups,
        group_scale=group_scale,
    )


def _generic_offsets(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
) -> torch.Tensor:
    if config.neighbor_mode in {NeighborMode.GRID, NeighborMode.DILATED}:
        if config.num_neighbors != 8:
            raise ValueError("generic GRID/DILATED mode requires eight neighbors")
        scale = float(config.dilation if config.neighbor_mode is NeighborMode.DILATED else 1)
        values = tuple((x * scale, y * scale) for x, y in GRID8_XY)
        return _base_offsets(values, current).unsqueeze(1)
    if config.affinity_mode is AffinityMode.STATIC:
        _, total = _static_offsets(config, inputs, current)
        return total.unsqueeze(1)
    if inputs.offsets is None:
        raise ValueError("OFFSET neighbor mode requires offsets")
    offsets = inputs.offsets.to(device=current.device, dtype=torch.float32)
    expected = (
        current.shape[0],
        config.iterations,
        config.num_neighbors,
        2,
        current.shape[2],
        current.shape[3],
    )
    if tuple(offsets.shape) != expected:
        raise ValueError(
            f"per-iteration offsets must have shape {expected}, got {tuple(offsets.shape)}"
        )
    selected = offsets[:, :, :, [1, 0]] if config.offset_mode is OffsetMode.RESIDUAL_YX else offsets
    if config.offset_mode is OffsetMode.ABSOLUTE_XY:
        return selected
    if not config.base_offsets_xy:
        raise ValueError("residual offsets require base_offsets_xy")
    return _base_offsets(config.base_offsets_xy, current).unsqueeze(1) + selected


def _generic_affinity(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
) -> torch.Tensor:
    if config.affinity_mode is AffinityMode.STATIC:
        return _neighbor_map(
            inputs.affinity,
            current,
            config.num_neighbors,
            "affinity",
        ).unsqueeze(1)
    tensor = inputs.affinity.to(device=current.device, dtype=torch.float32)
    expected = (
        current.shape[0],
        config.iterations,
        config.num_neighbors,
        current.shape[2],
        current.shape[3],
    )
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"per-iteration affinity must have shape {expected}, got {tuple(tensor.shape)}"
        )
    return tensor


def _pretransform_affinity(
    affinity: torch.Tensor,
    config: SPNConfig,
) -> torch.Tensor:
    if config.normalization not in {NormalizationMode.TC, NormalizationMode.TGASS}:
        return affinity
    scale = (
        float(config.num_neighbors)
        if config.normalization is NormalizationMode.TC
        else config.affinity_gamma * float(config.num_neighbors) + 1.0e-8
    )
    return torch.tanh(affinity / config.tanh_temperature) / scale


def _normalize_affinity(
    affinity: torch.Tensor,
    config: SPNConfig,
) -> torch.Tensor:
    if config.normalization is NormalizationMode.SOFTMAX:
        return torch.softmax(affinity, dim=2)
    if config.normalization is NormalizationMode.TC:
        return affinity
    if config.normalization is NormalizationMode.DYSPN_NLPM:
        raise ValueError("DYSPN_NLPM requires the named NLPM adapter")
    denominator = torch.sum(torch.abs(affinity), dim=2, keepdim=True) + config.eps
    if config.normalization in {NormalizationMode.ASS, NormalizationMode.TGASS}:
        denominator = torch.where(
            denominator < 1.0,
            torch.ones_like(denominator),
            denominator,
        )
    return affinity / denominator


def _sample_confidence_by_step(
    confidence: torch.Tensor,
    offsets: torch.Tensor,
    config: SPNConfig,
) -> torch.Tensor:
    batch, steps, neighbors, _, height, width = offsets.shape
    source = confidence[:, None].expand(batch, steps, 1, height, width)
    sampled = sample_neighbors(
        source.reshape(batch * steps, 1, height, width),
        offsets.reshape(batch * steps, neighbors, 2, height, width).detach(),
        config.sampling_mode,
        config.padding_mode,
        align_corners=config.align_corners,
    )
    return sampled[:, :, 0].reshape(batch, steps, neighbors, height, width)


def compile_generic_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    if config.affinity_layout is not AffinityLayout.TARGET:
        raise ValueError("generic plan accepts only target-layout affinity")
    if config.anchor_mode is AnchorMode.INITIAL_CURRENT:
        raise ValueError("generic INITIAL_CURRENT requires the named NLPM adapter")
    offsets = _generic_offsets(config, inputs, current)
    affinity = _pretransform_affinity(
        _generic_affinity(config, inputs, current),
        config,
    )
    if config.neighbor_confidence is NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR:
        if inputs.confidence is None:
            raise ValueError("neighbor confidence sampling requires confidence")
        confidence = as_nchw(inputs.confidence, "confidence", device=current.device)
        if confidence.shape != (current.shape[0], 1, current.shape[2], current.shape[3]):
            raise ValueError("confidence must have shape [B,1,H,W]")
        affinity = affinity * _sample_confidence_by_step(confidence, offsets, config)
    affinity = _normalize_affinity(affinity, config)
    residual = 1.0 - torch.sum(affinity, dim=2, keepdim=True)
    zeros = torch.zeros_like(residual)
    current_affinity = residual if config.anchor_mode is AnchorMode.CURRENT else zeros
    initial_affinity = residual if config.anchor_mode is AnchorMode.INITIAL else zeros

    pre_gate = pre_value = post_gate = post_value = None
    if config.sparse_fusion is not SparseFusionMode.NONE:
        sparse, mask = _sparse_and_mask(inputs, current)
        if config.sparse_fusion is SparseFusionMode.HARD_PRE:
            pre_gate, pre_value = mask.unsqueeze(1), sparse.unsqueeze(1)
        elif config.sparse_fusion is SparseFusionMode.HARD_POST:
            post_gate, post_value = mask.unsqueeze(1), sparse.unsqueeze(1)
        elif config.sparse_fusion is SparseFusionMode.CSPN_CODE_POST:
            post_gate, post_value = mask.unsqueeze(1), initial.unsqueeze(1)
        elif config.sparse_fusion is SparseFusionMode.SOFT_POST:
            if inputs.confidence is None:
                raise ValueError("SOFT_POST requires confidence")
            confidence = as_nchw(inputs.confidence, "confidence", device=current.device)
            if confidence.shape != current.shape:
                raise ValueError("confidence must have the same shape as current")
            post_gate = (confidence * mask).unsqueeze(1)
            post_value = sparse.unsqueeze(1)
    return _make_plan(
        config=config,
        current=current,
        offsets_xy=offsets,
        neighbor_affinity=affinity.unsqueeze(3),
        current_affinity=current_affinity,
        initial_affinity=initial_affinity,
        pre_gate=pre_gate,
        pre_value=pre_value,
        post_gate=post_gate,
        post_value=post_value,
        reduction_mode=ReductionMode.TORCH_SUM,
    )
