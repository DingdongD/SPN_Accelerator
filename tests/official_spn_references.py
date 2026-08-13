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

DYSPN_BASE_XY = {
    1: ((0.0, 0.0),),
    3: ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    5: ((0.0, -1.0), (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    9: (
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ),
}


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
    grid_align_corners = align_corners and h > 1 and w > 1
    if grid_align_corners:
        gx = 2.0 * sx / (w - 1) - 1.0
        gy = 2.0 * sy / (h - 1) - 1.0
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
        align_corners=grid_align_corners,
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
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, object]]:
    gate = _cspn_shifted_channels(guidance)
    gate = gate / torch.sum(torch.abs(gate), dim=1, keepdim=True)
    gate_sum = torch.sum(gate, dim=1)[:, :, 1:-1, 1:-1]
    state = initial
    outputs = []
    candidates = []
    pre_fused_states = []
    mask = sparse_depth.sign() if sparse_depth is not None else None
    for _ in range(iterations):
        pre_fused_states.append(state)
        shifted_state = _cspn_shifted_channels(state)
        neighbor = torch.sum(gate * shifted_state, dim=1)[:, :, 1:-1, 1:-1]
        state = neighbor + (1.0 - gate_sum) * initial
        candidates.append(state)
        if mask is not None:
            state = (1.0 - mask) * state + mask * initial
        outputs.append(state)
    batch, _, height, width = initial.shape
    target_offsets = tuple((-x, -y) for x, y in GRID8_XY)
    offsets = torch.tensor(target_offsets, device=initial.device).view(1, 8, 2, 1, 1)
    offsets = offsets.expand(batch, 8, 2, height, width)
    metadata = {
        "candidates": candidates,
        "offsets": offsets,
        "neighbor_affinity": gate[:, :, 0, 1:-1, 1:-1],
        "initial_affinity": 1.0 - gate_sum,
        "pre_fused_states": pre_fused_states,
        "post_fusion_gate": torch.zeros_like(initial) if mask is None else mask,
    }
    return state, outputs, metadata


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
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, object]]:
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
    pre_fused_states = []
    mask = None
    if preserve_input:
        if sparse_depth is None:
            raise ValueError("preserve_input requires sparse_depth")
        mask = (sparse_depth > 0.0).to(dtype=torch.float32)
    for _ in range(iterations):
        if mask is not None:
            state = (1.0 - mask) * state + mask * sparse_depth
        pre_fused_states.append(state)
        samples = _sample_absolute_xy(state, total_xy, align_corners=True)
        state = torch.sum(samples * affinity[:, :, None], dim=1) + center * state
        outputs.append(state)
    return state, outputs, affinity, center, {
        "offsets": total_xy,
        "candidates": outputs,
        "pre_fused_states": pre_fused_states,
        "pre_fusion_gate": torch.zeros_like(center) if mask is None else mask,
    }


def reference_dyspn(
    initial: torch.Tensor,
    residual_yx: torch.Tensor,
    logits: torch.Tensor,
    sparse_depth: torch.Tensor,
    confidence_logits: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict[str, object]]:
    b, iterations, neighbors, _, h, w = residual_yx.shape
    base = torch.tensor(
        DYSPN_BASE_XY[neighbors],
        dtype=torch.float32,
        device=initial.device,
    ).view(1, 1, neighbors, 2, 1, 1)
    residual_xy = residual_yx[:, :, :, [1, 0]]
    total_xy = base + residual_xy
    affinities = torch.softmax(logits, dim=2)
    confidence = torch.sigmoid(confidence_logits) * sparse_depth.sign()

    state = initial
    outputs = []
    candidates = []
    pre_fused_states = []
    effective_affinities = []
    for iteration in range(iterations):
        pre_fused_states.append(state)
        samples = _sample_absolute_xy(
            state,
            total_xy[:, iteration],
            align_corners=False,
        )
        candidate = torch.zeros_like(state)
        for neighbor in range(neighbors):
            candidate = (
                candidate
                + samples[:, neighbor]
                * affinities[:, iteration, neighbor : neighbor + 1]
            )
        state = (1.0 - confidence) * candidate + confidence * sparse_depth
        candidates.append(candidate)
        outputs.append(state)
        effective_affinities.append(affinities[:, iteration])
    return state, outputs, effective_affinities, {
        "candidates": candidates,
        "offsets": [total_xy[:, iteration] for iteration in range(iterations)],
        "pre_fused_states": pre_fused_states,
        "post_fusion_gate": confidence,
    }


def _edge_weight(kernel: int, device: torch.device) -> torch.Tensor:
    edge_indices = []
    for row in range(kernel):
        for column in range(kernel):
            if row in {0, kernel - 1} or column in {0, kernel - 1}:
                edge_indices.append(row * kernel + column)
    weight = torch.zeros(
        (1, len(edge_indices), kernel, kernel),
        dtype=torch.float32,
        device=device,
    )
    for channel, edge_index in enumerate(edge_indices):
        weight[
            0,
            channel,
            -(edge_index // kernel) - 1,
            (-edge_index) % kernel - 1,
        ] = 1.0
    return weight


def _edge_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.conv2d(value, _edge_weight(kernel, value.device), padding=kernel // 2)


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


def reference_dyspn_nlpm(
    initial: torch.Tensor,
    guidance: torch.Tensor,
    attention_logits: torch.Tensor,
    sparse_depth: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict[str, object]]:
    iterations = attention_logits.shape[1]
    batch, _, height, width = initial.shape
    abs_sums = torch.cat(
        (
            _edge_sum(torch.abs(guidance[:, 0:8]), 3),
            _edge_sum(torch.abs(guidance[:, 8:24]), 5),
            _edge_sum(torch.abs(guidance[:, 24:48]), 7),
            torch.ones((batch, 1, height, width), device=initial.device),
        ),
        dim=1,
    )
    signed_sums = torch.cat(
        (
            _edge_sum(guidance[:, 0:8], 3),
            _edge_sum(guidance[:, 8:24], 5),
            _edge_sum(guidance[:, 24:48], 7),
            torch.ones((batch, 1, height, width), device=initial.device),
        ),
        dim=1,
    )
    attention = torch.sigmoid(attention_logits)
    sparse_confidence = sparse_depth.sign() * confidence
    denominator_all = (
        torch.sum(attention * abs_sums[:, None], dim=2, keepdim=True) + 1.0e-4
    )
    initial_numerator_all = denominator_all - torch.sum(
        attention * signed_sums[:, None],
        dim=2,
        keepdim=True,
    )
    offsets_xy = _edge_offsets_xy(3) + _edge_offsets_xy(5) + _edge_offsets_xy(7)
    shifted_guidance = _shift_channels_to_target(guidance, offsets_xy)
    offsets = torch.tensor(
        offsets_xy,
        dtype=torch.float32,
        device=initial.device,
    ).view(1, 48, 2, 1, 1)
    offsets = offsets.expand(batch, 48, 2, height, width)

    state = initial
    candidates = []
    outputs = []
    pre_fused_states = []
    neighbor_affinities = []
    current_affinities = []
    initial_affinities = []
    for iteration in range(iterations):
        pre_fused_states.append(state)
        attn = attention[:, iteration]
        denominator = torch.sum(attn * abs_sums, dim=1, keepdim=True) + 1.0e-4
        guided = state * guidance
        neighbor = (
            attn[:, 0:1] * _edge_sum(guided[:, 0:8], 3)
            + attn[:, 1:2] * _edge_sum(guided[:, 8:24], 5)
            + attn[:, 2:3] * _edge_sum(guided[:, 24:48], 7)
            + attn[:, 3:4] * state
        )
        initial_weight = denominator - torch.sum(
            attn * signed_sums,
            dim=1,
            keepdim=True,
        )
        candidate = (neighbor + initial_weight * initial) / denominator
        neighbor_affinities.append(attn[:, 0:3] / denominator)
        current_affinities.append(attn[:, 3:4] / denominator)
        initial_affinities.append(initial_weight / denominator)
        candidates.append(candidate)
        state = (
            (1.0 - sparse_confidence) * candidate
            + sparse_confidence * sparse_depth
        )
        outputs.append(state)
    group_scale = attention[:, :, 0:3].unsqueeze(3) / denominator_all.unsqueeze(3)
    effective_neighbor_affinity = torch.cat(
        (
            shifted_guidance[:, None, 0:8, None] * group_scale[:, :, 0:1],
            shifted_guidance[:, None, 8:24, None] * group_scale[:, :, 1:2],
            shifted_guidance[:, None, 24:48, None] * group_scale[:, :, 2:3],
        ),
        dim=2,
    )
    return state, candidates, outputs, {
        "neighbor_affinities": neighbor_affinities,
        "current_affinities": current_affinities,
        "initial_affinities": initial_affinities,
        "shifted_guidance": shifted_guidance,
        "offsets": offsets,
        "group_scale": group_scale,
        "effective_neighbor_affinity": effective_neighbor_affinity,
        "current_affinity": attention[:, :, 3:4] / denominator_all,
        "initial_affinity": initial_numerator_all / denominator_all,
        "pre_fused_states": pre_fused_states,
        "post_fusion_gate": sparse_confidence,
    }
