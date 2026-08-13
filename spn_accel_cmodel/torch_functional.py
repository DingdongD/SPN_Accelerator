"""Unified FP32 Torch primitives for SPN propagation.

Only propagation is modeled here.  Networks that predict initial depth,
offsets, affinities, confidence, or attention stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborMode(Enum):
    GRID = auto()
    DILATED = auto()
    OFFSET = auto()


class SamplingMode(Enum):
    INTEGER = auto()
    BILINEAR = auto()


class PaddingMode(Enum):
    ZEROS = auto()
    BORDER = auto()


class OffsetMode(Enum):
    ABSOLUTE_XY = auto()
    RESIDUAL_XY = auto()
    RESIDUAL_YX = auto()


class AffinityMode(Enum):
    STATIC = auto()
    PER_ITERATION = auto()


class NormalizationMode(Enum):
    AS = auto()
    ASS = auto()
    TC = auto()
    TGASS = auto()
    SOFTMAX = auto()
    DYSPN_NLPM = auto()

    ABS_SUM = AS
    ABS_SUM_FLOOR = ASS


class AnchorMode(Enum):
    NONE = auto()
    INITIAL = auto()
    CURRENT = auto()
    INITIAL_CURRENT = auto()


class NeighborConfidenceMode(Enum):
    NONE = auto()
    SAMPLE_AT_NEIGHBOR = auto()


class SparseFusionMode(Enum):
    NONE = auto()
    CSPN_CODE_POST = auto()
    HARD_PRE = auto()
    HARD_POST = auto()
    SOFT_POST = auto()


class AffinityLayout(Enum):
    TARGET = auto()
    CSPN_SHIFTED_SOURCE = auto()


class SPNProfile(Enum):
    GENERIC = auto()
    CSPN = auto()
    NLSPN = auto()
    COMPLETIONFORMER = auto()
    DYSPN = auto()
    DYSPN_NLPM = auto()


_GRID8_XY = (
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
)

_DYSPN_BASE_XY = {
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


@dataclass(frozen=True)
class SPNConfig:
    iterations: int
    num_neighbors: int
    neighbor_mode: NeighborMode
    sampling_mode: SamplingMode
    padding_mode: PaddingMode
    offset_mode: OffsetMode
    affinity_mode: AffinityMode
    normalization: NormalizationMode
    anchor_mode: AnchorMode
    neighbor_confidence: NeighborConfidenceMode = NeighborConfidenceMode.NONE
    sparse_fusion: SparseFusionMode = SparseFusionMode.NONE
    affinity_layout: AffinityLayout = AffinityLayout.TARGET
    profile: SPNProfile = SPNProfile.GENERIC
    dilation: int = 1
    eps: float = 1.0e-4
    tanh_temperature: float = 1.0
    affinity_gamma: float = 0.5
    align_corners: bool = True
    legacy_confidence_offsets: bool = False
    base_offsets_xy: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")
        if self.num_neighbors < 1:
            raise ValueError("num_neighbors must be at least one")
        if self.dilation < 1:
            raise ValueError("dilation must be at least one")
        if self.tanh_temperature <= 0.0:
            raise ValueError("tanh_temperature must be positive")
        if self.affinity_gamma <= 0.0:
            raise ValueError("affinity_gamma must be positive")
        if self.base_offsets_xy and len(self.base_offsets_xy) != self.num_neighbors:
            raise ValueError("base_offsets_xy length must equal num_neighbors")

    @classmethod
    def cspn(
        cls,
        iterations: int = 24,
        *,
        preserve_code_mask: bool = False,
    ) -> "SPNConfig":
        return cls(
            iterations=iterations,
            num_neighbors=8,
            neighbor_mode=NeighborMode.GRID,
            sampling_mode=SamplingMode.INTEGER,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.STATIC,
            normalization=NormalizationMode.AS,
            anchor_mode=AnchorMode.INITIAL,
            sparse_fusion=(
                SparseFusionMode.CSPN_CODE_POST
                if preserve_code_mask
                else SparseFusionMode.NONE
            ),
            affinity_layout=AffinityLayout.CSPN_SHIFTED_SOURCE,
            profile=SPNProfile.CSPN,
            align_corners=True,
            base_offsets_xy=_GRID8_XY,
        )

    @classmethod
    def nlspn(
        cls,
        iterations: int = 18,
        *,
        normalization: NormalizationMode = NormalizationMode.TGASS,
        confidence: bool = True,
        preserve_input: bool = False,
        affinity_gamma: float = 0.5,
        legacy_confidence_offsets: bool = False,
    ) -> "SPNConfig":
        if normalization not in {
            NormalizationMode.AS,
            NormalizationMode.ASS,
            NormalizationMode.TC,
            NormalizationMode.TGASS,
        }:
            raise ValueError("NLSPN normalization must be AS, ASS, TC, or TGASS")
        return cls(
            iterations=iterations,
            num_neighbors=8,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.RESIDUAL_YX,
            affinity_mode=AffinityMode.STATIC,
            normalization=normalization,
            anchor_mode=AnchorMode.CURRENT,
            neighbor_confidence=(
                NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR
                if confidence
                else NeighborConfidenceMode.NONE
            ),
            sparse_fusion=(
                SparseFusionMode.HARD_PRE if preserve_input else SparseFusionMode.NONE
            ),
            profile=SPNProfile.NLSPN,
            tanh_temperature=1.0,
            affinity_gamma=affinity_gamma,
            align_corners=True,
            legacy_confidence_offsets=legacy_confidence_offsets,
            base_offsets_xy=_GRID8_XY,
        )

    @classmethod
    def completionformer(
        cls,
        iterations: int = 6,
        *,
        normalization: NormalizationMode = NormalizationMode.TGASS,
        confidence: bool = True,
        preserve_input: bool = False,
        affinity_gamma: float = 0.5,
        legacy_confidence_offsets: bool = False,
    ) -> "SPNConfig":
        cfg = cls.nlspn(
            iterations,
            normalization=normalization,
            confidence=confidence,
            preserve_input=preserve_input,
            affinity_gamma=affinity_gamma,
            legacy_confidence_offsets=legacy_confidence_offsets,
        )
        return cls(
            **{
                **cfg.__dict__,
                "profile": SPNProfile.COMPLETIONFORMER,
                "tanh_temperature": 100.0,
            }
        )

    @classmethod
    def dyspn(
        cls,
        iterations: int = 6,
        *,
        num_neighbors: int = 5,
    ) -> "SPNConfig":
        if num_neighbors not in _DYSPN_BASE_XY:
            raise ValueError("DySPN supports K in {1, 3, 5, 9}")
        return cls(
            iterations=iterations,
            num_neighbors=num_neighbors,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.RESIDUAL_YX,
            affinity_mode=AffinityMode.PER_ITERATION,
            normalization=NormalizationMode.SOFTMAX,
            anchor_mode=AnchorMode.NONE,
            sparse_fusion=SparseFusionMode.SOFT_POST,
            profile=SPNProfile.DYSPN,
            align_corners=False,
            base_offsets_xy=_DYSPN_BASE_XY[num_neighbors],
        )

    @classmethod
    def dyspn_nlpm(cls, iterations: int = 6) -> "SPNConfig":
        return cls(
            iterations=iterations,
            num_neighbors=48,
            neighbor_mode=NeighborMode.DILATED,
            sampling_mode=SamplingMode.INTEGER,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.PER_ITERATION,
            normalization=NormalizationMode.DYSPN_NLPM,
            anchor_mode=AnchorMode.INITIAL_CURRENT,
            sparse_fusion=SparseFusionMode.SOFT_POST,
            profile=SPNProfile.DYSPN_NLPM,
            align_corners=True,
        )


@dataclass
class SPNInputs:
    current: torch.Tensor
    affinity: torch.Tensor
    initial: torch.Tensor | None = None
    offsets: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    sparse_depth: torch.Tensor | None = None
    sparse_mask: torch.Tensor | None = None
    attention: torch.Tensor | None = None


@dataclass
class SPNTrace:
    outputs: list[torch.Tensor] = field(default_factory=list)
    candidates: list[torch.Tensor] = field(default_factory=list)
    offsets: list[torch.Tensor] = field(default_factory=list)
    neighbor_affinities: list[torch.Tensor] = field(default_factory=list)
    current_affinities: list[torch.Tensor | None] = field(default_factory=list)
    initial_affinities: list[torch.Tensor | None] = field(default_factory=list)


class UnifiedSPN(nn.Module):
    def __init__(self, config: SPNConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        inputs: SPNInputs,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SPNTrace]:
        current = _as_nchw(inputs.current, "current")
        initial = (
            _as_nchw(inputs.initial, "initial")
            if inputs.initial is not None
            else current
        )
        if initial.shape != current.shape:
            raise ValueError("initial and current must have identical shapes")

        if self.config.profile is SPNProfile.CSPN:
            state, trace = self._forward_cspn(inputs, current, initial)
        elif self.config.profile in {SPNProfile.NLSPN, SPNProfile.COMPLETIONFORMER}:
            state, trace = self._forward_nlspn(inputs, current, initial)
        else:
            state, trace = self._forward_general(inputs, current, initial)
        return (state, trace) if return_trace else state

    def _forward_cspn(
        self,
        inputs: SPNInputs,
        current: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, SPNTrace]:
        cfg = self.config
        guidance = _as_neighbor_map(
            inputs.affinity,
            current,
            cfg.num_neighbors,
            "affinity",
        )
        shifted_guidance = _cspn_shifted_channels(guidance)
        denom = torch.sum(torch.abs(shifted_guidance), dim=1, keepdim=True)
        normalized = shifted_guidance / denom
        neighbor_sum = torch.sum(normalized, dim=1)[:, :, 1:-1, 1:-1]
        initial_affinity = 1.0 - neighbor_sum

        trace = SPNTrace()
        state = current
        sparse = None
        mask = None
        if cfg.sparse_fusion is SparseFusionMode.CSPN_CODE_POST:
            sparse = _required_sparse(inputs, current)
            mask = sparse.sign()

        for _ in range(cfg.iterations):
            shifted_state = _cspn_shifted_channels(state)
            neighbor = torch.sum(normalized * shifted_state, dim=1)
            neighbor = neighbor[:, :, 1:-1, 1:-1]
            candidate = neighbor + initial_affinity * initial
            if mask is not None:
                state = (1.0 - mask) * candidate + mask * initial
            else:
                state = candidate
            trace.candidates.append(candidate)
            trace.outputs.append(state)
            trace.offsets.append(base_offsets_tensor(cfg.base_offsets_xy, current))
            trace.neighbor_affinities.append(
                normalized[:, :, 0, 1:-1, 1:-1]
            )
            trace.current_affinities.append(None)
            trace.initial_affinities.append(initial_affinity)
        return state, trace

    def _forward_nlspn(
        self,
        inputs: SPNInputs,
        current: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, SPNTrace]:
        cfg = self.config
        affinity = _as_neighbor_map(
            inputs.affinity,
            current,
            cfg.num_neighbors,
            "affinity",
        )
        residual_xy, total_xy = _static_residual_offsets(inputs, current, cfg)

        if cfg.normalization in {NormalizationMode.TC, NormalizationMode.TGASS}:
            scale = (
                float(cfg.num_neighbors)
                if cfg.normalization is NormalizationMode.TC
                else cfg.affinity_gamma * float(cfg.num_neighbors) + 1.0e-8
            )
            affinity = torch.tanh(affinity / cfg.tanh_temperature) / scale

        if cfg.neighbor_confidence is NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR:
            if inputs.confidence is None:
                raise ValueError("neighbor confidence sampling requires confidence")
            confidence = _as_nchw(inputs.confidence, "confidence")
            if (
                confidence.shape[1] != 1
                or confidence.shape[0] != current.shape[0]
                or confidence.shape[2:] != current.shape[2:]
            ):
                raise ValueError("confidence must have shape [B,1,H,W]")
            confidence_offsets = total_xy if cfg.legacy_confidence_offsets else residual_xy
            sampled_confidence = sample_neighbors(
                confidence,
                confidence_offsets.detach(),
                cfg.sampling_mode,
                cfg.padding_mode,
                align_corners=cfg.align_corners,
            )[:, :, 0]
            affinity = affinity * sampled_confidence

        denom = torch.sum(torch.abs(affinity), dim=1, keepdim=True) + cfg.eps
        if cfg.normalization in {NormalizationMode.ASS, NormalizationMode.TGASS}:
            denom = torch.where(denom < 1.0, torch.ones_like(denom), denom)
        if cfg.normalization in {
            NormalizationMode.AS,
            NormalizationMode.ASS,
            NormalizationMode.TGASS,
        }:
            affinity = affinity / denom
        current_affinity = 1.0 - torch.sum(affinity, dim=1, keepdim=True)

        sparse = None
        mask = None
        if cfg.sparse_fusion is SparseFusionMode.HARD_PRE:
            sparse = _required_sparse(inputs, current)
            mask = _sparse_mask(inputs, sparse, current)

        trace = SPNTrace()
        state = current
        for _ in range(cfg.iterations):
            if mask is not None:
                state = (1.0 - mask) * state + mask * sparse
            samples = sample_neighbors(
                state,
                total_xy,
                cfg.sampling_mode,
                cfg.padding_mode,
                align_corners=cfg.align_corners,
            )
            candidate = torch.sum(samples * affinity[:, :, None], dim=1)
            candidate = candidate + current_affinity * state
            state = candidate
            trace.candidates.append(candidate)
            trace.outputs.append(state)
            trace.offsets.append(total_xy)
            trace.neighbor_affinities.append(affinity)
            trace.current_affinities.append(current_affinity)
            trace.initial_affinities.append(None)
        return state, trace

    def _forward_general(
        self,
        inputs: SPNInputs,
        current: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, SPNTrace]:
        if self.config.profile is SPNProfile.DYSPN:
            return self._forward_dyspn(inputs, current)
        if self.config.profile is SPNProfile.DYSPN_NLPM:
            return self._forward_dyspn_nlpm(inputs, current, initial)
        raise NotImplementedError(f"unsupported generic profile {self.config.profile.name}")

    def _forward_dyspn(
        self,
        inputs: SPNInputs,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, SPNTrace]:
        cfg = self.config
        logits = _as_iteration_neighbor_map(inputs.affinity, current, cfg)
        if inputs.offsets is None:
            raise ValueError("DySPN requires per-iteration offsets")
        residual_yx = inputs.offsets.to(device=current.device, dtype=torch.float32)
        expected_offsets = (
            current.shape[0],
            cfg.iterations,
            cfg.num_neighbors,
            2,
            current.shape[2],
            current.shape[3],
        )
        if tuple(residual_yx.shape) != expected_offsets:
            raise ValueError(
                f"DySPN offsets must have shape {expected_offsets}, got "
                f"{tuple(residual_yx.shape)}"
            )
        residual_xy = residual_yx[:, :, :, [1, 0]]
        base = torch.tensor(
            cfg.base_offsets_xy,
            device=current.device,
            dtype=torch.float32,
        ).view(1, 1, cfg.num_neighbors, 2, 1, 1)
        total_xy = base + residual_xy
        affinities = torch.softmax(logits, dim=2)

        sparse = _required_sparse(inputs, current)
        if inputs.confidence is None:
            raise ValueError("DySPN soft-post fusion requires confidence logits")
        confidence_logits = _as_nchw(inputs.confidence, "confidence").to(
            device=current.device
        )
        if confidence_logits.shape != current.shape:
            raise ValueError("DySPN confidence must have the same shape as current")
        confidence = torch.sigmoid(confidence_logits) * sparse.sign()

        state = current
        trace = SPNTrace()
        for iteration in range(cfg.iterations):
            offsets = total_xy[:, iteration]
            affinity = affinities[:, iteration]
            samples = sample_neighbors(
                state,
                offsets,
                cfg.sampling_mode,
                cfg.padding_mode,
                align_corners=cfg.align_corners,
            )
            candidate = torch.sum(samples * affinity[:, :, None], dim=1)
            state = (1.0 - confidence) * candidate + confidence * sparse
            trace.candidates.append(candidate)
            trace.outputs.append(state)
            trace.offsets.append(offsets)
            trace.neighbor_affinities.append(affinity)
            trace.current_affinities.append(None)
            trace.initial_affinities.append(None)
        return state, trace

    def _forward_dyspn_nlpm(
        self,
        inputs: SPNInputs,
        current: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, SPNTrace]:
        cfg = self.config
        guidance = _as_neighbor_map(inputs.affinity, current, 48, "affinity")
        if inputs.attention is None:
            raise ValueError("DySPN NLPM requires per-iteration attention")
        attention_logits = inputs.attention.to(device=current.device, dtype=torch.float32)
        expected_attention = (
            current.shape[0],
            cfg.iterations,
            4,
            current.shape[2],
            current.shape[3],
        )
        if tuple(attention_logits.shape) != expected_attention:
            raise ValueError(
                f"DySPN NLPM attention must have shape {expected_attention}, got "
                f"{tuple(attention_logits.shape)}"
            )
        attention = torch.sigmoid(attention_logits)
        sparse = _required_sparse(inputs, current)
        if inputs.confidence is None:
            raise ValueError("DySPN NLPM requires sparse confidence")
        confidence = _as_nchw(inputs.confidence, "confidence").to(device=current.device)
        if confidence.shape != current.shape:
            raise ValueError("DySPN NLPM confidence must have the same shape as current")
        sparse_confidence = sparse.sign() * confidence

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

        state = current
        trace = SPNTrace()
        for iteration in range(cfg.iterations):
            attn = attention[:, iteration]
            denominator = torch.sum(attn * abs_sums, dim=1, keepdim=True) + cfg.eps
            guided = state * guidance
            neighbor = (
                attn[:, 0:1] * _edge_sum(guided[:, 0:8], 3)
                + attn[:, 1:2] * _edge_sum(guided[:, 8:24], 5)
                + attn[:, 2:3] * _edge_sum(guided[:, 24:48], 7)
                + attn[:, 3:4] * state
            )
            initial_numerator = denominator - torch.sum(
                attn * signed_sums,
                dim=1,
                keepdim=True,
            )
            candidate = (neighbor + initial_numerator * initial) / denominator
            state = (
                (1.0 - sparse_confidence) * candidate
                + sparse_confidence * sparse
            )
            trace.candidates.append(candidate)
            trace.outputs.append(state)
            trace.offsets.append(torch.empty(0, device=current.device))
            trace.neighbor_affinities.append(attn[:, 0:3] / denominator)
            trace.current_affinities.append(attn[:, 3:4] / denominator)
            trace.initial_affinities.append(initial_numerator / denominator)
        return state, trace


def _as_nchw(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"{name} must be NCHW, got {tuple(value.shape)}")
    return value.to(dtype=torch.float32)


def _as_neighbor_map(
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


def _as_iteration_neighbor_map(
    value: torch.Tensor,
    current: torch.Tensor,
    cfg: SPNConfig,
) -> torch.Tensor:
    tensor = value.to(device=current.device, dtype=torch.float32)
    expected = (
        current.shape[0],
        cfg.iterations,
        cfg.num_neighbors,
        current.shape[2],
        current.shape[3],
    )
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"per-iteration affinity must have shape {expected}, got {tuple(tensor.shape)}"
        )
    return tensor


def _static_residual_offsets(
    inputs: SPNInputs,
    current: torch.Tensor,
    cfg: SPNConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.offsets is None:
        raise ValueError("offset propagation requires offsets")
    offsets = inputs.offsets.to(device=current.device, dtype=torch.float32)
    expected = (
        current.shape[0],
        cfg.num_neighbors,
        2,
        current.shape[2],
        current.shape[3],
    )
    if tuple(offsets.shape) != expected:
        raise ValueError(f"offsets must have shape {expected}, got {tuple(offsets.shape)}")
    if cfg.offset_mode is OffsetMode.RESIDUAL_YX:
        residual_xy = offsets[:, :, [1, 0]]
    elif cfg.offset_mode is OffsetMode.RESIDUAL_XY:
        residual_xy = offsets
    else:
        residual_xy = offsets
    if cfg.offset_mode is OffsetMode.ABSOLUTE_XY:
        return residual_xy, residual_xy
    base = base_offsets_tensor(cfg.base_offsets_xy, current)
    return residual_xy, base + residual_xy


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
    channels = [value] * 8 if value.shape[1] == 1 else list(torch.chunk(value, 8, dim=1))
    return torch.stack(
        [F.pad(channel, padding) for channel, padding in zip(channels, paddings)],
        dim=1,
    )


def _edge_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
    edge_indices = []
    for row in range(kernel):
        for column in range(kernel):
            if row in {0, kernel - 1} or column in {0, kernel - 1}:
                edge_indices.append(row * kernel + column)
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


def _required_sparse(inputs: SPNInputs, current: torch.Tensor) -> torch.Tensor:
    if inputs.sparse_depth is None:
        raise ValueError("configured sparse fusion requires sparse_depth")
    sparse = _as_nchw(inputs.sparse_depth, "sparse_depth").to(device=current.device)
    if sparse.shape != current.shape:
        raise ValueError("sparse_depth must have the same shape as current")
    return sparse


def _sparse_mask(
    inputs: SPNInputs,
    sparse: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    if inputs.sparse_mask is None:
        return (sparse > 0.0).to(dtype=torch.float32)
    mask = _as_nchw(inputs.sparse_mask, "sparse_mask").to(device=current.device)
    if mask.shape != current.shape:
        raise ValueError("sparse_mask must have the same shape as current")
    return (mask > 0.0).to(dtype=torch.float32)


def _normalized_coordinate(
    coordinate: torch.Tensor,
    size: int,
    align_corners: bool,
) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(coordinate)
    if align_corners:
        return 2.0 * coordinate / float(size - 1) - 1.0
    return 2.0 * (coordinate + 0.5) / float(size) - 1.0


def sample_neighbors(
    state: torch.Tensor,
    offsets_xy: torch.Tensor,
    sampling_mode: SamplingMode,
    padding_mode: PaddingMode,
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Sample `[B,K,2,H,W]` absolute `(dx,dy)` displacements.

    Returns `[B,K,C,H,W]`.  Coordinates are never clamped; padding behavior is
    delegated to ``grid_sample``.
    """

    if state.ndim != 4:
        raise ValueError(f"state must be NCHW, got {tuple(state.shape)}")
    if offsets_xy.ndim != 5 or offsets_xy.shape[2] != 2:
        raise ValueError(
            "offsets_xy must have shape [B,K,2,H,W], got "
            f"{tuple(offsets_xy.shape)}"
        )

    state = state.to(dtype=torch.float32)
    offsets_xy = offsets_xy.to(device=state.device, dtype=torch.float32)
    b, c, h, w = state.shape
    bo, k, _, ho, wo = offsets_xy.shape
    if (bo, ho, wo) != (b, h, w):
        raise ValueError(
            "offset batch/spatial dimensions must match state: "
            f"state={tuple(state.shape)}, offsets={tuple(offsets_xy.shape)}"
        )

    yy, xx = torch.meshgrid(
        torch.arange(h, device=state.device, dtype=torch.float32),
        torch.arange(w, device=state.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xx.view(1, 1, h, w) + offsets_xy[:, :, 0]
    sample_y = yy.view(1, 1, h, w) + offsets_xy[:, :, 1]
    grid_x = _normalized_coordinate(sample_x, w, align_corners)
    grid_y = _normalized_coordinate(sample_y, h, align_corners)
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(b * k, h, w, 2)
    source = state[:, None].expand(b, k, c, h, w).reshape(b * k, c, h, w)

    mode = "nearest" if sampling_mode is SamplingMode.INTEGER else "bilinear"
    torch_padding = "zeros" if padding_mode is PaddingMode.ZEROS else "border"
    sampled = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=torch_padding,
        align_corners=align_corners,
    )
    return sampled.reshape(b, k, c, h, w)


def base_offsets_tensor(
    values: Sequence[tuple[float, float]],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return an expanded `[B,K,2,H,W]` `(dx,dy)` base stencil."""

    b, _, h, w = reference.shape
    base = torch.tensor(values, device=reference.device, dtype=torch.float32)
    return base.view(1, len(values), 2, 1, 1).expand(b, -1, -1, h, w)
