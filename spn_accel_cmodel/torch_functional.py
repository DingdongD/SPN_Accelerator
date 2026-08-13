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
    ) -> "SPNConfig":
        cfg = cls.nlspn(
            iterations,
            normalization=normalization,
            confidence=confidence,
            preserve_input=preserve_input,
            affinity_gamma=affinity_gamma,
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
        raise NotImplementedError("propagation profiles are implemented in the next TDD task")


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
