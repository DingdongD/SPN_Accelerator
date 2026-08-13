"""Shared public types for Torch SPN plan compilation and propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch


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


class ReductionMode(Enum):
    TORCH_SUM = auto()
    SEQUENTIAL = auto()
    GROUPED = auto()


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
        if self.normalization is NormalizationMode.SOFTMAX and self.anchor_mode is not AnchorMode.NONE:
            raise ValueError("SOFTMAX requires AnchorMode.NONE")
        if self.normalization is NormalizationMode.DYSPN_NLPM and self.profile is not SPNProfile.DYSPN_NLPM:
            raise ValueError("DYSPN_NLPM normalization requires the named NLPM profile")
        if self.anchor_mode is AnchorMode.INITIAL_CURRENT and self.profile is not SPNProfile.DYSPN_NLPM:
            raise ValueError("INITIAL_CURRENT requires the named NLPM profile")

    @classmethod
    def cspn(cls, iterations: int = 24, *, preserve_code_mask: bool = False) -> "SPNConfig":
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
            sparse_fusion=SparseFusionMode.CSPN_CODE_POST if preserve_code_mask else SparseFusionMode.NONE,
            affinity_layout=AffinityLayout.CSPN_SHIFTED_SOURCE,
            profile=SPNProfile.CSPN,
            align_corners=True,
            base_offsets_xy=GRID8_XY,
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
            neighbor_confidence=NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR if confidence else NeighborConfidenceMode.NONE,
            sparse_fusion=SparseFusionMode.HARD_PRE if preserve_input else SparseFusionMode.NONE,
            profile=SPNProfile.NLSPN,
            tanh_temperature=1.0,
            affinity_gamma=affinity_gamma,
            align_corners=True,
            legacy_confidence_offsets=legacy_confidence_offsets,
            base_offsets_xy=GRID8_XY,
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
        config = cls.nlspn(
            iterations,
            normalization=normalization,
            confidence=confidence,
            preserve_input=preserve_input,
            affinity_gamma=affinity_gamma,
            legacy_confidence_offsets=legacy_confidence_offsets,
        )
        return cls(**{**config.__dict__, "profile": SPNProfile.COMPLETIONFORMER, "tanh_temperature": 100.0})

    @classmethod
    def dyspn(cls, iterations: int = 6, *, num_neighbors: int = 5) -> "SPNConfig":
        if num_neighbors not in DYSPN_BASE_XY:
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
            base_offsets_xy=DYSPN_BASE_XY[num_neighbors],
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


@dataclass(frozen=True)
class CanonicalSPNPlan:
    iterations: int
    offsets_xy: torch.Tensor
    neighbor_affinity: torch.Tensor
    current_affinity: torch.Tensor
    initial_affinity: torch.Tensor
    pre_fusion_gate: torch.Tensor
    pre_fusion_value: torch.Tensor
    post_fusion_gate: torch.Tensor
    post_fusion_value: torch.Tensor
    sampling_mode: SamplingMode
    padding_mode: PaddingMode
    align_corners: bool
    reduction_mode: ReductionMode
    reduction_groups: tuple[tuple[int, int], ...]
    group_scale: torch.Tensor


@dataclass
class SPNTrace:
    pre_fused_states: list[torch.Tensor] = field(default_factory=list)
    candidates: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    offsets: list[torch.Tensor] = field(default_factory=list)
    neighbor_affinities: list[torch.Tensor] = field(default_factory=list)
    current_affinities: list[torch.Tensor | None] = field(default_factory=list)
    initial_affinities: list[torch.Tensor | None] = field(default_factory=list)
    pre_fusion_gates: list[torch.Tensor] = field(default_factory=list)
    post_fusion_gates: list[torch.Tensor] = field(default_factory=list)
    group_scales: list[torch.Tensor] = field(default_factory=list)
