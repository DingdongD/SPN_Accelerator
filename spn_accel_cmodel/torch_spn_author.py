"""Author-level parameter decoders for unified SPN propagation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .torch_spn_core import as_nchw
from .torch_spn_decoded import DecodedSPNParameters
from .torch_spn_types import (
    CompletionFormerRawInputs,
    DySPNRawInputs,
    NLSPNRawInputs,
    NormalizationMode,
    SPNConfig,
    SPNProfile,
)


def _guidance(
    value: torch.Tensor,
    current: torch.Tensor,
    channels: int,
    name: str,
) -> torch.Tensor:
    tensor = value.to(device=current.device, dtype=torch.float32)
    expected = (current.shape[0], channels, current.shape[2], current.shape[3])
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    return tensor


class _OfficialParameterLoader:
    _official_parameter_names: tuple[str, ...] = ()

    def load_official_parameters(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        prefix: str,
    ) -> None:
        if not prefix:
            raise ValueError("prefix must be explicit and non-empty")
        local = dict(self.named_parameters())
        with torch.no_grad():
            for name in self._official_parameter_names:
                key = prefix + name
                if key not in state_dict:
                    raise ValueError(f"missing official parameter {key}")
                source = state_dict[key]
                target = local[name]
                if source.shape != target.shape:
                    raise ValueError(
                        f"{key} must have shape {tuple(target.shape)}, "
                        f"got {tuple(source.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))


class _NLSPNFamilyAuthor(nn.Module, _OfficialParameterLoader):
    input_type: type
    profile: SPNProfile
    initial_field: str
    sparse_field: str
    _official_parameter_names = (
        "conv_offset_aff.weight",
        "conv_offset_aff.bias",
        "aff_scale_const",
    )

    def __init__(self, config: SPNConfig):
        super().__init__()
        if config.profile is not self.profile:
            raise ValueError(f"{type(self).__name__} requires {self.profile.name} config")
        self.config = config
        neighbors = config.num_neighbors
        self.conv_offset_aff = nn.Conv2d(
            neighbors,
            3 * neighbors,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()
        if config.normalization is NormalizationMode.TC:
            scale = float(neighbors)
        elif config.normalization is NormalizationMode.TGASS:
            scale = config.affinity_gamma * float(neighbors)
        else:
            scale = 1.0
        self.aff_scale_const = nn.Parameter(torch.tensor([scale]))
        self.aff_scale_const.requires_grad = (
            config.normalization is NormalizationMode.TGASS
        )

    def decode(self, inputs) -> DecodedSPNParameters:
        if not isinstance(inputs, self.input_type):
            raise TypeError(
                f"{self.profile.name} expects {self.input_type.__name__}, "
                f"got {type(inputs).__name__}"
            )
        initial_value = getattr(inputs, self.initial_field)
        guidance_value = inputs.guidance
        confidence = inputs.confidence_probability
        sparse = getattr(inputs, self.sparse_field)
        current = as_nchw(initial_value, "initial depth")
        guidance = _guidance(
            guidance_value,
            current,
            self.config.num_neighbors,
            "guidance",
        )
        raw = self.conv_offset_aff(guidance)
        o1, o2, affinity = torch.chunk(raw, 3, dim=1)
        offsets = torch.cat((o1, o2), dim=1).view(
            current.shape[0],
            self.config.num_neighbors,
            2,
            current.shape[2],
            current.shape[3],
        )
        return DecodedSPNParameters(
            current=current,
            initial=current,
            raw_affinity=affinity,
            residual_offsets_yx=offsets,
            confidence_probability=confidence,
            sparse_depth=sparse,
            affinity_scale=self.aff_scale_const,
        )


class NLSPNAuthor(_NLSPNFamilyAuthor):
    input_type = NLSPNRawInputs
    profile = SPNProfile.NLSPN
    initial_field = "feat_init"
    sparse_field = "feat_fix"


class CompletionFormerAuthor(_NLSPNFamilyAuthor):
    input_type = CompletionFormerRawInputs
    profile = SPNProfile.COMPLETIONFORMER
    initial_field = "pred_init"
    sparse_field = "sparse_depth"


class DySPNAuthor(nn.Module, _OfficialParameterLoader):
    input_type = DySPNRawInputs
    profile = SPNProfile.DYSPN
    _official_parameter_names = (
        "conv_offset_aff.weight",
        "conv_offset_aff.bias",
    )

    def __init__(self, config: SPNConfig):
        super().__init__()
        if config.profile is not self.profile:
            raise ValueError("DySPNAuthor requires DYSPN config")
        self.config = config
        channels = config.iterations * config.num_neighbors
        self.conv_offset_aff = nn.Conv2d(
            channels,
            3 * channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv_offset_aff.weight.data.zero_()
        self.conv_offset_aff.bias.data.zero_()

    def decode(self, inputs: DySPNRawInputs) -> DecodedSPNParameters:
        if not isinstance(inputs, self.input_type):
            raise TypeError(
                f"DYSPN expects DySPNRawInputs, got {type(inputs).__name__}"
            )
        current = as_nchw(inputs.feat_init, "feat_init")
        channels = self.config.iterations * self.config.num_neighbors
        guide = _guidance(inputs.guide, current, channels, "guide")
        sparse = as_nchw(inputs.sparse_depth, "sparse_depth", device=current.device)
        confidence = as_nchw(
            inputs.confidence_logits,
            "confidence_logits",
            device=current.device,
        )
        if sparse.shape != current.shape:
            raise ValueError("sparse_depth must have the same shape as feat_init")
        if confidence.shape != current.shape:
            raise ValueError("confidence_logits must have the same shape as feat_init")
        raw = self.conv_offset_aff(guide)
        offset_flat, affinity_flat = torch.split(
            raw,
            [2 * channels, channels],
            dim=1,
        )
        b, _, h, w = current.shape
        offsets = offset_flat.view(
            b,
            self.config.iterations,
            self.config.num_neighbors,
            2,
            h,
            w,
        )
        affinity = affinity_flat.view(
            b,
            self.config.iterations,
            self.config.num_neighbors,
            h,
            w,
        )
        return DecodedSPNParameters(
            current=current,
            initial=current,
            raw_affinity=affinity,
            residual_offsets_yx=offsets,
            confidence_logits=confidence,
            sparse_depth=sparse,
        )
