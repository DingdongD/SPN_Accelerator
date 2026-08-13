"""Public Torch API for canonical FP32 SPN propagation.

Prediction heads remain outside this boundary. Named model profiles compile
author-specific metadata into one :class:`CanonicalSPNPlan`; every profile then
executes the same :func:`propagate_canonical` recurrence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .torch_spn_adapters import (
    compile_completionformer_plan,
    compile_cspn_plan,
    compile_dyspn_nlpm_plan,
    compile_dyspn_plan,
    compile_generic_plan,
    compile_nlspn_plan,
)
from .torch_spn_core import (
    as_nchw,
    propagate_canonical,
    sample_neighbors,
    validate_plan,
)
from .torch_spn_types import (
    AffinityLayout,
    AffinityMode,
    AnchorMode,
    CanonicalSPNPlan,
    NeighborConfidenceMode,
    NeighborMode,
    NormalizationMode,
    OffsetMode,
    PaddingMode,
    ReductionMode,
    SPNConfig,
    SPNInputs,
    SPNProfile,
    SPNTrace,
    SamplingMode,
    SparseFusionMode,
)


_PLAN_COMPILERS = {
    SPNProfile.CSPN: compile_cspn_plan,
    SPNProfile.NLSPN: compile_nlspn_plan,
    SPNProfile.COMPLETIONFORMER: compile_completionformer_plan,
    SPNProfile.DYSPN: compile_dyspn_plan,
    SPNProfile.DYSPN_NLPM: compile_dyspn_nlpm_plan,
    SPNProfile.GENERIC: compile_generic_plan,
}


class UnifiedSPN(nn.Module):
    """Compile one model profile and execute the canonical recurrence once."""

    def __init__(self, config: SPNConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        inputs: SPNInputs,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SPNTrace]:
        current = as_nchw(inputs.current, "current")
        initial = (
            current
            if inputs.initial is None
            else as_nchw(inputs.initial, "initial", device=current.device)
        )
        if initial.shape != current.shape:
            raise ValueError("initial and current must have identical shapes")
        compiler = _PLAN_COMPILERS[self.config.profile]
        plan = compiler(self.config, inputs, current, initial)
        return propagate_canonical(current, initial, plan, return_trace)


__all__ = [
    "AffinityLayout",
    "AffinityMode",
    "AnchorMode",
    "CanonicalSPNPlan",
    "NeighborConfidenceMode",
    "NeighborMode",
    "NormalizationMode",
    "OffsetMode",
    "PaddingMode",
    "ReductionMode",
    "SPNConfig",
    "SPNInputs",
    "SPNProfile",
    "SPNTrace",
    "SamplingMode",
    "SparseFusionMode",
    "UnifiedSPN",
    "as_nchw",
    "compile_completionformer_plan",
    "compile_cspn_plan",
    "compile_dyspn_nlpm_plan",
    "compile_dyspn_plan",
    "compile_generic_plan",
    "compile_nlspn_plan",
    "propagate_canonical",
    "sample_neighbors",
    "validate_plan",
]
