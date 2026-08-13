"""Public Torch API for canonical FP32 SPN propagation.

Prediction heads remain outside this boundary. Named model profiles compile
author-specific metadata into one :class:`CanonicalSPNPlan`; every profile then
executes the same :func:`propagate_canonical` recurrence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .torch_spn_author import (
    AUTHOR_BUILDERS,
    CSPNAuthor,
    CompletionFormerAuthor,
    DySPNNLPMAuthor,
    DySPNAuthor,
    NLSPNAuthor,
)
from .torch_spn_adapters import (
    compile_completionformer_plan,
    compile_cspn_plan,
    compile_dyspn_nlpm_plan,
    compile_dyspn_plan,
    compile_nlspn_plan,
)
from .torch_spn_core import (
    propagate_canonical,
    sample_neighbors,
    validate_plan,
)
from .torch_spn_types import (
    AffinityLayout,
    AffinityMode,
    AnchorMode,
    CanonicalSPNPlan,
    CSPNRawInputs,
    CompletionFormerRawInputs,
    DySPNNLPMRawInputs,
    DySPNRawInputs,
    NeighborConfidenceMode,
    NeighborMode,
    NormalizationMode,
    OffsetMode,
    PaddingMode,
    ReductionMode,
    NLSPNRawInputs,
    SPNConfig,
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
}

AuthorInputs = (
    CSPNRawInputs
    | NLSPNRawInputs
    | CompletionFormerRawInputs
    | DySPNRawInputs
    | DySPNNLPMRawInputs
)


class UnifiedSPN(nn.Module):
    """Compile one model profile and execute the canonical recurrence once."""

    def __init__(self, config: SPNConfig):
        super().__init__()
        self.config = config
        self.author = AUTHOR_BUILDERS[config.profile](config)

    def forward(
        self,
        inputs: AuthorInputs,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SPNTrace]:
        decoded = self.author.decode(inputs)
        compiler = _PLAN_COMPILERS[self.config.profile]
        plan = compiler(self.config, decoded)
        return propagate_canonical(
            decoded.current,
            decoded.initial,
            plan,
            return_trace,
        )


__all__ = [
    "AffinityLayout",
    "AffinityMode",
    "AnchorMode",
    "CSPNRawInputs",
    "CSPNAuthor",
    "CanonicalSPNPlan",
    "CompletionFormerRawInputs",
    "CompletionFormerAuthor",
    "DySPNNLPMRawInputs",
    "DySPNNLPMAuthor",
    "DySPNRawInputs",
    "DySPNAuthor",
    "NeighborConfidenceMode",
    "NeighborMode",
    "NormalizationMode",
    "NLSPNRawInputs",
    "NLSPNAuthor",
    "OffsetMode",
    "PaddingMode",
    "ReductionMode",
    "SPNConfig",
    "SPNProfile",
    "SPNTrace",
    "SamplingMode",
    "SparseFusionMode",
    "UnifiedSPN",
    "compile_completionformer_plan",
    "compile_cspn_plan",
    "compile_dyspn_nlpm_plan",
    "compile_dyspn_plan",
    "compile_nlspn_plan",
    "propagate_canonical",
    "sample_neighbors",
    "validate_plan",
]
