from .config import NPUConfig
from .simulator import NPUSimulator, Workload, completionformer_dec2_nlspn_workload
from .spn_engine import OffsetPattern
from .trace import ArrayOffsetProvider
from .tensor_engine import ConvMapping


_TORCH_EXPORTS = {
    "CanonicalSPNPlan",
    "CSPNRawInputs",
    "CSPNAuthor",
    "CompletionFormerRawInputs",
    "CompletionFormerAuthor",
    "DySPNNLPMRawInputs",
    "DySPNNLPMAuthor",
    "DySPNRawInputs",
    "DySPNAuthor",
    "NLSPNRawInputs",
    "NLSPNAuthor",
    "NormalizationMode",
    "ReductionMode",
    "SPNConfig",
    "SPNTrace",
    "UnifiedSPN",
    "compile_completionformer_plan",
    "compile_cspn_plan",
    "compile_dyspn_nlpm_plan",
    "compile_dyspn_plan",
    "compile_nlspn_plan",
    "propagate_canonical",
}


def __getattr__(name: str):
    """Load the optional Torch functional model only when it is requested."""

    if name in _TORCH_EXPORTS:
        from . import torch_functional

        return getattr(torch_functional, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NPUConfig",
    "NPUSimulator",
    "Workload",
    "completionformer_dec2_nlspn_workload",
    "OffsetPattern",
    "ArrayOffsetProvider",
    "ConvMapping",
    "NormalizationMode",
    "CanonicalSPNPlan",
    "CSPNRawInputs",
    "CSPNAuthor",
    "CompletionFormerRawInputs",
    "CompletionFormerAuthor",
    "DySPNNLPMRawInputs",
    "DySPNNLPMAuthor",
    "DySPNRawInputs",
    "DySPNAuthor",
    "NLSPNRawInputs",
    "NLSPNAuthor",
    "ReductionMode",
    "SPNConfig",
    "SPNTrace",
    "UnifiedSPN",
    "compile_completionformer_plan",
    "compile_cspn_plan",
    "compile_dyspn_nlpm_plan",
    "compile_dyspn_plan",
    "compile_nlspn_plan",
    "propagate_canonical",
]
