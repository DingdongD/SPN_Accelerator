from .config import NPUConfig
from .simulator import NPUSimulator, Workload, completionformer_dec2_nlspn_workload
from .spn_engine import OffsetPattern
from .trace import ArrayOffsetProvider
from .tensor_engine import ConvMapping


_TORCH_EXPORTS = {
    "NormalizationMode",
    "SPNConfig",
    "SPNInputs",
    "SPNTrace",
    "UnifiedSPN",
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
    "SPNConfig",
    "SPNInputs",
    "SPNTrace",
    "UnifiedSPN",
]
