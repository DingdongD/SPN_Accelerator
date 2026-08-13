"""Internal post-parameter-generation tensor contract for SPN adapters."""

from dataclasses import dataclass

import torch


@dataclass
class DecodedSPNParameters:
    current: torch.Tensor
    initial: torch.Tensor
    raw_affinity: torch.Tensor
    residual_offsets_yx: torch.Tensor | None = None
    confidence_probability: torch.Tensor | None = None
    confidence_logits: torch.Tensor | None = None
    sparse_depth: torch.Tensor | None = None
    sparse_mask: torch.Tensor | None = None
    attention_logits: torch.Tensor | None = None
    affinity_scale: torch.Tensor | None = None
