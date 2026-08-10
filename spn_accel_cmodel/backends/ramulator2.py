from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MemoryRequest:
    cycle: int
    address: int
    is_write: bool = False


class Ramulator2Backend:
    """Boundary for replacing analytical DMA timing with Ramulator2."""

    def run(self, requests: Sequence[MemoryRequest]) -> Sequence[int]:
        raise NotImplementedError("Ramulator2 bridge is intentionally external to the dependency-free MVP")
