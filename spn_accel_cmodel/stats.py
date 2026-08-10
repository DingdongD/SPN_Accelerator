from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class TensorStats:
    cycles: int = 0
    compute_cycles: int = 0
    dma_read_cycles: int = 0
    dma_write_cycles: int = 0
    act_bytes: int = 0
    weight_bytes: int = 0
    output_bytes: int = 0
    macs: int = 0
    array_active_mac_cycles: int = 0
    tiles: int = 0

    @property
    def utilization(self) -> float:
        if self.compute_cycles <= 0:
            return 0.0
        return self.array_active_mac_cycles / max(1, self.compute_cycles)


@dataclass
class SPNStats:
    cycles: int = 0
    iterations: int = 0
    agu_busy_cycles: int = 0
    gather_busy_cycles: int = 0
    interp_busy_cycles: int = 0
    affinity_busy_cycles: int = 0
    gather_reads: int = 0
    bank_conflict_stall_cycles: int = 0
    state_reads: int = 0
    state_writes: int = 0
    packets: int = 0

    @property
    def bank_conflict_rate(self) -> float:
        denom = self.gather_busy_cycles
        return self.bank_conflict_stall_cycles / denom if denom else 0.0


@dataclass
class SimulationStats:
    total_cycles: int = 0
    tensor: Dict[str, TensorStats] = field(default_factory=dict)
    spn: Dict[str, SPNStats] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for value in out["tensor"].values():
            cycles = value.get("compute_cycles", 0)
            value["utilization"] = (
                value.get("array_active_mac_cycles", 0) / cycles if cycles else 0.0
            )
        for value in out["spn"].values():
            cycles = value.get("gather_busy_cycles", 0)
            value["bank_conflict_rate"] = (
                value.get("bank_conflict_stall_cycles", 0) / cycles if cycles else 0.0
            )
        return out
