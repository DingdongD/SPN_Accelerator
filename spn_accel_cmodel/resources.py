from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Tuple

from .config import DMAConfig, SRAMConfig


@dataclass
class SingleResource:
    name: str
    available_cycle: int = 0
    busy_cycles: int = 0

    def schedule(self, ready_cycle: int, duration: int) -> Tuple[int, int]:
        start = max(ready_cycle, self.available_cycle)
        end = start + max(0, duration)
        self.available_cycle = end
        self.busy_cycles += max(0, duration)
        return start, end


@dataclass
class MultiChannelResource:
    name: str
    channels: int
    available_cycles: List[int] = field(init=False)
    busy_cycles: int = 0

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        self.available_cycles = [0] * self.channels

    def schedule(self, ready_cycle: int, duration: int) -> Tuple[int, int, int]:
        idx = min(range(self.channels), key=self.available_cycles.__getitem__)
        start = max(ready_cycle, self.available_cycles[idx])
        end = start + max(0, duration)
        self.available_cycles[idx] = end
        self.busy_cycles += max(0, duration)
        return start, end, idx

    @property
    def available_cycle(self) -> int:
        return min(self.available_cycles)

    @property
    def drain_cycle(self) -> int:
        return max(self.available_cycles)


class DMAEngine(MultiChannelResource):
    def __init__(self, config: DMAConfig):
        super().__init__(name="dma", channels=config.channels)
        self.config = config
        self.read_bytes = 0
        self.write_bytes = 0
        self.read_busy_cycles = 0
        self.write_busy_cycles = 0

    def transfer_cycles(self, num_bytes: int) -> int:
        if num_bytes <= 0:
            return 0
        return self.config.setup_cycles + math.ceil(
            num_bytes / self.config.bandwidth_bytes_per_cycle
        )

    def read(self, ready_cycle: int, num_bytes: int) -> Tuple[int, int, int]:
        self.read_bytes += num_bytes
        duration = self.transfer_cycles(num_bytes)
        self.read_busy_cycles += duration
        return self.schedule(ready_cycle, duration)

    def write(self, ready_cycle: int, num_bytes: int) -> Tuple[int, int, int]:
        self.write_bytes += num_bytes
        duration = self.transfer_cycles(num_bytes)
        self.write_busy_cycles += duration
        return self.schedule(ready_cycle, duration)


@dataclass
class SRAMServiceResult:
    end_cycle: int
    service_cycles: int
    ideal_cycles: int
    conflict_stall_cycles: int
    accesses: int


class BankedSRAM:
    """Cycle-level bank/port service model for address traces.

    Requests are presented in issue groups. Each issue group represents scalar
    reads that the upstream gather stage would like to issue in one cycle.
    Per-bank queues persist across issue cycles, so hot-bank bursts create
    backpressure rather than being reduced to a single average-bandwidth term.
    """

    def __init__(self, config: SRAMConfig):
        self.config = config
        self.read_accesses = 0
        self.write_accesses = 0

    def bank(self, address: int) -> int:
        if address < 0:
            raise ValueError("address must be non-negative")
        word = address // self.config.word_bytes
        return word % self.config.banks

    def service_read_groups(
        self,
        groups: Iterable[List[int]],
        start_cycle: int,
    ) -> SRAMServiceResult:
        bank_available = [start_cycle] * self.config.banks
        issue_cycle = start_cycle
        accesses = 0
        ideal_finish = start_cycle

        for group in groups:
            if not group:
                issue_cycle += 1
                continue
            counts: Dict[int, int] = {}
            for addr in group:
                b = self.bank(addr)
                counts[b] = counts.get(b, 0) + 1
            accesses += len(group)
            ideal_group = math.ceil(
                len(group) / (self.config.banks * self.config.ports_per_bank)
            )
            ideal_finish = max(ideal_finish, issue_cycle + ideal_group)
            for bank_id, count in counts.items():
                slots = math.ceil(count / self.config.ports_per_bank)
                begin = max(issue_cycle, bank_available[bank_id])
                bank_available[bank_id] = begin + slots
            issue_cycle += 1

        data_ready = max(bank_available, default=start_cycle) + self.config.read_latency
        service_cycles = max(0, data_ready - start_cycle)
        ideal_cycles = max(0, ideal_finish + self.config.read_latency - start_cycle)
        conflict_stall = max(0, service_cycles - ideal_cycles)
        self.read_accesses += accesses
        return SRAMServiceResult(
            end_cycle=data_ready,
            service_cycles=service_cycles,
            ideal_cycles=ideal_cycles,
            conflict_stall_cycles=conflict_stall,
            accesses=accesses,
        )
