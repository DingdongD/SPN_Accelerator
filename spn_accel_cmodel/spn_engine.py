from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterator, List, Sequence, Tuple

from .config import SPNEngineConfig
from .resources import BankedSRAM, SingleResource
from .stats import SPNStats
from .workload import Prop2DOp


NeighborOffset = Tuple[float, float]
DEFAULT_NEIGHBORS: Tuple[NeighborOffset, ...] = (
    (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
    (0.0, -1.0),                (0.0, 1.0),
    (1.0, -1.0),  (1.0, 0.0),  (1.0, 1.0),
)


@dataclass(frozen=True)
class OffsetPattern:
    kind: str = "zero"
    seed: int = 1
    amplitude: float = 0.75


class SPNTraceGenerator:
    def __init__(self, op: Prop2DOp, pattern: OffsetPattern):
        self.op = op
        self.pattern = pattern

    def _learned_delta(self, y: int, x: int, k: int) -> NeighborOffset:
        if self.pattern.kind == "zero":
            return 0.0, 0.0
        if self.pattern.kind == "random":
            key = (self.pattern.seed * 1000003 + y * 9176 + x * 131 + k * 17) & 0xFFFFFFFF
            rng = random.Random(key)
            a = self.pattern.amplitude
            return rng.uniform(-a, a), rng.uniform(-a, a)
        if self.pattern.kind == "checker":
            s = self.pattern.amplitude if ((x + y + k) & 1) == 0 else -self.pattern.amplitude
            return 0.5 * s, s
        raise ValueError(f"unknown offset pattern: {self.pattern.kind}")

    def _addr(self, y: int, x: int) -> int:
        y = min(max(y, 0), self.op.height - 1)
        x = min(max(x, 0), self.op.width - 1)
        return (y * self.op.width + x) * self.op.state_bytes

    def packet_addresses(self, y0: int, x0: int, count: int) -> Tuple[List[int], int]:
        addresses: List[int] = []
        samples = 0
        linear0 = y0 * self.op.width + x0
        for linear in range(linear0, min(linear0 + count, self.op.pixels)):
            y, x = divmod(linear, self.op.width)
            addresses.append(self._addr(y, x))
            for k in range(self.op.neighbors):
                base_dy, base_dx = DEFAULT_NEIGHBORS[k]
                ddy, ddx = self._learned_delta(y, x, k)
                sy = y + base_dy + ddy
                sx = x + base_dx + ddx
                fy = math.floor(sy)
                fx = math.floor(sx)
                addresses.extend((self._addr(fy, fx), self._addr(fy, fx + 1), self._addr(fy + 1, fx), self._addr(fy + 1, fx + 1)))
                samples += 1
        return addresses, samples


class SPNEngine:
    """Packetized AGU -> gather -> interpolate -> affinity/reduction cycle model."""

    def __init__(self, config: SPNEngineConfig):
        self.config = config

    def _groups(self, addresses: Sequence[int]) -> Iterator[List[int]]:
        lanes = self.config.gather_lanes
        for i in range(0, len(addresses), lanes):
            yield list(addresses[i : i + lanes])

    def run(self, op: Prop2DOp, pattern: OffsetPattern | None = None, ready_cycle: int = 0) -> SPNStats:
        if op.neighbors != self.config.neighbors:
            raise ValueError("current SPN engine expects configured neighbor count")
        if op.metadata_bytes > self.config.metadata_sram.capacity_bytes:
            raise ValueError(f"metadata requires {op.metadata_bytes} B > metadata SRAM; tiled-SPN mapping is required")
        if op.pingpong_bytes > self.config.state_sram.capacity_bytes:
            raise ValueError(f"ping-pong state requires {op.pingpong_bytes} B > state SRAM")

        initial_ready = ready_cycle
        pattern = pattern or OffsetPattern()
        trace = SPNTraceGenerator(op, pattern)
        agu = SingleResource("spn_agu")
        gather = SingleResource("spn_gather")
        interp = SingleResource("spn_interp")
        affinity = SingleResource("spn_affinity")
        state = BankedSRAM(self.config.state_sram)
        stats = SPNStats(iterations=op.steps)
        final_end = ready_cycle

        for _step in range(op.steps):
            for linear0 in range(0, op.pixels, self.config.packet_pixels):
                pixel_count = min(self.config.packet_pixels, op.pixels - linear0)
                y0, x0 = divmod(linear0, op.width)
                addresses, samples = trace.packet_addresses(y0, x0, pixel_count)
                agu_cycles = self.config.agu_latency + math.ceil(samples / self.config.agu_lanes)
                _, agu_end = agu.schedule(ready_cycle, agu_cycles)

                gather_start = max(agu_end, gather.available_cycle)
                service = state.service_read_groups(self._groups(addresses), gather_start)
                gather_duration = max(1, service.end_cycle - gather_start)
                _, gather_end = gather.schedule(gather_start, gather_duration)

                interp_cycles = self.config.interp_latency + math.ceil(samples / self.config.interp_lanes)
                _, interp_end = interp.schedule(gather_end, interp_cycles)
                affinity_ops = pixel_count * (op.neighbors + 1)
                aff_cycles = self.config.affinity_latency + math.ceil(affinity_ops / self.config.affinity_lanes)
                _, aff_end = affinity.schedule(interp_end, aff_cycles)

                stats.gather_reads += service.accesses
                stats.bank_conflict_stall_cycles += service.conflict_stall_cycles
                stats.state_reads += service.accesses
                stats.state_writes += pixel_count
                stats.packets += 1
                final_end = max(final_end, aff_end)
            ready_cycle = final_end

        stats.cycles = final_end - initial_ready
        stats.agu_busy_cycles = agu.busy_cycles
        stats.gather_busy_cycles = gather.busy_cycles
        stats.interp_busy_cycles = interp.busy_cycles
        stats.affinity_busy_cycles = affinity.busy_cycles
        return stats
