from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .config import DMAConfig, TensorEngineConfig
from .resources import DMAEngine, SingleResource
from .stats import TensorStats
from .workload import Conv2DOp, ConvTile


class TensorBackend(Protocol):
    def conv_compute_cycles(self, m: int, n: int, k: int) -> int:
        ...

    def active_mac_cycles(self, m: int, n: int, k: int) -> int:
        ...


class AnalyticalSystolicBackend:
    """Output-stationary 2-D systolic-array cycle model."""

    def __init__(self, config: TensorEngineConfig):
        self.config = config

    def conv_compute_cycles(self, m: int, n: int, k: int) -> int:
        cycles = 0
        for m0 in range(0, m, self.config.rows):
            mr = min(self.config.rows, m - m0)
            for n0 in range(0, n, self.config.cols):
                nr = min(self.config.cols, n - n0)
                cycles += k + mr + nr - 2 + self.config.pipeline_depth
        return cycles

    def active_mac_cycles(self, m: int, n: int, k: int) -> int:
        macs = m * n * k
        return math.ceil(macs / (self.config.rows * self.config.cols))


@dataclass(frozen=True)
class ConvMapping:
    tile_oh: int = 16
    tile_ow: int = 16
    tile_cin: int = 32
    tile_cout: int = 32


class TensorEngine:
    def __init__(self, config: TensorEngineConfig, dma_config: DMAConfig, backend: TensorBackend | None = None):
        self.config = config
        self.dma = DMAEngine(dma_config)
        self.compute = SingleResource("tensor")
        self.backend = backend or AnalyticalSystolicBackend(config)

    def _check_tile(self, op: Conv2DOp, tile: ConvTile) -> None:
        in_h = (tile.out_h - 1) * op.stride + op.kernel
        in_w = (tile.out_w - 1) * op.stride + op.kernel
        act = in_h * in_w * tile.cin * self.config.act_bytes
        weight = op.kernel * op.kernel * tile.cin * tile.cout * self.config.weight_bytes
        psum = tile.out_h * tile.out_w * tile.cout * self.config.psum_bytes
        if act > self.config.act_sram_bytes:
            raise ValueError(f"activation tile requires {act} B > ABUF")
        if weight > self.config.weight_sram_bytes:
            raise ValueError(f"weight tile requires {weight} B > WBUF")
        if psum > self.config.psum_sram_bytes:
            raise ValueError(f"psum tile requires {psum} B > PBUF")

    def run_conv(self, op: Conv2DOp, mapping: ConvMapping, ready_cycle: int = 0) -> TensorStats:
        stats = TensorStats(macs=op.macs)
        buffer_free = [ready_cycle, ready_cycle] if self.config.double_buffer else [ready_cycle]
        final_end = ready_cycle
        tile_index = 0

        for oh0 in range(0, op.out_h, mapping.tile_oh):
            toh = min(mapping.tile_oh, op.out_h - oh0)
            for ow0 in range(0, op.out_w, mapping.tile_ow):
                tow = min(mapping.tile_ow, op.out_w - ow0)
                for co0 in range(0, op.cout, mapping.tile_cout):
                    tco = min(mapping.tile_cout, op.cout - co0)
                    last_compute_end = ready_cycle
                    for ci0 in range(0, op.cin, mapping.tile_cin):
                        tci = min(mapping.tile_cin, op.cin - ci0)
                        tile = ConvTile(toh, tow, tci, tco)
                        self._check_tile(op, tile)
                        in_h = (toh - 1) * op.stride + op.kernel
                        in_w = (tow - 1) * op.stride + op.kernel
                        act_bytes = in_h * in_w * tci * self.config.act_bytes
                        weight_bytes = op.kernel * op.kernel * tci * tco * self.config.weight_bytes
                        stats.act_bytes += act_bytes
                        stats.weight_bytes += weight_bytes

                        slot = tile_index % len(buffer_free)
                        load_ready = max(ready_cycle, buffer_free[slot])
                        _, act_end, _ = self.dma.read(load_ready, act_bytes)
                        _, weight_end, _ = self.dma.read(load_ready, weight_bytes)
                        load_end = max(act_end, weight_end)

                        m = toh * tow
                        n = tco
                        k = op.kernel * op.kernel * tci
                        compute_cycles = self.backend.conv_compute_cycles(m, n, k)
                        _, compute_end = self.compute.schedule(max(load_end, last_compute_end), compute_cycles)
                        stats.compute_cycles += compute_cycles
                        stats.array_active_mac_cycles += self.backend.active_mac_cycles(m, n, k)
                        last_compute_end = compute_end
                        buffer_free[slot] = compute_end
                        tile_index += 1
                        stats.tiles += 1

                    output_bytes = toh * tow * tco * self.config.act_bytes
                    stats.output_bytes += output_bytes
                    _, store_end, _ = self.dma.write(last_compute_end, output_bytes)
                    final_end = max(final_end, store_end)

        stats.cycles = final_end - ready_cycle
        stats.dma_read_cycles = self.dma.read_busy_cycles
        stats.dma_write_cycles = self.dma.write_busy_cycles
        return stats
