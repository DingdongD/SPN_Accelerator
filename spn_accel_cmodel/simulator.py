from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import NPUConfig
from .spn_engine import OffsetPattern, SPNEngine
from .stats import SimulationStats
from .tensor_engine import ConvMapping, TensorBackend, TensorEngine
from .workload import Conv2DOp, Prop2DOp


@dataclass(frozen=True)
class Workload:
    convs: Tuple[Conv2DOp, ...] = ()
    propagation: Tuple[Prop2DOp, ...] = ()


class NPUSimulator:
    def __init__(self, config: NPUConfig, tensor_backend: TensorBackend | None = None):
        self.config = config
        self.tensor_backend = tensor_backend

    def run(self, workload: Workload, conv_mapping: ConvMapping | None = None, offset_pattern: OffsetPattern | None = None) -> SimulationStats:
        stats = SimulationStats()
        current_cycle = 0
        conv_mapping = conv_mapping or ConvMapping()

        for op in workload.convs:
            engine = TensorEngine(self.config.tensor, self.config.dma, backend=self.tensor_backend)
            result = engine.run_conv(op, conv_mapping, ready_cycle=current_cycle)
            stats.tensor[op.name] = result
            current_cycle += result.cycles

        for op in workload.propagation:
            engine = SPNEngine(self.config.spn)
            result = engine.run(op, offset_pattern, ready_cycle=current_cycle)
            stats.spn[op.name] = result
            current_cycle += result.cycles

        stats.total_cycles = current_cycle
        stats.metadata.update({"frequency_hz": self.config.frequency_hz, "latency_ms": current_cycle / self.config.frequency_hz * 1e3})
        return stats


def completionformer_dec2_nlspn_workload(prop_steps: int = 12) -> Workload:
    return Workload(
        convs=(
            Conv2DOp("dec2_upconv_160x32", 128, 128, 160, 32, 3),
            Conv2DOp("dec2_block_conv0_32x32", 128, 128, 32, 32, 3),
            Conv2DOp("dec2_block_conv1_32x32", 128, 128, 32, 32, 3),
        ),
        propagation=(Prop2DOp("nlspn_prop", 128, 128, neighbors=8, steps=prop_steps),),
    )
