from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class SRAMConfig:
    capacity_bytes: int
    banks: int
    ports_per_bank: int = 1
    word_bytes: int = 2
    read_latency: int = 1
    write_latency: int = 1


@dataclass(frozen=True)
class DMAConfig:
    channels: int = 2
    bandwidth_bytes_per_cycle: float = 32.0
    setup_cycles: int = 12


@dataclass(frozen=True)
class TensorEngineConfig:
    rows: int = 64
    cols: int = 64
    pipeline_depth: int = 2
    act_sram_bytes: int = 256 * 1024
    weight_sram_bytes: int = 512 * 1024
    psum_sram_bytes: int = 256 * 1024
    double_buffer: bool = True
    act_bytes: int = 1
    weight_bytes: int = 1
    psum_bytes: int = 4


@dataclass(frozen=True)
class SPNEngineConfig:
    agu_lanes: int = 16
    gather_lanes: int = 32
    interp_lanes: int = 16
    affinity_lanes: int = 16
    agu_latency: int = 2
    interp_latency: int = 3
    affinity_latency: int = 2
    packet_pixels: int = 32
    neighbors: int = 8
    metadata_sram: SRAMConfig = field(
        default_factory=lambda: SRAMConfig(768 * 1024, banks=16, word_bytes=2)
    )
    state_sram: SRAMConfig = field(
        default_factory=lambda: SRAMConfig(64 * 1024, banks=16, word_bytes=2)
    )


@dataclass(frozen=True)
class NPUConfig:
    frequency_hz: float = 1.0e9
    tensor: TensorEngineConfig = field(default_factory=TensorEngineConfig)
    spn: SPNEngineConfig = field(default_factory=SPNEngineConfig)
    dma: DMAConfig = field(default_factory=DMAConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NPUConfig":
        tensor = TensorEngineConfig(**data.get("tensor", {}))
        dma = DMAConfig(**data.get("dma", {}))
        spn_raw = dict(data.get("spn", {}))
        if "metadata_sram" in spn_raw:
            spn_raw["metadata_sram"] = SRAMConfig(**spn_raw["metadata_sram"])
        if "state_sram" in spn_raw:
            spn_raw["state_sram"] = SRAMConfig(**spn_raw["state_sram"])
        spn = SPNEngineConfig(**spn_raw)
        return NPUConfig(
            frequency_hz=float(data.get("frequency_hz", 1.0e9)),
            tensor=tensor,
            spn=spn,
            dma=dma,
        )

    @staticmethod
    def load(path: str | Path) -> "NPUConfig":
        with open(path, "r", encoding="utf-8") as f:
            return NPUConfig.from_dict(json.load(f))
