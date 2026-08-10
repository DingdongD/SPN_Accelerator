from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conv2DOp:
    name: str
    out_h: int
    out_w: int
    cin: int
    cout: int
    kernel: int = 3
    stride: int = 1
    padding: int = 1

    @property
    def macs(self) -> int:
        return (
            self.out_h
            * self.out_w
            * self.cin
            * self.cout
            * self.kernel
            * self.kernel
        )


@dataclass(frozen=True)
class ConvTile:
    out_h: int
    out_w: int
    cin: int
    cout: int


@dataclass(frozen=True)
class Prop2DOp:
    name: str
    height: int
    width: int
    neighbors: int = 8
    steps: int = 12
    mode: str = "DEFORM_BILINEAR"
    state_bytes: int = 2
    offset_bytes: int = 2
    affinity_bytes: int = 2
    max_offset: float = 2.0

    @property
    def pixels(self) -> int:
        return self.height * self.width

    @property
    def metadata_bytes(self) -> int:
        offset = self.pixels * self.neighbors * 2 * self.offset_bytes
        affinity = self.pixels * self.neighbors * self.affinity_bytes
        return offset + affinity

    @property
    def pingpong_bytes(self) -> int:
        return 2 * self.pixels * self.state_bytes
