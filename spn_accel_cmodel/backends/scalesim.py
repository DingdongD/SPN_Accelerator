from __future__ import annotations

import shutil

from ..tensor_engine import TensorBackend


class ScaleSimBackend(TensorBackend):
    """Adapter contract for future SCALE-Sim integration."""

    def __init__(self, executable: str = "scalesim"):
        self.executable = executable
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable!r} not found. Install SCALE-Sim or use AnalyticalSystolicBackend.")

    def conv_compute_cycles(self, m: int, n: int, k: int) -> int:
        raise NotImplementedError("SCALE-Sim subprocess/config adapter is a validation hook in the MVP")

    def active_mac_cycles(self, m: int, n: int, k: int) -> int:
        raise NotImplementedError
