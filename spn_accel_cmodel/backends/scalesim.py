from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..config import TensorEngineConfig
from ..tensor_engine import TensorBackend


@dataclass(frozen=True)
class ScaleSimCase:
    """One GEMM microkernel used to cross-check tensor-array timing."""

    name: str
    m: int
    n: int
    k: int

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.m, self.n, self.k


@dataclass(frozen=True)
class ScaleSimConvCase:
    """One valid-convolution microkernel for cycle + traffic validation."""

    name: str
    ifmap_h: int
    ifmap_w: int
    filter_h: int
    filter_w: int
    channels: int
    filters: int
    stride: int = 1

    @property
    def out_h(self) -> int:
        return (self.ifmap_h - self.filter_h) // self.stride + 1

    @property
    def out_w(self) -> int:
        return (self.ifmap_w - self.filter_w) // self.stride + 1

    @property
    def gemm_key(self) -> Tuple[int, int, int]:
        return (
            self.out_h * self.out_w,
            self.filters,
            self.filter_h * self.filter_w * self.channels,
        )


@dataclass(frozen=True)
class ScaleSimComputeResult:
    layer_id: int
    total_cycles_including_prefetch: int
    total_cycles: int
    stall_cycles: int
    overall_util_pct: float
    mapping_efficiency_pct: float
    compute_util_pct: float


@dataclass(frozen=True)
class ScaleSimAccessResult:
    layer_id: int
    sram_ifmap_reads: int
    sram_filter_reads: int
    sram_ofmap_writes: int
    dram_ifmap_reads: int
    dram_filter_reads: int
    dram_ofmap_writes: int


@dataclass(frozen=True)
class ScaleSimCaseResult:
    case: ScaleSimCase | ScaleSimConvCase
    compute: ScaleSimComputeResult
    access: ScaleSimAccessResult | None = None


def _clean_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}


def _int_value(row: Mapping[str, str], key: str) -> int:
    value = row[key].strip()
    return int(float(value))


def _float_value(row: Mapping[str, str], key: str) -> float:
    return float(row[key].strip())


def parse_compute_report(path: str | Path) -> List[ScaleSimComputeResult]:
    """Parse SCALE-Sim v3 COMPUTE_REPORT.csv.

    The field names are taken from the current SCALE-Sim v3 report generator.
    Trailing empty CSV fields are tolerated.
    """

    results: List[ScaleSimComputeResult] = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = _clean_row(raw)
            if not row or not row.get("LayerID", ""):
                continue
            results.append(
                ScaleSimComputeResult(
                    layer_id=_int_value(row, "LayerID"),
                    total_cycles_including_prefetch=_int_value(
                        row, "Total Cycles (incl. prefetch)"
                    ),
                    total_cycles=_int_value(row, "Total Cycles"),
                    stall_cycles=_int_value(row, "Stall Cycles"),
                    overall_util_pct=_float_value(row, "Overall Util %"),
                    mapping_efficiency_pct=_float_value(row, "Mapping Efficiency %"),
                    compute_util_pct=_float_value(row, "Compute Util %"),
                )
            )
    return results


def parse_detailed_access_report(path: str | Path) -> List[ScaleSimAccessResult]:
    results: List[ScaleSimAccessResult] = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = _clean_row(raw)
            if not row or not row.get("LayerID", ""):
                continue
            results.append(
                ScaleSimAccessResult(
                    layer_id=_int_value(row, "LayerID"),
                    sram_ifmap_reads=_int_value(row, "SRAM IFMAP Reads"),
                    sram_filter_reads=_int_value(row, "SRAM Filter Reads"),
                    sram_ofmap_writes=_int_value(row, "SRAM OFMAP Writes"),
                    dram_ifmap_reads=_int_value(row, "DRAM IFMAP Reads"),
                    dram_filter_reads=_int_value(row, "DRAM Filter Reads"),
                    dram_ofmap_writes=_int_value(row, "DRAM OFMAP Writes"),
                )
            )
    return results


def write_gemm_topology(path: str | Path, cases: Sequence[ScaleSimCase]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer Name", "M", "N", "K", ""])
        for case in cases:
            writer.writerow([case.name, case.m, case.n, case.k, ""])


def write_conv_topology(path: str | Path, cases: Sequence[ScaleSimConvCase]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Layer name", "IFMAP Height", "IFMAP Width", "Filter Height",
            "Filter Width", "Channels", "Num Filter", "Strides", ""
        ])
        for case in cases:
            writer.writerow([
                case.name, case.ifmap_h, case.ifmap_w, case.filter_h, case.filter_w,
                case.channels, case.filters, case.stride, ""
            ])


def write_compat_layout(
    path: str | Path, cases: Sequence[ScaleSimCase | ScaleSimConvCase]
) -> None:
    """Write a layout file accepted by current SCALE-Sim.

    Custom layouts remain disabled in the generated config, so these values do
    not change the baseline dataflow; SCALE-Sim currently still requires a
    syntactically valid layout file on the CLI path.
    """

    header = [
        "Layer name",
        "IFMAP Height Intraline Factor",
        "IFMAP Width Intraline Factor",
        "Filter Height Intraline Factor",
        "Filter Width Intraline Factor",
        "Channel Intraline Factor",
        "Num Filter Intraline Factor",
        "IFMAP Height Intraline Order",
        "IFMAP Width Intraline Order",
        "Channel Intraline Order",
        "IFMAP Height Interline Order",
        "IFMAP Width Interline Order",
        "Channel Interline Order",
        "Num Filter Intraline Order",
        "Channel Intraline Order",
        "Filter Height Intraline Order",
        "Filter Width Intraline Order",
        "Num Filter Interline Order",
        "Channel Interline Order",
        "Filter Height Interline Order",
        "Filter Width Interline Order",
        "",
    ]
    # Matches the structure of SCALE-Sim's layouts/conv_nets/test.csv.  The
    # factors are inert when custom layout support is disabled.
    values = [2, 2, 1, 1, 16, 4, 0, 1, 2, 4, 5, 3, 3, 2, 1, 0, 4, 5, 6, 7, ""]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for case in cases:
            writer.writerow([case.name, *values])


def write_scalesim_config(
    path: str | Path,
    config: TensorEngineConfig,
    run_name: str = "spn_cmodel_validation",
    dataflow: str = "os",
) -> None:
    if dataflow not in {"os", "ws", "is"}:
        raise ValueError("SCALE-Sim dataflow must be os/ws/is")

    def kb(num_bytes: int) -> int:
        return max(1, math.ceil(num_bytes / 1024))

    text = f"""[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight:    {config.rows}
ArrayWidth:     {config.cols}
IfmapSramSzkB:  {kb(config.act_sram_bytes)}
FilterSramSzkB: {kb(config.weight_sram_bytes)}
OfmapSramSzkB:  {kb(config.psum_sram_bytes)}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 1000000
Dataflow : {dataflow}
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 1000000
IfmapSRAMBankNum: 1
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 1000000
FilterSRAMBankNum: 1
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
"""
    Path(path).write_text(text, encoding="utf-8")


class ScaleSimRunner:
    """Run SCALE-Sim v3 as an external validation golden.

    This is intentionally a validation runner rather than the default timing
    backend.  A validation sweep can be slow; its results can later be cached in
    a LUT and consumed by :class:`ScaleSimReportBackend`.
    """

    def __init__(
        self,
        tensor_config: TensorEngineConfig,
        python_executable: str = sys.executable,
        dataflow: str = "os",
    ):
        self.tensor_config = tensor_config
        self.python_executable = python_executable
        self.dataflow = dataflow

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("scalesim") is not None

    def _run_cases(
        self,
        cases: Sequence[ScaleSimCase | ScaleSimConvCase],
        input_type: str,
        output_root: str | Path | None,
        keep_inputs: bool,
    ) -> List[ScaleSimCaseResult]:
        if not cases:
            return []
        if not self.available():
            raise RuntimeError(
                "SCALE-Sim is not importable. Install scalesim-project/SCALE-Sim "
                "and rerun validation, or parse an existing report."
            )
        if input_type not in {"gemm", "conv"}:
            raise ValueError("input_type must be gemm or conv")

        owned_tmp = output_root is None
        tmp_ctx = tempfile.TemporaryDirectory(prefix="spn_scalesim_") if owned_tmp else None
        root = Path(tmp_ctx.name if tmp_ctx is not None else output_root)  # type: ignore[arg-type]
        root.mkdir(parents=True, exist_ok=True)
        run_name = "spn_cmodel_validation"
        cfg = root / "scalesim.cfg"
        topo = root / "microkernels.csv"
        layout = root / "layout.csv"
        logs = root / "outputs"
        logs.mkdir(parents=True, exist_ok=True)
        write_scalesim_config(cfg, self.tensor_config, run_name, self.dataflow)
        if input_type == "gemm":
            write_gemm_topology(topo, cases)  # type: ignore[arg-type]
        else:
            write_conv_topology(topo, cases)  # type: ignore[arg-type]
        write_compat_layout(layout, cases)

        cmd = [
            self.python_executable, "-m", "scalesim.scale",
            "-c", str(cfg), "-t", str(topo), "-l", str(layout),
            "-p", str(logs), "-i", input_type, "-s", "N",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "SCALE-Sim failed with exit code "
                f"{proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        run_dir = logs / run_name
        compute = parse_compute_report(run_dir / "COMPUTE_REPORT.csv")
        access_path = run_dir / "DETAILED_ACCESS_REPORT.csv"
        access = parse_detailed_access_report(access_path) if access_path.exists() else []
        compute_by_id = {x.layer_id: x for x in compute}
        access_by_id = {x.layer_id: x for x in access}
        out: List[ScaleSimCaseResult] = []
        for layer_id, case in enumerate(cases):
            if layer_id not in compute_by_id:
                raise RuntimeError(f"SCALE-Sim report missing layer {layer_id} ({case.name})")
            out.append(ScaleSimCaseResult(case, compute_by_id[layer_id], access_by_id.get(layer_id)))

        if tmp_ctx is not None and not keep_inputs:
            tmp_ctx.cleanup()
        return out

    def run(
        self,
        cases: Sequence[ScaleSimCase],
        output_root: str | Path | None = None,
        keep_inputs: bool = False,
    ) -> List[ScaleSimCaseResult]:
        return self._run_cases(cases, "gemm", output_root, keep_inputs)

    def run_conv(
        self,
        cases: Sequence[ScaleSimConvCase],
        output_root: str | Path | None = None,
        keep_inputs: bool = False,
    ) -> List[ScaleSimCaseResult]:
        return self._run_cases(cases, "conv", output_root, keep_inputs)


class ScaleSimReportBackend(TensorBackend):
    """TensorBackend backed by previously generated SCALE-Sim microkernel data."""

    def __init__(
        self,
        config: TensorEngineConfig,
        results: Iterable[ScaleSimCaseResult],
        include_stalls: bool = True,
    ):
        self.config = config
        self.include_stalls = include_stalls
        self._cycles: Dict[Tuple[int, int, int], int] = {}
        for result in results:
            if not isinstance(result.case, ScaleSimCase):
                continue
            # InterfaceBandwidth=CALC is recommended for compute calibration.
            cycles = result.compute.total_cycles
            if not include_stalls:
                cycles = max(0, cycles - result.compute.stall_cycles)
            self._cycles[result.case.key] = cycles

    def conv_compute_cycles(self, m: int, n: int, k: int) -> int:
        key = (m, n, k)
        if key not in self._cycles:
            raise KeyError(f"no SCALE-Sim golden for GEMM M,N,K={key}")
        return self._cycles[key]

    def active_mac_cycles(self, m: int, n: int, k: int) -> int:
        return math.ceil((m * n * k) / (self.config.rows * self.config.cols))
