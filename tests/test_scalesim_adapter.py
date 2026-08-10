import tempfile
import unittest
from pathlib import Path

from spn_accel_cmodel.backends.scalesim import (
    ScaleSimCase,
    ScaleSimConvCase,
    parse_compute_report,
    parse_detailed_access_report,
    write_gemm_topology,
    write_conv_topology,
    write_scalesim_config,
)
from spn_accel_cmodel.config import TensorEngineConfig


class ScaleSimAdapterTest(unittest.TestCase):
    def test_parses_current_v3_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "COMPUTE_REPORT.csv").write_text(
                "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n"
                "0, 110, 100, 3, 80.0, 90.0, 85.0,\n",
                encoding="utf-8",
            )
            (root / "DETAILED_ACCESS_REPORT.csv").write_text(
                "LayerID, SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n"
                "0,0,1,10,0,1,20,0,1,30,0,1,4,0,1,5,0,1,6,\n",
                encoding="utf-8",
            )
            c = parse_compute_report(root / "COMPUTE_REPORT.csv")[0]
            a = parse_detailed_access_report(root / "DETAILED_ACCESS_REPORT.csv")[0]
            self.assertEqual(c.total_cycles, 100)
            self.assertEqual(c.stall_cycles, 3)
            self.assertEqual(a.sram_filter_reads, 20)
            self.assertEqual(a.dram_ofmap_writes, 6)

    def test_generates_mnk_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            topo = root / "x.csv"
            conv_topo = root / "conv.csv"
            cfg = root / "x.cfg"
            write_gemm_topology(topo, [ScaleSimCase("x", 64, 32, 288)])
            write_conv_topology(conv_topo, [ScaleSimConvCase("c", 18, 18, 3, 3, 32, 32)])
            write_scalesim_config(cfg, TensorEngineConfig(rows=64, cols=32))
            self.assertIn("Layer Name,M,N,K", topo.read_text())
            self.assertIn("IFMAP Height", conv_topo.read_text())
            text = cfg.read_text()
            self.assertIn("ArrayHeight:    64", text)
            self.assertIn("ArrayWidth:     32", text)
            self.assertIn("Dataflow : os", text)
            self.assertIn("InterfaceBandwidth: CALC", text)


if __name__ == "__main__":
    unittest.main()
