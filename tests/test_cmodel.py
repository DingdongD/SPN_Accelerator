import unittest

from spn_accel_cmodel.config import DMAConfig, NPUConfig, SPNEngineConfig, SRAMConfig, TensorEngineConfig
from spn_accel_cmodel.resources import BankedSRAM
from spn_accel_cmodel.simulator import NPUSimulator, Workload
from spn_accel_cmodel.spn_engine import OffsetPattern, SPNEngine, SPNTraceGenerator
from spn_accel_cmodel.tensor_engine import ConvMapping, TensorEngine
from spn_accel_cmodel.workload import Conv2DOp, Prop2DOp


class CModelTest(unittest.TestCase):
    def test_hot_bank_slower_than_distributed(self):
        cfg = SRAMConfig(capacity_bytes=4096, banks=8, ports_per_bank=1, word_bytes=2)
        sram = BankedSRAM(cfg)
        hot = [[i * 16 for i in range(16)]]
        dist = [[i * 2 for i in range(16)]]
        hot_r = sram.service_read_groups(hot, 0)
        dist_r = sram.service_read_groups(dist, 0)
        self.assertGreater(hot_r.service_cycles, dist_r.service_cycles)
        self.assertGreater(hot_r.conflict_stall_cycles, 0)

    def test_larger_array_reduces_cycles(self):
        op = Conv2DOp("conv", 32, 32, 64, 64, 3)
        mapping = ConvMapping(16, 16, 32, 32)
        small = TensorEngine(TensorEngineConfig(rows=16, cols=16), DMAConfig())
        large = TensorEngine(TensorEngineConfig(rows=64, cols=64), DMAConfig())
        self.assertGreater(small.run_conv(op, mapping).cycles, large.run_conv(op, mapping).cycles)

    def test_metadata_reused_across_iterations(self):
        op = Prop2DOp("p", 8, 8, neighbors=8, steps=2)
        trace = SPNTraceGenerator(op, OffsetPattern("random", seed=7))
        a, _ = trace.packet_addresses(0, 0, 8)
        b, _ = trace.packet_addresses(0, 0, 8)
        self.assertEqual(a, b)

    def test_spn_expected_read_count(self):
        op = Prop2DOp("p", 8, 8, neighbors=8, steps=1)
        cfg = SPNEngineConfig(
            packet_pixels=8,
            metadata_sram=SRAMConfig(op.metadata_bytes, banks=8, word_bytes=2),
            state_sram=SRAMConfig(op.pingpong_bytes, banks=8, word_bytes=2),
        )
        result = SPNEngine(cfg).run(op, OffsetPattern("zero"))
        self.assertEqual(result.gather_reads, op.pixels * (1 + 4 * op.neighbors))
        self.assertEqual(result.state_writes, op.pixels)

    def test_small_end_to_end(self):
        wl = Workload(
            convs=(Conv2DOp("c", 8, 8, 16, 16),),
            propagation=(Prop2DOp("p", 8, 8, steps=1),),
        )
        result = NPUSimulator(NPUConfig()).run(wl, ConvMapping(8, 8, 16, 16), OffsetPattern("zero"))
        self.assertGreater(result.total_cycles, 0)
        self.assertIn("c", result.tensor)
        self.assertIn("p", result.spn)


if __name__ == "__main__":
    unittest.main()
