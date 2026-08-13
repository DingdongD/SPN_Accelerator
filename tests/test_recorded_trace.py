import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from spn_accel_cmodel.config import SPNEngineConfig, SRAMConfig
from spn_accel_cmodel.functional import propagate_once
from spn_accel_cmodel.spn_engine import SPNEngine
from spn_accel_cmodel.trace import ArrayOffsetProvider
from spn_accel_cmodel.workload import Prop2DOp


@unittest.skipIf(np is None, "numpy optional validation dependency is unavailable")
class RecordedTraceTest(unittest.TestCase):
    def test_completionformer_18ch_offset_skips_center(self):
        arr = np.zeros((1, 18, 4, 5), dtype=np.float32)
        arr[:, 0, :, :] = 0.25
        arr[:, 1, :, :] = -0.5
        # Center pair channels 8,9 should be discarded.
        arr[:, 8, :, :] = 99.0
        arr[:, 9, :, :] = 99.0
        provider = ArrayOffsetProvider.from_numpy(arr, neighbors=8, center_index=4)
        self.assertEqual((provider.height, provider.width, provider.neighbors), (4, 5, 8))
        self.assertEqual(provider.delta(0, 0, 0), (0.25, -0.5))
        self.assertNotEqual(provider.delta(0, 0, 4), (99.0, 99.0))

    def test_center_only_affinity_is_identity(self):
        h = w = 4
        state = np.arange(h * w, dtype=np.float32).reshape(h, w)
        offset = np.zeros((16, h, w), dtype=np.float32)
        aff = np.zeros((9, h, w), dtype=np.float32)
        aff[4] = 1.0
        out = propagate_once(state, offset, aff)
        np.testing.assert_allclose(out, state, rtol=0, atol=0)

    def test_spn_engine_accepts_recorded_provider(self):
        h = w = 8
        provider = ArrayOffsetProvider.from_numpy(np.zeros((16, h, w), dtype=np.float32))
        op = Prop2DOp("p", h, w, steps=1)
        cfg = SPNEngineConfig(
            packet_pixels=8,
            metadata_sram=SRAMConfig(op.metadata_bytes, banks=8),
            state_sram=SRAMConfig(op.pingpong_bytes, banks=8),
        )
        result = SPNEngine(cfg).run(op, offset_provider=provider)
        self.assertGreater(result.cycles, 0)
        self.assertEqual(result.gather_reads, op.pixels * 33)

    def test_npz_loader(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trace.npz"
            np.savez(path, offset=np.zeros((1, 16, 3, 7), dtype=np.float32))
            provider = ArrayOffsetProvider.from_npz(path)
            self.assertEqual((provider.height, provider.width), (3, 7))


try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


@unittest.skipIf(np is None or torch is None, "numpy/torch unavailable")
class TorchFunctionalCrossCheck(unittest.TestCase):
    def test_bilinear_matches_grid_sample(self):
        from spn_accel_cmodel.functional import bilinear_sample_zero

        state = np.arange(30, dtype=np.float32).reshape(5, 6) / 7.0
        tensor = torch.from_numpy(state).view(1, 1, 5, 6)
        for y, x in [(-0.4, 1.2), (0.0, 0.0), (1.3, 2.7), (4.2, 5.1), (5.0, -1.0)]:
            ny = 2.0 * y / 4.0 - 1.0
            nx = 2.0 * x / 5.0 - 1.0
            grid = torch.tensor([[[[nx, ny]]]], dtype=torch.float32)
            golden = F.grid_sample(
                tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )[0, 0, 0, 0].item()
            actual = bilinear_sample_zero(state, y, x)
            self.assertAlmostEqual(actual, golden, places=5)

    def test_full_neighbor_sum_matches_grid_sample_formula(self):
        from spn_accel_cmodel.functional import propagate_once
        from spn_accel_cmodel.spn_engine import DEFAULT_NEIGHBORS

        rng = np.random.default_rng(4)
        h, w = 4, 5
        state = rng.normal(size=(h, w)).astype(np.float32)
        offsets = rng.uniform(-0.4, 0.4, size=(16, h, w)).astype(np.float32)
        # Use explicit 9-channel affinity (including center).
        aff = rng.normal(scale=0.05, size=(9, h, w)).astype(np.float32)
        aff[4] += 0.7
        actual = propagate_once(state, offsets, aff)

        off = offsets.reshape(8, 2, h, w)
        src = torch.from_numpy(state).view(1, 1, h, w)
        golden = torch.from_numpy(aff[4]).clone()
        golden = golden * torch.from_numpy(state)
        aidx = 0
        ys = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
        xs = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
        for k, (bdy, bdx) in enumerate(DEFAULT_NEIGHBORS):
            if aidx == 4:
                aidx += 1
            sy = ys + bdy + torch.from_numpy(off[k, 0])
            sx = xs + bdx + torch.from_numpy(off[k, 1])
            ny = 2.0 * sy / max(h - 1, 1) - 1.0
            nx = 2.0 * sx / max(w - 1, 1) - 1.0
            grid = torch.stack((nx, ny), dim=-1).unsqueeze(0)
            sampled = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
            golden += torch.from_numpy(aff[aidx]) * sampled
            aidx += 1
        np.testing.assert_allclose(actual, golden.numpy(), rtol=1e-5, atol=1e-5)

    def test_unified_torch_matches_numpy_reference(self):
        from spn_accel_cmodel import (
            NormalizationMode,
            SPNConfig,
            compile_nlspn_plan,
            propagate_canonical,
        )
        from spn_accel_cmodel.functional import propagate_once
        from spn_accel_cmodel.torch_spn_decoded import DecodedSPNParameters

        rng = np.random.default_rng(90210)
        h, w = 5, 6
        state = rng.normal(size=(h, w)).astype(np.float32)
        residual_yx = rng.uniform(-0.35, 0.35, size=(8, 2, h, w)).astype(np.float32)
        raw_affinity = rng.normal(scale=0.3, size=(8, h, w)).astype(np.float32)
        affinity = raw_affinity / (
            np.sum(np.abs(raw_affinity), axis=0, keepdims=True) + 1.0e-4
        )
        center = 1.0 - affinity.sum(axis=0, keepdims=True)
        affinity9 = np.concatenate((affinity[:4], center, affinity[4:]), axis=0)

        numpy_out = propagate_once(state, residual_yx.reshape(16, h, w), affinity9)
        current = torch.from_numpy(state).view(1, 1, h, w)
        config = SPNConfig.nlspn(
            iterations=1,
            confidence=False,
            normalization=NormalizationMode.AS,
        )
        decoded = DecodedSPNParameters(
            current=current,
            initial=current,
            raw_affinity=torch.from_numpy(raw_affinity).unsqueeze(0),
            residual_offsets_yx=torch.from_numpy(residual_yx).unsqueeze(0),
        )
        torch_out = propagate_canonical(
            current,
            current,
            compile_nlspn_plan(config, decoded),
        )
        np.testing.assert_allclose(
            torch_out[0, 0].numpy(),
            numpy_out,
            rtol=1.0e-5,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
