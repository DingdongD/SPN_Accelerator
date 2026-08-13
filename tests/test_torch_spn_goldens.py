import unittest

import torch

from spn_accel_cmodel.torch_functional import (
    AffinityMode,
    AnchorMode,
    NeighborConfidenceMode,
    NormalizationMode,
    OffsetMode,
    PaddingMode,
    SPNConfig,
    SamplingMode,
    SparseFusionMode,
    sample_neighbors,
)


class TorchSPNSamplingTest(unittest.TestCase):
    def test_zero_padding_does_not_replicate_border(self):
        state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        offsets = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        offsets[0, 0, :, 0, 0] = torch.tensor([-0.5, -0.5])

        for align_corners in (True, False):
            actual = sample_neighbors(
                state,
                offsets,
                SamplingMode.BILINEAR,
                PaddingMode.ZEROS,
                align_corners=align_corners,
            )
            torch.testing.assert_close(
                actual[0, 0, 0, 0, 0],
                torch.tensor(0.25),
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_border_padding_is_explicit_non_author_behavior(self):
        state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        offsets = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        offsets[0, 0, :, 0, 0] = torch.tensor([-0.5, -0.5])
        actual = sample_neighbors(
            state,
            offsets,
            SamplingMode.BILINEAR,
            PaddingMode.BORDER,
            align_corners=True,
        )
        torch.testing.assert_close(actual[0, 0, 0, 0, 0], torch.tensor(1.0))

    def test_nlspn_profile_uses_residual_yx_offsets(self):
        cfg = SPNConfig.nlspn(
            iterations=2,
            normalization=NormalizationMode.AS,
        )
        self.assertEqual(cfg.offset_mode, OffsetMode.RESIDUAL_YX)
        self.assertEqual(cfg.anchor_mode, AnchorMode.CURRENT)
        self.assertEqual(cfg.affinity_mode, AffinityMode.STATIC)
        self.assertEqual(
            cfg.neighbor_confidence,
            NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR,
        )

    def test_completionformer_and_dyspn_profiles_capture_author_defaults(self):
        completionformer = SPNConfig.completionformer()
        self.assertEqual(completionformer.iterations, 6)
        self.assertEqual(completionformer.normalization, NormalizationMode.TGASS)
        self.assertEqual(completionformer.tanh_temperature, 100.0)
        self.assertEqual(completionformer.sparse_fusion, SparseFusionMode.NONE)

        dyspn = SPNConfig.dyspn()
        self.assertEqual(dyspn.iterations, 6)
        self.assertEqual(dyspn.num_neighbors, 5)
        self.assertEqual(dyspn.normalization, NormalizationMode.SOFTMAX)
        self.assertEqual(dyspn.anchor_mode, AnchorMode.NONE)
        self.assertEqual(dyspn.affinity_mode, AffinityMode.PER_ITERATION)
        self.assertFalse(dyspn.align_corners)

    def test_invalid_dyspn_neighbor_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "DySPN supports K in"):
            SPNConfig.dyspn(num_neighbors=4)


if __name__ == "__main__":
    unittest.main()
