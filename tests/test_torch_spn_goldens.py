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
    SPNInputs,
    SamplingMode,
    SparseFusionMode,
    UnifiedSPN,
    sample_neighbors,
)

from official_spn_references import reference_cspn, reference_nlspn


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


def _randn(generator, shape, scale=1.0):
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


class CSPNGoldenTest(unittest.TestCase):
    def _case(self, preserve_code_mask):
        generator = torch.Generator().manual_seed(1207)
        initial = _randn(generator, (1, 1, 4, 5))
        guidance = _randn(generator, (1, 8, 4, 5)) + 0.2
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 9.0

        expected, expected_trace = reference_cspn(
            initial,
            guidance,
            iterations=3,
            sparse_depth=sparse if preserve_code_mask else None,
        )
        actual, trace = UnifiedSPN(
            SPNConfig.cspn(
                iterations=3,
                preserve_code_mask=preserve_code_mask,
            )
        )(
            SPNInputs(
                current=initial,
                affinity=guidance,
                sparse_depth=sparse if preserve_code_mask else None,
            ),
            return_trace=True,
        )

        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
        self.assertEqual(len(trace.outputs), 3)
        for actual_step, expected_step in zip(
            trace.outputs,
            expected_trace,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_step,
                expected_step,
                rtol=1.0e-6,
                atol=1.0e-6,
            )

    def test_matches_released_cspn_without_sparse_mask(self):
        self._case(False)

    def test_matches_released_cspn_code_post_mask_behavior(self):
        self._case(True)


class NLSPNGoldenTest(unittest.TestCase):
    def _inputs(self):
        generator = torch.Generator().manual_seed(1207)
        initial = _randn(generator, (1, 1, 5, 6))
        affinity = _randn(generator, (1, 8, 5, 6), scale=0.4)
        residual_yx = _randn(generator, (1, 8, 2, 5, 6), scale=0.25)
        confidence = torch.sigmoid(_randn(generator, (1, 1, 5, 6)))
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 1] = 2.5
        sparse[:, :, 3, 4] = 7.0
        return initial, affinity, residual_yx, confidence, sparse

    def test_all_nlspn_affinity_modes_match_author_formula(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        for mode in (
            NormalizationMode.AS,
            NormalizationMode.ASS,
            NormalizationMode.TC,
            NormalizationMode.TGASS,
        ):
            with self.subTest(mode=mode.name):
                expected, expected_trace, expected_aff, expected_center = reference_nlspn(
                    initial,
                    affinity,
                    residual_yx,
                    iterations=2,
                    mode=mode.name,
                    confidence=confidence,
                )
                actual, trace = UnifiedSPN(
                    SPNConfig.nlspn(iterations=2, normalization=mode)
                )(
                    SPNInputs(
                        current=initial,
                        affinity=affinity,
                        offsets=residual_yx,
                        confidence=confidence,
                    ),
                    return_trace=True,
                )
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
                torch.testing.assert_close(
                    trace.neighbor_affinities[0], expected_aff, rtol=1.0e-5, atol=1.0e-6
                )
                torch.testing.assert_close(
                    trace.current_affinities[0], expected_center, rtol=1.0e-5, atol=1.0e-6
                )
                for actual_step, expected_step in zip(
                    trace.outputs,
                    expected_trace,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_step,
                        expected_step,
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )

    def test_hard_pre_sparse_preservation_matches_author_order(self):
        initial, affinity, residual_yx, confidence, sparse = self._inputs()
        expected, expected_trace, _, _ = reference_nlspn(
            initial,
            affinity,
            residual_yx,
            iterations=3,
            mode="TGASS",
            confidence=confidence,
            sparse_depth=sparse,
            preserve_input=True,
        )
        actual, trace = UnifiedSPN(
            SPNConfig.nlspn(iterations=3, preserve_input=True)
        )(
            SPNInputs(
                current=initial,
                affinity=affinity,
                offsets=residual_yx,
                confidence=confidence,
                sparse_depth=sparse,
            ),
            return_trace=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        for actual_step, expected_step in zip(trace.outputs, expected_trace, strict=True):
            torch.testing.assert_close(actual_step, expected_step, rtol=1.0e-5, atol=1.0e-6)

    def test_completionformer_temperature_100_matches_its_fork(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        expected, expected_trace, _, _ = reference_nlspn(
            initial,
            affinity,
            residual_yx,
            iterations=2,
            mode="TGASS",
            confidence=confidence,
            temperature=100.0,
        )
        actual, trace = UnifiedSPN(SPNConfig.completionformer(iterations=2))(
            SPNInputs(
                current=initial,
                affinity=affinity,
                offsets=residual_yx,
                confidence=confidence,
            ),
            return_trace=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        for actual_step, expected_step in zip(trace.outputs, expected_trace, strict=True):
            torch.testing.assert_close(actual_step, expected_step, rtol=1.0e-5, atol=1.0e-6)

    def test_legacy_confidence_offsets_include_the_base_stencil(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        expected, _, _, _ = reference_nlspn(
            initial,
            affinity,
            residual_yx,
            iterations=1,
            mode="TGASS",
            confidence=confidence,
            legacy_confidence_offsets=True,
        )
        actual = UnifiedSPN(
            SPNConfig.nlspn(iterations=1, legacy_confidence_offsets=True)
        )(
            SPNInputs(
                current=initial,
                affinity=affinity,
                offsets=residual_yx,
                confidence=confidence,
            )
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
