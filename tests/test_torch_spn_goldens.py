import unittest

try:
    import torch

    from spn_accel_cmodel.torch_functional import (
        AffinityMode,
        AffinityLayout,
        AnchorMode,
        NeighborConfidenceMode,
        NeighborMode,
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
    from official_spn_references import (
        reference_cspn,
        reference_dyspn,
        reference_dyspn_nlpm,
        reference_nlspn,
    )
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
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

    def test_singleton_dimensions_preserve_absolute_zero_padding(self):
        cases = (
            (
                torch.tensor([[[[4.0]]]]),
                torch.tensor([1.0, 0.0]),
                torch.tensor(0.0),
            ),
            (
                torch.tensor([[[[2.0, 6.0]]]]),
                torch.tensor([0.0, 0.5]),
                torch.tensor(1.0),
            ),
            (
                torch.tensor([[[[2.0], [6.0]]]]),
                torch.tensor([0.5, 0.0]),
                torch.tensor(1.0),
            ),
        )
        for state, xy, expected in cases:
            with self.subTest(shape=tuple(state.shape), displacement=xy.tolist()):
                offsets = torch.zeros(
                    (1, 1, 2, state.shape[2], state.shape[3]),
                    dtype=torch.float32,
                )
                offsets[0, 0, :, 0, 0] = xy
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
                        expected,
                        rtol=0.0,
                        atol=1.0e-6,
                    )

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

    def test_incompatible_softmax_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SOFTMAX requires AnchorMode.NONE"):
            SPNConfig(
                iterations=1,
                num_neighbors=1,
                neighbor_mode=NeighborMode.OFFSET,
                sampling_mode=SamplingMode.BILINEAR,
                padding_mode=PaddingMode.ZEROS,
                offset_mode=OffsetMode.ABSOLUTE_XY,
                affinity_mode=AffinityMode.STATIC,
                normalization=NormalizationMode.SOFTMAX,
                anchor_mode=AnchorMode.INITIAL,
            )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class GenericSPNTest(unittest.TestCase):
    def test_explicit_config_runs_general_recurrence_and_hard_post_fusion(self):
        cfg = SPNConfig(
            iterations=1,
            num_neighbors=1,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.STATIC,
            normalization=NormalizationMode.AS,
            anchor_mode=AnchorMode.INITIAL,
            sparse_fusion=SparseFusionMode.HARD_POST,
            affinity_layout=AffinityLayout.TARGET,
        )
        current = torch.ones((1, 1, 2, 3))
        initial = torch.zeros_like(current)
        sparse = torch.zeros_like(current)
        sparse[:, :, 0, 1] = 9.0
        actual = UnifiedSPN(cfg)(
            SPNInputs(
                current=current,
                initial=initial,
                affinity=torch.full((1, 1, 2, 3), 0.5),
                offsets=torch.zeros((1, 1, 2, 2, 3)),
                sparse_depth=sparse,
            )
        )
        coefficient = 0.5 / (0.5 + 1.0e-4)
        expected = torch.full_like(current, coefficient)
        expected[:, :, 0, 1] = 9.0
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_generic_tgass_applies_confidence_after_tanh_transform(self):
        cfg = SPNConfig(
            iterations=1,
            num_neighbors=1,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.STATIC,
            normalization=NormalizationMode.TGASS,
            anchor_mode=AnchorMode.INITIAL,
            neighbor_confidence=NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR,
            affinity_gamma=0.5,
        )
        current = torch.full((1, 1, 2, 2), 4.0)
        actual = UnifiedSPN(cfg)(
            SPNInputs(
                current=current,
                initial=torch.zeros_like(current),
                affinity=torch.full((1, 1, 2, 2), 2.0),
                offsets=torch.zeros((1, 1, 2, 2, 2)),
                confidence=torch.full((1, 1, 2, 2), 0.25),
            )
        )
        effective = torch.tanh(torch.tensor(2.0)) / 0.5 * 0.25
        torch.testing.assert_close(
            actual,
            torch.full_like(actual, 4.0 * effective),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class TorchSPNValidationTest(unittest.TestCase):
    def test_nlspn_rejects_missing_confidence(self):
        with self.assertRaisesRegex(ValueError, "requires confidence"):
            UnifiedSPN(SPNConfig.nlspn(iterations=1))(
                SPNInputs(
                    current=torch.zeros((1, 1, 2, 2)),
                    affinity=torch.zeros((1, 8, 2, 2)),
                    offsets=torch.zeros((1, 8, 2, 2, 2)),
                )
            )

    def test_dyspn_rejects_static_affinity_shape(self):
        with self.assertRaisesRegex(ValueError, "per-iteration affinity"):
            UnifiedSPN(SPNConfig.dyspn(iterations=2))(
                SPNInputs(
                    current=torch.zeros((1, 1, 2, 2)),
                    affinity=torch.zeros((1, 5, 2, 2)),
                    offsets=torch.zeros((1, 2, 5, 2, 2, 2)),
                    confidence=torch.zeros((1, 1, 2, 2)),
                    sparse_depth=torch.zeros((1, 1, 2, 2)),
                )
            )

    def test_nlspn_rejects_wrong_affinity_channels(self):
        with self.assertRaisesRegex(ValueError, "expected 8 channels"):
            UnifiedSPN(SPNConfig.nlspn(iterations=1, confidence=False))(
                SPNInputs(
                    current=torch.zeros((1, 1, 2, 3)),
                    affinity=torch.zeros((1, 7, 2, 3)),
                    offsets=torch.zeros((1, 8, 2, 2, 3)),
                )
            )

    def test_nlspn_rejects_wrong_offset_shape(self):
        with self.assertRaisesRegex(ValueError, "offsets"):
            UnifiedSPN(SPNConfig.nlspn(iterations=1, confidence=False))(
                SPNInputs(
                    current=torch.zeros((1, 1, 2, 3)),
                    affinity=torch.zeros((1, 8, 2, 3)),
                    offsets=torch.zeros((1, 8, 2, 2, 2)),
                )
            )

    def test_dyspn_rejects_iteration_mismatch(self):
        with self.assertRaisesRegex(ValueError, "DySPN offsets"):
            UnifiedSPN(SPNConfig.dyspn(iterations=2))(
                SPNInputs(
                    current=torch.zeros((1, 1, 2, 3)),
                    affinity=torch.zeros((1, 2, 5, 2, 3)),
                    offsets=torch.zeros((1, 1, 5, 2, 2, 3)),
                    confidence=torch.zeros((1, 1, 2, 3)),
                    sparse_depth=torch.zeros((1, 1, 2, 3)),
                )
            )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_metadata_is_moved_to_cuda_state_device(self):
        current = torch.randn((1, 1, 3, 4), device="cuda")
        output = UnifiedSPN(SPNConfig.nlspn(iterations=1))(
            SPNInputs(
                current=current,
                initial=torch.randn((1, 1, 3, 4)),
                affinity=torch.randn((1, 8, 3, 4)),
                offsets=torch.randn((1, 8, 2, 3, 4)) * 0.1,
                confidence=torch.sigmoid(torch.randn((1, 1, 3, 4))),
            )
        )
        self.assertEqual(output.device.type, "cuda")


def _randn(generator, shape, scale=1.0):
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CSPNGoldenTest(unittest.TestCase):
    def _case(self, preserve_code_mask):
        generator = torch.Generator().manual_seed(1207)
        initial = _randn(generator, (1, 1, 4, 5))
        guidance = _randn(generator, (1, 8, 4, 5)) + 0.2
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 9.0

        expected, expected_trace, expected_metadata = reference_cspn(
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

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(len(trace.outputs), 3)
        for actual_step, expected_step in zip(
            trace.outputs,
            expected_trace,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_step,
                expected_step,
                rtol=0.0,
                atol=0.0,
            )
        for candidate, expected_candidate in zip(
            trace.candidates,
            expected_metadata["candidates"],
            strict=True,
        ):
            torch.testing.assert_close(candidate, expected_candidate, rtol=0.0, atol=0.0)
        for offsets in trace.offsets:
            torch.testing.assert_close(offsets, expected_metadata["offsets"], rtol=0.0, atol=0.0)
        for affinity in trace.neighbor_affinities:
            torch.testing.assert_close(
                affinity,
                expected_metadata["neighbor_affinity"],
                rtol=0.0,
                atol=0.0,
            )
        for affinity in trace.initial_affinities:
            torch.testing.assert_close(
                affinity,
                expected_metadata["initial_affinity"],
                rtol=0.0,
                atol=0.0,
            )

    def test_matches_released_cspn_without_sparse_mask(self):
        self._case(False)

    def test_matches_released_cspn_code_post_mask_behavior(self):
        self._case(True)

    def test_shifted_source_channel_mapping_has_hand_computed_interior_value(self):
        initial = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
        guidance = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1, 1)
        guidance = guidance.expand(1, 8, 5, 5)
        actual = UnifiedSPN(SPNConfig.cspn(iterations=1))(
            SPNInputs(initial, guidance)
        )
        source_values = torch.tensor(
            [18.0, 17.0, 16.0, 13.0, 11.0, 8.0, 7.0, 6.0]
        )
        expected = torch.sum(torch.arange(1, 9) * source_values) / 36.0
        torch.testing.assert_close(actual[0, 0, 2, 2], expected, rtol=0.0, atol=1.0e-6)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
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

    def _assert_trace(
        self,
        trace,
        expected_outputs,
        expected_affinity,
        expected_center,
        metadata,
    ):
        iterations = len(expected_outputs)
        for field in (
            "outputs",
            "candidates",
            "offsets",
            "neighbor_affinities",
            "current_affinities",
            "initial_affinities",
        ):
            self.assertEqual(len(getattr(trace, field)), iterations, field)
        for actual_step, expected_step in zip(
            trace.outputs,
            expected_outputs,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_step,
                expected_step,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        for candidate, expected_candidate in zip(
            trace.candidates,
            metadata["candidates"],
            strict=True,
        ):
            torch.testing.assert_close(
                candidate,
                expected_candidate,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        for offsets in trace.offsets:
            torch.testing.assert_close(
                offsets,
                metadata["offsets"],
                rtol=0.0,
                atol=0.0,
            )
        for neighbor, current, initial in zip(
            trace.neighbor_affinities,
            trace.current_affinities,
            trace.initial_affinities,
            strict=True,
        ):
            torch.testing.assert_close(
                neighbor,
                expected_affinity,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                current,
                expected_center,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            self.assertIsNone(initial)

    def test_all_nlspn_affinity_modes_match_author_formula(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        for mode in (
            NormalizationMode.AS,
            NormalizationMode.ASS,
            NormalizationMode.TC,
            NormalizationMode.TGASS,
        ):
            with self.subTest(mode=mode.name):
                expected, expected_trace, expected_aff, expected_center, expected_metadata = reference_nlspn(
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
                self._assert_trace(
                    trace,
                    expected_trace,
                    expected_aff,
                    expected_center,
                    expected_metadata,
                )

    def test_hard_pre_sparse_preservation_matches_author_order(self):
        initial, affinity, residual_yx, confidence, sparse = self._inputs()
        (
            expected,
            expected_trace,
            expected_aff,
            expected_center,
            expected_metadata,
        ) = reference_nlspn(
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
        self._assert_trace(
            trace,
            expected_trace,
            expected_aff,
            expected_center,
            expected_metadata,
        )

    def test_completionformer_temperature_100_matches_its_fork(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        (
            expected,
            expected_trace,
            expected_aff,
            expected_center,
            expected_metadata,
        ) = reference_nlspn(
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
        self._assert_trace(
            trace,
            expected_trace,
            expected_aff,
            expected_center,
            expected_metadata,
        )

    def test_singleton_height_matches_absolute_zero_padding_oracle(self):
        initial = torch.tensor([[[[2.0, 4.0, 8.0]]]])
        affinity = torch.zeros((1, 8, 1, 3))
        affinity[:, 3] = 1.0
        residual_yx = torch.zeros((1, 8, 2, 1, 3))
        residual_yx[:, 3, 0] = 0.5
        for name, config, temperature in (
            (
                "NLSPN",
                SPNConfig.nlspn(
                    iterations=1,
                    normalization=NormalizationMode.AS,
                    confidence=False,
                ),
                1.0,
            ),
            (
                "CompletionFormer",
                SPNConfig.completionformer(
                    iterations=1,
                    normalization=NormalizationMode.AS,
                    confidence=False,
                ),
                100.0,
            ),
        ):
            with self.subTest(profile=name):
                expected, _, _, _, _ = reference_nlspn(
                    initial,
                    affinity,
                    residual_yx,
                    iterations=1,
                    mode="AS",
                    temperature=temperature,
                )
                actual = UnifiedSPN(config)(
                    SPNInputs(initial, affinity, offsets=residual_yx)
                )
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_legacy_confidence_offsets_include_the_base_stencil(self):
        initial, affinity, residual_yx, confidence, _ = self._inputs()
        expected, _, _, _, _ = reference_nlspn(
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

    def test_accepts_released_16_and_18_channel_offset_traces(self):
        initial, affinity, residual_yx, _, _ = self._inputs()
        model = UnifiedSPN(
            SPNConfig.nlspn(
                iterations=1,
                confidence=False,
                normalization=NormalizationMode.AS,
            )
        )
        canonical = model(
            SPNInputs(initial, affinity, offsets=residual_yx)
        )
        flattened16 = model(
            SPNInputs(initial, affinity, offsets=residual_yx.reshape(1, 16, 5, 6))
        )
        with_center = torch.cat(
            (
                residual_yx[:, :4],
                torch.zeros((1, 1, 2, 5, 6)),
                residual_yx[:, 4:],
            ),
            dim=1,
        ).reshape(1, 18, 5, 6)
        flattened18 = model(SPNInputs(initial, affinity, offsets=with_center))
        torch.testing.assert_close(flattened16, canonical)
        torch.testing.assert_close(flattened18, canonical)

    def test_residual_yx_order_and_base_grid_have_hand_computed_value(self):
        initial = torch.tensor(
            [[[[0.0, 1.0, 2.0, 3.0],
               [10.0, 11.0, 12.0, 13.0],
               [20.0, 21.0, 22.0, 23.0]]]]
        )
        affinity = torch.zeros((1, 8, 3, 4))
        affinity[:, 3] = 1.0  # base displacement (dx=-1, dy=0)
        residual_yx = torch.zeros((1, 8, 2, 3, 4))
        residual_yx[:, 3, 0] = 0.25  # dy
        residual_yx[:, 3, 1] = 0.50  # dx
        actual = UnifiedSPN(
            SPNConfig.nlspn(
                iterations=1,
                confidence=False,
                normalization=NormalizationMode.AS,
            )
        )(SPNInputs(initial, affinity, offsets=residual_yx))
        neighbor_weight = 1.0 / 1.0001
        expected = neighbor_weight * 14.0 + (1.0 - neighbor_weight) * 12.0
        torch.testing.assert_close(
            actual[0, 0, 1, 2],
            torch.tensor(expected),
            rtol=0.0,
            atol=2.0e-6,
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNGoldenTest(unittest.TestCase):
    def _case(self, neighbors):
        generator = torch.Generator().manual_seed(9183 + neighbors)
        initial = _randn(generator, (1, 1, 4, 6))
        residual_yx = _randn(generator, (1, 3, neighbors, 2, 4, 6), scale=0.3)
        logits = _randn(generator, (1, 3, neighbors, 4, 6), scale=0.7)
        sparse = torch.zeros_like(initial)
        sparse[:, :, 0, 0] = 3.0
        sparse[:, :, 2, 4] = 8.0
        confidence_logits = _randn(generator, (1, 1, 4, 6))
        expected, expected_trace, expected_affinities, expected_metadata = reference_dyspn(
            initial,
            residual_yx,
            logits,
            sparse,
            confidence_logits,
        )
        actual, trace = UnifiedSPN(
            SPNConfig.dyspn(iterations=3, num_neighbors=neighbors)
        )(
            SPNInputs(
                current=initial,
                affinity=logits,
                offsets=residual_yx,
                confidence=confidence_logits,
                sparse_depth=sparse,
            ),
            return_trace=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        for actual_step, expected_step in zip(trace.outputs, expected_trace, strict=True):
            torch.testing.assert_close(actual_step, expected_step, rtol=1.0e-5, atol=1.0e-6)
        for actual_aff, expected_aff in zip(
            trace.neighbor_affinities,
            expected_affinities,
            strict=True,
        ):
            torch.testing.assert_close(actual_aff, expected_aff, rtol=1.0e-6, atol=1.0e-7)
        for candidate, expected_candidate in zip(
            trace.candidates,
            expected_metadata["candidates"],
            strict=True,
        ):
            torch.testing.assert_close(candidate, expected_candidate, rtol=1.0e-5, atol=1.0e-6)
        for offsets, expected_offsets in zip(
            trace.offsets,
            expected_metadata["offsets"],
            strict=True,
        ):
            torch.testing.assert_close(offsets, expected_offsets, rtol=0.0, atol=0.0)
        self.assertTrue(all(value is None for value in trace.current_affinities))
        self.assertTrue(all(value is None for value in trace.initial_affinities))

    def test_k5_matches_current_default_author_module(self):
        self._case(5)

    def test_k9_matches_current_author_module(self):
        self._case(9)

    def test_k9_uses_author_sequential_fp32_accumulation_order(self):
        values = torch.tensor(
            [8.0, 1.0e7, -1.0e8, -1.0e7, 1.0e8, 4.0, 1.0e7, -2.0, -2.0],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        logits = torch.zeros((1, 1, 9, 3, 3))
        offsets = torch.zeros((1, 1, 9, 2, 3, 3))
        inputs = SPNInputs(
            current=values,
            affinity=logits,
            offsets=offsets,
            confidence=torch.zeros_like(values),
            sparse_depth=torch.zeros_like(values),
        )
        actual = UnifiedSPN(SPNConfig.dyspn(iterations=1, num_neighbors=9))(inputs)
        expected, _, _, _ = reference_dyspn(
            values,
            offsets,
            logits,
            torch.zeros_like(values),
            torch.zeros_like(values),
        )
        torch.testing.assert_close(
            actual[0, 0, 1, 1],
            expected[0, 0, 1, 1],
            rtol=0.0,
            atol=0.0,
        )

    def test_each_iteration_uses_its_own_metadata(self):
        generator = torch.Generator().manual_seed(771)
        initial = _randn(generator, (1, 1, 3, 4))
        residual = _randn(generator, (1, 2, 5, 2, 3, 4), scale=0.2)
        logits = _randn(generator, (1, 2, 5, 3, 4))
        sparse = torch.zeros_like(initial)
        confidence = torch.zeros_like(initial)
        model = UnifiedSPN(SPNConfig.dyspn(iterations=2, num_neighbors=5))
        normal = model(
            SPNInputs(initial, logits, offsets=residual, confidence=confidence, sparse_depth=sparse)
        )
        swapped = model(
            SPNInputs(
                initial,
                logits.flip(1),
                offsets=residual.flip(1),
                confidence=confidence,
                sparse_depth=sparse,
            )
        )
        self.assertFalse(torch.allclose(normal, swapped))

    def test_k1_out_of_bounds_sample_uses_zero_padding(self):
        initial = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        residual_yx = torch.zeros((1, 1, 1, 2, 2, 2))
        residual_yx[0, 0, 0, :, 0, 0] = torch.tensor([-0.5, -0.5])
        actual = UnifiedSPN(SPNConfig.dyspn(iterations=1, num_neighbors=1))(
            SPNInputs(
                current=initial,
                affinity=torch.zeros((1, 1, 1, 2, 2)),
                offsets=residual_yx,
                confidence=torch.zeros_like(initial),
                sparse_depth=torch.zeros_like(initial),
            )
        )
        torch.testing.assert_close(actual[0, 0, 0, 0], torch.tensor(0.25), atol=1.0e-6, rtol=0.0)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNNLPMGoldenTest(unittest.TestCase):
    def test_7x7_naive_nlpm_matches_author_operation_order(self):
        generator = torch.Generator().manual_seed(555)
        initial = _randn(generator, (1, 1, 7, 8))
        guidance = _randn(generator, (1, 48, 7, 8), scale=0.15)
        attention = _randn(generator, (1, 2, 4, 7, 8))
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 5.0
        confidence = torch.sigmoid(_randn(generator, (1, 1, 7, 8)))
        expected, expected_candidates, expected_outputs, expected_metadata = reference_dyspn_nlpm(
            initial,
            guidance,
            attention,
            sparse,
            confidence,
        )
        actual, trace = UnifiedSPN(SPNConfig.dyspn_nlpm(iterations=2))(
            SPNInputs(
                current=initial,
                affinity=guidance,
                attention=attention,
                confidence=confidence,
                sparse_depth=sparse,
            ),
            return_trace=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        for actual_step, expected_step in zip(
            trace.candidates,
            expected_candidates,
            strict=True,
        ):
            torch.testing.assert_close(actual_step, expected_step, rtol=1.0e-5, atol=1.0e-6)
        for actual_step, expected_step in zip(trace.outputs, expected_outputs, strict=True):
            torch.testing.assert_close(actual_step, expected_step, rtol=1.0e-5, atol=1.0e-6)
        for field in (
            "neighbor_affinities",
            "current_affinities",
            "initial_affinities",
        ):
            for actual_value, expected_value in zip(
                getattr(trace, field),
                expected_metadata[field],
                strict=True,
            ):
                torch.testing.assert_close(actual_value, expected_value, rtol=1.0e-5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
