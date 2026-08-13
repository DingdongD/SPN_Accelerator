import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from official_spn_references import (
        reference_completionformer_author_decode,
        reference_cspn,
        reference_dyspn,
        reference_dyspn_author_decode,
        reference_dyspn_nlpm,
        reference_nlspn,
        reference_nlspn_author_decode,
    )
    from spn_accel_cmodel.torch_functional import (
        AffinityMode,
        AnchorMode,
        CSPNRawInputs,
        CompletionFormerRawInputs,
        DySPNNLPMRawInputs,
        DySPNRawInputs,
        NLSPNRawInputs,
        NeighborConfidenceMode,
        NormalizationMode,
        OffsetMode,
        PaddingMode,
        SPNConfig,
        SamplingMode,
        SparseFusionMode,
        UnifiedSPN,
        sample_neighbors,
    )
    from spn_accel_cmodel.torch_spn_adapters import (
        compile_completionformer_plan,
        compile_nlspn_plan,
    )


def _randn(generator, shape, scale=1.0):
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def _fill_author_conv(model, generator):
    with torch.no_grad():
        model.author.conv_offset_aff.weight.copy_(
            _randn(generator, model.author.conv_offset_aff.weight.shape, 0.08)
        )
        model.author.conv_offset_aff.bias.copy_(
            _randn(generator, model.author.conv_offset_aff.bias.shape, 0.05)
        )


def _assert_steps(test, actual, expected, *, rtol=1.0e-5, atol=1.0e-6):
    test.assertEqual(len(actual), len(expected))
    for actual_step, expected_step in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_step,
            expected_step,
            rtol=rtol,
            atol=atol,
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class TorchSPNSamplingTest(unittest.TestCase):
    def test_zero_and_border_padding_are_distinct(self):
        state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        offsets = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        offsets[0, 0, :, 0, 0] = torch.tensor([-0.5, -0.5])
        zero = sample_neighbors(
            state,
            offsets,
            SamplingMode.BILINEAR,
            PaddingMode.ZEROS,
            align_corners=True,
        )
        border = sample_neighbors(
            state,
            offsets,
            SamplingMode.BILINEAR,
            PaddingMode.BORDER,
            align_corners=True,
        )
        torch.testing.assert_close(zero[0, 0, 0, 0, 0], torch.tensor(0.25))
        torch.testing.assert_close(border[0, 0, 0, 0, 0], torch.tensor(1.0))

    def test_named_profiles_capture_author_defaults(self):
        nlspn = SPNConfig.nlspn()
        self.assertEqual(nlspn.offset_mode, OffsetMode.RESIDUAL_YX)
        self.assertEqual(nlspn.anchor_mode, AnchorMode.CURRENT)
        self.assertEqual(nlspn.affinity_mode, AffinityMode.STATIC)
        self.assertEqual(
            nlspn.neighbor_confidence,
            NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR,
        )

        completionformer = SPNConfig.completionformer()
        self.assertEqual(completionformer.iterations, 6)
        self.assertEqual(completionformer.tanh_temperature, 100.0)
        self.assertEqual(completionformer.sparse_fusion, SparseFusionMode.NONE)

        dyspn = SPNConfig.dyspn()
        self.assertEqual(dyspn.num_neighbors, 5)
        self.assertEqual(dyspn.normalization, NormalizationMode.SOFTMAX)
        self.assertEqual(dyspn.anchor_mode, AnchorMode.NONE)
        self.assertFalse(dyspn.align_corners)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CSPNAuthorGoldenTest(unittest.TestCase):
    def test_raw_author_interface_matches_release_formula_with_and_without_mask(self):
        generator = torch.Generator().manual_seed(1207)
        initial = _randn(generator, (1, 1, 4, 5))
        guidance = _randn(generator, (1, 8, 4, 5)) + 0.2
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 9.0
        for preserve in (False, True):
            with self.subTest(preserve_code_mask=preserve):
                expected, expected_steps, metadata = reference_cspn(
                    initial,
                    guidance,
                    iterations=3,
                    sparse_depth=sparse if preserve else None,
                )
                actual, trace = UnifiedSPN(
                    SPNConfig.cspn(
                        iterations=3,
                        preserve_code_mask=preserve,
                    )
                )(
                    CSPNRawInputs(guidance, initial, sparse if preserve else None),
                    return_trace=True,
                )

                torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)
                _assert_steps(self, trace.outputs, expected_steps, rtol=0.0)
                _assert_steps(self, trace.candidates, metadata["candidates"], rtol=0.0)
                for offsets, affinity, initial_affinity, post_gate in zip(
                    trace.offsets,
                    trace.neighbor_affinities,
                    trace.initial_affinities,
                    trace.post_fusion_gates,
                    strict=True,
                ):
                    torch.testing.assert_close(offsets, metadata["offsets"])
                    torch.testing.assert_close(
                        affinity[:, :, 0],
                        metadata["neighbor_affinity"],
                    )
                    torch.testing.assert_close(
                        initial_affinity,
                        metadata["initial_affinity"],
                    )
                    torch.testing.assert_close(post_gate, metadata["post_fusion_gate"])

    def test_shifted_source_integer_microcase(self):
        initial = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
        guidance = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1, 1)
        guidance = guidance.expand(1, 8, 5, 5)
        actual = UnifiedSPN(SPNConfig.cspn(iterations=1))(
            CSPNRawInputs(guidance, initial)
        )
        sources = torch.tensor([18.0, 17.0, 16.0, 13.0, 11.0, 8.0, 7.0, 6.0])
        expected = torch.sum(torch.arange(1, 9) * sources) / 36.0
        torch.testing.assert_close(actual[0, 0, 2, 2], expected, rtol=0.0, atol=1.0e-6)


class _NLSPNFamilyAuthorGolden:
    config_factory = None
    input_type = None
    decode_reference = None
    temperature = None

    compiler = None

    def _cases(self):
        return ((NormalizationMode.TGASS, True, False, False, 3),)

    def test_raw_author_interface_matches_all_declared_formula_variants(self):
        for index, (mode, use_confidence, legacy, preserve, iterations) in enumerate(
            self._cases()
        ):
            with self.subTest(
                mode=mode.name,
                confidence=use_confidence,
                legacy=legacy,
                preserve=preserve,
                iterations=iterations,
            ):
                generator = torch.Generator().manual_seed(4107 + index)
                config = self.config_factory(
                    iterations=iterations,
                    normalization=mode,
                    confidence=use_confidence,
                    legacy_confidence_offsets=legacy,
                    preserve_input=preserve,
                )
                model = UnifiedSPN(config)
                _fill_author_conv(model, generator)
                initial = _randn(generator, (1, 1, 5, 6))
                guidance = _randn(generator, (1, 8, 5, 6))
                confidence = torch.sigmoid(_randn(generator, (1, 1, 5, 6)))
                sparse = torch.zeros_like(initial)
                sparse[:, :, 2, 3] = 4.0
                expected_decoded = self.decode_reference(
                    guidance,
                    model.author.conv_offset_aff.weight,
                    model.author.conv_offset_aff.bias,
                )
                expected, expected_steps, expected_affinity, expected_center, metadata = (
                    reference_nlspn(
                        initial,
                        expected_decoded["raw_affinity"],
                        expected_decoded["residual_offsets_yx"],
                        iterations=iterations,
                        mode=mode.name,
                        confidence=confidence if use_confidence else None,
                        sparse_depth=sparse if preserve else None,
                        preserve_input=preserve,
                        temperature=self.temperature,
                        legacy_confidence_offsets=legacy,
                    )
                )
                if self.input_type is NLSPNRawInputs:
                    inputs = self.input_type(
                        initial,
                        guidance,
                        confidence_probability=confidence if use_confidence else None,
                        feat_fix=sparse if preserve else None,
                    )
                else:
                    inputs = self.input_type(initial, guidance, confidence, sparse)

                decoded = model.author.decode(inputs)
                torch.testing.assert_close(
                    decoded.raw_affinity,
                    expected_decoded["raw_affinity"],
                )
                torch.testing.assert_close(
                    decoded.residual_offsets_yx,
                    expected_decoded["residual_offsets_yx"],
                )
                plan = self.compiler(config, decoded)
                torch.testing.assert_close(plan.offsets_xy[:, 0], metadata["offsets"])
                torch.testing.assert_close(
                    plan.neighbor_affinity[:, 0, :, 0],
                    expected_affinity,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(plan.current_affinity[:, 0], expected_center)
                torch.testing.assert_close(
                    plan.pre_fusion_gate[:, 0],
                    metadata["pre_fusion_gate"],
                )

                actual, trace = model(inputs, return_trace=True)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
                _assert_steps(self, trace.outputs, expected_steps)
                _assert_steps(self, trace.candidates, metadata["candidates"])


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class NLSPNAuthorGoldenTest(_NLSPNFamilyAuthorGolden, unittest.TestCase):
    config_factory = SPNConfig.nlspn
    input_type = NLSPNRawInputs
    decode_reference = staticmethod(reference_nlspn_author_decode)
    temperature = 1.0
    compiler = staticmethod(compile_nlspn_plan)

    def _cases(self):
        return (
            (NormalizationMode.AS, False, False, False, 2),
            (NormalizationMode.ASS, True, False, True, 2),
            (NormalizationMode.TC, True, True, False, 2),
            (NormalizationMode.TGASS, True, False, False, 3),
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerAuthorGoldenTest(
    _NLSPNFamilyAuthorGolden,
    unittest.TestCase,
):
    config_factory = SPNConfig.completionformer
    input_type = CompletionFormerRawInputs
    decode_reference = staticmethod(reference_completionformer_author_decode)
    temperature = 100.0
    compiler = staticmethod(compile_completionformer_plan)

    def _cases(self):
        return (
            (NormalizationMode.TC, True, False, False, 6),
            (NormalizationMode.TGASS, True, False, False, 6),
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNAuthorGoldenTest(unittest.TestCase):
    def test_all_released_stencils_match_decode_metadata_and_propagation(self):
        for neighbors in (1, 3, 5, 9):
            with self.subTest(neighbors=neighbors):
                generator = torch.Generator().manual_seed(9183 + neighbors)
                iterations = 3
                config = SPNConfig.dyspn(
                    iterations=iterations,
                    num_neighbors=neighbors,
                )
                model = UnifiedSPN(config)
                _fill_author_conv(model, generator)
                initial = _randn(generator, (1, 1, 4, 6))
                guide = _randn(generator, (1, iterations * neighbors, 4, 6))
                sparse = torch.zeros_like(initial)
                sparse[:, :, 2, 4] = 8.0
                confidence_logits = _randn(generator, (1, 1, 4, 6))
                expected_decoded = reference_dyspn_author_decode(
                    guide,
                    model.author.conv_offset_aff.weight,
                    model.author.conv_offset_aff.bias,
                    iterations=iterations,
                    num_neighbors=neighbors,
                )
                expected, expected_steps, expected_affinities, metadata = (
                    reference_dyspn(
                        initial,
                        expected_decoded["residual_offsets_yx"],
                        expected_decoded["raw_affinity"],
                        sparse,
                        confidence_logits,
                    )
                )
                inputs = DySPNRawInputs(
                    initial,
                    guide,
                    sparse,
                    confidence_logits,
                )
                decoded = model.author.decode(inputs)
                torch.testing.assert_close(
                    decoded.raw_affinity,
                    expected_decoded["raw_affinity"],
                )
                torch.testing.assert_close(
                    decoded.residual_offsets_yx,
                    expected_decoded["residual_offsets_yx"],
                )

                actual, trace = model(inputs, return_trace=True)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
                _assert_steps(self, trace.outputs, expected_steps)
                _assert_steps(self, trace.candidates, metadata["candidates"])
                _assert_steps(self, trace.offsets, metadata["offsets"], rtol=0.0, atol=0.0)
                _assert_steps(
                    self,
                    [value[:, :, 0] for value in trace.neighbor_affinities],
                    expected_affinities,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                )
                for post_gate in trace.post_fusion_gates:
                    torch.testing.assert_close(
                        post_gate,
                        metadata["post_fusion_gate"],
                    )

    def test_k9_preserves_author_sequential_fp32_reduction_order(self):
        values = torch.tensor(
            [8.0, 1.0e7, -1.0e8, -1.0e7, 1.0e8, 4.0, 1.0e7, -2.0, -2.0],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        guide = torch.zeros((1, 9, 3, 3))
        zeros = torch.zeros_like(values)
        model = UnifiedSPN(SPNConfig.dyspn(iterations=1, num_neighbors=9))
        actual = model(DySPNRawInputs(values, guide, zeros, zeros))
        expected, _, _, _ = reference_dyspn(
            values,
            torch.zeros((1, 1, 9, 2, 3, 3)),
            torch.zeros((1, 1, 9, 3, 3)),
            zeros,
            zeros,
        )
        torch.testing.assert_close(
            actual[0, 0, 1, 1],
            expected[0, 0, 1, 1],
            rtol=0.0,
            atol=0.0,
        )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNNLPMAuthorGoldenTest(unittest.TestCase):
    def test_raw_author_interface_matches_release_formula_each_iteration(self):
        generator = torch.Generator().manual_seed(555)
        initial = _randn(generator, (1, 1, 7, 8))
        guidance = _randn(generator, (1, 48, 7, 8), scale=0.15)
        dynamic_logits = _randn(generator, (1, 8, 7, 8))
        attention_logits = dynamic_logits.view(1, 2, 4, 7, 8)
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 5.0
        confidence = torch.sigmoid(_randn(generator, (1, 1, 7, 8)))
        expected, expected_candidates, expected_steps, metadata = reference_dyspn_nlpm(
            initial,
            guidance,
            attention_logits,
            sparse,
            confidence,
        )

        model = UnifiedSPN(SPNConfig.dyspn_nlpm(iterations=2))
        inputs = DySPNNLPMRawInputs(
            initial,
            guidance,
            dynamic_logits,
            sparse,
            confidence,
        )
        decoded = model.author.decode(inputs)
        torch.testing.assert_close(decoded.raw_affinity, guidance)
        torch.testing.assert_close(decoded.attention_logits, attention_logits)
        torch.testing.assert_close(decoded.confidence_probability, confidence)

        actual, trace = model(
            inputs,
            return_trace=True,
        )

        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        _assert_steps(self, trace.candidates, expected_candidates)
        _assert_steps(self, trace.outputs, expected_steps)
        for iteration in range(2):
            torch.testing.assert_close(
                trace.neighbor_affinities[iteration],
                metadata["effective_neighbor_affinity"][:, iteration],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                trace.current_affinities[iteration],
                metadata["current_affinity"][:, iteration],
            )
            torch.testing.assert_close(
                trace.initial_affinities[iteration],
                metadata["initial_affinity"][:, iteration],
            )
            torch.testing.assert_close(
                trace.group_scales[iteration],
                metadata["group_scale"][:, iteration],
            )
            torch.testing.assert_close(
                trace.post_fusion_gates[iteration],
                metadata["post_fusion_gate"],
            )


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
class TorchSPNDeviceTest(unittest.TestCase):
    def test_all_author_profiles_accept_cpu_metadata_with_cuda_state(self):
        state = torch.ones((1, 1, 3, 4), device="cuda")
        zeros = torch.zeros((1, 1, 3, 4))
        cases = (
            (
                SPNConfig.cspn(iterations=1),
                CSPNRawInputs(torch.ones((1, 8, 3, 4)), state),
            ),
            (
                SPNConfig.nlspn(iterations=1),
                NLSPNRawInputs(state, torch.ones((1, 8, 3, 4)), zeros),
            ),
            (
                SPNConfig.completionformer(iterations=1),
                CompletionFormerRawInputs(
                    state,
                    torch.ones((1, 8, 3, 4)),
                    zeros,
                    zeros,
                ),
            ),
            (
                SPNConfig.dyspn(iterations=1),
                DySPNRawInputs(
                    state,
                    torch.ones((1, 5, 3, 4)),
                    zeros,
                    zeros,
                ),
            ),
            (
                SPNConfig.dyspn_nlpm(iterations=1),
                DySPNNLPMRawInputs(
                    state,
                    torch.ones((1, 48, 3, 4)),
                    torch.zeros((1, 4, 3, 4)),
                    zeros,
                    zeros,
                ),
            ),
        )
        for config, inputs in cases:
            with self.subTest(profile=config.profile.name):
                output, trace = UnifiedSPN(config).cuda()(
                    inputs,
                    return_trace=True,
                )
                self.assertEqual(output.device.type, "cuda")
                self.assertEqual(trace.outputs[0].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
