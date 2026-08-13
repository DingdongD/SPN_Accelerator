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
    def test_raw_author_interface_matches_release_formula_each_iteration(self):
        generator = torch.Generator().manual_seed(1207)
        initial = _randn(generator, (1, 1, 4, 5))
        guidance = _randn(generator, (1, 8, 4, 5)) + 0.2
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 2] = 9.0
        expected, expected_steps, _ = reference_cspn(
            initial,
            guidance,
            iterations=3,
            sparse_depth=sparse,
        )

        actual, trace = UnifiedSPN(
            SPNConfig.cspn(iterations=3, preserve_code_mask=True)
        )(
            CSPNRawInputs(guidance, initial, sparse),
            return_trace=True,
        )

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)
        _assert_steps(self, trace.outputs, expected_steps, rtol=0.0)


class _NLSPNFamilyAuthorGolden:
    config_factory = None
    input_type = None
    decode_reference = None
    temperature = None

    def test_raw_author_interface_matches_conv_decode_and_propagation(self):
        generator = torch.Generator().manual_seed(4107)
        config = self.config_factory(iterations=3)
        model = UnifiedSPN(config)
        _fill_author_conv(model, generator)
        initial = _randn(generator, (1, 1, 5, 6))
        guidance = _randn(generator, (1, 8, 5, 6))
        confidence = torch.sigmoid(_randn(generator, (1, 1, 5, 6)))
        sparse = torch.zeros_like(initial)
        decoded = self.decode_reference(
            guidance,
            model.author.conv_offset_aff.weight,
            model.author.conv_offset_aff.bias,
        )
        expected, expected_steps, expected_affinity, _, _ = reference_nlspn(
            initial,
            decoded["raw_affinity"],
            decoded["residual_offsets_yx"],
            iterations=3,
            mode="TGASS",
            confidence=confidence,
            temperature=self.temperature,
        )
        if self.input_type is NLSPNRawInputs:
            inputs = self.input_type(
                initial,
                guidance,
                confidence_probability=confidence,
                feat_fix=sparse,
            )
        else:
            inputs = self.input_type(initial, guidance, confidence, sparse)

        actual, trace = model(inputs, return_trace=True)

        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        _assert_steps(self, trace.outputs, expected_steps)
        for actual_affinity in trace.neighbor_affinities:
            torch.testing.assert_close(
                actual_affinity[:, :, 0],
                expected_affinity,
                rtol=1.0e-5,
                atol=1.0e-6,
            )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class NLSPNAuthorGoldenTest(_NLSPNFamilyAuthorGolden, unittest.TestCase):
    config_factory = SPNConfig.nlspn
    input_type = NLSPNRawInputs
    decode_reference = staticmethod(reference_nlspn_author_decode)
    temperature = 1.0


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerAuthorGoldenTest(
    _NLSPNFamilyAuthorGolden,
    unittest.TestCase,
):
    config_factory = SPNConfig.completionformer
    input_type = CompletionFormerRawInputs
    decode_reference = staticmethod(reference_completionformer_author_decode)
    temperature = 100.0


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNAuthorGoldenTest(unittest.TestCase):
    def test_raw_author_interface_matches_conv_decode_and_propagation(self):
        generator = torch.Generator().manual_seed(9188)
        iterations = 3
        neighbors = 5
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
        decoded = reference_dyspn_author_decode(
            guide,
            model.author.conv_offset_aff.weight,
            model.author.conv_offset_aff.bias,
            iterations=iterations,
            num_neighbors=neighbors,
        )
        expected, expected_steps, expected_affinities, _ = reference_dyspn(
            initial,
            decoded["residual_offsets_yx"],
            decoded["raw_affinity"],
            sparse,
            confidence_logits,
        )

        actual, trace = model(
            DySPNRawInputs(initial, guide, sparse, confidence_logits),
            return_trace=True,
        )

        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        _assert_steps(self, trace.outputs, expected_steps)
        _assert_steps(
            self,
            [value[:, :, 0] for value in trace.neighbor_affinities],
            expected_affinities,
            rtol=1.0e-6,
            atol=1.0e-7,
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
        expected, expected_candidates, expected_steps, _ = reference_dyspn_nlpm(
            initial,
            guidance,
            attention_logits,
            sparse,
            confidence,
        )

        actual, trace = UnifiedSPN(SPNConfig.dyspn_nlpm(iterations=2))(
            DySPNNLPMRawInputs(
                initial,
                guidance,
                dynamic_logits,
                sparse,
                confidence,
            ),
            return_trace=True,
        )

        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        _assert_steps(self, trace.candidates, expected_candidates)
        _assert_steps(self, trace.outputs, expected_steps)


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
