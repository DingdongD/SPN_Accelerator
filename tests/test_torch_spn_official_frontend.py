import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    import torch.nn.functional as F

    from spn_accel_cmodel.torch_spn_official_frontend import (
        OFFICIAL_FRONTEND_BUILDERS,
        CSPNOfficialFrontend,
        CompletionFormerOfficialFrontend,
        DySPNNLPMOfficialFrontend,
        DySPNOfficialFrontend,
        NLSPNOfficialFrontend,
    )
    from spn_accel_cmodel.torch_spn_decoded import DecodedSPNParameters
    from spn_accel_cmodel.torch_spn_types import (
        CSPNRawInputs,
        CompletionFormerRawInputs,
        DySPNNLPMRawInputs,
        DySPNRawInputs,
        NLSPNRawInputs,
        SPNConfig,
        SPNProfile,
    )
    from official_spn_references import (
        reference_completionformer_official_decode,
        reference_dyspn_official_decode,
        reference_nlspn_official_decode,
    )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class OfficialInputTypeTest(unittest.TestCase):
    def test_profile_inputs_name_confidence_semantics_explicitly(self):
        image = torch.zeros((1, 1, 3, 4))
        nlspn = NLSPNRawInputs(image, torch.zeros((1, 8, 3, 4)))
        self.assertIs(nlspn.feat_init, image)
        self.assertIsNone(nlspn.confidence_probability)
        self.assertIs(
            DySPNRawInputs(image, image, image, image).confidence_logits,
            image,
        )
        self.assertIs(
            DySPNNLPMRawInputs(
                image,
                torch.zeros((1, 48, 3, 4)),
                torch.zeros((1, 24, 3, 4)),
                image,
                image,
            ).confidence_probability,
            image,
        )
        self.assertIs(CSPNRawInputs(image, image).blur_depth, image)
        self.assertIs(
            CompletionFormerRawInputs(image, image, image, image).pred_init,
            image,
        )

    def test_decoded_parameters_keep_post_decoder_fields_unambiguous(self):
        image = torch.zeros((1, 1, 3, 4))
        decoded = DecodedSPNParameters(
            current=image,
            initial=image,
            raw_affinity=torch.zeros((1, 8, 3, 4)),
            residual_offsets_yx=torch.zeros((1, 8, 2, 3, 4)),
            affinity_scale=torch.ones(1),
        )
        self.assertEqual(
            decoded.residual_offsets_yx.shape,
            (1, 8, 2, 3, 4),
        )
        self.assertIsNone(decoded.attention_logits)
        self.assertIsNone(decoded.confidence_logits)


class _NLSPNFamilyDecoderContract:
    frontend_type = None
    config_factory = None
    input_type = None

    def _case(self):
        generator = torch.Generator().manual_seed(8102)
        config = self.config_factory(iterations=2)
        frontend = self.frontend_type(config)
        with torch.no_grad():
            frontend.conv_offset_aff.weight.copy_(
                torch.randn(
                    frontend.conv_offset_aff.weight.shape,
                    generator=generator,
                )
            )
            frontend.conv_offset_aff.bias.copy_(
                torch.randn(
                    frontend.conv_offset_aff.bias.shape,
                    generator=generator,
                )
            )
            frontend.aff_scale_const.copy_(torch.tensor([3.25]))
        initial = torch.randn((1, 1, 3, 4), generator=generator)
        guidance = torch.randn((1, 8, 3, 4), generator=generator)
        confidence = torch.sigmoid(
            torch.randn((1, 1, 3, 4), generator=generator)
        )
        sparse = torch.zeros_like(initial)
        if self.input_type is NLSPNRawInputs:
            inputs = self.input_type(
                initial,
                guidance,
                confidence_probability=confidence,
                feat_fix=sparse,
            )
        else:
            inputs = self.input_type(initial, guidance, confidence, sparse)
        return config, frontend, inputs, initial, guidance

    def test_decoder_matches_official_conv_chunk_cat_view_order(self):
        _, frontend, inputs, initial, guidance = self._case()
        raw = F.conv2d(
            guidance,
            frontend.conv_offset_aff.weight,
            frontend.conv_offset_aff.bias,
            padding=1,
        )
        o1, o2, expected_affinity = torch.chunk(raw, 3, dim=1)
        expected_offsets = torch.cat((o1, o2), dim=1).view(1, 8, 2, 3, 4)

        decoded = frontend.decode(inputs)

        torch.testing.assert_close(decoded.raw_affinity, expected_affinity)
        torch.testing.assert_close(decoded.residual_offsets_yx, expected_offsets)
        torch.testing.assert_close(decoded.affinity_scale, frontend.aff_scale_const)
        torch.testing.assert_close(decoded.current, initial)
        torch.testing.assert_close(decoded.initial, initial)

    def test_official_parameter_shapes_and_zero_initialization_match_release(self):
        config = self.config_factory(iterations=1)
        frontend = self.frontend_type(config)
        self.assertEqual(tuple(frontend.conv_offset_aff.weight.shape), (24, 8, 3, 3))
        self.assertEqual(tuple(frontend.conv_offset_aff.bias.shape), (24,))
        self.assertEqual(tuple(frontend.aff_scale_const.shape), (1,))
        self.assertEqual(torch.count_nonzero(frontend.conv_offset_aff.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(frontend.conv_offset_aff.bias).item(), 0)

    def test_explicit_prefix_loads_only_exact_official_parameters(self):
        config = self.config_factory(iterations=1)
        frontend = self.frontend_type(config)
        generator = torch.Generator().manual_seed(8103)
        weight = torch.randn(frontend.conv_offset_aff.weight.shape, generator=generator)
        bias = torch.randn(frontend.conv_offset_aff.bias.shape, generator=generator)
        scale = torch.tensor([2.75])
        state = {
            "module.prop_layer.conv_offset_aff.weight": weight,
            "module.prop_layer.conv_offset_aff.bias": bias,
            "module.prop_layer.aff_scale_const": scale,
        }
        frontend.load_official_parameters(state, prefix="module.prop_layer.")
        torch.testing.assert_close(frontend.conv_offset_aff.weight, weight)
        torch.testing.assert_close(frontend.conv_offset_aff.bias, bias)
        torch.testing.assert_close(frontend.aff_scale_const, scale)

        missing = dict(state)
        del missing["module.prop_layer.aff_scale_const"]
        with self.assertRaisesRegex(ValueError, "aff_scale_const"):
            frontend.load_official_parameters(missing, prefix="module.prop_layer.")

        wrong = dict(state)
        wrong["module.prop_layer.conv_offset_aff.bias"] = torch.zeros(23)
        with self.assertRaisesRegex(ValueError, "conv_offset_aff.bias"):
            frontend.load_official_parameters(wrong, prefix="module.prop_layer.")


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class NLSPNOfficialFrontendDecoderTest(_NLSPNFamilyDecoderContract, unittest.TestCase):
    frontend_type = NLSPNOfficialFrontend
    config_factory = SPNConfig.nlspn
    input_type = NLSPNRawInputs


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerOfficialFrontendDecoderTest(
    _NLSPNFamilyDecoderContract,
    unittest.TestCase,
):
    frontend_type = CompletionFormerOfficialFrontend
    config_factory = SPNConfig.completionformer
    input_type = CompletionFormerRawInputs


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNOfficialFrontendDecoderTest(unittest.TestCase):
    def test_all_released_k_values_match_official_conv_split_and_view(self):
        generator = torch.Generator().manual_seed(8201)
        for neighbors in (1, 3, 5, 9):
            with self.subTest(neighbors=neighbors):
                config = SPNConfig.dyspn(iterations=2, num_neighbors=neighbors)
                frontend = DySPNOfficialFrontend(config)
                with torch.no_grad():
                    frontend.conv_offset_aff.weight.copy_(
                        torch.randn(
                            frontend.conv_offset_aff.weight.shape,
                            generator=generator,
                        )
                    )
                    frontend.conv_offset_aff.bias.copy_(
                        torch.randn(
                            frontend.conv_offset_aff.bias.shape,
                            generator=generator,
                        )
                    )
                initial = torch.randn((1, 1, 3, 4), generator=generator)
                guide = torch.randn(
                    (1, 2 * neighbors, 3, 4),
                    generator=generator,
                )
                sparse = torch.zeros_like(initial)
                confidence_logits = torch.randn(
                    initial.shape,
                    generator=generator,
                )
                decoded = frontend.decode(
                    DySPNRawInputs(
                        initial,
                        guide,
                        sparse,
                        confidence_logits,
                    )
                )

                raw = F.conv2d(
                    guide,
                    frontend.conv_offset_aff.weight,
                    frontend.conv_offset_aff.bias,
                    padding=1,
                )
                offset_flat, affinity_flat = torch.split(
                    raw,
                    [4 * neighbors, 2 * neighbors],
                    dim=1,
                )
                expected_offsets = offset_flat.view(
                    1, 2, neighbors, 2, 3, 4
                )
                expected_logits = affinity_flat.view(1, 2, neighbors, 3, 4)
                torch.testing.assert_close(
                    decoded.residual_offsets_yx,
                    expected_offsets,
                )
                torch.testing.assert_close(decoded.raw_affinity, expected_logits)
                self.assertIs(decoded.confidence_logits, confidence_logits)
                self.assertEqual(
                    tuple(frontend.conv_offset_aff.weight.shape),
                    (6 * neighbors, 2 * neighbors, 3, 3),
                )

    def test_explicit_prefix_loads_exact_dyspn_conv_parameters(self):
        frontend = DySPNOfficialFrontend(SPNConfig.dyspn(iterations=2, num_neighbors=5))
        generator = torch.Generator().manual_seed(8202)
        weight = torch.randn(frontend.conv_offset_aff.weight.shape, generator=generator)
        bias = torch.randn(frontend.conv_offset_aff.bias.shape, generator=generator)
        state = {
            "module.dyspn_2_5.conv_offset_aff.weight": weight,
            "module.dyspn_2_5.conv_offset_aff.bias": bias,
        }
        frontend.load_official_parameters(state, prefix="module.dyspn_2_5.")
        torch.testing.assert_close(frontend.conv_offset_aff.weight, weight)
        torch.testing.assert_close(frontend.conv_offset_aff.bias, bias)

        missing = dict(state)
        del missing["module.dyspn_2_5.conv_offset_aff.bias"]
        with self.assertRaisesRegex(ValueError, "conv_offset_aff.bias"):
            frontend.load_official_parameters(missing, prefix="module.dyspn_2_5.")


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CSPNOfficialFrontendDecoderTest(unittest.TestCase):
    def test_maps_official_inputs_without_generating_parameters(self):
        guidance = torch.randn((1, 8, 3, 4))
        initial = torch.randn((1, 1, 3, 4))
        sparse = torch.zeros_like(initial)
        frontend = CSPNOfficialFrontend(SPNConfig.cspn(iterations=2))
        decoded = frontend.decode(CSPNRawInputs(guidance, initial, sparse))
        torch.testing.assert_close(decoded.current, initial)
        torch.testing.assert_close(decoded.initial, initial)
        torch.testing.assert_close(decoded.raw_affinity, guidance)
        self.assertIs(decoded.sparse_depth, sparse)
        self.assertIsNone(decoded.residual_offsets_yx)
        self.assertIsNone(decoded.attention_logits)
        self.assertEqual(sum(parameter.numel() for parameter in frontend.parameters()), 0)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNNLPMOfficialFrontendDecoderTest(unittest.TestCase):
    def test_reshapes_dynamic_logits_without_sigmoid(self):
        generator = torch.Generator().manual_seed(8301)
        initial = torch.randn((1, 1, 3, 4), generator=generator)
        guidance = torch.randn((1, 48, 3, 4), generator=generator)
        dynamic = torch.randn((1, 8, 3, 4), generator=generator)
        sparse = torch.zeros_like(initial)
        confidence = torch.sigmoid(torch.randn(initial.shape, generator=generator))
        frontend = DySPNNLPMOfficialFrontend(SPNConfig.dyspn_nlpm(iterations=2))
        decoded = frontend.decode(
            DySPNNLPMRawInputs(
                initial,
                guidance,
                dynamic,
                sparse,
                confidence,
            )
        )
        torch.testing.assert_close(
            decoded.attention_logits,
            dynamic.view(1, 2, 4, 3, 4),
        )
        torch.testing.assert_close(decoded.raw_affinity, guidance)
        self.assertIs(decoded.confidence_probability, confidence)
        self.assertEqual(sum(parameter.numel() for parameter in frontend.parameters()), 0)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class OfficialInputValidationTest(unittest.TestCase):
    def test_builder_table_contains_exactly_five_official_profiles(self):
        self.assertEqual(
            set(OFFICIAL_FRONTEND_BUILDERS),
            {
                SPNProfile.CSPN,
                SPNProfile.NLSPN,
                SPNProfile.COMPLETIONFORMER,
                SPNProfile.DYSPN,
                SPNProfile.DYSPN_NLPM,
            },
        )

    def test_each_official_rejects_another_profiles_input_type(self):
        image = torch.zeros((1, 1, 3, 4))
        wrong = CSPNRawInputs(torch.zeros((1, 8, 3, 4)), image)
        cases = (
            NLSPNOfficialFrontend(SPNConfig.nlspn()),
            CompletionFormerOfficialFrontend(SPNConfig.completionformer()),
            DySPNOfficialFrontend(SPNConfig.dyspn()),
            DySPNNLPMOfficialFrontend(SPNConfig.dyspn_nlpm()),
        )
        for frontend in cases:
            with self.subTest(frontend=type(frontend).__name__):
                with self.assertRaisesRegex(TypeError, "expects"):
                    frontend.decode(wrong)

    def test_cspn_and_nlpm_reject_wrong_official_channels(self):
        image = torch.zeros((1, 1, 3, 4))
        with self.assertRaisesRegex(ValueError, "guidance"):
            CSPNOfficialFrontend(SPNConfig.cspn()).decode(
                CSPNRawInputs(torch.zeros((1, 7, 3, 4)), image)
            )
        nlpm = DySPNNLPMOfficialFrontend(SPNConfig.dyspn_nlpm(iterations=2))
        with self.assertRaisesRegex(ValueError, "guidance"):
            nlpm.decode(
                DySPNNLPMRawInputs(
                    image,
                    torch.zeros((1, 47, 3, 4)),
                    torch.zeros((1, 8, 3, 4)),
                    image,
                    image,
                )
            )
        with self.assertRaisesRegex(ValueError, "dynamic_logits"):
            nlpm.decode(
                DySPNNLPMRawInputs(
                    image,
                    torch.zeros((1, 48, 3, 4)),
                    torch.zeros((1, 7, 3, 4)),
                    image,
                    image,
                )
            )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class IndependentOfficialDecodeTest(unittest.TestCase):
    def test_nlspn_family_matches_independent_conv_formula(self):
        generator = torch.Generator().manual_seed(8401)
        cases = (
            (
                NLSPNOfficialFrontend,
                SPNConfig.nlspn(iterations=2),
                NLSPNRawInputs,
                reference_nlspn_official_decode,
            ),
            (
                CompletionFormerOfficialFrontend,
                SPNConfig.completionformer(iterations=2),
                CompletionFormerRawInputs,
                reference_completionformer_official_decode,
            ),
        )
        for frontend_type, config, input_type, reference in cases:
            with self.subTest(profile=config.profile.name):
                frontend = frontend_type(config)
                with torch.no_grad():
                    frontend.conv_offset_aff.weight.copy_(
                        torch.randn(
                            frontend.conv_offset_aff.weight.shape,
                            generator=generator,
                        )
                    )
                    frontend.conv_offset_aff.bias.copy_(
                        torch.randn(
                            frontend.conv_offset_aff.bias.shape,
                            generator=generator,
                        )
                    )
                initial = torch.randn((1, 1, 3, 4), generator=generator)
                guidance = torch.randn((1, 8, 3, 4), generator=generator)
                confidence = torch.sigmoid(
                    torch.randn(initial.shape, generator=generator)
                )
                sparse = torch.zeros_like(initial)
                if input_type is NLSPNRawInputs:
                    inputs = input_type(
                        initial,
                        guidance,
                        confidence_probability=confidence,
                        feat_fix=sparse,
                    )
                else:
                    inputs = input_type(initial, guidance, confidence, sparse)
                decoded = frontend.decode(inputs)
                expected = reference(
                    guidance,
                    frontend.conv_offset_aff.weight,
                    frontend.conv_offset_aff.bias,
                )
                torch.testing.assert_close(
                    decoded.residual_offsets_yx,
                    expected["residual_offsets_yx"],
                )
                torch.testing.assert_close(
                    decoded.raw_affinity,
                    expected["raw_affinity"],
                )

    def test_dyspn_matches_independent_conv_formula(self):
        generator = torch.Generator().manual_seed(8402)
        config = SPNConfig.dyspn(iterations=2, num_neighbors=5)
        frontend = DySPNOfficialFrontend(config)
        with torch.no_grad():
            frontend.conv_offset_aff.weight.copy_(
                torch.randn(
                    frontend.conv_offset_aff.weight.shape,
                    generator=generator,
                )
            )
            frontend.conv_offset_aff.bias.copy_(
                torch.randn(
                    frontend.conv_offset_aff.bias.shape,
                    generator=generator,
                )
            )
        initial = torch.randn((1, 1, 3, 4), generator=generator)
        guide = torch.randn((1, 10, 3, 4), generator=generator)
        decoded = frontend.decode(
            DySPNRawInputs(initial, guide, torch.zeros_like(initial), initial)
        )
        expected = reference_dyspn_official_decode(
            guide,
            frontend.conv_offset_aff.weight,
            frontend.conv_offset_aff.bias,
            iterations=2,
            num_neighbors=5,
        )
        torch.testing.assert_close(
            decoded.residual_offsets_yx,
            expected["residual_offsets_yx"],
        )
        torch.testing.assert_close(
            decoded.raw_affinity,
            expected["raw_affinity"],
        )


if __name__ == "__main__":
    unittest.main()
