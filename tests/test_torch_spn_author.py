import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    import torch.nn.functional as F

    from spn_accel_cmodel.torch_spn_author import (
        AUTHOR_BUILDERS,
        CSPNAuthor,
        CompletionFormerAuthor,
        DySPNNLPMAuthor,
        DySPNAuthor,
        NLSPNAuthor,
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


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class AuthorInputTypeTest(unittest.TestCase):
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
    author_type = None
    config_factory = None
    input_type = None

    def _case(self):
        generator = torch.Generator().manual_seed(8102)
        config = self.config_factory(iterations=2)
        author = self.author_type(config)
        with torch.no_grad():
            author.conv_offset_aff.weight.copy_(
                torch.randn(
                    author.conv_offset_aff.weight.shape,
                    generator=generator,
                )
            )
            author.conv_offset_aff.bias.copy_(
                torch.randn(
                    author.conv_offset_aff.bias.shape,
                    generator=generator,
                )
            )
            author.aff_scale_const.copy_(torch.tensor([3.25]))
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
        return config, author, inputs, initial, guidance

    def test_decoder_matches_author_conv_chunk_cat_view_order(self):
        _, author, inputs, initial, guidance = self._case()
        raw = F.conv2d(
            guidance,
            author.conv_offset_aff.weight,
            author.conv_offset_aff.bias,
            padding=1,
        )
        o1, o2, expected_affinity = torch.chunk(raw, 3, dim=1)
        expected_offsets = torch.cat((o1, o2), dim=1).view(1, 8, 2, 3, 4)

        decoded = author.decode(inputs)

        torch.testing.assert_close(decoded.raw_affinity, expected_affinity)
        torch.testing.assert_close(decoded.residual_offsets_yx, expected_offsets)
        torch.testing.assert_close(decoded.affinity_scale, author.aff_scale_const)
        torch.testing.assert_close(decoded.current, initial)
        torch.testing.assert_close(decoded.initial, initial)

    def test_author_parameter_shapes_and_zero_initialization_match_release(self):
        config = self.config_factory(iterations=1)
        author = self.author_type(config)
        self.assertEqual(tuple(author.conv_offset_aff.weight.shape), (24, 8, 3, 3))
        self.assertEqual(tuple(author.conv_offset_aff.bias.shape), (24,))
        self.assertEqual(tuple(author.aff_scale_const.shape), (1,))
        self.assertEqual(torch.count_nonzero(author.conv_offset_aff.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(author.conv_offset_aff.bias).item(), 0)

    def test_explicit_prefix_loads_only_exact_official_parameters(self):
        config = self.config_factory(iterations=1)
        author = self.author_type(config)
        generator = torch.Generator().manual_seed(8103)
        weight = torch.randn(author.conv_offset_aff.weight.shape, generator=generator)
        bias = torch.randn(author.conv_offset_aff.bias.shape, generator=generator)
        scale = torch.tensor([2.75])
        state = {
            "module.prop_layer.conv_offset_aff.weight": weight,
            "module.prop_layer.conv_offset_aff.bias": bias,
            "module.prop_layer.aff_scale_const": scale,
        }
        author.load_official_parameters(state, prefix="module.prop_layer.")
        torch.testing.assert_close(author.conv_offset_aff.weight, weight)
        torch.testing.assert_close(author.conv_offset_aff.bias, bias)
        torch.testing.assert_close(author.aff_scale_const, scale)

        missing = dict(state)
        del missing["module.prop_layer.aff_scale_const"]
        with self.assertRaisesRegex(ValueError, "aff_scale_const"):
            author.load_official_parameters(missing, prefix="module.prop_layer.")

        wrong = dict(state)
        wrong["module.prop_layer.conv_offset_aff.bias"] = torch.zeros(23)
        with self.assertRaisesRegex(ValueError, "conv_offset_aff.bias"):
            author.load_official_parameters(wrong, prefix="module.prop_layer.")


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class NLSPNAuthorDecoderTest(_NLSPNFamilyDecoderContract, unittest.TestCase):
    author_type = NLSPNAuthor
    config_factory = SPNConfig.nlspn
    input_type = NLSPNRawInputs


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerAuthorDecoderTest(
    _NLSPNFamilyDecoderContract,
    unittest.TestCase,
):
    author_type = CompletionFormerAuthor
    config_factory = SPNConfig.completionformer
    input_type = CompletionFormerRawInputs


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNAuthorDecoderTest(unittest.TestCase):
    def test_all_released_k_values_match_author_conv_split_and_view(self):
        generator = torch.Generator().manual_seed(8201)
        for neighbors in (1, 3, 5, 9):
            with self.subTest(neighbors=neighbors):
                config = SPNConfig.dyspn(iterations=2, num_neighbors=neighbors)
                author = DySPNAuthor(config)
                with torch.no_grad():
                    author.conv_offset_aff.weight.copy_(
                        torch.randn(
                            author.conv_offset_aff.weight.shape,
                            generator=generator,
                        )
                    )
                    author.conv_offset_aff.bias.copy_(
                        torch.randn(
                            author.conv_offset_aff.bias.shape,
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
                decoded = author.decode(
                    DySPNRawInputs(
                        initial,
                        guide,
                        sparse,
                        confidence_logits,
                    )
                )

                raw = F.conv2d(
                    guide,
                    author.conv_offset_aff.weight,
                    author.conv_offset_aff.bias,
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
                    tuple(author.conv_offset_aff.weight.shape),
                    (6 * neighbors, 2 * neighbors, 3, 3),
                )

    def test_explicit_prefix_loads_exact_dyspn_conv_parameters(self):
        author = DySPNAuthor(SPNConfig.dyspn(iterations=2, num_neighbors=5))
        generator = torch.Generator().manual_seed(8202)
        weight = torch.randn(author.conv_offset_aff.weight.shape, generator=generator)
        bias = torch.randn(author.conv_offset_aff.bias.shape, generator=generator)
        state = {
            "module.dyspn_2_5.conv_offset_aff.weight": weight,
            "module.dyspn_2_5.conv_offset_aff.bias": bias,
        }
        author.load_official_parameters(state, prefix="module.dyspn_2_5.")
        torch.testing.assert_close(author.conv_offset_aff.weight, weight)
        torch.testing.assert_close(author.conv_offset_aff.bias, bias)

        missing = dict(state)
        del missing["module.dyspn_2_5.conv_offset_aff.bias"]
        with self.assertRaisesRegex(ValueError, "conv_offset_aff.bias"):
            author.load_official_parameters(missing, prefix="module.dyspn_2_5.")


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CSPNAuthorDecoderTest(unittest.TestCase):
    def test_maps_official_inputs_without_generating_parameters(self):
        guidance = torch.randn((1, 8, 3, 4))
        initial = torch.randn((1, 1, 3, 4))
        sparse = torch.zeros_like(initial)
        author = CSPNAuthor(SPNConfig.cspn(iterations=2))
        decoded = author.decode(CSPNRawInputs(guidance, initial, sparse))
        torch.testing.assert_close(decoded.current, initial)
        torch.testing.assert_close(decoded.initial, initial)
        torch.testing.assert_close(decoded.raw_affinity, guidance)
        self.assertIs(decoded.sparse_depth, sparse)
        self.assertIsNone(decoded.residual_offsets_yx)
        self.assertIsNone(decoded.attention_logits)
        self.assertEqual(sum(parameter.numel() for parameter in author.parameters()), 0)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNNLPMAuthorDecoderTest(unittest.TestCase):
    def test_reshapes_dynamic_logits_without_sigmoid(self):
        generator = torch.Generator().manual_seed(8301)
        initial = torch.randn((1, 1, 3, 4), generator=generator)
        guidance = torch.randn((1, 48, 3, 4), generator=generator)
        dynamic = torch.randn((1, 8, 3, 4), generator=generator)
        sparse = torch.zeros_like(initial)
        confidence = torch.sigmoid(torch.randn(initial.shape, generator=generator))
        author = DySPNNLPMAuthor(SPNConfig.dyspn_nlpm(iterations=2))
        decoded = author.decode(
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
        self.assertEqual(sum(parameter.numel() for parameter in author.parameters()), 0)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class AuthorInputValidationTest(unittest.TestCase):
    def test_builder_table_contains_exactly_five_author_profiles(self):
        self.assertEqual(
            set(AUTHOR_BUILDERS),
            {
                SPNProfile.CSPN,
                SPNProfile.NLSPN,
                SPNProfile.COMPLETIONFORMER,
                SPNProfile.DYSPN,
                SPNProfile.DYSPN_NLPM,
            },
        )

    def test_each_author_rejects_another_profiles_input_type(self):
        image = torch.zeros((1, 1, 3, 4))
        wrong = CSPNRawInputs(torch.zeros((1, 8, 3, 4)), image)
        cases = (
            NLSPNAuthor(SPNConfig.nlspn()),
            CompletionFormerAuthor(SPNConfig.completionformer()),
            DySPNAuthor(SPNConfig.dyspn()),
            DySPNNLPMAuthor(SPNConfig.dyspn_nlpm()),
        )
        for author in cases:
            with self.subTest(author=type(author).__name__):
                with self.assertRaisesRegex(TypeError, "expects"):
                    author.decode(wrong)

    def test_cspn_and_nlpm_reject_wrong_author_channels(self):
        image = torch.zeros((1, 1, 3, 4))
        with self.assertRaisesRegex(ValueError, "guidance"):
            CSPNAuthor(SPNConfig.cspn()).decode(
                CSPNRawInputs(torch.zeros((1, 7, 3, 4)), image)
            )
        nlpm = DySPNNLPMAuthor(SPNConfig.dyspn_nlpm(iterations=2))
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


if __name__ == "__main__":
    unittest.main()
