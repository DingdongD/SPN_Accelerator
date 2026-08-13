import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from spn_accel_cmodel.torch_spn_decoded import DecodedSPNParameters
    from spn_accel_cmodel.torch_spn_types import (
        CSPNRawInputs,
        CompletionFormerRawInputs,
        DySPNNLPMRawInputs,
        DySPNRawInputs,
        NLSPNRawInputs,
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


if __name__ == "__main__":
    unittest.main()
