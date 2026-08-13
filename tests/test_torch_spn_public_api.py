import unittest
from unittest import mock


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    import spn_accel_cmodel.torch_functional as torch_functional
    import spn_accel_cmodel.torch_spn_adapters as torch_spn_adapters
    import spn_accel_cmodel.torch_spn_types as torch_spn_types
    from spn_accel_cmodel.torch_functional import UnifiedSPN
    from spn_accel_cmodel.torch_spn_types import (
        CSPNRawInputs,
        CompletionFormerRawInputs,
        DySPNNLPMRawInputs,
        DySPNRawInputs,
        NLSPNRawInputs,
        SPNConfig,
    )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class PublicAuthorAPIContractTest(unittest.TestCase):
    def test_legacy_generic_api_is_removed(self):
        old_input = "SPN" + "Inputs"
        old_profile = "GEN" + "ERIC"
        old_compiler = "compile_" + "generic_plan"
        self.assertFalse(hasattr(torch_spn_types, old_input))
        self.assertFalse(hasattr(torch_spn_types.SPNProfile, old_profile))
        self.assertFalse(hasattr(torch_spn_adapters, old_compiler))
        self.assertNotIn(old_input, torch_functional.__all__)
        self.assertNotIn(old_compiler, torch_functional.__all__)

    def test_each_author_profile_calls_canonical_core_once(self):
        state = torch.ones((1, 1, 3, 4))
        zeros = torch.zeros_like(state)
        cases = (
            (
                SPNConfig.cspn(iterations=1),
                CSPNRawInputs(torch.ones((1, 8, 3, 4)), state),
            ),
            (
                SPNConfig.nlspn(iterations=1),
                NLSPNRawInputs(state, torch.ones((1, 8, 3, 4)), state),
            ),
            (
                SPNConfig.completionformer(iterations=1),
                CompletionFormerRawInputs(
                    state,
                    torch.ones((1, 8, 3, 4)),
                    state,
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
                with mock.patch.object(
                    torch_functional,
                    "propagate_canonical",
                    return_value=state,
                ) as canonical:
                    UnifiedSPN(config)(inputs)
                canonical.assert_called_once()


if __name__ == "__main__":
    unittest.main()
