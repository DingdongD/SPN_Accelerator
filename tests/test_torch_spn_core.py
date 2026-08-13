import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from spn_accel_cmodel import torch_spn_types
    from spn_accel_cmodel.torch_functional import (
        CanonicalSPNPlan,
        ReductionMode,
        SPNConfig,
        SPNProfile,
    )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CanonicalTypeTest(unittest.TestCase):
    def test_named_profiles_and_canonical_types_are_public(self):
        self.assertIs(SPNConfig.cspn().profile, SPNProfile.CSPN)
        self.assertIs(SPNConfig.nlspn().profile, SPNProfile.NLSPN)
        self.assertIs(
            SPNConfig.completionformer().profile,
            SPNProfile.COMPLETIONFORMER,
        )

    def test_public_types_are_the_canonical_type_objects(self):
        self.assertIs(SPNConfig, torch_spn_types.SPNConfig)
        self.assertIs(SPNProfile, torch_spn_types.SPNProfile)
        self.assertIs(SPNConfig.dyspn().profile, SPNProfile.DYSPN)
        self.assertTrue(hasattr(CanonicalSPNPlan, "__dataclass_fields__"))
        self.assertEqual(
            {member.name for member in ReductionMode},
            {"TORCH_SUM", "SEQUENTIAL", "GROUPED"},
        )


if __name__ == "__main__":
    unittest.main()
