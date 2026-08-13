import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from official_spn_references import reference_cspn
    from spn_accel_cmodel.torch_spn_adapters import compile_cspn_plan
    from spn_accel_cmodel.torch_spn_core import propagate_canonical
    from spn_accel_cmodel.torch_spn_types import (
        ReductionMode,
        SPNConfig,
        SPNInputs,
    )


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CSPNPlanTest(unittest.TestCase):
    def test_shifted_source_guidance_compiles_to_target_plan(self):
        initial = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
        guidance = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1, 1)
        guidance = guidance.expand(1, 8, 3, 3)
        sparse = torch.zeros_like(initial)
        sparse[:, :, 1, 1] = 7.0
        config = SPNConfig.cspn(iterations=2, preserve_code_mask=True)
        inputs = SPNInputs(initial, guidance, sparse_depth=sparse)
        expected, _, metadata = reference_cspn(
            initial,
            guidance,
            iterations=2,
            sparse_depth=sparse,
        )

        plan = compile_cspn_plan(config, inputs, initial, initial)

        self.assertEqual(plan.offsets_xy.shape, (1, 1, 8, 2, 3, 3))
        torch.testing.assert_close(
            plan.offsets_xy[0, 0, 0, :, 1, 1],
            torch.tensor([1.0, 1.0]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            plan.neighbor_affinity[:, 0, :, 0],
            metadata["neighbor_affinity"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            plan.initial_affinity[:, 0],
            metadata["initial_affinity"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            plan.current_affinity,
            torch.zeros_like(plan.current_affinity),
        )
        torch.testing.assert_close(plan.post_fusion_gate[:, 0], sparse.sign())
        torch.testing.assert_close(plan.post_fusion_value[:, 0], initial)
        torch.testing.assert_close(
            plan.pre_fusion_gate,
            torch.zeros_like(plan.pre_fusion_gate),
        )
        self.assertIs(plan.reduction_mode, ReductionMode.TORCH_SUM)
        actual = propagate_canonical(initial, initial, plan)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
