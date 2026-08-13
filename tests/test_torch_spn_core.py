import unittest
import ast
import inspect
from dataclasses import replace


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from spn_accel_cmodel import (
        torch_functional,
        torch_spn_adapters,
        torch_spn_author,
        torch_spn_core,
        torch_spn_types,
    )
    from spn_accel_cmodel.torch_spn_core import sample_neighbors as core_sample_neighbors
    from spn_accel_cmodel.torch_functional import (
        CanonicalSPNPlan,
        PaddingMode,
        ReductionMode,
        SPNConfig,
        SPNProfile,
        SamplingMode,
        UnifiedSPN,
        propagate_canonical,
        validate_plan,
    )


def one_neighbor_plan(**overrides):
    zeros = torch.zeros((1, 1, 1, 1, 3), dtype=torch.float32)
    values = dict(
        iterations=1,
        offsets_xy=torch.zeros((1, 1, 1, 2, 1, 3), dtype=torch.float32),
        neighbor_affinity=torch.ones((1, 1, 1, 1, 1, 3), dtype=torch.float32),
        current_affinity=zeros.clone(),
        initial_affinity=zeros.clone(),
        pre_fusion_gate=zeros.clone(),
        pre_fusion_value=zeros.clone(),
        post_fusion_gate=zeros.clone(),
        post_fusion_value=zeros.clone(),
        sampling_mode=SamplingMode.BILINEAR,
        padding_mode=PaddingMode.ZEROS,
        align_corners=True,
        reduction_mode=ReductionMode.TORCH_SUM,
        reduction_groups=((0, 1),),
        group_scale=torch.ones((1, 1, 1, 1, 1, 3), dtype=torch.float32),
    )
    values.update(overrides)
    return CanonicalSPNPlan(**values)


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

    def test_package_exports_canonical_plan_and_core(self):
        from spn_accel_cmodel import (
            CanonicalSPNPlan as PackagePlan,
            ReductionMode as PackageReductionMode,
            propagate_canonical as package_core,
        )

        self.assertIs(PackagePlan, CanonicalSPNPlan)
        self.assertIs(PackageReductionMode, ReductionMode)
        self.assertIs(package_core, propagate_canonical)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CanonicalCoreTest(unittest.TestCase):
    def test_one_core_applies_pre_anchor_and_post_fusion(self):
        current = torch.tensor([[[[2.0, 4.0, 8.0]]]])
        initial = torch.ones((1, 1, 1, 3))
        plan = one_neighbor_plan(
            neighbor_affinity=torch.zeros((1, 1, 1, 1, 1, 3)),
            current_affinity=torch.full((1, 1, 1, 1, 3), 0.25),
            initial_affinity=torch.full((1, 1, 1, 1, 3), 0.75),
            pre_fusion_gate=torch.tensor([[[[[0.0, 1.0, 0.0]]]]]),
            pre_fusion_value=torch.tensor([[[[[0.0, 6.0, 0.0]]]]]),
            post_fusion_gate=torch.tensor([[[[[0.0, 0.0, 1.0]]]]]),
            post_fusion_value=torch.tensor([[[[[0.0, 0.0, 9.0]]]]]),
        )
        actual, trace = propagate_canonical(
            current,
            initial,
            plan,
            return_trace=True,
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([[[[1.25, 2.25, 9.0]]]]),
        )
        self.assertEqual(len(trace.pre_fused_states), 1)
        self.assertEqual(len(trace.group_scales), 1)

    def test_plan_rejects_wrong_time_dimension(self):
        state = torch.zeros((1, 1, 1, 3))
        plan = one_neighbor_plan(
            offsets_xy=torch.zeros((1, 2, 1, 2, 1, 3)),
        )
        with self.assertRaisesRegex(ValueError, "offsets_xy time dimension"):
            validate_plan(state, state, plan)

    def test_plan_rejects_low_rank_fields_with_named_value_error(self):
        state = torch.zeros((1, 1, 1, 3))
        fields = (
            "offsets_xy",
            "neighbor_affinity",
            "current_affinity",
            "initial_affinity",
            "pre_fusion_gate",
            "pre_fusion_value",
            "post_fusion_gate",
            "post_fusion_value",
            "group_scale",
        )
        for name in fields:
            for malformed in (torch.tensor(0.0), torch.zeros(1)):
                with self.subTest(field=name, rank=malformed.ndim):
                    plan = replace(one_neighbor_plan(), **{name: malformed})
                    with self.assertRaisesRegex(ValueError, name):
                        validate_plan(state, state, plan)

    def test_grouped_reduction_preserves_declared_group_order(self):
        current = torch.tensor([[[[1.0, 2.0, 4.0]]]])
        offsets = torch.zeros((1, 1, 3, 2, 1, 3))
        offsets[0, 0, 0, 0] = -1.0
        offsets[0, 0, 2, 0] = 1.0
        plan = one_neighbor_plan(
            offsets_xy=offsets,
            neighbor_affinity=torch.ones((1, 1, 3, 1, 1, 3)),
            reduction_mode=ReductionMode.GROUPED,
            reduction_groups=((0, 2), (2, 3)),
            group_scale=torch.tensor([2.0, 3.0])
            .view(1, 1, 2, 1, 1, 1)
            .expand(1, 1, 2, 1, 1, 3),
        )
        actual = propagate_canonical(current, current, plan)
        torch.testing.assert_close(actual[0, 0, 0, 1], torch.tensor(18.0))

    def test_sequential_reduction_matches_explicit_fp32_accumulation(self):
        current = torch.tensor(
            [[[[8.0, 1.0e7, -1.0e8],
               [-1.0e7, 1.0e8, 4.0],
               [1.0e7, -2.0, -2.0]]]]
        )
        offsets = torch.tensor(
            [
                (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
                (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
                (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0),
            ]
        ).view(1, 1, 9, 2, 1, 1).expand(1, 1, 9, 2, 3, 3)
        zeros = torch.zeros((1, 1, 1, 3, 3))
        plan = CanonicalSPNPlan(
            iterations=1,
            offsets_xy=offsets,
            neighbor_affinity=torch.full((1, 1, 9, 1, 3, 3), 1.0 / 9.0),
            current_affinity=zeros,
            initial_affinity=zeros,
            pre_fusion_gate=zeros,
            pre_fusion_value=zeros,
            post_fusion_gate=zeros,
            post_fusion_value=zeros,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            align_corners=False,
            reduction_mode=ReductionMode.SEQUENTIAL,
            reduction_groups=((0, 9),),
            group_scale=torch.ones((1, 1, 1, 1, 3, 3)),
        )
        actual = propagate_canonical(current, current, plan)
        samples = core_sample_neighbors(
            current,
            offsets[:, 0],
            SamplingMode.BILINEAR,
            PaddingMode.ZEROS,
            align_corners=False,
        )
        accumulator = torch.tensor(0.0)
        for value in samples[0, :, 0, 1, 1]:
            accumulator = accumulator + value / 9.0
        torch.testing.assert_close(
            actual[0, 0, 1, 1],
            accumulator,
            rtol=0.0,
            atol=0.0,
        )

    def test_plan_allows_author_nan_values(self):
        state = torch.zeros((1, 1, 1, 3))
        plan = replace(
            one_neighbor_plan(),
            neighbor_affinity=torch.full((1, 1, 1, 1, 1, 3), torch.nan),
        )
        validate_plan(state, state, plan)
        self.assertTrue(torch.isnan(propagate_canonical(state, state, plan)).all())


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class StructuralUnificationTest(unittest.TestCase):
    def test_only_canonical_core_loops_over_iterations(self):
        offenders = []

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function = None

            def visit_FunctionDef(self, node):
                previous = self.function
                self.function = node.name
                self.generic_visit(node)
                self.function = previous

            def visit_For(self, node):
                iterator = ast.unparse(node.iter)
                if "iterations" in iterator:
                    offenders.append((self.function, iterator))
                self.generic_visit(node)

        for module in (
            torch_functional,
            torch_spn_author,
            torch_spn_core,
            torch_spn_adapters,
        ):
            LoopVisitor().visit(ast.parse(inspect.getsource(module)))
        self.assertEqual(
            offenders,
            [("propagate_canonical", "range(plan.iterations)")],
        )

    def test_profile_specific_forward_methods_are_removed(self):
        forbidden = {
            "_forward_cspn",
            "_forward_nlspn",
            "_forward_dyspn",
            "_forward_dyspn_nlpm",
            "_forward_generic",
        }
        self.assertTrue(forbidden.isdisjoint(dir(UnifiedSPN)))

if __name__ == "__main__":
    unittest.main()
