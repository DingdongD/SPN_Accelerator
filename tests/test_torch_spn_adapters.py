import unittest


try:
    import torch
except ImportError:  # pragma: no cover - optional validation dependency
    torch = None

if torch is not None:
    from official_spn_references import (
        reference_cspn,
        reference_dyspn,
        reference_dyspn_nlpm,
        reference_nlspn,
    )
    from spn_accel_cmodel.torch_spn_adapters import (
        compile_completionformer_plan,
        compile_cspn_plan,
        compile_dyspn_plan,
        compile_dyspn_nlpm_plan,
        compile_nlspn_plan,
    )
    from spn_accel_cmodel.torch_spn_decoded import DecodedSPNParameters
    from spn_accel_cmodel.torch_spn_core import propagate_canonical
    from spn_accel_cmodel.torch_spn_types import (
        NormalizationMode,
        ReductionMode,
        SPNConfig,
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
        decoded = DecodedSPNParameters(
            current=initial,
            initial=initial,
            raw_affinity=guidance,
            sparse_depth=sparse,
        )
        expected, _, metadata = reference_cspn(
            initial,
            guidance,
            iterations=2,
            sparse_depth=sparse,
        )

        plan = compile_cspn_plan(config, decoded)

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


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class NLSPNPlanTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(4201)
        self.initial = torch.randn((1, 1, 4, 5), generator=generator)
        self.affinity = torch.randn((1, 8, 4, 5), generator=generator) * 0.4
        self.residual_yx = torch.randn(
            (1, 8, 2, 4, 5), generator=generator
        ) * 0.2
        self.confidence = torch.sigmoid(
            torch.randn((1, 1, 4, 5), generator=generator)
        )

    def _decoded(self, mode=NormalizationMode.TGASS, **overrides):
        if mode is NormalizationMode.TC:
            scale = 8.0
        elif mode is NormalizationMode.TGASS:
            scale = 4.0
        else:
            scale = 1.0
        values = dict(
            current=self.initial,
            initial=self.initial,
            raw_affinity=self.affinity,
            residual_offsets_yx=self.residual_yx,
            confidence_probability=self.confidence,
            affinity_scale=torch.tensor([scale]),
        )
        values.update(overrides)
        return DecodedSPNParameters(**values)

    def test_all_normalizations_compile_official_coefficients(self):
        for mode in (
            NormalizationMode.AS,
            NormalizationMode.ASS,
            NormalizationMode.TC,
            NormalizationMode.TGASS,
        ):
            with self.subTest(mode=mode.name):
                config = SPNConfig.nlspn(iterations=2, normalization=mode)
                expected, _, affinity, center, metadata = reference_nlspn(
                    self.initial,
                    self.affinity,
                    self.residual_yx,
                    iterations=2,
                    mode=mode.name,
                    confidence=self.confidence,
                )
                plan = compile_nlspn_plan(
                    config,
                    self._decoded(mode),
                )
                torch.testing.assert_close(
                    plan.offsets_xy[:, 0],
                    metadata["offsets"],
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    plan.neighbor_affinity[:, 0, :, 0],
                    affinity,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    plan.current_affinity[:, 0],
                    center,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    plan.initial_affinity,
                    torch.zeros_like(plan.initial_affinity),
                )
                actual = propagate_canonical(self.initial, self.initial, plan)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_preserve_input_compiles_to_pre_fusion(self):
        sparse = torch.zeros_like(self.initial)
        sparse[:, :, 1, 2] = 6.0
        plan = compile_nlspn_plan(
            SPNConfig.nlspn(iterations=2, preserve_input=True),
            self._decoded(sparse_depth=sparse),
        )
        torch.testing.assert_close(plan.pre_fusion_gate[:, 0], (sparse > 0).float())
        torch.testing.assert_close(plan.pre_fusion_value[:, 0], sparse)

    def test_tgass_uses_loaded_affinity_scale(self):
        config = SPNConfig.nlspn(iterations=1)
        loaded_scale = 3.25
        expected, _, affinity, center, _ = reference_nlspn(
            self.initial,
            self.affinity,
            self.residual_yx,
            iterations=1,
            mode="TGASS",
            confidence=self.confidence,
            gamma=loaded_scale / config.num_neighbors,
        )
        decoded = self._decoded(
            affinity_scale=torch.tensor([loaded_scale]),
        )
        plan = compile_nlspn_plan(config, decoded)

        torch.testing.assert_close(
            plan.neighbor_affinity[:, 0, :, 0],
            affinity,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(plan.current_affinity[:, 0], center)
        actual = propagate_canonical(self.initial, self.initial, plan)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerPlanTest(unittest.TestCase):
    setUp = NLSPNPlanTest.setUp
    _decoded = NLSPNPlanTest._decoded

    def test_temperature_100_compiles_completionformer_coefficients(self):
        config = SPNConfig.completionformer(iterations=2)
        expected, _, affinity, center, _ = reference_nlspn(
            self.initial,
            self.affinity,
            self.residual_yx,
            iterations=2,
            mode="TGASS",
            confidence=self.confidence,
            temperature=100.0,
        )
        plan = compile_completionformer_plan(
            config,
            self._decoded(),
        )
        torch.testing.assert_close(
            plan.neighbor_affinity[:, 0, :, 0],
            affinity,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(plan.current_affinity[:, 0], center)
        actual = propagate_canonical(self.initial, self.initial, plan)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNPlanTest(unittest.TestCase):
    def test_all_released_stencils_compile_per_iteration_metadata(self):
        for neighbors in (1, 3, 5, 9):
            with self.subTest(neighbors=neighbors):
                generator = torch.Generator().manual_seed(8100 + neighbors)
                steps, height, width = 3, 3, 4
                initial = torch.randn((1, 1, height, width), generator=generator)
                residual_yx = torch.randn(
                    (1, steps, neighbors, 2, height, width),
                    generator=generator,
                ) * 0.2
                logits = torch.randn(
                    (1, steps, neighbors, height, width),
                    generator=generator,
                )
                sparse = torch.zeros_like(initial)
                sparse[:, :, 1, 2] = 5.0
                confidence_logits = torch.randn(
                    (1, 1, height, width),
                    generator=generator,
                )
                decoded = DecodedSPNParameters(
                    current=initial,
                    initial=initial,
                    raw_affinity=logits,
                    residual_offsets_yx=residual_yx,
                    confidence_logits=confidence_logits,
                    sparse_depth=sparse,
                )
                expected, _, _, metadata = reference_dyspn(
                    initial,
                    residual_yx,
                    logits,
                    sparse,
                    confidence_logits,
                )
                plan = compile_dyspn_plan(
                    SPNConfig.dyspn(iterations=steps, num_neighbors=neighbors),
                    decoded,
                )
                self.assertEqual(
                    plan.offsets_xy.shape,
                    (1, steps, neighbors, 2, height, width),
                )
                self.assertEqual(
                    plan.neighbor_affinity.shape,
                    (1, steps, neighbors, 1, height, width),
                )
                torch.testing.assert_close(
                    plan.neighbor_affinity[:, :, :, 0],
                    torch.softmax(logits, dim=2),
                )
                for iteration, expected_offsets in enumerate(metadata["offsets"]):
                    torch.testing.assert_close(
                        plan.offsets_xy[:, iteration],
                        expected_offsets,
                        rtol=0.0,
                        atol=0.0,
                    )
                self.assertIs(plan.reduction_mode, ReductionMode.SEQUENTIAL)
                torch.testing.assert_close(
                    plan.post_fusion_gate[:, 0],
                    torch.sigmoid(confidence_logits) * sparse.sign(),
                )
                torch.testing.assert_close(plan.post_fusion_value[:, 0], sparse)
                torch.testing.assert_close(
                    plan.current_affinity,
                    torch.zeros_like(plan.current_affinity),
                )
                actual = propagate_canonical(initial, initial, plan)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class DySPNNLPMPlanTest(unittest.TestCase):
    def test_multiscale_groups_compile_to_48_canonical_neighbors(self):
        generator = torch.Generator().manual_seed(7721)
        steps, height, width = 3, 5, 6
        initial = torch.randn((1, 1, height, width), generator=generator)
        guidance = torch.randn((1, 48, height, width), generator=generator) * 0.2
        attention = torch.randn(
            (1, steps, 4, height, width),
            generator=generator,
        )
        sparse = torch.zeros_like(initial)
        sparse[:, :, 2, 3] = 7.0
        confidence = torch.sigmoid(
            torch.randn((1, 1, height, width), generator=generator)
        )
        expected, _, _, metadata = reference_dyspn_nlpm(
            initial,
            guidance,
            attention,
            sparse,
            confidence,
        )
        config = SPNConfig.dyspn_nlpm(iterations=steps)
        plan = compile_dyspn_nlpm_plan(
            config,
            DecodedSPNParameters(
                current=initial,
                initial=initial,
                raw_affinity=guidance,
                attention_logits=attention,
                sparse_depth=sparse,
                confidence_probability=confidence,
            ),
        )
        self.assertEqual(plan.reduction_groups, ((0, 8), (8, 24), (24, 48)))
        self.assertIs(plan.reduction_mode, ReductionMode.GROUPED)
        self.assertEqual(plan.offsets_xy.shape, (1, 1, 48, 2, height, width))
        self.assertEqual(plan.group_scale.shape, (1, steps, 3, 1, height, width))
        torch.testing.assert_close(
            plan.neighbor_affinity[:, 0, :, 0],
            metadata["shifted_guidance"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            plan.group_scale,
            metadata["group_scale"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            plan.current_affinity,
            metadata["current_affinity"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            plan.initial_affinity,
            metadata["initial_affinity"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        actual = propagate_canonical(initial, initial, plan)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

if __name__ == "__main__":
    unittest.main()
