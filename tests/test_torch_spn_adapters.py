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
        compile_generic_plan,
        compile_nlspn_plan,
    )
    from spn_accel_cmodel.torch_spn_core import propagate_canonical
    from spn_accel_cmodel.torch_spn_types import (
        AffinityLayout,
        AffinityMode,
        AnchorMode,
        NeighborConfidenceMode,
        NeighborMode,
        NormalizationMode,
        OffsetMode,
        PaddingMode,
        ReductionMode,
        SPNConfig,
        SPNInputs,
        SamplingMode,
        SparseFusionMode,
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

    def _inputs(self, **overrides):
        values = dict(
            current=self.initial,
            affinity=self.affinity,
            offsets=self.residual_yx,
            confidence=self.confidence,
        )
        values.update(overrides)
        return SPNInputs(**values)

    def test_all_normalizations_compile_author_coefficients(self):
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
                    self._inputs(),
                    self.initial,
                    self.initial,
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
            self._inputs(sparse_depth=sparse),
            self.initial,
            self.initial,
        )
        torch.testing.assert_close(plan.pre_fusion_gate[:, 0], (sparse > 0).float())
        torch.testing.assert_close(plan.pre_fusion_value[:, 0], sparse)

    def test_released_18_channel_offset_layout_discards_center(self):
        with_center = torch.cat(
            (
                self.residual_yx[:, :4],
                torch.full((1, 1, 2, 4, 5), 99.0),
                self.residual_yx[:, 4:],
            ),
            dim=1,
        ).reshape(1, 18, 4, 5)
        config = SPNConfig.nlspn(iterations=1, confidence=False)
        canonical = compile_nlspn_plan(
            config,
            self._inputs(confidence=None),
            self.initial,
            self.initial,
        )
        flattened = compile_nlspn_plan(
            config,
            self._inputs(offsets=with_center, confidence=None),
            self.initial,
            self.initial,
        )
        torch.testing.assert_close(flattened.offsets_xy, canonical.offsets_xy)


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class CompletionFormerPlanTest(NLSPNPlanTest):
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
            self._inputs(),
            self.initial,
            self.initial,
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
                inputs = SPNInputs(
                    initial,
                    logits,
                    offsets=residual_yx,
                    confidence=confidence_logits,
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
                    inputs,
                    initial,
                    initial,
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
            SPNInputs(
                initial,
                guidance,
                attention=attention,
                sparse_depth=sparse,
                confidence=confidence,
            ),
            initial,
            initial,
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


@unittest.skipIf(torch is None, "torch optional validation dependency is unavailable")
class GenericPlanTest(unittest.TestCase):
    def test_absolute_initial_anchor_and_hard_post_compile_to_plan(self):
        config = SPNConfig(
            iterations=1,
            num_neighbors=1,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.STATIC,
            normalization=NormalizationMode.AS,
            anchor_mode=AnchorMode.INITIAL,
            sparse_fusion=SparseFusionMode.HARD_POST,
            affinity_layout=AffinityLayout.TARGET,
        )
        current = torch.ones((1, 1, 2, 3))
        initial = torch.zeros_like(current)
        sparse = torch.zeros_like(current)
        sparse[:, :, 0, 1] = 9.0
        inputs = SPNInputs(
            current,
            torch.full((1, 1, 2, 3), 0.5),
            initial=initial,
            offsets=torch.zeros((1, 1, 2, 2, 3)),
            sparse_depth=sparse,
        )
        plan = compile_generic_plan(config, inputs, current, initial)
        coefficient = 0.5 / (0.5 + 1.0e-4)
        torch.testing.assert_close(
            plan.neighbor_affinity,
            torch.full_like(plan.neighbor_affinity, coefficient),
        )
        torch.testing.assert_close(
            plan.initial_affinity,
            torch.full_like(plan.initial_affinity, 1.0 - coefficient),
        )
        torch.testing.assert_close(plan.post_fusion_gate[:, 0], (sparse > 0).float())
        actual = propagate_canonical(current, initial, plan)
        expected = torch.full_like(current, coefficient)
        expected[:, :, 0, 1] = 9.0
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_tgass_samples_confidence_after_tanh(self):
        config = SPNConfig(
            iterations=1,
            num_neighbors=1,
            neighbor_mode=NeighborMode.OFFSET,
            sampling_mode=SamplingMode.BILINEAR,
            padding_mode=PaddingMode.ZEROS,
            offset_mode=OffsetMode.ABSOLUTE_XY,
            affinity_mode=AffinityMode.STATIC,
            normalization=NormalizationMode.TGASS,
            anchor_mode=AnchorMode.INITIAL,
            neighbor_confidence=NeighborConfidenceMode.SAMPLE_AT_NEIGHBOR,
            affinity_gamma=0.5,
        )
        current = torch.full((1, 1, 2, 2), 4.0)
        initial = torch.zeros_like(current)
        plan = compile_generic_plan(
            config,
            SPNInputs(
                current,
                torch.full((1, 1, 2, 2), 2.0),
                initial=initial,
                offsets=torch.zeros((1, 1, 2, 2, 2)),
                confidence=torch.full((1, 1, 2, 2), 0.25),
            ),
            current,
            initial,
        )
        effective = torch.tanh(torch.tensor(2.0)) / 0.5 * 0.25
        torch.testing.assert_close(
            plan.neighbor_affinity,
            torch.full_like(plan.neighbor_affinity, effective),
        )


if __name__ == "__main__":
    unittest.main()
