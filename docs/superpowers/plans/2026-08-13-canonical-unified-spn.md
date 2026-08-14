# Canonical Unified SPN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CSPN, NLSPN, CompletionFormer, and DySPN profile-specific propagation loops with metadata adapters and one canonical Torch FP32 recurrence.

**Architecture:** Split the current monolithic Torch module into shared types, a profile-agnostic canonical core, and metadata-only adapters. `UnifiedSPN` retains the public API, selects one adapter, and calls `propagate_canonical()` exactly once; only that core may iterate the evolving state.

**Tech Stack:** Python 3.11, PyTorch FP32, dataclasses, enums, unittest, AST structural tests.

---

## File Structure

- Create `spn_accel_cmodel/torch_spn_types.py` for enums, configuration,
  inputs, traces, stencil constants, and `CanonicalSPNPlan`.
- Create `spn_accel_cmodel/torch_spn_core.py` for plan validation, sampling,
  reduction, fusion, and the sole propagation loop.
- Create `spn_accel_cmodel/torch_spn_adapters.py` for author-layout-to-plan
  compilers. These functions never propagate evolving depth.
- Replace `spn_accel_cmodel/torch_functional.py` with compatibility exports and
  the small `UnifiedSPN` wrapper.
- Create `tests/test_torch_spn_core.py` and
  `tests/test_torch_spn_adapters.py` for direct core and adapter tests.
- Modify `tests/official_spn_references.py`,
  `tests/test_torch_spn_goldens.py`, and `tests/test_recorded_trace.py` for
  independent metadata and end-to-end validation.
- Modify `spn_accel_cmodel/__init__.py` and `README.md` for exports and usage.

### Task 1: Introduce canonical types without changing propagation

**Files:**
- Create: `spn_accel_cmodel/torch_spn_types.py`
- Modify: `spn_accel_cmodel/torch_functional.py`
- Create: `tests/test_torch_spn_core.py`

- [ ] **Step 1: Write the failing public-type test**

```python
import unittest

from spn_accel_cmodel.torch_functional import (
    CanonicalSPNPlan,
    ReductionMode,
    SPNConfig,
    SPNProfile,
)


class CanonicalTypeTest(unittest.TestCase):
    def test_named_profiles_and_canonical_types_are_public(self):
        self.assertIs(SPNConfig.cspn().profile, SPNProfile.CSPN)
        self.assertIs(SPNConfig.nlspn().profile, SPNProfile.NLSPN)
        self.assertIs(SPNConfig.completionformer().profile, SPNProfile.COMPLETIONFORMER)
        self.assertIs(SPNConfig.dyspn().profile, SPNProfile.DYSPN)
        self.assertTrue(hasattr(CanonicalSPNPlan, "__dataclass_fields__"))
        self.assertEqual(
            {member.name for member in ReductionMode},
            {"TORCH_SUM", "SEQUENTIAL", "GROUPED"},
        )
```

- [ ] **Step 2: Run the test and verify the missing imports fail**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_core.CanonicalTypeTest -v
```

Expected: import failure for `CanonicalSPNPlan` or `ReductionMode`.

- [ ] **Step 3: Extract shared types and add the canonical dataclasses**

Move every existing enum, stencil constant, `SPNConfig`, and `SPNInputs` into
`torch_spn_types.py`. Define and re-export these new types:

```python
class ReductionMode(Enum):
    TORCH_SUM = auto()
    SEQUENTIAL = auto()
    GROUPED = auto()


@dataclass(frozen=True)
class CanonicalSPNPlan:
    iterations: int
    offsets_xy: torch.Tensor
    neighbor_affinity: torch.Tensor
    current_affinity: torch.Tensor
    initial_affinity: torch.Tensor
    pre_fusion_gate: torch.Tensor
    pre_fusion_value: torch.Tensor
    post_fusion_gate: torch.Tensor
    post_fusion_value: torch.Tensor
    sampling_mode: SamplingMode
    padding_mode: PaddingMode
    align_corners: bool
    reduction_mode: ReductionMode
    reduction_groups: tuple[tuple[int, int], ...]
    group_scale: torch.Tensor


@dataclass
class SPNTrace:
    pre_fused_states: list[torch.Tensor] = field(default_factory=list)
    candidates: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    offsets: list[torch.Tensor] = field(default_factory=list)
    neighbor_affinities: list[torch.Tensor] = field(default_factory=list)
    current_affinities: list[torch.Tensor] = field(default_factory=list)
    initial_affinities: list[torch.Tensor] = field(default_factory=list)
    pre_fusion_gates: list[torch.Tensor] = field(default_factory=list)
    post_fusion_gates: list[torch.Tensor] = field(default_factory=list)
    group_scales: list[torch.Tensor] = field(default_factory=list)
```

Import and re-export these names from `torch_functional.py`. Do not move or
change the current propagation functions in this task.

- [ ] **Step 4: Run the new test and all existing tests**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_core.CanonicalTypeTest -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: the new test and all existing 47 tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_types.py \
  spn_accel_cmodel/torch_functional.py tests/test_torch_spn_core.py
git commit -m "refactor: introduce canonical SPN plan types"
```

### Task 2: Implement the profile-agnostic canonical core

**Files:**
- Create: `spn_accel_cmodel/torch_spn_core.py`
- Modify: `spn_accel_cmodel/torch_functional.py`
- Modify: `tests/test_torch_spn_core.py`

- [ ] **Step 1: Add failing validation and recurrence tests**

Add a `one_neighbor_plan()` fixture with `B=C=H=1`, `W=3`, K=1, S=1,
zero fusion/anchors, unit neighbor affinity, group `((0,1),)`, and unit group
scale. Test a plan that uses current/initial anchors, a middle-pixel pre-fusion,
and a final-pixel post-fusion:

```python
actual, trace = propagate_canonical(
    torch.tensor([[[[2.0, 4.0, 8.0]]]]),
    torch.ones((1, 1, 1, 3)),
    one_neighbor_plan(
        neighbor_affinity=torch.zeros((1, 1, 1, 1, 1, 3)),
        current_affinity=torch.full((1, 1, 1, 1, 3), 0.25),
        initial_affinity=torch.full((1, 1, 1, 1, 3), 0.75),
        pre_fusion_gate=torch.tensor([[[[[0.0, 1.0, 0.0]]]]]),
        pre_fusion_value=torch.tensor([[[[[0.0, 6.0, 0.0]]]]]),
        post_fusion_gate=torch.tensor([[[[[0.0, 0.0, 1.0]]]]]),
        post_fusion_value=torch.tensor([[[[[0.0, 0.0, 9.0]]]]]),
    ),
    return_trace=True,
)
torch.testing.assert_close(actual, torch.tensor([[[[1.25, 2.25, 9.0]]]]))
self.assertEqual(len(trace.pre_fused_states), 1)
```

Add a wrong-time-dimension test that calls
`validate_plan(state, state, plan)` and expects
`ValueError("offsets_xy time dimension")`; a K=3 grouped reduction with groups
`((0,2),(2,3))`, scales `(2,3)`, samples `(1,2,4)`, and expected result 18;
the existing adversarial K=9 DySPN values for exact sequential reduction; and
a NaN-affinity plan that validates successfully and propagates NaN unchanged.

- [ ] **Step 2: Run the tests and verify core imports fail**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_core.CanonicalCoreTest -v
```

Expected: missing `propagate_canonical` and `validate_plan`.

- [ ] **Step 3: Implement validation, reduction, and the sole recurrence**

Move the corrected `sample_neighbors()` into `torch_spn_core.py`. The INTEGER
path uses explicit integer indexing with zero/border validity handling, not
nearest `grid_sample`. The BILINEAR path keeps absolute-pixel coordinates and
the singleton-dimension correction. Implement:

```python
def _step(tensor, iteration):
    return tensor[:, 0 if tensor.shape[1] == 1 else iteration]


def _blend(base, value, gate):
    return (1.0 - gate) * base + gate * value


def propagate_canonical(current, initial, plan, return_trace=False):
    validate_plan(current, initial, plan)
    state = current
    trace = SPNTrace()
    for iteration in range(plan.iterations):
        pre_gate = _step(plan.pre_fusion_gate, iteration)
        propagated = _blend(state, _step(plan.pre_fusion_value, iteration), pre_gate)
        offsets = _step(plan.offsets_xy, iteration)
        samples = sample_neighbors(
            propagated, offsets, plan.sampling_mode, plan.padding_mode,
            align_corners=plan.align_corners,
        )
        neighbor = _step(plan.neighbor_affinity, iteration)
        scale = _step(plan.group_scale, iteration)
        candidate = reduce_neighbors(
            samples * neighbor, scale, plan.reduction_groups, plan.reduction_mode
        )
        candidate += _step(plan.current_affinity, iteration) * propagated
        candidate += _step(plan.initial_affinity, iteration) * initial
        post_gate = _step(plan.post_fusion_gate, iteration)
        state = _blend(candidate, _step(plan.post_fusion_value, iteration), post_gate)
        append_trace(trace, propagated, candidate, state, plan, iteration)
    return (state, trace) if return_trace else state
```

Expose `as_nchw(value, name, device=None)` for wrapper and adapter input
canonicalization. `validate_plan(current, initial, plan)` checks matching
initial/current shapes, B/H/W/K/Ca, time dimension `1` or T, FP32 device,
fusion broadcasting, and complete non-overlapping groups. It deliberately does
not reject NaN/Inf produced by official formulas. `reduce_neighbors()` supports
single `torch.sum`, sequential K accumulation, and ordered grouped reduction.

- [ ] **Step 4: Run direct core and sampling tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_core.CanonicalCoreTest \
  test_torch_spn_goldens.TorchSPNSamplingTest -v
```

Expected: recurrence, validation, grouped/sequential reduction, singleton, and
padding tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_core.py \
  spn_accel_cmodel/torch_functional.py tests/test_torch_spn_core.py
git commit -m "feat: add canonical Torch SPN recurrence"
```

### Task 3: Compile CSPN metadata into a canonical plan

**Files:**
- Create: `spn_accel_cmodel/torch_spn_adapters.py`
- Create: `tests/test_torch_spn_adapters.py`
- Modify: `tests/official_spn_references.py`

- [ ] **Step 1: Write the failing CSPN adapter test**

Extend `reference_cspn()` metadata with the complete target-layout affinity,
fixed offsets, and initial coefficient. For a 3x3 depth tensor, guidance
channels `1..8`, two iterations, and one nonzero sparse position, assert:

```python
plan = compile_cspn_plan(
    SPNConfig.cspn(iterations=2, preserve_code_mask=True),
    SPNInputs(initial, guidance, sparse_depth=sparse),
    initial,
    initial,
)
self.assertEqual(plan.offsets_xy.shape, (1, 1, 8, 2, 3, 3))
torch.testing.assert_close(
    plan.neighbor_affinity[:, 0, :, 0],
    metadata["neighbor_affinity"],
    rtol=0.0,
    atol=0.0,
)
torch.testing.assert_close(plan.initial_affinity[:, 0], metadata["initial_affinity"])
torch.testing.assert_close(plan.post_fusion_gate[:, 0], sparse.sign())
torch.testing.assert_close(plan.post_fusion_value[:, 0], initial)
```

Also assert current affinity, pre-fusion gate, and pre-fusion value are zero;
reduction is `TORCH_SUM`; and the adapter returns no propagated depth.

- [ ] **Step 2: Run the adapter test and verify the compiler is missing**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_adapters.CSPNPlanTest -v
```

Expected: import failure for `compile_cspn_plan`.

- [ ] **Step 3: Implement metadata-only CSPN compilation**

Create `torch_spn_adapters.py`. Move the source-channel shift helper from the
old module and expose this consistent adapter signature:

```python
def compile_cspn_plan(
    config: SPNConfig,
    inputs: SPNInputs,
    current: torch.Tensor,
    initial: torch.Tensor,
) -> CanonicalSPNPlan:
    guidance = as_cspn_guidance(inputs.affinity, current)
    shifted = shift_cspn_source_channels(guidance)
    normalized = shifted / torch.sum(torch.abs(shifted), dim=1, keepdim=True)
    neighbor = normalized[:, :, :, 1:-1, 1:-1].unsqueeze(1)
    initial_affinity = 1.0 - torch.sum(neighbor, dim=2)
    return make_plan(
        config=config,
        current=current,
        offsets_xy=base_offsets_tensor(config.base_offsets_xy, current).unsqueeze(1),
        neighbor_affinity=neighbor,
        current_affinity=torch.zeros_like(initial_affinity),
        initial_affinity=initial_affinity,
        post_gate=cspn_post_gate(config, inputs, current),
        post_value=initial,
        reduction_mode=ReductionMode.TORCH_SUM,
    )
```

`make_plan()` fills no-fusion gates with zeros, values on the current device,
one reduction group `(0,K)`, and unit group scale. It does not sample or update
depth.

- [ ] **Step 4: Run CSPN adapter and existing author goldens**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.CSPNPlanTest \
  test_torch_spn_goldens.CSPNGoldenTest -v
```

Expected: plan mapping and all existing CSPN goldens pass.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_adapters.py \
  tests/test_torch_spn_adapters.py tests/official_spn_references.py
git commit -m "feat: compile CSPN into canonical propagation plan"
```

### Task 4: Compile NLSPN and CompletionFormer metadata

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_adapters.py`
- Modify: `tests/test_torch_spn_adapters.py`
- Modify: `tests/official_spn_references.py`

- [ ] **Step 1: Write failing plan tests for all released modes**

For AS, ASS, TC, and TGASS, compare plan offsets, neighbor affinity, current
affinity, zero initial affinity, and fusion fields against independently
computed `reference_nlspn()` metadata:

```python
for mode in (
    NormalizationMode.AS,
    NormalizationMode.ASS,
    NormalizationMode.TC,
    NormalizationMode.TGASS,
):
    plan = compile_nlspn_plan(
        SPNConfig.nlspn(iterations=3, normalization=mode),
        inputs,
        initial,
        initial,
    )
    torch.testing.assert_close(plan.offsets_xy[:, 0], metadata["offsets"])
    torch.testing.assert_close(
        plan.neighbor_affinity[:, 0, :, 0],
        expected_affinity,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(plan.current_affinity[:, 0], expected_center)
```

Add CompletionFormer TC/TGASS assertions for temperature 100, tests for
residual-only released confidence offsets versus total legacy offsets, 16/18
channel input conversion, and singleton H/W. For `preserve_input=True`, assert
`pre_gate == (sparse > 0).float()` and `pre_value == sparse`.

- [ ] **Step 2: Run tests and verify both compilers are missing**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.NLSPNPlanTest \
  test_torch_spn_adapters.CompletionFormerPlanTest -v
```

Expected: missing compiler-function failures.

- [ ] **Step 3: Implement one shared NLSPN-family metadata compiler**

Move 16/18-channel parsing, YX-to-XY conversion, base-grid addition, confidence
sampling, and normalization into adapters. Use these public wrappers:

```python
def compile_nlspn_plan(config, inputs, current, initial):
    return compile_nlspn_like_plan(config, inputs, current, initial, temperature=1.0)


def compile_completionformer_plan(config, inputs, current, initial):
    return compile_nlspn_like_plan(
        config, inputs, current, initial, temperature=100.0
    )
```

Inside `compile_nlspn_like_plan`, apply tanh before sampled confidence, then
AS/ASS/TC/TGASS normalization in released order. Produce static S=1 metadata,
current coefficient `1-sum(neighbor)`, zero initial coefficient, and canonical
hard-pre fusion. Do not loop over iterations.

- [ ] **Step 4: Run adapter, singleton, and author tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.NLSPNPlanTest \
  test_torch_spn_adapters.CompletionFormerPlanTest \
  test_torch_spn_goldens.NLSPNGoldenTest -v
```

Expected: every affinity mode, temperature, confidence convention, offset
layout, sparse pre-fusion, and singleton case passes.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_adapters.py \
  tests/test_torch_spn_adapters.py tests/official_spn_references.py
git commit -m "feat: compile NLSPN profiles into canonical plans"
```

### Task 5: Compile current DySPN metadata

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_adapters.py`
- Modify: `tests/test_torch_spn_adapters.py`

- [ ] **Step 1: Write failing K=1/3/5/9 and per-step tests**

Use fixed-seed inputs and assert:

```python
plan = compile_dyspn_plan(config, inputs, initial, initial)
self.assertEqual(plan.offsets_xy.shape, (1, steps, k, 2, h, w))
self.assertEqual(plan.neighbor_affinity.shape, (1, steps, k, 1, h, w))
torch.testing.assert_close(
    plan.neighbor_affinity[:, :, :, 0],
    torch.softmax(logits, dim=2),
)
self.assertIs(plan.reduction_mode, ReductionMode.SEQUENTIAL)
torch.testing.assert_close(
    plan.post_fusion_gate[:, 0],
    torch.sigmoid(confidence_logits) * sparse.sign(),
)
torch.testing.assert_close(plan.post_fusion_value[:, 0], sparse)
```

Assert each iteration retains different metadata and all anchors/pre-fusion
fields are zero.

- [ ] **Step 2: Run and verify the compiler is missing**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_adapters.DySPNPlanTest -v
```

Expected: missing `compile_dyspn_plan`.

- [ ] **Step 3: Implement vectorized DySPN plan compilation**

Convert residual YX to XY, add the configured K base stencil, apply softmax
across K, and build per-step neighbor tensors. Use zero anchors/pre-fusion,
static S=1 sparse post-fusion, one group `(0,K)`, and
`ReductionMode.SEQUENTIAL`. The function contains no iteration loop.

- [ ] **Step 4: Run adapter and DySPN golden tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.DySPNPlanTest \
  test_torch_spn_goldens.DySPNGoldenTest -v
```

Expected: all stencil sizes, per-step metadata, boundary behavior, and exact
adversarial sequential accumulation pass.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_adapters.py tests/test_torch_spn_adapters.py
git commit -m "feat: compile DySPN into canonical propagation plan"
```

### Task 6: Compile NLPM and generic metadata

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_adapters.py`
- Modify: `tests/official_spn_references.py`
- Modify: `tests/test_torch_spn_adapters.py`

- [ ] **Step 1: Write failing full-48-neighbor NLPM tests**

Extend the independent NLPM oracle to expose target-layout shifted guidance,
48 fixed offsets, group scale, current coefficient, and initial coefficient.
Assert:

```python
plan = compile_dyspn_nlpm_plan(config, inputs, initial, initial)
self.assertEqual(plan.reduction_groups, ((0, 8), (8, 24), (24, 48)))
self.assertIs(plan.reduction_mode, ReductionMode.GROUPED)
self.assertEqual(plan.offsets_xy.shape, (1, 1, 48, 2, h, w))
self.assertEqual(plan.group_scale.shape, (1, steps, 3, 1, h, w))
torch.testing.assert_close(
    plan.neighbor_affinity[:, 0, :, 0],
    metadata["shifted_guidance"],
)
torch.testing.assert_close(plan.group_scale, metadata["group_scale"])
torch.testing.assert_close(plan.current_affinity, metadata["current_affinity"])
torch.testing.assert_close(plan.initial_affinity, metadata["initial_affinity"])
```

Add a generic-plan test covering absolute offsets, TGASS confidence transform
order, initial anchoring, and hard-post sparse fusion.

- [ ] **Step 2: Run and verify both compilers are missing**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.DySPNNLPMPlanTest \
  test_torch_spn_adapters.GenericPlanTest -v
```

Expected: missing `compile_dyspn_nlpm_plan` and `compile_generic_plan`.

- [ ] **Step 3: Implement grouped NLPM and generic plan compilation**

Generate 3x3, 5x5, and 7x7 edge offsets in author channel order. Shift each
guidance channel to target layout without reading evolving state. With
`attention = sigmoid(attention_logits)`, compute:

```python
denominator = torch.sum(attention * abs_sums, dim=2, keepdim=True) + config.eps
group_scale = attention[:, :, 0:3] / denominator
current_affinity = attention[:, :, 3:4] / denominator
initial_affinity = (
    denominator - torch.sum(attention * signed_sums, dim=2, keepdim=True)
) / denominator
```

Set post gate to `confidence * sparse.sign()` and post value to sparse depth.
Implement `compile_generic_plan()` with this explicit coefficient mapping:

```python
neighbor = normalize_affinity(pretransform(raw_affinity, config), config)
residual = 1.0 - torch.sum(neighbor, dim=2)
current_affinity = residual if config.anchor_mode is AnchorMode.CURRENT else zeros
initial_affinity = residual if config.anchor_mode is AnchorMode.INITIAL else zeros
```

For sampled confidence, multiply after `pretransform` and before normalization.
Map HARD_PRE to pre gate/value, HARD_POST to sparse post gate/value,
CSPN_CODE_POST to initial post value, and SOFT_POST to confidence post gate with
sparse value. Reject generic `INITIAL_CURRENT`; only NLPM defines that
factorization.

- [ ] **Step 4: Run NLPM/generic adapter and existing golden tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_adapters.DySPNNLPMPlanTest \
  test_torch_spn_adapters.GenericPlanTest \
  test_torch_spn_goldens.DySPNNLPMGoldenTest \
  test_torch_spn_goldens.GenericSPNTest -v
```

Expected: canonical metadata and author outputs pass within the approved FP32
tolerances.

- [ ] **Step 5: Commit**

```bash
git add spn_accel_cmodel/torch_spn_adapters.py \
  tests/official_spn_references.py tests/test_torch_spn_adapters.py
git commit -m "feat: compile remaining SPN profiles into canonical plans"
```

### Task 7: Route every profile through the one canonical core

**Files:**
- Replace: `spn_accel_cmodel/torch_functional.py`
- Modify: `spn_accel_cmodel/__init__.py`
- Modify: `tests/test_torch_spn_core.py`
- Modify: `tests/test_torch_spn_goldens.py`
- Modify: `tests/test_recorded_trace.py`

- [ ] **Step 1: Write failing structural unification tests**

Use AST to find production loops whose iterator expression contains
`iterations`, and assert the only match is
`propagate_canonical: range(plan.iterations)`. Also assert these methods do not
exist on `UnifiedSPN`:

```python
forbidden = {
    "_forward_cspn",
    "_forward_nlspn",
    "_forward_dyspn",
    "_forward_dyspn_nlpm",
    "_forward_generic",
}
self.assertTrue(forbidden.isdisjoint(dir(UnifiedSPN)))
```

Patch `torch_functional.propagate_canonical` with a spy, run each named profile
once, and assert one call per `UnifiedSPN.forward()`.

- [ ] **Step 2: Run structural tests and verify old methods fail**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_core.StructuralUnificationTest -v
```

Expected: failures list the five existing `_forward_*` methods and their
profile-specific loops.

- [ ] **Step 3: Replace the public module with one compile-and-run path**

Delete every old recurrence and moved helper. Keep compatibility re-exports and
define:

```python
_PLAN_COMPILERS = {
    SPNProfile.CSPN: compile_cspn_plan,
    SPNProfile.NLSPN: compile_nlspn_plan,
    SPNProfile.COMPLETIONFORMER: compile_completionformer_plan,
    SPNProfile.DYSPN: compile_dyspn_plan,
    SPNProfile.DYSPN_NLPM: compile_dyspn_nlpm_plan,
    SPNProfile.GENERIC: compile_generic_plan,
}


class UnifiedSPN(nn.Module):
    def __init__(self, config: SPNConfig):
        super().__init__()
        self.config = config

    def forward(self, inputs: SPNInputs, *, return_trace=False):
        current = as_nchw(inputs.current, "current")
        initial = current if inputs.initial is None else as_nchw(
            inputs.initial, "initial", device=current.device
        )
        if initial.shape != current.shape:
            raise ValueError("initial and current must have identical shapes")
        plan = _PLAN_COMPILERS[self.config.profile](
            self.config, inputs, current, initial
        )
        return propagate_canonical(current, initial, plan, return_trace)
```

Export `CanonicalSPNPlan`, `ReductionMode`, `validate_plan`, and
`propagate_canonical` lazily through `spn_accel_cmodel/__init__.py`.

- [ ] **Step 4: Expand every profile's trace comparison**

Assert all trace lists have `iterations` entries. Compare pre-fused state,
candidate, output, offsets, effective K-neighbor affinity, current/initial
affinity, both fusion gates, and group scale with independent oracle metadata.
CSPN remains exact; bilinear and normalized paths retain their approved
tolerances. Parameterize the existing available-CUDA test so each named adapter
accepts CPU metadata with CUDA current/initial state and returns CUDA output.

- [ ] **Step 5: Run structural, author, and NumPy cross-check tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_core.StructuralUnificationTest \
  test_torch_spn_goldens \
  test_recorded_trace.TorchFunctionalCrossCheck -v
```

Expected: the AST test proves one recurrence and every author/NumPy output
passes.

- [ ] **Step 6: Commit**

```bash
git add spn_accel_cmodel/torch_functional.py spn_accel_cmodel/__init__.py \
  tests/test_torch_spn_core.py tests/test_torch_spn_goldens.py \
  tests/test_recorded_trace.py
git commit -m "refactor: route every SPN profile through one recurrence"
```

### Task 8: Document, audit, review, and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-13-canonical-unified-spn-design.md`
- Test: complete repository

- [ ] **Step 1: Update architecture and usage documentation**

Document both supported paths:

```python
model = UnifiedSPN(SPNConfig.nlspn(iterations=6))
output, trace = model(inputs, return_trace=True)

plan = compile_nlspn_plan(config, inputs, current, initial)
output, trace = propagate_canonical(current, initial, plan, return_trace=True)
```

State that DCNv2 and prediction heads remain outside propagation.

- [ ] **Step 2: Audit stale branches and unfinished markers**

```bash
rg -n "def _forward_|NotImplemented|TODO|TBD|FIXME" \
  spn_accel_cmodel tests README.md
```

Expected: no profile-specific recurrence, incomplete marker, or generic
not-implemented branch remains.

- [ ] **Step 3: Verify syntax and whitespace**

```bash
python -m compileall -q spn_accel_cmodel tests
git diff --check
```

Expected: both commands exit zero without output.

- [ ] **Step 4: Run the three canonical validation layers**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_core test_torch_spn_adapters test_torch_spn_goldens -v
```

Expected: all direct-core, adapter, structural, and independent author-golden
tests pass.

- [ ] **Step 5: Run the complete repository suite**

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: all tests pass, including C-model, SCALE-Sim, validation,
recorded-trace, and NumPy/Torch cross-check suites.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-13-canonical-unified-spn-design.md
git commit -m "docs: describe canonical unified SPN execution"
```

- [ ] **Step 7: Request final review and repeat verification**

Ask the reviewer to inspect the complete diff from `2941546` through `HEAD`,
especially forbidden adapter recurrence, CSPN source layout, NLSPN confidence
offsets, DySPN sequential reduction, NLPM group order, and trace completeness.
Resolve all critical and important findings. Rerun Steps 2 through 5 on the
final commit and require a clean `git status --short`.
