# SPN OfficialFrontend v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overloaded post-decoder `SPNInputs` facade with five official-frontend PyTorch interfaces that decode official SPN parameters and execute the existing single canonical propagation recurrence.

**Architecture:** Profile-specific `nn.Module` frontends own only the official plan-producing parameters (`conv_offset_aff` and, for the NLSPN family, `aff_scale_const`). They emit one internal `DecodedSPNParameters` contract; metadata-only adapters compile that contract to `CanonicalSPNPlan`; `propagate_canonical()` remains the sole evolving-state loop. The implementation is an FP32 propagation reference only and contains no DCNv2 dependency, prediction backbone, checkpoint download, non-FP32 path, or accelerator modeling.

**Tech Stack:** Python 3.11+, PyTorch, `dataclasses`, `unittest`, `ast`, Git.

---

## File Map

- Modify `spn_accel_cmodel/torch_spn_types.py`: add five public RawInputs, remove `SPNInputs` and the `GENERIC` profile at cutover.
- Create `spn_accel_cmodel/torch_spn_decoded.py`: one internal decoded-parameter dataclass.
- Create `spn_accel_cmodel/torch_spn_official_frontend.py`: five official frontends and strict explicit-prefix parameter loading.
- Modify `spn_accel_cmodel/torch_spn_adapters.py`: consume decoded parameters, honor loaded affinity scale, remove the generic compiler.
- Modify `spn_accel_cmodel/torch_functional.py`: official-frontend `UnifiedSPN` facade.
- Modify `spn_accel_cmodel/__init__.py`: public RawInputs and official-module exports; remove old exports.
- Create `tests/test_torch_spn_official_frontend.py`: decoder, parameter-loading, raw-type, and no-recurrence tests.
- Modify `tests/official_spn_references.py`: independent official-boundary convolution/decode formulas.
- Modify `tests/test_torch_spn_adapters.py`: construct `DecodedSPNParameters` directly.
- Modify `tests/test_torch_spn_core.py`: assert the new facade calls the canonical core once and old API is absent.
- Modify `tests/test_torch_spn_goldens.py`: drive all five profiles from RawInputs and compare every iteration.
- Modify `tests/test_recorded_trace.py`: use the decoded compiler/core boundary for already-decoded trace tensors.
- Modify `README.md` and `validation/README.md`: document the official and canonical entry points and precise equivalence scope.

Do not stage or alter unrelated existing whitespace-only worktree changes in
`spn_accel_cmodel/trace.py`, `spn_accel_cmodel/torch_spn_adapters.py`, or
`tests/official_spn_references.py`; preserve them while staging each intended
hunk explicitly.

### Task 1: Add profile-specific raw and internal decoded types

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_types.py`
- Create: `spn_accel_cmodel/torch_spn_decoded.py`
- Test: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Write the failing public-type and decoded-contract tests**

Create `tests/test_torch_spn_official_frontend.py` with optional-Torch import handling
matching the other Torch suites. Add a test that constructs each exact type:

```python
class OfficialInputTypeTest(unittest.TestCase):
    def test_profile_inputs_name_confidence_semantics_explicitly(self):
        image = torch.zeros((1, 1, 3, 4))
        self.assertEqual(
            NLSPNRawInputs(image, torch.zeros((1, 8, 3, 4))).feat_init,
            image,
        )
        self.assertIsNone(
            NLSPNRawInputs(image, torch.zeros((1, 8, 3, 4)))
            .confidence_probability
        )
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

    def test_decoded_parameters_keep_post_decoder_fields_unambiguous(self):
        image = torch.zeros((1, 1, 3, 4))
        decoded = DecodedSPNParameters(
            current=image,
            initial=image,
            raw_affinity=torch.zeros((1, 8, 3, 4)),
            residual_offsets_yx=torch.zeros((1, 8, 2, 3, 4)),
            affinity_scale=torch.ones(1),
        )
        self.assertEqual(decoded.residual_offsets_yx.shape, (1, 8, 2, 3, 4))
        self.assertIsNone(decoded.attention_logits)
```

Import `CSPNRawInputs`, `NLSPNRawInputs`, `CompletionFormerRawInputs`,
`DySPNRawInputs`, and `DySPNNLPMRawInputs` from `torch_spn_types`, and import
`DecodedSPNParameters` from the new internal module.

- [ ] **Step 2: Run the type tests and verify RED**

Run:

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_official_frontend.OfficialInputTypeTest -v
```

Expected: import failure because the RawInputs and decoded contract do not
exist.

- [ ] **Step 3: Add the five RawInputs dataclasses**

Append these exact dataclasses to `torch_spn_types.py` without removing the old
type yet; removal happens in the atomic API cutover:

```python
@dataclass
class CSPNRawInputs:
    guidance: torch.Tensor
    blur_depth: torch.Tensor
    sparse_depth: torch.Tensor | None = None


@dataclass
class NLSPNRawInputs:
    feat_init: torch.Tensor
    guidance: torch.Tensor
    confidence_probability: torch.Tensor | None = None
    feat_fix: torch.Tensor | None = None
    rgb: torch.Tensor | None = None


@dataclass
class CompletionFormerRawInputs:
    pred_init: torch.Tensor
    guidance: torch.Tensor
    confidence_probability: torch.Tensor
    sparse_depth: torch.Tensor
    rgb: torch.Tensor | None = None


@dataclass
class DySPNRawInputs:
    feat_init: torch.Tensor
    guide: torch.Tensor
    sparse_depth: torch.Tensor
    confidence_logits: torch.Tensor


@dataclass
class DySPNNLPMRawInputs:
    feat_init: torch.Tensor
    guidance: torch.Tensor
    dynamic_logits: torch.Tensor
    sparse_depth: torch.Tensor
    confidence_probability: torch.Tensor
```

- [ ] **Step 4: Add the internal decoded contract**

Create `torch_spn_decoded.py`:

```python
from dataclasses import dataclass

import torch


@dataclass
class DecodedSPNParameters:
    current: torch.Tensor
    initial: torch.Tensor
    raw_affinity: torch.Tensor
    residual_offsets_yx: torch.Tensor | None = None
    confidence_probability: torch.Tensor | None = None
    confidence_logits: torch.Tensor | None = None
    sparse_depth: torch.Tensor | None = None
    sparse_mask: torch.Tensor | None = None
    attention_logits: torch.Tensor | None = None
    affinity_scale: torch.Tensor | None = None
```

Do not export this internal type from `spn_accel_cmodel/__init__.py`.

- [ ] **Step 5: Run the type tests and existing suite**

Run:

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_official_frontend.OfficialInputTypeTest -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: the new tests pass and the existing 71 tests remain green.

- [ ] **Step 6: Commit the type layer**

```bash
git add spn_accel_cmodel/torch_spn_types.py \
  spn_accel_cmodel/torch_spn_decoded.py tests/test_torch_spn_official_frontend.py
git commit -m "refactor: define official-frontend SPN input contracts"
```

### Task 2: Implement NLSPN-family parameter decoders

**Files:**
- Create: `spn_accel_cmodel/torch_spn_official_frontend.py`
- Modify: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Write failing decoder-equivalence tests**

For both `NLSPNOfficialFrontend` and `CompletionFormerOfficialFrontend`, use a fixed generator,
assign random FP32 `conv_offset_aff.weight`, `conv_offset_aff.bias`, and
`aff_scale_const`, and calculate the independent result with `F.conv2d`:

```python
raw = F.conv2d(
    guidance,
    official.conv_offset_aff.weight,
    official.conv_offset_aff.bias,
    padding=1,
)
o1, o2, expected_affinity = torch.chunk(raw, 3, dim=1)
expected_offsets = torch.cat((o1, o2), dim=1).view(b, 8, 2, h, w)
decoded = official.decode(inputs)
torch.testing.assert_close(decoded.raw_affinity, expected_affinity)
torch.testing.assert_close(decoded.residual_offsets_yx, expected_offsets)
torch.testing.assert_close(decoded.affinity_scale, official.aff_scale_const)
```

Also assert exact layer shapes `(24,8,3,3)` and `(24,)`, zero
initialization, decoded current/initial identity, and FP32/device movement.

- [ ] **Step 2: Write failing strict official-parameter loading tests**

Use a synthetic full state dict and mandatory non-empty prefix:

```python
state = {
    "module.prop_layer.conv_offset_aff.weight": weight,
    "module.prop_layer.conv_offset_aff.bias": bias,
    "module.prop_layer.aff_scale_const": scale,
}
official.load_official_parameters(state, prefix="module.prop_layer.")
torch.testing.assert_close(official.conv_offset_aff.weight, weight)
torch.testing.assert_close(official.aff_scale_const, scale)
```

Add separate assertions that a missing exact key and a wrong-shaped exact key
raise `ValueError` naming that key. Do not test aliases or fallback prefixes.

- [ ] **Step 3: Run the NLSPN-family tests and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.NLSPNOfficialFrontendDecoderTest \
  test_torch_spn_official_frontend.CompletionFormerOfficialFrontendDecoderTest -v
```

Expected: missing `torch_spn_official_frontend` or missing official classes.

- [ ] **Step 4: Implement the shared NLSPN-family base**

Create `torch_spn_official_frontend.py`. Implement a private `_NLSPNFamilyOfficialFrontend` that:

- accepts `SPNConfig` and one exact RawInputs type;
- creates `conv_offset_aff` with official name, dimensions, padding, and zero
  initialization;
- creates `aff_scale_const` as a scalar `nn.Parameter`, initialized to K for
  TC, `affinity_gamma*K` for TGASS, and one for AS/ASS;
- makes the parameter trainable only for TGASS, matching the official module;
- implements `decode()` using the exact `chunk -> cat -> view` order;
- returns `DecodedSPNParameters` without normalizing affinity or sampling
  evolving depth;
- requires the selected RawInputs class with `TypeError` on mismatch.

Implement `load_official_parameters()` with only these exact names:

```python
names = (
    "conv_offset_aff.weight",
    "conv_offset_aff.bias",
    "aff_scale_const",
)
```

For every name, form `prefix + name`, require the key, require exact shape, and
copy under `torch.no_grad()`. Do not search or strip prefixes automatically.

- [ ] **Step 5: Add `NLSPNOfficialFrontend` and `CompletionFormerOfficialFrontend`**

Subclasses specify their exact input type and map fields as follows:

```text
NLSPN:
  current/initial <- feat_init
  confidence_probability <- confidence_probability
  sparse_depth <- feat_fix

CompletionFormer:
  current/initial <- pred_init
  confidence_probability <- confidence_probability
  sparse_depth <- sparse_depth
```

`rgb` is accepted at the official interface but not included in decoded
parameters because the fixed upstream propagation modules do not consume it in
their parameter formulas.

- [ ] **Step 6: Run focused and existing tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.NLSPNOfficialFrontendDecoderTest \
  test_torch_spn_official_frontend.CompletionFormerOfficialFrontendDecoderTest -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: decoder/loading tests pass and the existing suite remains green.

- [ ] **Step 7: Commit NLSPN-family decoders**

```bash
git add spn_accel_cmodel/torch_spn_official_frontend.py tests/test_torch_spn_official_frontend.py
git commit -m "feat: decode NLSPN official parameters"
```

### Task 3: Implement current DySPN parameter decoding

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_official_frontend.py`
- Modify: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Write failing K=1/3/5/9 decoder tests**

For each K and fixed T=2, construct `guide` with `T*K` channels and independently
compute:

```python
raw = F.conv2d(
    guide,
    official.conv_offset_aff.weight,
    official.conv_offset_aff.bias,
    padding=1,
)
offset_flat, affinity_flat = torch.split(raw, [2 * t * k, t * k], dim=1)
expected_offsets = offset_flat.view(b, t, k, 2, h, w)
expected_logits = affinity_flat.view(b, t, k, h, w)
decoded = official.decode(inputs)
```

Compare both tensors exactly, assert layer shape
`(3*T*K,T*K,3,3)`, zero initialization, current/initial mapping, sparse-depth
mapping, and that `confidence_logits` is sigmoid-free in decoded parameters.

- [ ] **Step 2: Write failing explicit-prefix loading tests**

Use prefix `module.dyspn_2_5.` and require only
`conv_offset_aff.weight/bias`. Verify missing and wrong-shaped keys raise named
`ValueError`.

- [ ] **Step 3: Run and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.DySPNOfficialFrontendDecoderTest -v
```

Expected: missing `DySPNOfficialFrontend` behavior.

- [ ] **Step 4: Implement `DySPNOfficialFrontend`**

Create the official-shaped zero-initialized convolution using
`channels=config.iterations*config.num_neighbors`. Decode with exact
`torch.split` and contiguous `view` expressions. Return raw per-step logits,
raw residual YX offsets, sparse depth, and unchanged confidence logits in
`DecodedSPNParameters`. Do not add base offsets, softmax, sigmoid, sampling, or
propagation in this class.

- [ ] **Step 5: Run focused and existing tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.DySPNOfficialFrontendDecoderTest -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: all pass.

- [ ] **Step 6: Commit the DySPN decoder**

```bash
git add spn_accel_cmodel/torch_spn_official_frontend.py tests/test_torch_spn_official_frontend.py
git commit -m "feat: decode DySPN official parameters"
```

### Task 4: Implement CSPN and NLPM official interfaces

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_official_frontend.py`
- Modify: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Write failing CSPN official mapping test**

Assert `CSPNOfficialFrontend.decode(CSPNRawInputs(...))` maps blur depth to current and
initial, guidance to raw affinity, sparse depth to sparse depth, and produces no
offset, attention, confidence, or affinity-scale tensors.

- [ ] **Step 2: Write failing NLPM reshape and confidence test**

For `T=2`, require raw `dynamic_logits` shape `[B,8,H,W]` and independently
reshape with:

```python
expected = dynamic_logits.view(b, t, 4, h, w)
```

Assert guidance remains `[B,48,H,W]`, confidence probability is unchanged, and
no sigmoid is applied in the official module.

- [ ] **Step 3: Write failing input-type and shape error tests**

For every official class, pass another profile's RawInputs and expect `TypeError`
naming both configured profile and expected type. Add field-named `ValueError`
tests for CSPN guidance channels, NLPM 48 guidance channels, and NLPM `4*T`
dynamic channels.

- [ ] **Step 4: Run and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.CSPNOfficialFrontendDecoderTest \
  test_torch_spn_official_frontend.DySPNNLPMOfficialFrontendDecoderTest \
  test_torch_spn_official_frontend.OfficialInputValidationTest -v
```

Expected: missing CSPN/NLPM official classes or validation.

- [ ] **Step 5: Implement the two parameter-free official classes**

Both are `nn.Module` classes with `decode()` but no learned parameters.
Canonicalize state-like tensors to FP32 on the initial state's device. Require
exact official channel and spatial shapes. NLPM reshapes dynamic logits but does
not apply sigmoid or group normalization.

- [ ] **Step 6: Add the exact official builder table**

In `torch_spn_official_frontend.py` define:

```python
OFFICIAL_FRONTEND_BUILDERS = {
    SPNProfile.CSPN: CSPNOfficialFrontend,
    SPNProfile.NLSPN: NLSPNOfficialFrontend,
    SPNProfile.COMPLETIONFORMER: CompletionFormerOfficialFrontend,
    SPNProfile.DYSPN: DySPNOfficialFrontend,
    SPNProfile.DYSPN_NLPM: DySPNNLPMOfficialFrontend,
}
```

Do not add a default lookup, generic official, or unknown-profile fallback.

- [ ] **Step 7: Run all official tests and existing suite**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_official_frontend -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: all official tests and the existing suite pass.

- [ ] **Step 8: Commit parameter-free official interfaces**

```bash
git add spn_accel_cmodel/torch_spn_official_frontend.py tests/test_torch_spn_official_frontend.py
git commit -m "feat: add CSPN and NLPM official interfaces"
```

### Task 5: Add independent official-boundary formulas

**Files:**
- Modify: `tests/official_spn_references.py`
- Modify: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Write failing raw-decoder oracle tests**

Add tests that import `reference_nlspn_official_decode`,
`reference_completionformer_official_decode`, and
`reference_dyspn_official_decode`. Compare their outputs with each production
official decoder using fixed random weights and biases. The production decoder
test must not pass its decoded tensors into the oracle.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend.IndependentOfficialDecodeTest -v
```

Expected: missing official decode reference functions.

- [ ] **Step 3: Implement independent `F.conv2d` decoder formulas**

In `official_spn_references.py`, add standalone functions that accept raw
profile tensors plus explicit weight/bias tensors and return dictionaries with
`conv_output`, `residual_offsets_yx`, and `raw_affinity`. Use only `F.conv2d`,
`torch.chunk`/`torch.split`, and direct `view`; do not import production official,
adapter, decoded, or core modules.

For NLSPN and CompletionFormer use:

```python
conv_output = F.conv2d(guidance, weight, bias, padding=1)
o1, o2, affinity = torch.chunk(conv_output, 3, dim=1)
offsets = torch.cat((o1, o2), dim=1).view(b, k, 2, h, w)
```

For DySPN use the exact `2*T*K`/`T*K` split and views.

- [ ] **Step 4: Run oracle and official tests**

```bash
PYTHONPATH=.:tests python -m unittest test_torch_spn_official_frontend -v
```

Expected: all pass.

- [ ] **Step 5: Commit independent decoder references**

Stage only intended hunks from the already-modified reference file:

```bash
git add -p tests/official_spn_references.py
git add tests/test_torch_spn_official_frontend.py
git commit -m "test: add independent SPN official decoder formulas"
```

### Task 6: Atomically cut adapters and facade over to OfficialFrontend v1

**Files:**
- Modify: `spn_accel_cmodel/torch_spn_types.py`
- Modify: `spn_accel_cmodel/torch_spn_adapters.py`
- Modify: `spn_accel_cmodel/torch_functional.py`
- Modify: `spn_accel_cmodel/__init__.py`
- Modify: `tests/test_torch_spn_adapters.py`
- Modify: `tests/test_torch_spn_core.py`
- Modify: `tests/test_recorded_trace.py`
- Modify: `tests/test_torch_spn_goldens.py`

This is one atomic cutover: do not commit until every existing test consumer has
been migrated and no production compatibility shim exists.

- [ ] **Step 1: Write failing raw-facade structural tests**

In `StructuralUnificationTest`, assert:

```python
self.assertFalse(hasattr(torch_spn_types, "SPNInputs"))
self.assertNotIn("GENERIC", SPNProfile.__members__)
self.assertFalse(hasattr(torch_spn_adapters, "compile_generic_plan"))
```

Patch `torch_functional.propagate_canonical` and run each of the five RawInputs
through `UnifiedSPN`; assert exactly one call per profile. Also assert passing a
different profile's RawInputs raises `TypeError` before core invocation.

- [ ] **Step 2: Run the structural test and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_core.StructuralUnificationTest -v
```

Expected: old `SPNInputs`, `GENERIC`, and generic compiler are still present and
the facade does not accept RawInputs.

- [ ] **Step 3: Refactor plan compilers to the decoded contract**

Change every named compiler signature to:

```python
def compile_nlspn_plan(
    config: SPNConfig,
    decoded: DecodedSPNParameters,
) -> CanonicalSPNPlan:
```

Use `decoded.current` and `decoded.initial` internally. Apply the same pattern
to CSPN, CompletionFormer, DySPN, and NLPM. Remove all `SPNInputs` imports and
field reads.

For NLSPN/CompletionFormer TC and TGASS, require
`decoded.affinity_scale` and calculate:

```python
affinity = torch.tanh(
    affinity / config.tanh_temperature
) / (
    decoded.affinity_scale.to(current.device, torch.float32)
    + (1.0e-8 if config.normalization is NormalizationMode.TGASS else 0.0)
)
```

AS/ASS ignore the scale. Preserve the existing confidence sampling,
normalization floor, center separation, sparse pre-fusion, offset conventions,
and tolerances.

For DySPN, apply softmax to decoded per-step logits and sigmoid only to decoded
confidence logits. For NLPM, apply sigmoid only to decoded attention logits and
consume confidence probability unchanged.

Delete `_generic_offsets`, `_generic_affinity`, `_pretransform_affinity`,
`_normalize_affinity`, `_sample_confidence_by_step`, and
`compile_generic_plan` when they have no named-profile callers.

- [ ] **Step 4: Migrate adapter tests to decoded parameters**

Replace every `SPNInputs` construction in `test_torch_spn_adapters.py` with
`DecodedSPNParameters`. Call compilers with `(config, decoded)`. Remove
`GenericPlanTest`; direct canonical behavior remains covered in
`CanonicalCoreTest`.

Add an explicit loaded-scale test where config gamma and
`decoded.affinity_scale` differ, proving the compiler uses the decoded
checkpoint parameter.

- [ ] **Step 5: Replace `UnifiedSPN` with the strict official facade**

Implement:

```python
class UnifiedSPN(nn.Module):
    def __init__(self, config: SPNConfig):
        super().__init__()
        self.config = config
        self.official_frontend = OFFICIAL_FRONTEND_BUILDERS[config.profile](config)

    def forward(self, inputs, *, return_trace=False):
        decoded = self.official_frontend.decode(inputs)
        plan = _PLAN_COMPILERS[self.config.profile](self.config, decoded)
        return propagate_canonical(
            decoded.current,
            decoded.initial,
            plan,
            return_trace,
        )
```

The compiler table contains exactly five named profiles. Do not catch
`KeyError`, reinterpret input types, or invoke a fallback compiler.

- [ ] **Step 6: Remove the old public and generic API**

Delete `SPNInputs`, `SPNProfile.GENERIC`, the default generic profile, generic
exports, and any generic configuration tests. Move `profile` before optional
dataclass fields so `SPNConfig` requires an explicit profile; all supported
construction goes through the five named class methods.

Export all five RawInputs and five official classes through `torch_functional.py`
and lazy package exports. Keep `CanonicalSPNPlan`, `SPNConfig`, `SPNTrace`, and
`propagate_canonical` public. Do not export `DecodedSPNParameters` at package
top level.

- [ ] **Step 7: Migrate recorded-trace and low-level consumers**

`test_recorded_trace.py` already owns decoded NLSPN offsets/affinity. Construct
`DecodedSPNParameters`, call `compile_nlspn_plan`, then call
`propagate_canonical`; do not synthesize guidance or use an official frontend for
post-decoder data.

Keep sampling-only tests pointed at `sample_neighbors`. Replace any generic
facade test with direct `CanonicalSPNPlan` core coverage.

- [ ] **Step 8: Migrate existing goldens to official RawInputs**

For CSPN and NLPM, rename arguments directly into their RawInputs.

For every existing post-decoder NLSPN/CompletionFormer/DySPN test, either:

- move it to adapter-level coverage by constructing `DecodedSPNParameters`, or
- create a deterministic 1x1/3x3 official convolution whose output is part of the
  test and compare against the independent official oracle.

End-to-end `UnifiedSPN` golden cases must begin from guidance/guide tensors and
set the official's fixed random weights; they may not inject precomputed offsets
or affinities.

- [ ] **Step 9: Run the atomic cutover suites**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend test_torch_spn_adapters test_torch_spn_core \
  test_torch_spn_goldens test_recorded_trace -v
```

Expected: all migrated official, adapter, core, golden, and trace tests pass.

- [ ] **Step 10: Audit forbidden compatibility paths**

```bash
rg -n "SPNInputs|SPNProfile\.GENERIC|compile_generic_plan|def _forward_" \
  spn_accel_cmodel tests
rg -n "range\([^\n]*iterations|for [^\n]*iterations" \
  spn_accel_cmodel/torch_spn*.py
```

Expected: the first command has no matches. The second command reports only the
canonical propagation loop; decoder reshape code must not loop over
propagation steps.

- [ ] **Step 11: Commit the breaking cutover**

Stage intended adapter/reference hunks without staging pre-existing whitespace:

```bash
git add spn_accel_cmodel/torch_spn_types.py \
  spn_accel_cmodel/torch_functional.py spn_accel_cmodel/__init__.py \
  tests/test_torch_spn_adapters.py tests/test_torch_spn_core.py \
  tests/test_torch_spn_goldens.py tests/test_recorded_trace.py
git add -p spn_accel_cmodel/torch_spn_adapters.py
git commit -m "refactor: expose official-frontend unified SPN propagation"
```

### Task 7: Prove full official-boundary equivalence

**Files:**
- Modify: `tests/official_spn_references.py`
- Modify: `tests/test_torch_spn_goldens.py`
- Modify: `tests/test_torch_spn_official_frontend.py`

- [ ] **Step 1: Add failing end-to-end official formula tests**

Create one fixed-seed official-boundary case for every profile and extend coverage
as follows:

- CSPN: code-mask disabled/enabled and shifted-source integer microcase;
- NLSPN: AS, ASS, TC, TGASS, probability confidence on/off, legacy on/off,
  preserve-input on/off;
- CompletionFormer: TC and TGASS with temperature 100 and default T=6;
- DySPN: K=1/3/5/9, distinct per-step metadata, sequential reduction
  adversarial case, confidence-logit sparse fusion;
- NLPM: T=2, all three groups, dynamic-logit sigmoid, probability confidence,
  current/initial anchors.

For every case compare decoded tensors, canonical plan tensors, candidate, and
every output iteration.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_goldens.OfficialBoundaryGoldenTest -v
```

Expected: missing raw official reference functions or missing full-trace metadata.

- [ ] **Step 3: Extend independent official formulas through propagation**

Compose the new independent decoder formulas with the existing independent
official propagation formulas inside `official_spn_references.py`. Pass explicit
weight, bias, and affinity-scale tensors. Do not import production modules or
reuse `CanonicalSPNPlan`.

Return both decoder metadata and one list per propagation iteration so tests
can compare the complete boundary, not only final output.

- [ ] **Step 4: Run all official and golden tests**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend test_torch_spn_goldens -v
```

Expected: all pass using `rtol=1e-5, atol=1e-6` for normalized/bilinear paths
and exact checks for integer/operation-order microcases.

- [ ] **Step 5: Run CUDA cross-device coverage**

Parameterize the existing available-CUDA test with all five RawInputs. Decoder
weights and raw metadata begin on CPU; move only the `UnifiedSPN` module and
state-like input to CUDA according to normal PyTorch module semantics. Verify
the output and canonical trace use CUDA. Skip only when CUDA is unavailable.

Run:

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_goldens.TorchSPNValidationTest -v
```

Expected in the current environment: CUDA case runs and passes, not skips.

- [ ] **Step 6: Commit complete official equivalence**

```bash
git add -p tests/official_spn_references.py
git add tests/test_torch_spn_goldens.py tests/test_torch_spn_official_frontend.py
git commit -m "test: prove official-frontend SPN equivalence"
```

### Task 8: Document, audit, review, and verify

**Files:**
- Modify: `README.md`
- Modify: `validation/README.md`
- Test: complete repository

- [ ] **Step 1: Replace README usage with RawInputs**

Document the official-frontend path:

```python
model = UnifiedSPN(SPNConfig.completionformer(iterations=6))
output, trace = model(
    CompletionFormerRawInputs(
        pred_init=pred_init,
        guidance=guidance,
        confidence_probability=confidence,
        sparse_depth=sparse_depth,
    ),
    return_trace=True,
)
```

Document the separate direct canonical path for callers that already own
decoded coefficients. State explicitly that the package is a propagation-only
FP32 operator reference; backbones, DCNv2 carrier compilation, real checkpoint
distribution, non-FP32 arithmetic, and accelerator models are outside this API.

- [ ] **Step 2: Update validation scope and upstream pins**

In `validation/README.md`, describe both decoder and propagation equivalence,
the four fixed upstream commits, fixed-random-weight unit tests, and the absence
of downloaded checkpoints. Retain the existing numerical tolerances.

- [ ] **Step 3: Run syntax, whitespace, and stale-API audits**

```bash
python -m compileall -q spn_accel_cmodel tests
git diff --check 2941546..HEAD
rg -n "SPNInputs|SPNProfile\.GENERIC|compile_generic_plan|def _forward_|TODO|TBD|FIXME" \
  spn_accel_cmodel tests README.md validation/README.md
rg -n "range\([^\n]*iterations|for [^\n]*iterations" \
  spn_accel_cmodel/torch_spn*.py
```

Expected: no stale old API/profile-specific forward/incomplete markers; only
`propagate_canonical()` loops over propagation iterations. Ignore only the
pre-existing external Ramulator bridge `NotImplementedError`, which is outside
the searched markers and this propagation task.

- [ ] **Step 4: Run the three propagation validation layers**

```bash
PYTHONPATH=.:tests python -m unittest \
  test_torch_spn_official_frontend test_torch_spn_core \
  test_torch_spn_adapters test_torch_spn_goldens -v
```

Expected: all decoder, structural, adapter, and independent official-formula
tests pass.

- [ ] **Step 5: Run the complete repository suite**

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: every repository test passes.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md validation/README.md
git commit -m "docs: describe official-frontend unified SPN reference"
```

- [ ] **Step 7: Request independent review**

Ask the reviewer to inspect the full OfficialFrontend range, specifically:

- official convolution parameter names and channel order;
- `aff_scale_const` checkpoint use;
- probability versus logit fields;
- NLSPN/CompletionFormer confidence offset convention;
- DySPN T/K reshape, base stencils, softmax, and sequential reduction;
- NLPM dynamic sigmoid and group order;
- absence of decoder/adapter state recurrence and old fallback APIs.

Resolve all Critical and Important findings using failing regression tests.

- [ ] **Step 8: Repeat final verification on the reviewed HEAD**

Repeat Steps 3 through 5. Confirm committed files are clean with:

```bash
git status --short
```

If the three known whitespace-only user changes remain, report them explicitly
as preserved and uncommitted rather than deleting or staging them. Push the
reviewed commits to `agent/unified-torch-spn` so PR #2 updates.
