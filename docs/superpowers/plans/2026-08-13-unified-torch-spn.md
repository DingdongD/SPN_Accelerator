# Unified Torch SPN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one Torch FP32 propagation implementation whose named profiles match the released CSPN, NLSPN, CompletionFormer, and current-default DySPN propagation paths against independent golden references.

**Architecture:** `spn_accel_cmodel/torch_functional.py` owns typed configuration, coordinate/layout adapters, sampling, normalization, anchoring, sparse fusion, and tracing. `tests/official_spn_references.py` independently restates author-code propagation formulas; `tests/test_torch_spn_goldens.py` compares every intermediate state and targeted edge cases without reusing production helpers. The existing NumPy and timing models remain unchanged except for a Torch cross-check and documentation.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy validation extra, `unittest`, setuptools optional dependencies.

---

### Task 1: Establish the Typed API and Sampling Contract

**Files:**
- Create: `tests/test_torch_spn_goldens.py`
- Create: `spn_accel_cmodel/torch_functional.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing API, validation, and sampling tests**

Add tests that import every public enum/dataclass, construct canonical configs,
reject incompatible shapes, verify `(dy,dx)` residual conversion, and check an
out-of-bounds bilinear microcase for both `align_corners=True` and `False`:

```python
class TorchSPNSamplingTest(unittest.TestCase):
    def test_zero_padding_does_not_replicate_border(self):
        state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        offsets = torch.tensor([[[[[-0.5, -0.5], [-0.5, -0.5]],
                                  [[-0.5, -0.5], [-0.5, -0.5]]]]])
        actual = sample_neighbors(state, offsets, SamplingMode.BILINEAR,
                                  PaddingMode.ZEROS, align_corners=True)
        torch.testing.assert_close(actual[0, 0, 0, 0, 0], torch.tensor(0.25))

    def test_nlspn_profile_uses_residual_yx_offsets(self):
        cfg = SPNConfig.nlspn(iterations=2, normalization=NormalizationMode.AS)
        self.assertEqual(cfg.offset_mode, OffsetMode.RESIDUAL_YX)
        self.assertEqual(cfg.anchor_mode, AnchorMode.CURRENT)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. python -m unittest tests.test_torch_spn_goldens.TorchSPNSamplingTest -v
```

Expected: import failure because `spn_accel_cmodel.torch_functional` does not
exist.

- [ ] **Step 3: Implement enums, dataclasses, canonical constructors, tensor validation, offset adapters, and sampler**

Implement these public definitions:

```python
class NeighborMode(Enum): GRID = auto(); DILATED = auto(); OFFSET = auto()
class SamplingMode(Enum): INTEGER = auto(); BILINEAR = auto()
class PaddingMode(Enum): ZEROS = auto(); BORDER = auto()
class OffsetMode(Enum): ABSOLUTE_XY = auto(); RESIDUAL_XY = auto(); RESIDUAL_YX = auto()
class AffinityMode(Enum): STATIC = auto(); PER_ITERATION = auto()
class NormalizationMode(Enum): ABS_SUM = auto(); ABS_SUM_FLOOR = auto(); TC = auto(); TGASS = auto(); SOFTMAX = auto(); DYSPN_NLPM = auto()
class AnchorMode(Enum): NONE = auto(); INITIAL = auto(); CURRENT = auto(); INITIAL_CURRENT = auto()
class NeighborConfidenceMode(Enum): NONE = auto(); SAMPLE_AT_NEIGHBOR = auto()
class SparseFusionMode(Enum): NONE = auto(); CSPN_CODE_POST = auto(); HARD_PRE = auto(); HARD_POST = auto(); SOFT_POST = auto()
class AffinityLayout(Enum): TARGET = auto(); CSPN_SHIFTED_SOURCE = auto()
```

Implement `SPNConfig`, `SPNInputs`, and `SPNTrace` with named constructors, and
implement `sample_neighbors()` using un-clamped absolute coordinates and the
correct normalized-coordinate formula:

```python
if align_corners:
    normalized = 2.0 * coord / (size - 1) - 1.0
else:
    normalized = 2.0 * (coord + 0.5) / size - 1.0
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all `TorchSPNSamplingTest` tests pass.

- [ ] **Step 5: Add the optional Torch dependency**

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
validation = ["numpy>=1.24"]
torch-validation = ["numpy>=1.24", "torch>=2.0"]
```

- [ ] **Step 6: Commit the API and sampler**

```bash
git add pyproject.toml spn_accel_cmodel/torch_functional.py tests/test_torch_spn_goldens.py
git commit -m "feat: add typed Torch SPN sampling API"
```

### Task 2: Match CSPN, NLSPN, and CompletionFormer

**Files:**
- Create: `tests/official_spn_references.py`
- Modify: `tests/test_torch_spn_goldens.py`
- Modify: `spn_accel_cmodel/torch_functional.py`

- [ ] **Step 1: Write independent official references**

Implement direct reference functions without importing production helpers:

```python
def reference_cspn(initial, guidance, iterations, sparse_depth=None): ...
def reference_nlspn(initial, raw_affinity, residual_yx, iterations,
                    mode, confidence=None, sparse_depth=None,
                    preserve_input=False, temperature=1.0,
                    gamma=0.5): ...
```

Each reference builds its own grids and calls only PyTorch primitives. CSPN
must shift both guidance and state like the released code and reproduce the
released post-mask use of initial prediction. NLSPN must sample confidence at
detached offsets, normalize after confidence multiplication, and insert its
current-state coefficient.

- [ ] **Step 2: Write fixed-seed differential tests for CSPN and all NLSPN modes**

Add tests that use `torch.Generator().manual_seed(1207)`, compare two or three
iterations, and compare `SPNTrace.outputs` element by element:

```python
for actual, expected in zip(trace.outputs, expected_trace, strict=True):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

Cover CSPN without and with its code-exact post mask; NLSPN AS, ASS, TC, and
TGASS; confidence propagation; hard-pre preserve input; and CompletionFormer's
temperature-100 difference.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=. python -m unittest \
  tests.test_torch_spn_goldens.CSPNGoldenTest \
  tests.test_torch_spn_goldens.NLSPNGoldenTest -v
```

Expected: failures because `UnifiedSPN.forward()` and profile normalization are
not implemented.

- [ ] **Step 4: Implement recurrence, normalization, anchors, sampled confidence, and CSPN layout**

Implement `UnifiedSPN.forward()` so each iteration selects static or dynamic
metadata, optionally applies hard-pre fusion, samples neighbors, applies
effective coefficients, then performs the configured post fusion. Preserve the
official operation order in each profile. Store effective offsets, neighbor
coefficients, self coefficients, initial coefficients, candidates, and outputs
in `SPNTrace`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 3 command. Expected: all CSPN/NLSPN/CompletionFormer tests pass.

- [ ] **Step 6: Commit the fixed-affinity profiles**

```bash
git add spn_accel_cmodel/torch_functional.py tests/official_spn_references.py tests/test_torch_spn_goldens.py
git commit -m "feat: align Torch CSPN and NLSPN propagation"
```

### Task 3: Match Current DySPN and the Named NLPM Compatibility Path

**Files:**
- Modify: `tests/official_spn_references.py`
- Modify: `tests/test_torch_spn_goldens.py`
- Modify: `spn_accel_cmodel/torch_functional.py`

- [ ] **Step 1: Add independent DySPN references**

Implement:

```python
def reference_dyspn(initial, residual_yx, logits, sparse_depth,
                     confidence_logits, base_offsets): ...
def reference_dyspn_nlpm(initial, guidance, attention_logits,
                          sparse_depth, confidence, iterations): ...
```

The default reference uses per-iteration `softmax(dim=2)`,
`align_corners=False`, zero padding, and post-step confidence fusion. The NLPM
reference independently performs the 3x3/5x5/7x7 edge reductions, dynamic
group attention, initial residual, and post fusion in author order.

- [ ] **Step 2: Write K=5, K=9, boundary, and NLPM differential tests**

Use fixed random inputs with nonzero metadata on every iteration. Include one
test that swaps iteration metadata and proves the result changes, preventing a
static-affinity implementation from passing.

- [ ] **Step 3: Run tests and verify RED**

```bash
PYTHONPATH=. python -m unittest \
  tests.test_torch_spn_goldens.DySPNGoldenTest \
  tests.test_torch_spn_goldens.DySPNNLPMGoldenTest -v
```

Expected: failures for missing dynamic softmax and NLPM behavior.

- [ ] **Step 4: Implement per-iteration metadata, DySPN stencils, soft-post fusion, and NLPM groups**

`SPNConfig.dyspn()` must accept K in `{1,3,5,9}`, create the author base
stencil including center, and require `[B,T,K,H,W]` logits plus
`[B,T,K,2,H,W]` residual offsets. `SPNConfig.dyspn_nlpm()` must require 48
guidance channels and `[B,T,4,H,W]` group attention.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 3 command. Expected: all DySPN tests pass.

- [ ] **Step 6: Commit dynamic propagation support**

```bash
git add spn_accel_cmodel/torch_functional.py tests/official_spn_references.py tests/test_torch_spn_goldens.py
git commit -m "feat: align Torch DySPN propagation"
```

### Task 4: Cross-Check Existing NumPy Semantics and Harden Validation

**Files:**
- Modify: `tests/test_recorded_trace.py`
- Modify: `tests/test_torch_spn_goldens.py`
- Modify: `spn_accel_cmodel/__init__.py`

- [ ] **Step 1: Write a failing NumPy-to-Torch cross-check**

Create the same static eight-neighbor trace in NumPy and Torch, use explicit
center/current affinity, and require:

```python
np.testing.assert_allclose(torch_out.numpy(), numpy_out,
                           rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run the cross-check and verify RED**

```bash
PYTHONPATH=. python -m unittest \
  tests.test_recorded_trace.TorchFunctionalCrossCheck.test_unified_torch_matches_numpy_reference -v
```

Expected: failure until the new public module is wired to the existing trace
convention.

- [ ] **Step 3: Add public exports and trace-layout adapter**

Export `UnifiedSPN`, `SPNConfig`, and `SPNInputs` from
`spn_accel_cmodel.__init__` without importing Torch eagerly when the optional
dependency is absent. Add only the adapter needed to consume existing
CompletionFormer/NLSPN trace layouts.

- [ ] **Step 4: Run validation tests and verify GREEN**

```bash
PYTHONPATH=. python -m unittest tests.test_recorded_trace tests.test_torch_spn_goldens -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit cross-validation wiring**

```bash
git add spn_accel_cmodel/__init__.py tests/test_recorded_trace.py tests/test_torch_spn_goldens.py
git commit -m "test: cross-check Torch and NumPy SPN semantics"
```

### Task 5: Document and Verify the Complete Change

**Files:**
- Modify: `README.md`
- Modify: `validation/README.md`

- [ ] **Step 1: Update user documentation**

Document the supported profiles, the propagation-only boundary, exact commands,
the FP32 tolerances, and the optional role of DCNv2:

```bash
PYTHONPATH=. python -m unittest tests.test_torch_spn_goldens -v
PYTHONPATH=. python -m unittest discover -s tests -v
```

- [ ] **Step 2: Run formatting and syntax checks**

```bash
python -m compileall -q spn_accel_cmodel tests
git diff --check
```

Expected: both commands exit zero with no diagnostics.

- [ ] **Step 3: Run the complete fresh verification suite**

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Expected: every existing and new test passes with zero failures and zero errors.

- [ ] **Step 4: Review requirements against the design specification**

Read `docs/superpowers/specs/2026-08-13-unified-torch-spn-design.md` and verify
that each in-scope profile, trace field, error case, tolerance, and documentation
requirement has a corresponding passing test or documented optional gate.

- [ ] **Step 5: Commit documentation and final verification state**

```bash
git add README.md validation/README.md
git commit -m "docs: describe Torch SPN golden validation"
```
