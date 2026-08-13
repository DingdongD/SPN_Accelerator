# Unified Torch SPN Design

## Objective

Add a PyTorch FP32 propagation model to `SPN_Accelerator` that expresses the
propagation portions of CSPN, NLSPN, CompletionFormer, and DySPN through one
explicit recurrence.  CNN and Transformer heads that predict offsets,
affinities, confidence, or sparse-depth anchors remain outside the model.

The implementation must be tested against independent, model-specific
references constructed from the authors' released propagation source.  It must
not claim end-to-end model equivalence or include training-network behavior.

## Scope

The first implementation covers these author-code profiles:

- CSPN 3x3 propagation with eight non-center integer neighbors, initial-depth
  anchoring, zero padding, and the released PyTorch code's optional
  post-propagation replacement behavior.
- NLSPN with deformable non-center neighbors, AS/ASS/TC/TGASS affinity modes,
  neighbor-position confidence sampling, current-state anchoring, zero padding,
  and optional pre-propagation input preservation.
- CompletionFormer's NLSPN derivative, including its `tanh(raw / 100)`
  difference and its default TGASS configuration.
- The current default DySPN module: per-iteration offsets and logits, softmax
  across K sampled positions (including the center), zero-padded bilinear
  sampling with `align_corners=False`, and post-propagation sparse-confidence
  fusion.
- The DySPN 7x7 NLPM/naive profile as a separately named compatibility profile,
  because it is not the repository's current default DySPN path.

The following are outside this change:

- offset, affinity, confidence, and initial-depth prediction heads;
- complete pretrained-network inference and dataset metrics;
- CUDA kernel performance or gradient-performance benchmarking;
- quantized or fixed-point propagation semantics;
- modifications to the architectural cycle model.

## Architecture

Create `spn_accel_cmodel/torch_functional.py`.  The module has three layers:

1. Typed configuration and input structures describe sampling, offset layout,
   affinity lifetime and normalization, anchoring, neighbor-confidence use, and
   sparse fusion independently.
2. Small Torch primitives perform coordinate conversion, zero-padded sampling,
   affinity normalization, and sparse fusion.
3. `UnifiedSPN` applies one recurrence and records an optional trace containing
   every intermediate state and the effective per-iteration coefficients.

The broad recurrence is:

```text
pre_state[t] = optional_pre_sparse_constraint(state[t])
candidate[t] = sum_k(weight[t,k] * sample(pre_state[t], position[t,k]))
             + self_weight[t] * pre_state[t]
             + initial_weight[t] * initial
state[t+1]   = optional_post_sparse_fusion(candidate[t], sparse_depth, confidence)
```

This separation is required because the author implementations place sparse
constraints and confidence in different parts of the recurrence.

## Public Types

The configuration exposes independent enums rather than model-named behavior:

```text
NeighborMode: GRID, DILATED, OFFSET
SamplingMode: INTEGER, BILINEAR
PaddingMode: ZEROS, BORDER
OffsetMode: ABSOLUTE_XY, RESIDUAL_XY, RESIDUAL_YX
AffinityMode: STATIC, PER_ITERATION
NormalizationMode: ABS_SUM, ABS_SUM_FLOOR, TC, TGASS, SOFTMAX, DYSPN_NLPM
AnchorMode: NONE, INITIAL, CURRENT, INITIAL_CURRENT
NeighborConfidenceMode: NONE, SAMPLE_AT_NEIGHBOR
SparseFusionMode: NONE, HARD_PRE, HARD_POST, SOFT_POST
AffinityLayout: TARGET, CSPN_SHIFTED_SOURCE
```

`SPNInputs` accepts NCHW FP32 state tensors plus optional initial state,
affinity/logits, offsets, confidence, sparse depth, sparse mask, group attention,
and center attention.  Static tensors omit the iteration dimension; dynamic
tensors contain an explicit T dimension.

Named constructors provide canonical profiles:

```text
SPNConfig.cspn(...)
SPNConfig.nlspn(...)
SPNConfig.completionformer(...)
SPNConfig.dyspn(...)
SPNConfig.dyspn_nlpm(...)
```

These constructors prevent callers from reconstructing subtle author defaults
by hand.  Low-level explicit construction remains available for architecture
experiments.

## Coordinate and Sampling Contract

Offsets inside `UnifiedSPN` use absolute pixel displacements ordered `(dx,dy)`.
Adapters convert author tensors before sampling:

- CSPN uses fixed integer 3x3 displacements without a center sample.
- NLSPN and CompletionFormer use DCNv2 residual offsets ordered `(dy,dx)` and
  add the corresponding deformable-kernel base displacement.
- DySPN uses the repository's per-iteration residual layout and base stencil;
  K may be 1, 3, 5, or 9 and includes the center where present.

Normalized coordinates use the correct formula for the selected
`align_corners` value.  Zero padding never clamps coordinates.  Border mode is
supported only as an explicit non-author experiment.

## Model Profiles

### CSPN

Neighbor weights use absolute-sum normalization.  The residual coefficient
`1 - sum(neighbor_weights)` multiplies the initial prediction on every
iteration.  Guidance layout conversion matches the author's shifted guidance
semantics.  The released PyTorch code forms a mask from `sparse_depth` but
post-replaces with `raw_depth_input` (the initial prediction), not with the
sparse-depth values.  The canonical `cspn()` profile reproduces that code.  A
separately selected generic hard-post fusion can express the sparse-value
replacement stated in the paper without mislabeling it as source-code exact.

### NLSPN

The eight predicted affinities are static across propagation iterations.  The
center coefficient `1 - sum(neighbor_weights)` multiplies the current pre-state.
When confidence propagation is enabled, the single-channel confidence map is
sampled at each non-center offset with detached coordinates before affinity
normalization.  AS, ASS, TC, and TGASS remain distinct:

- AS: absolute-sum normalization;
- ASS: absolute-sum denominator floored to one;
- TC: `tanh(raw) / K` without subsequent absolute-sum normalization;
- TGASS: `tanh(raw) / (gamma * K)` followed by floor-one absolute-sum
  normalization.

Optional input preservation replaces valid sparse positions in the state before
each propagation step.

### CompletionFormer

CompletionFormer uses the same propagation structure as NLSPN.  Its TC/TGASS
pre-transform is `tanh(raw / 100)` rather than NLSPN's `tanh(raw)`.  The
canonical profile defaults to six iterations, TGASS, gamma 0.5, neighbor
confidence enabled, and no input preservation.

### DySPN

The default profile accepts offsets and logits for every iteration.  Softmax is
applied independently across K positions for each iteration and pixel.  There
is no separately synthesized anchor coefficient because the sampling stencil
contains its center entry.  After propagation:

```text
c = sigmoid(confidence) * valid_sparse_mask
state = (1 - c) * candidate + c * sparse_depth
```

Sampling uses zero padding, bilinear interpolation, and
`align_corners=False`.

### DySPN NLPM

The 3x3, 5x5, and 7x7 edge groups and the current-state group have separate
per-iteration attention.  The normalized residual term anchors the initial
prediction.  This profile is named explicitly and is never selected by
`SPNConfig.dyspn()`.

## Validation Design

Create `tests/official_spn_references.py` with direct, model-specific reference
functions.  These functions must not import or call `UnifiedSPN` helpers.  They
mirror the author source's order of operations so shared implementation bugs
cannot make the golden and implementation agree accidentally.

Create `tests/test_torch_spn_goldens.py` with fixed-seed differential tests for:

- one and multiple iterations;
- intermediate states, not only final output;
- image boundaries and samples partially or fully outside the image;
- static and per-iteration affinity/offset inputs;
- `(dy,dx)` to `(dx,dy)` conversion and base-grid addition;
- all NLSPN affinity modes and CompletionFormer's temperature difference;
- sampled neighbor confidence with detached offsets;
- hard-pre, hard-post, and soft-post sparse behavior;
- DySPN K=5 and K=9 stencils;
- the separately named DySPN NLPM path;
- invalid tensor shapes and incompatible configuration combinations.

Hand-computed deterministic microcases cover coordinate order, zero padding,
center insertion, and sparse-fusion timing.  Random differential tests use
fixed seeds and values deliberately chosen away from interpolation
discontinuities.

Acceptance criteria are:

- integer CSPN microcases use exact equality where operation order permits it;
- bilinear and normalization paths use
  `torch.testing.assert_close(rtol=1e-5, atol=1e-6)`;
- every profile's complete intermediate trace matches its independent oracle;
- the existing NumPy CompletionFormer/NLSPN path remains passing and receives a
  Torch cross-check for its supported subset;
- the full existing repository test suite remains passing.

An optional CUDA/DCNv2 test may be added when the official extension is already
installed.  It compares only the propagation gather/reduce result for identical
state, offset, and affinity tensors.  It is not a required CI dependency,
because the pure-Torch references already validate the propagation semantics.

## Error Handling

The module raises `ValueError` before computation when required inputs are
absent, iteration dimensions disagree with the configuration, channel counts do
not match the stencil, offsets use an unsupported shape, or a normalization is
paired with an incompatible anchor mode.  Inputs are converted to FP32 on the
state device; batch and spatial broadcasting is limited to explicitly
documented scalar/static metadata forms.

## Documentation

Update the repository README and validation README to state which author-code
profiles are covered, how to run the Torch golden suite, and why DCNv2 is only
an optional propagation-carrier check rather than part of the prediction head.
Do not describe tolerance-based floating-point agreement as bitwise equality.
