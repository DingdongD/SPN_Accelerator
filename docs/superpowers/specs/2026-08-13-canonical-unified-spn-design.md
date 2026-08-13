# Canonical Unified Torch SPN Design

**Date:** 2026-08-13

## Goal

Refactor the Torch FP32 propagation model so CSPN, NLSPN,
CompletionFormer, and DySPN execute one canonical propagation recurrence.
Named model profiles may adapt author-specific tensor layouts and coefficients,
but they may not implement their own propagation loops or sample the evolving
state.

The CNN and Transformer heads that predict guidance, offsets, affinities,
attention, confidence, and sparse anchors remain outside this module. DCNv2 is
not part of the canonical algorithm; it is only the gather carrier used by the
released NLSPN and CompletionFormer implementations.

## Implementation Map

The implemented boundary follows this specification directly:

- `spn_accel_cmodel/torch_spn_types.py` defines `CanonicalSPNPlan` and the
  public configuration/input/trace types;
- `spn_accel_cmodel/torch_spn_adapters.py` compiles author metadata without
  advancing the propagated state;
- `spn_accel_cmodel/torch_spn_core.py` owns canonical sampling, reduction,
  fusion, validation, and the only iteration loop;
- `spn_accel_cmodel/torch_functional.py` preserves `UnifiedSPN` as a thin
  public compile-and-execute wrapper.

Callers may use `UnifiedSPN` or compile a plan and invoke
`propagate_canonical()` directly. Both paths reach the same recurrence.

## Pre-refactor Problem

The existing `UnifiedSPN` exposes a common API but dispatches propagation to
`_forward_cspn`, `_forward_nlspn`, `_forward_dyspn`,
`_forward_dyspn_nlpm`, or `_forward_generic`. This is profile-level code reuse,
not algorithm unification. It also permits configuration fields to be
descriptive placeholders when a named profile bypasses the generic path.

The refactor must remove these profile-specific recurrence functions. A profile
identifier may select an adapter once before execution; it must never select a
different propagation loop.

## Canonical Recurrence

For state `x_t`, initial prediction `x_0`, and step `t`, the single propagation
core computes:

```text
p_t       = (1 - g_pre_t) * x_t + g_pre_t * v_pre_t
s_t,k     = sample(p_t, offset_t,k)
c_t       = reduce_k(a_t,k * s_t,k)
            + a_current_t * p_t
            + a_initial_t * x_0
x_(t+1)   = (1 - g_post_t) * c_t + g_post_t * v_post_t
```

All tensors use FP32. The sampling and reduction policies are explicit plan
fields because coordinate convention, padding, and FP32 accumulation order are
observable parts of the released implementations. These policies change how
the common operators execute; they do not introduce a different recurrence.

## Canonical Plan

Adapters compile model inputs into one validated data contract:

```python
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
```

Canonical shapes are:

```text
offsets_xy          [B,S,K,2,H,W]
neighbor_affinity   [B,S,K,Ca,H,W]
current_affinity    [B,S,Ca,H,W]
initial_affinity    [B,S,Ca,H,W]
fusion gates        [B,S,1,H,W]
fusion values       [B,S,C,H,W]
group_scale         [B,S,G,Ca,H,W]
```

`S` is either `1` for static metadata or `iterations` for per-step metadata;
the core selects step zero for a static tensor without materializing repeated
copies. `Ca` is either `1` for channel-shared coefficients or `C` for
per-channel coefficients. Fusion gates broadcast across state channels.

`reduction_groups` partitions K into contiguous ranges. `group_scale` is one
for an ordinary single-group reduction. It retains NLPM's three group scales
and FP32 group order without adding an NLPM recurrence. The effective canonical
coefficient for neighbor K remains its neighbor affinity multiplied by its
group scale.

No-fusion is represented by a zero gate. Therefore the recurrence does not
need profile-specific optional branches. Every plan field is consumed by the
core or rejected during plan validation; unused configuration placeholders are
removed.

## Components and Responsibilities

### Profile adapters

Each adapter performs only input validation and metadata compilation:

- coordinate and channel-layout conversion;
- fixed-stencil generation;
- affinity transforms and normalization;
- sampling of metadata such as NLSPN confidence;
- construction of current/initial anchor coefficients;
- construction of pre/post fusion gates and values.

An adapter may use vectorized operations over all iterations and may sample a
confidence map. It may not loop over propagation iterations, sample the
evolving depth state, calculate a candidate depth, or produce output depth.

The adapters are:

```text
compile_cspn_plan
compile_nlspn_plan
compile_completionformer_plan
compile_dyspn_plan
compile_dyspn_nlpm_plan   # explicitly named compatibility path
compile_generic_plan      # existing configurable API, same canonical core
```

CompletionFormer may call the common NLSPN metadata compiler with a different
temperature, but it still exposes its own named adapter for traceability.

### Canonical propagation core

`propagate_canonical(current, initial, plan, return_trace)` is the only function
that owns the propagation iteration loop. It:

1. selects static or per-step metadata;
2. applies canonical pre-fusion;
3. samples all neighbors;
4. applies the configured FP32 within-group and between-group reduction order;
5. adds current and initial anchors;
6. applies canonical post-fusion;
7. records the canonical parameters, candidate, and output.

It does not inspect `SPNProfile`, normalization modes, confidence modes, sparse
fusion modes, or author-specific tensor layouts.

### Public wrapper

`UnifiedSPN.forward()` preserves the public `SPNConfig` and `SPNInputs` entry
point. It selects one adapter, compiles one `CanonicalSPNPlan`, and invokes
`propagate_canonical()` exactly once. This is the only permitted profile
dispatch.

## Exact Model Mappings

### CSPN

- Shift the eight source-layout guidance channels into target layout before
  building the plan.
- Normalize using the released absolute-sum formula and operation order.
- Use the fixed non-center 3x3 integer offsets, zero padding, static metadata,
  initial affinity `1 - sum(neighbor_affinity)`, and zero current affinity.
- The released code's optional sparse mask becomes post-fusion with
  `gate = sparse_depth.sign()` and `value = initial`, preserving the author
  behavior rather than substituting sparse depth values.

### NLSPN

- Convert residual `(dy,dx)` input to `(dx,dy)` and add the fixed 3x3
  non-center base stencil.
- Implement AS, ASS, TC, and TGASS normalization during plan compilation.
- For TC/TGASS, apply `tanh(raw / temperature)` before multiplying sampled
  confidence. Released confidence sampling uses residual-only offsets; the
  named legacy option uses base-plus-residual offsets.
- Use bilinear zero-padded sampling with `align_corners=True`, current affinity
  `1 - sum(neighbor_affinity)`, and zero initial affinity.
- `preserve_input` becomes pre-fusion with
  `gate = (sparse_depth > 0).float()` and `value = sparse_depth`.

The canonical sampler retains absolute-pixel semantics for `H=1` or `W=1`.

### CompletionFormer

CompletionFormer uses the NLSPN mapping with `temperature=100` for TC/TGASS.
Its released 16-channel and 18-channel offset layouts are canonicalized before
plan construction. No CompletionFormer-specific propagation code remains.

### Current DySPN

- Convert each iteration's residual `(dy,dx)` offsets to absolute `(dx,dy)`
  offsets using the K=1, 3, 5, or 9 base stencil.
- Apply softmax across K independently at each iteration and pixel.
- Set current and initial affinities to zero because the DySPN stencil includes
  its center when applicable.
- Use bilinear zero-padded sampling with `align_corners=False`.
- Preserve the released sequential FP32 neighbor accumulation order through
  `ReductionMode.SEQUENTIAL`.
- Use post-fusion with
  `gate = sigmoid(confidence_logits) * sparse_depth.sign()` and
  `value = sparse_depth`.

### DySPN NLPM compatibility profile

The 3x3, 5x5, and 7x7 edge groups are expanded into 48 canonical target-layout
neighbors. The plan records group ranges `(0,8)`, `(8,24)`, and `(24,48)`.
For each step, the released denominator is absorbed into group scale, current
affinity, and initial affinity; the three attention values remain explicit
group scales so multiplication occurs after each within-group reduction.
Sparse confidence becomes canonical post-fusion. The compatibility profile
therefore uses the same core and does not retain a grouped propagation loop.

## Reduction and Sampling Policies

`SamplingMode.INTEGER` uses exact integer gathering for CSPN and NLPM.
`SamplingMode.BILINEAR` uses absolute pixel coordinates with explicit zero or
border padding. Coordinate normalization is an implementation detail of the
sampler, including the singleton-dimension correction.

`ReductionMode.TORCH_SUM` reproduces profiles whose released carrier performs a
single K reduction. `ReductionMode.SEQUENTIAL` reproduces DySPN's explicit
Python-loop accumulation order. `ReductionMode.GROUPED` performs the declared
within-group reductions, applies `group_scale`, and reduces the group results
in declared order for NLPM. All modes execute inside the same canonical
candidate calculation.

## Trace Contract

`SPNTrace` records one entry per propagation iteration for:

```text
pre_fused_states
candidates
outputs
offsets
neighbor_affinities
current_affinities
initial_affinities
pre_fusion_gates
post_fusion_gates
```

Trace affinities always use canonical K-neighbor layout. The trace records
effective coefficients after group scaling and also records group scales. In
particular, NLPM records all 48 effective neighbor coefficients rather than
only its three group attention values. This makes traces comparable across
profiles and directly usable by the accelerator model.

## Validation and Golden Tests

Validation has three independent layers:

1. Adapter tests compare every canonical plan tensor against model-specific,
   independently computed metadata.
2. Core tests feed hand-computed plans directly to `propagate_canonical()` and
   verify recurrence, sampling, reduction, anchors, and fusion without any
   profile adapter.
3. End-to-end fixed-seed tests compare every output and trace step against the
   independent CSPN, NLSPN, CompletionFormer, DySPN, and NLPM author-formula
   oracles.

Structural regression tests enforce that:

- `_forward_cspn`, `_forward_nlspn`, `_forward_dyspn`,
  `_forward_dyspn_nlpm`, and `_forward_generic` do not exist;
- `propagate_canonical()` is the only production function containing a loop
  over propagation iterations;
- each named profile invokes the canonical core exactly once;
- adapters return a plan and never return propagated depth.

Numerical acceptance remains exact where author operation order permits it and
uses `rtol=1e-5, atol=1e-6` for bilinear/normalization paths. Tests cover CPU,
available CUDA, singleton dimensions, partial and complete out-of-bounds
sampling, static/per-step metadata, invalid shapes, and all released affinity
modes. The existing NumPy functional cross-check and complete repository test
suite must remain passing.

## Error Handling

Plan compilation raises `ValueError` before propagation when required author
inputs are absent or malformed. Canonical validation rejects:

- a time dimension other than `1` or `iterations`;
- K, spatial, batch, or coordinate dimensions that disagree;
- affinity channel count other than `1` or state channel count;
- incompatible dtype or device after canonicalization;
- a fusion gate/value shape that cannot broadcast to the state;
- invalid or overlapping reduction-group ranges.

Errors identify the adapter and field. The core assumes a validated plan and
contains no profile-specific recovery behavior. It does not clamp, replace, or
reject NaN/Inf values produced by an official formula; normal PyTorch FP32
propagation semantics apply.

## Compatibility and Scope

The public constructors `SPNConfig.cspn()`, `nlspn()`,
`completionformer()`, `dyspn()`, and `dyspn_nlpm()` remain available.
`UnifiedSPN.forward()` and optional trace return behavior remain available.
Configuration fields are retained only when an adapter or canonical execution
policy consumes them.

This refactor does not implement prediction heads, train a model, load model
weights, depend on DCNv2, or change the architectural timing model. Its scope is
the exact FP32 propagation operator and the canonical metadata consumed by that
operator.

## Completion Criteria

The refactor is complete only when:

1. all named models compile to `CanonicalSPNPlan`;
2. all propagation iterations execute in `propagate_canonical()`;
3. no profile-specific propagation loop remains;
4. adapters contain no evolving-state propagation;
5. complete canonical traces match independent author-formula goldens;
6. the full repository test suite passes with a clean worktree.
