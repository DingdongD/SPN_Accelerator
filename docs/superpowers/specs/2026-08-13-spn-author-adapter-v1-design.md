# SPN AuthorAdapter v1 Design

**Date:** 2026-08-13

## Goal

Extend the existing canonical FP32 SPN reference from a post-parameter-generation
boundary to the author module boundary for all five supported profiles:

- CSPN;
- NLSPN;
- CompletionFormer;
- current DySPN;
- DySPN NLPM compatibility.

The completed data path is:

```text
Official/profile-specific tensors
        -> author parameter decode
        -> decoded SPN parameters
        -> CanonicalSPNPlan
        -> propagate_canonical()
```

`propagate_canonical()` remains the only function that advances the evolving
depth state. Prediction backbones, pretrained checkpoint distribution, DCNv2,
non-FP32 execution, and any system-level accelerator modeling remain outside
this change. AuthorAdapter v1 is solely a PyTorch reference for the unified SPN
propagation operator.

## Upstream Baseline

The author interfaces and parameter decoders follow these fixed upstream
commits:

- XinJCheng/CSPN `b3e487bdcdcd8a63333656e69b3268698e543181`;
- zzangjinsun/NLSPN_ECCV20
  `ba33fa5d9ea62ca970026a145ab18fab76d79d4a`;
- youmi-zym/CompletionFormer
  `2744eddee9b57595dc3064f7d342569736a6803b`;
- Kyakaka/DySPN `d4871eeabc8797d821873a2a41daf359466a7255`.

Tests use fixed random weights and inputs. They do not download, commit, or
require a pretrained checkpoint. Checkpoint compatibility means that the
author modules expose the official plan-producing parameter names and tensor
shapes and can copy those exact keys from a caller-supplied official state
dict. DCNv2 carrier constants such as `w`, `b`, and `w_conf` are not duplicated
because the canonical core does not execute that carrier.

## Layered Architecture

```text
Profile-specific RawInputs
        |
        v
Author Reference Module
  - official conv_offset_aff where applicable
  - official aff_scale_const where applicable
  - official channel split and reshape
  - explicit confidence interpretation
        |
        v
DecodedSPNParameters (internal)
        |
        v
Profile Plan Compiler
  - source-to-target layout conversion
  - YX-to-XY conversion
  - center separation
  - affinity normalization
  - sparse fusion mapping
        |
        v
CanonicalSPNPlan
        |
        v
propagate_canonical()
```

### File responsibilities

- `spn_accel_cmodel/torch_spn_types.py` defines configuration, canonical plan,
  trace, and the five public raw-input dataclasses. The overloaded public
  `SPNInputs` type is removed.
- `spn_accel_cmodel/torch_spn_decoded.py` defines the internal decoded tensor
  contract used only between author modules and plan compilers.
- `spn_accel_cmodel/torch_spn_author.py` defines five author reference modules
  and the profile-to-module construction table.
- `spn_accel_cmodel/torch_spn_adapters.py` accepts decoded parameters and emits
  `CanonicalSPNPlan`. It owns no learned layers and never advances depth state.
- `spn_accel_cmodel/torch_spn_core.py` retains validation, sampling, reduction,
  fusion, tracing, and the sole propagation iteration loop.
- `spn_accel_cmodel/torch_functional.py` exposes the breaking author-level
  `UnifiedSPN` facade and public types.

## Public Raw Interfaces

The public input contract is a profile-specific tagged set rather than one
overloaded dataclass:

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

The field names deliberately distinguish probability tensors from logits.
There is no automatic confidence interpretation.

`UnifiedSPN(config)` constructs exactly one matching author module under the
public `author` attribute. `forward()` accepts only that module's raw-input
type:

```python
model = UnifiedSPN(SPNConfig.nlspn())
model.author.load_official_parameters(
    full_state_dict,
    prefix="module.prop_layer.",
)
output, trace = model(raw_inputs, return_trace=True)
```

`prefix` is mandatory and identifies the exact upstream SPN submodule. The
loader reads only its explicitly declared plan-producing keys, requires every
one of them with the official shape, and copies them into the author module.
AuthorAdapter v1 performs no heuristic key search, prefix guessing, alternate
key alias, or fallback loading.

## Internal Decoded Contract

`DecodedSPNParameters` is an internal dataclass with unambiguous
post-parameter-generation fields:

```python
@dataclass
class DecodedSPNParameters:
    current: torch.Tensor
    initial: torch.Tensor
    raw_affinity: torch.Tensor
    residual_offsets_yx: torch.Tensor | None = None
    confidence_probability: torch.Tensor | None = None
    sparse_depth: torch.Tensor | None = None
    sparse_mask: torch.Tensor | None = None
    attention_logits: torch.Tensor | None = None
    affinity_scale: torch.Tensor | None = None
```

Profile compilers accept this internal contract. Fields that do not apply to a
profile remain `None`; the selected compiler requires its exact fields and
does not reinterpret another field as a substitute.

## Profile Decode Semantics

### CSPN

The author interface is `guidance`, `blur_depth`, and optional
`sparse_depth`. CSPN has no learned parameter decoder in the propagation
module. The author module maps:

- `blur_depth` to both canonical current and initial state;
- `guidance` to the CSPN shifted-source affinity layout;
- `sparse_depth.sign()` to the post-fusion gate when code-mask preservation is
  configured;
- `blur_depth` rather than sparse depth to the post-fusion value, matching the
  released implementation.

Fixed 3x3 non-center offsets remain compile-time profile metadata.

### NLSPN

`NLSPNAuthor` owns:

```python
self.conv_offset_aff = nn.Conv2d(
    num_neighbors,
    3 * num_neighbors,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
)
self.aff_scale_const = nn.Parameter(...)
```

The convolution is zero-initialized as in the released module. Its result is
split with `torch.chunk(result, 3, dim=1)` into `o1`, `o2`, and raw affinity.
The offset channels are reconstructed in the exact author order with
`torch.cat((o1, o2), dim=1).view(B, K, 2, H, W)`.

The plan compiler then performs the existing canonical operations:

- residual deformable offset YX to canonical XY;
- addition of the fixed 3x3 non-center base stencil;
- AS, ASS, TC, or TGASS affinity processing;
- confidence sampling with released residual-only behavior or explicit legacy
  base-plus-residual behavior;
- current anchor `1 - sum(neighbor_affinity)`;
- optional hard pre-fusion of `feat_fix`.

For TGASS, normalization uses the loaded `aff_scale_const` parameter, not a
recomputed `affinity_gamma * K` value. TC retains the official fixed K scale.

### CompletionFormer

`CompletionFormerAuthor` shares the NLSPN family implementation and owns the
same official parameter names and shapes. It differs only in author-specific
configuration:

- TC and TGASS use `tanh(raw_affinity / 100)`;
- the default propagation count is six;
- confidence is a probability produced by the backbone;
- sparse depth is passed as the author `feat_fix` input.

No CompletionFormer-specific propagation loop or sampling implementation is
introduced.

### Current DySPN

`DySPNAuthor` owns the released parameter generator:

```python
channels = iterations * num_neighbors
self.conv_offset_aff = nn.Conv2d(
    channels,
    3 * channels,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
)
```

The convolution is zero-initialized. Its output is split into `2*T*K` offset
channels and `T*K` affinity-logit channels. Offsets are reshaped to
`[B,T,K,2,H,W]` in author YX order; logits are reshaped to `[B,T,K,H,W]`.
The existing compiler adds the selected K=1/3/5/9 base stencil, changes YX to
XY, applies softmax along K, selects sequential FP32 reduction, and constructs
the post-fusion gate from `sigmoid(confidence_logits) * sparse_depth.sign()`.

### DySPN NLPM compatibility

The author interface names the released tensors directly:

- 48-channel regular-grid guidance;
- `4*T` dynamic logits;
- sparse-depth confidence probability.

The author module reshapes dynamic logits to `[B,T,4,H,W]`; the compiler owns
the existing sigmoid, 3x3/5x5/7x7 group coefficients, current and initial
anchors, and sparse post-fusion mapping. It retains the canonical groups
`(0,8)`, `(8,24)`, and `(24,48)` and introduces no learned convolution.

## Canonical Contract

The existing dense FP32 plan remains unchanged:

| Tensor | Shape |
|---|---|
| `offsets_xy` | `[B,S,K,2,H,W]` |
| `neighbor_affinity` | `[B,S,K,Ca,H,W]` |
| `current_affinity` | `[B,S,Ca,H,W]` |
| `initial_affinity` | `[B,S,Ca,H,W]` |
| `pre_fusion_gate` | `[B,S,1,H,W]` |
| `pre_fusion_value` | `[B,S,C,H,W]` |
| `post_fusion_gate` | `[B,S,1,H,W]` |
| `post_fusion_value` | `[B,S,C,H,W]` |
| `group_scale` | `[B,S,G,Ca,H,W]` |

`S` is one for frame-static metadata and `T` for per-iteration metadata.
Constant offsets are materialized as dense tensors because this contract is a
PyTorch numerical reference, not a physical storage or execution description.

## Validation and Errors

The implementation is intentionally strict:

- a raw-input type that does not match the configured profile raises
  `TypeError`;
- missing required tensors, invalid channel counts, invalid spatial shapes,
  and invalid per-iteration dimensions raise field-named `ValueError`;
- confidence logits and probabilities are never inferred from numeric range;
- no tensor is resized, padded to a different author layout, or silently
  substituted;
- no DCNv2, alternate sampling carrier, checkpoint-prefix heuristic, legacy
  `SPNInputs`, or profile fallback path is provided.

Normal Torch dtype/device canonicalization to FP32 on the current-state device
is retained. Author formula NaN/Inf behavior is not clamped or replaced.

## Test Strategy

### Decoder equivalence

For NLSPN, CompletionFormer, and DySPN, tests initialize the production
`conv_offset_aff` with fixed random FP32 weights and biases. An independent
oracle applies `torch.nn.functional.conv2d` with the same tensors and restates
the released split/reshape expressions. Tests compare:

- raw convolution output;
- residual offsets in author order;
- raw affinity or per-step affinity logits;
- loaded/fixed affinity scale behavior.

A synthetic full-model state dict with a non-empty explicit prefix verifies
the strict official-key copy path and missing/wrong-shape failures. It contains
fixed random tensors rather than pretrained weights.

The oracle does not call author frontend helpers or plan compilers.

### End-to-end author-boundary equivalence

Independent formulas consume each profile's RawInputs and the fixed random
decoder weights. They compare the compiled canonical metadata and every
propagation iteration for CSPN, NLSPN, CompletionFormer, current DySPN, and
NLPM. Coverage includes all supported NLSPN affinity modes, confidence modes,
legacy offsets, sparse fusion, DySPN K=1/3/5/9, and NLPM group order.

Normalized and bilinear FP32 paths use `rtol=1e-5, atol=1e-6`. Integer and
operation-order microcases use exact comparison where the author and canonical
expressions execute identical operations.

### Structural invariants

AST tests require:

- `propagate_canonical()` is the only production function that loops over
  propagation iterations while advancing state;
- author modules contain parameter decode but no evolving-state recurrence;
- adapters contain no learned layers and return only `CanonicalSPNPlan`;
- every `UnifiedSPN` forward invokes the canonical core exactly once.

The complete repository suite and CUDA CPU-metadata/device tests remain
required.

## Compatibility and Migration

This is an intentional breaking API change:

- public `SPNInputs` is removed;
- callers must use the profile-specific RawInputs type;
- existing low-level tests and callers that already possess decoded offsets or
  affinity migrate to internal decoded/compiler tests or construct a
  `CanonicalSPNPlan` and call `propagate_canonical()` directly;
- `CanonicalSPNPlan`, `propagate_canonical()`, `SPNConfig`, and trace behavior
  remain public.

There is no compatibility shim, deprecated alias, automatic conversion, or
fallback branch.

## Completion Criteria

AuthorAdapter v1 is complete only when:

1. all five profiles accept their author-level RawInputs;
2. NLSPN, CompletionFormer, and DySPN own official-shaped, zero-initialized
   parameter generators with official parameter names;
3. fixed random state dicts load strictly and decoder tensors match independent
   author formulas;
4. all five profiles compile to `CanonicalSPNPlan` and call the single
   recurrence once;
5. every traced propagation iteration matches its independent author formula;
6. `SPNInputs` and its fallback semantics are absent from the public and
   production API;
7. all focused and complete repository tests pass.
