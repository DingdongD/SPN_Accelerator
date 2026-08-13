# SPN Accelerator CModel

A tile/event-level architectural simulator for exploring **generic NPU tensor execution plus spatial propagation network (SPN) acceleration**. The first target is the CompletionFormer/NLSPN hot path, with CSPN and DySPN intended to share the same propagation abstraction.

## Architecture

```text
                    Command / schedule
                           |
           +---------------+----------------+
           |               |                |
      Tensor Engine       VPU*         SPN Engine
      Conv / GEMM                    AGU -> Gather
           |                         -> Bilinear
      ABUF/WBUF/PBUF                 -> Aff/Reduce
           |                              |
           +---------- shared memory -----+
                           |
                          DMA
                           |
                        DRAM*
```

`AnalyticalSystolicBackend` models array waves, fill/drain, Conv tiling, DMA overlap, ABUF/WBUF/PBUF capacity and utilization. `SPNEngine` models packetized `AGU -> banked gather -> bilinear -> affinity/reduction`, including real address traces and SRAM-bank conflicts. The default 128x128 INT16 NLSPN scratchpad budget is 512 KiB offset + 256 KiB affinity + 64 KiB depth ping-pong.

## Reproducible third-party stack

External architecture tools used by calibration and DSE are pinned under `third_party/` as git submodules:

| Module | Role |
|---|---|
| SCALE-Sim v3 | tensor-array cycle/traffic golden |
| Ramulator2 | standalone modern DDR/HBM timing backend |
| BookSim2 | NoC contention |
| Accelergy | action-count energy framework |
| HWComponents | component-model stack |
| HWComponents-CACTI | CACTI-backed SRAM/cache/DRAM models; recursively pulls CACTI |
| Timeloop | optional mapper/tiling cross-validation |

Exact upstream commits are recorded in `third_party/manifest.json`. SCALE-Sim's own nested legacy Ramulator is separately pinned for SCALE-Sim internal use; `third_party/ramulator2` is the standalone CModel memory backend. Reported experiments must not use `git submodule update --remote`.

## One-shot local setup

For the **complete stack**, use Python 3.12 because the pinned HWComponents packages require it:

```bash
git clone --branch agent/add-architectural-cmodel --recurse-submodules \
  https://github.com/DingdongD/SPN_Accelerator.git
cd SPN_Accelerator

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[validation]'

bash third_party/bootstrap.sh
python third_party/check.py
```

Equivalent:

```bash
make setup
make check-third-party
```

If only the timing/calibration stack is required, Python 3.10/3.11 is also supported:

```bash
make setup-core
make check-third-party-core
```

If the repository was cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
bash third_party/bootstrap.sh
```

The default bootstrap installs SCALE-Sim, Accelergy, HWComponents and HWComponents-CACTI into the active venv, builds/installs Ramulator2, and builds BookSim2. Timeloop source is initialized but not built automatically because upstream additionally requires system ISL/Barvinok dependencies. BookSim2 requires `flex` and `bison`.

See `third_party/README.md` for `--core`, `--init-only`, `--no-build`, exact pin verification and update policy.

## Smoke test

```bash
make test
PYTHONPATH=. python examples/run_dec2_nlspn.py
PYTHONPATH=. python examples/sweep_spn.py
```

Representative workload:

1. CompletionFormer `dec2` 3x3 Conv `160 -> 32` at 128x128;
2. two residual 3x3 Conv `32 -> 32` at 128x128;
3. deformable 8-neighbor NLSPN for configurable propagation steps.

The current baseline result is an **architecture-model prediction**, not a calibrated silicon claim.

## Torch propagation goldens

`spn_accel_cmodel.torch_functional` provides a propagation-only FP32 reference
for the released CSPN, NLSPN, CompletionFormer, DySPN, and DySPN-NLPM
implementations. This Torch operator is independent of the repository's FPGA,
timing, and accelerator models. CNN/Transformer prediction backbones and DCNv2
are outside its boundary.

All five profiles execute the same recurrence. `UnifiedSPN` first applies the
official propagation module's parameter decoder, including `conv_offset_aff`
where the released module owns it. A metadata adapter then creates a
`CanonicalSPNPlan`; it never advances the depth state. `propagate_canonical()`
is the sole owner of the propagation iteration loop:

```text
official RawInputs -> author parameter decode -> CanonicalSPNPlan
                                                    |
decoded current + initial --------------------------+
                                                    v
                                      propagate_canonical()
```

The implementation is split accordingly:

- `torch_spn_types.py`: public configuration, plan, and trace types;
- `torch_spn_author.py`: official propagation-module parameter decoders;
- `torch_spn_decoded.py`: internal post-decoder tensor contract;
- `torch_spn_adapters.py`: CSPN/NLSPN/CompletionFormer/DySPN/NLPM metadata
  compilers;
- `torch_spn_core.py`: sampling, reduction, sparse fusion, and the one canonical
  recurrence;
- `torch_functional.py`: stable public exports and the `UnifiedSPN` wrapper.

Canonical profiles preserve the author-code distinctions rather than treating
all SPNs as an eight-neighbor absolute-sum kernel:

| Profile | Propagation semantics |
|---|---|
| `SPNConfig.cspn()` | shifted 3x3 integer stencil, initial anchor, zero padding, optional code-exact post mask |
| `SPNConfig.nlspn()` | deformable eight-neighbor sampling, AS/ASS/TC/TGASS, current anchor, sampled confidence |
| `SPNConfig.completionformer()` | NLSPN fork with `tanh(raw / 100)` and six-step default |
| `SPNConfig.dyspn()` | per-iteration offsets/logits, K=1/3/5/9 including center, softmax, sparse-confidence post fusion |
| `SPNConfig.dyspn_nlpm()` | 3x3/5x5/7x7 grouped nonlinear propagation with per-step dynamic logits |

Example:

```python
from spn_accel_cmodel import (
    CompletionFormerRawInputs,
    SPNConfig,
    UnifiedSPN,
)

model = UnifiedSPN(SPNConfig.completionformer(iterations=6))
output, trace = model(
    CompletionFormerRawInputs(
        pred_init=pred_init,                    # [B,1,H,W]
        guidance=guidance,                      # [B,8,H,W]
        confidence_probability=confidence,      # [B,1,H,W]
        sparse_depth=sparse_depth,
    ),
    return_trace=True,
)
```

`model.author` owns the same `conv_offset_aff` and `aff_scale_const` parameter
names as the official propagation module. Parameters can therefore be copied
from a checkpoint already present locally with an explicit prefix:

```python
model.author.load_official_parameters(
    local_state_dict,
    prefix="module.prop_layer.",
)
```

No checkpoint is required to validate propagation mathematics; the unit tests
use fixed random FP32 parameters and identical random inputs on both sides.

Callers that already own decoded coefficients can use the lower boundary
directly:

```python
from spn_accel_cmodel import compile_nlspn_plan, propagate_canonical
from spn_accel_cmodel.torch_spn_decoded import DecodedSPNParameters

decoded = DecodedSPNParameters(
    current=current,
    initial=initial,
    raw_affinity=raw_affinity,
    residual_offsets_yx=residual_offsets_yx,
    confidence_probability=confidence,
    affinity_scale=affinity_scale,
)
plan = compile_nlspn_plan(config, decoded)
output, trace = propagate_canonical(current, initial, plan, return_trace=True)
```

Run the independent author-formula differential suite:

```bash
python -m pip install -e '.[torch-validation]'
PYTHONPATH=. python -m unittest discover -s tests -p 'test_torch_spn_goldens.py' -v
```

The golden functions in `tests/official_spn_references.py` do not import the
unified implementation.  Fixed-seed tests compare every iteration as well as
effective affinity and anchor coefficients.  Integer microcases use exact
checks where operation ordering allows; interpolated FP32 paths use
`rtol=1e-5, atol=1e-6`.

Released NLSPN and CompletionFormer repositories use DCNv2 as an implementation
carrier for deformable gather and weighted reduction. DCNv2 is not part of the
propagation abstraction or a dependency of this Torch model. Unit tests compare
the independently restated author decoder and propagation formulas using the
same fixed random inputs and parameters; no CUDA extension or downloaded
checkpoint is involved.

## Validation chain

```text
PyTorch / CompletionFormer trace
          |
          +--> NumPy SPN functional golden ---- output-value check
          |
          +--> CModel real-address SPN trace -- address/bank/cycle check

Tensor CModel ----> SCALE-Sim GEMM/Conv -------- compute/traffic check
SPN CModel -------> RTL vector contract -------- address/bank exact check
Subgraph CModel --> ACTSim / board JSON -------- latency/traffic check
```

Run every locally available gate:

```bash
make calibrate
```

`make calibrate` first verifies all third-party git pins. Once real SPN/RTL/ACTSim/board goldens are supplied:

```bash
make calibrate-strict
```

Useful direct commands:

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py
PYTHONPATH=. python validation/validate_conv_scalesim.py

PYTHONPATH=. python validation/validate_spn_trace.py trace.npz \
  --offset-key offset --affinity-key aff \
  --state-key pred_init --golden-key pred --steps 12

PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --trace-npz trace.npz --offset-key offset \
  --out validation/out/spn_vectors.csv

PYTHONPATH=. python validation/compare_spn_rtl_trace.py \
  validation/out/spn_vectors.csv rtl_vectors.csv
```

See `CALIBRATION_PLAN.md` for calibration/hold-out splits, fitting order and quantitative gates. See `validation/README.md` for semantic assumptions and external-golden schemas.

## Calibration boundary

Do not calibrate Tensor/SPN cycles to board wall time. Keep accelerator core, modeled DMA/memory, package/model load, launch/runtime, CPU glue, quant/dequant and CPU fallback separately accounted. Only components represented in the architectural CModel may be used to fit CModel parameters.

## Fidelity roadmap

```text
functional semantics
    -> SCALE-Sim tensor calibration
    -> SRAM/DMA/Ramulator2 calibration
    -> SPN exact RTL address contract
    -> SPN cycle calibration
    -> ACTSim subgraphs
    -> board subgraphs
    -> full CompletionFormer/CSPN/NLSPN/DySPN validation
    -> Timeloop/BookSim/energy-assisted architecture DSE
```

Ramulator2, BookSim2, Accelergy/HWComponents-CACTI and Timeloop are already pinned in `third_party/` so later fidelity stages do not require changing the source-dependency baseline.
