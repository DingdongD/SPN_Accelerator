# Validation and calibration chain

The simulator is intentionally split into **functional semantics**, **microarchitecture timing**, and **external calibration**. Each layer is validated independently so end-to-end agreement cannot hide compensating errors in compute, traffic, or SPN bank-conflict models.

The current chain is:

```text
PyTorch / CompletionFormer trace
          |
          +--> NumPy SPN functional reference  ---- output-value check
          |
          +--> CModel real-address SPN trace   ---- address/bank/cycle check

Tensor CModel ----> SCALE-Sim GEMM/Conv golden ---- compute/traffic check

SPN CModel -------> RTL vector contract ---------- address/bank exact check

Subgraph CModel --> ACTSim / board JSON ---------- latency/traffic check
```

## Status

| Level | Target | Implemented now | External golden required |
|---|---|---|---|
| L0 | SPN functional semantics | Yes: NumPy reference + unified Torch author-formula goldens | No |
| L1a | Tensor compute cycles | Yes: SCALE-Sim v3 GEMM adapter/report parser | SCALE-Sim installation or saved report |
| L1b | Conv traffic + cycles | Yes: SCALE-Sim Conv-tile adapter | SCALE-Sim installation |
| L2 | SRAM banks / DMA | Yes: synthetic bank-conflict tests | Optional Ramulator2 for DRAM timing |
| L3a | Real SPN address behavior | Yes: CompletionFormer-style NPZ offset loader | Real model trace NPZ |
| L3b | SPN RTL contract | Yes: vector exporter + exact trace comparator | RTL-produced CSV |
| L4 | ACTSim / board subgraph | Yes: generic nested-JSON comparator/schema | ACTSim/board golden JSON |
| L5 | Full model | Framework ready; not calibrated yet | Calibrated L1--L4 results |

The repository does **not** claim calibrated cycle accuracy until the corresponding external golden has actually been run.

---

## Level 0 — functional SPN semantics

`spn_accel_cmodel/functional.py` implements a dependency-light NumPy reference for the propagation kernel:

```text
absolute offset
  -> zero-padded bilinear sample
  -> per-neighbor affinity multiply
  -> center + 8-neighbor reduction
  -> repeated propagation
```

It is aligned to the CompletionFormer/NLSPN fallback convention used by the current reference path: absolute coordinates are converted to `grid_sample` coordinates with `align_corners=True`, bilinear interpolation, and zero padding.

Run the regression suite:

```bash
python -m pip install -e '.[validation]'
python -m unittest discover -s tests -v
```

When PyTorch is available, `tests/test_recorded_trace.py` additionally checks:

- scalar bilinear samples against `torch.nn.functional.grid_sample` including out-of-bound coordinates;
- a full 8-neighbor random offset/affinity propagation step against a direct PyTorch construction;
- 18-channel CompletionFormer offset tensors with the inserted center pair removed correctly.

For quantized hardware, add the accelerator's fixed-point rounding/saturation rules as a second functional reference instead of changing this floating-point semantic golden.

### Unified Torch author-code profiles

The broader Torch functional model lives in
`spn_accel_cmodel/torch_functional.py`.  Its independent test oracles live in
`tests/official_spn_references.py` and mirror these released source paths:

- XinJCheng/CSPN commit `b3e487bdcdcd8a63333656e69b3268698e543181`;
- zzangjinsun/NLSPN_ECCV20 commit `ba33fa5d9ea62ca970026a145ab18fab76d79d4a`;
- youmi-zym/CompletionFormer commit `2744eddee9b57595dc3064f7d342569736a6803b`;
- Kyakaka/DySPN commit `d4871eeabc8797d821873a2a41daf359466a7255`.

Run only the Torch profile goldens:

```bash
PYTHONPATH=. python -m unittest discover \
  -s tests -p 'test_torch_spn_goldens.py' -v
```

Run Torch plus the existing NumPy cross-check:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

The NLSPN/CompletionFormer `legacy` switch is observable at the propagation
boundary.  Current `legacy=False` source samples confidence with residual-only
offsets in its 1x1 deformable gather; `legacy=True` adds the 3x3 base stencil.
Both paths are tested.  Propagated state always uses base plus residual offsets.

Agreement here is FP32 semantic agreement, not a claim of bitwise equality
with a CUDA DCNv2 kernel.  The acceptance threshold for interpolated paths is
`rtol=1e-5, atol=1e-6`; deterministic integer microcases use exact equality
where the reduction order is identical.

---

## Level 1a — tensor compute cycles vs SCALE-Sim

The adapter targets the current SCALE-Sim report/input contracts:

- GEMM topology: `Layer Name, M, N, K`;
- `COMPUTE_REPORT.csv`: total cycles, stall cycles, utilization, mapping efficiency;
- `DETAILED_ACCESS_REPORT.csv`: SRAM/DRAM reads and writes.

Install SCALE-Sim in the validation environment, then run:

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py \
  --config configs/baseline.json \
  --output-dir validation/out/tensor_scalesim
```

The script compares representative CompletionFormer tile GEMMs against `AnalyticalSystolicBackend` and emits:

```text
validation/out/tensor_scalesim/summary.json
validation/out/tensor_scalesim/summary.md
```

If SCALE-Sim has already been run elsewhere, copy its report directory and use:

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py \
  --report-dir /path/to/scalesim/run_directory
```

### What is compared

- tensor-array cycles;
- SCALE-Sim stall cycles (recorded as metadata);
- overall utilization;
- mapping efficiency;
- compute utilization.

The default engineering acceptance target is <=5% cycle error **after** mapping/dataflow conventions are aligned. Failure here should be fixed in the tensor backend rather than hidden with a global end-to-end scale factor.

---

## Level 1b — Conv tile traffic and cycles vs SCALE-Sim

A GEMM representation is sufficient for systolic compute-cycle cross-checking, but it is **not** a same-semantics traffic golden for Conv because GEMM exposes an im2col-like input matrix. Therefore Conv traffic is validated separately using SCALE-Sim's Conv topology.

Run:

```bash
PYTHONPATH=. python validation/validate_conv_scalesim.py \
  --config configs/baseline.json \
  --output-dir validation/out/conv_scalesim
```

The supplied microkernels are defined so each SCALE-Sim Conv corresponds to one complete CModel tile. The IFMAP dimensions include exactly the receptive halo needed by that output tile (for example 18x18 input -> 16x16 output for a valid 3x3 stride-1 tile).

The script compares:

- compute cycles;
- DRAM IFMAP elements vs CModel activation-tile elements;
- DRAM filter elements vs CModel weight elements;
- DRAM OFMAP elements vs CModel output elements.

Traffic is expected to be exact once both tools use the same tile and memory-residency assumptions. If SCALE-Sim retains data due to its SRAM policy, record that explicitly and compare at the correct residency boundary rather than loosening the threshold silently.

---

## Level 2 — SRAM and DMA microbenchmarks

`BankedSRAM` is already exercised with deterministic synthetic traces. Add/retain cases for:

```text
sequential
stride-2 / stride-4 / stride-N
uniform-bank
single-hot-bank
random
```

Required exact checks are:

- address -> bank mapping;
- scalar request count;
- ideal service cycles;
- bank-conflict stall cycles.

The current DMA model is analytical (`setup + bytes/bandwidth`). Ramulator2 should be introduced only for experiments where DRAM command/row/bank timing matters; do not fit SRAM bank penalties into the DMA bandwidth parameter.

---

## Level 3a — real CompletionFormer/NLSPN trace validation

The timing model can now consume a real offset tensor instead of a synthetic offset pattern. Supported layouts include:

```text
[1, 16, H, W]       # 8 neighbors x (dy,dx)
[16, H, W]
[H, W, 16]
[8, 2, H, W]
[H, W, 8, 2]

[1, 18, H, W]       # center offset pair already inserted
[18, H, W]
```

For 18-channel tensors the center pair is removed before timing, matching the 8 deformable non-center neighbors.

Run timing-only validation:

```bash
PYTHONPATH=. python validation/validate_spn_trace.py \
  traces/completionformer_sample0.npz \
  --offset-key offset \
  --steps 12 \
  --output validation/out/spn_real_trace.json
```

If the NPZ also contains initial depth, affinity, and a golden propagated result:

```bash
PYTHONPATH=. python validation/validate_spn_trace.py \
  traces/completionformer_sample0.npz \
  --offset-key offset \
  --state-key pred_init \
  --affinity-key aff \
  --golden-key pred \
  --steps 12
```

The output reports both:

```text
functional:
  max_abs / mean_abs / RMSE

timing:
  cycles
  gather_reads
  bank_conflict_stall_cycles
  bank_conflict_rate
  AGU/gather/interp/affinity busy cycles
```

### Boundary timing policy

The floating-point functional golden uses true zero padding. The current timing SRAM model issues a fixed four-point read packet per deformable neighbor and clamps out-of-range addresses to valid SRAM locations; architecturally this should be interpreted as a **fixed-four-read pipeline with boundary masking**. No clamped value may contribute to the functional output. The RTL validation model must implement the same request/mask policy, or the timing model should be extended with an explicit `boundary_policy` before cycle numbers are compared.

---

## Level 3b — CModel <-> SPN RTL exact trace contract

Generate deterministic CModel vectors:

```bash
PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --height 16 --width 16 \
  --offset-pattern checker \
  --out validation/out/spn_vectors.csv
```

or use a recorded model trace:

```bash
PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --trace-npz traces/completionformer_sample0.npz \
  --offset-key offset \
  --out validation/out/spn_vectors.csv
```

The CSV is the RTL contract at the gather boundary and includes packet/group/lane/address/bank fields. After the RTL/Verilator testbench dumps the same schema:

```bash
PYTHONPATH=. python validation/compare_spn_rtl_trace.py \
  validation/out/spn_vectors.csv \
  /path/to/rtl_vectors.csv
```

This comparison is intentionally **exact**. The first mismatches are printed with row/field details so errors cannot be hidden in an average bank-conflict rate.

After address/bank traces match, add cycle columns from RTL for the five pipeline stages:

```text
AGU -> Gather -> Bilinear -> Affinity/Reduce -> State write
```

and calibrate only the fixed pipeline latency/queue-depth parameters.

---

## Level 4 — ACTSim and board subgraph validation

External tools should export a small normalized JSON record rather than forcing the CModel to parse unstable log text. See `golden_schema.example.json`.

A typical record should separate accelerator-core and software/runtime terms:

```json
{
  "tensor": {
    "dec2": {
      "cycles": 0,
      "dram_read_bytes": 0,
      "dram_write_bytes": 0
    }
  },
  "spn": {
    "propagation": {
      "cycles": 0,
      "state_reads": 0,
      "state_writes": 0
    }
  }
}
```

Compare nested metrics with:

```bash
PYTHONPATH=. python validation/compare_external.py \
  validation/out/model.json \
  validation/golden_actsim.json \
  --name dec2_actsim \
  --rel 0.08
```

For board measurements keep these terms separate:

```text
accelerator inference
DMA/boundary movement
package/model load
host Q/DQ and glue
software launch/runtime
```

Do **not** tune tensor/SPN engine cycles to match a board end-to-end number containing package loading or CPU glue.

---

## Level 5 — full-model validation

Only after Levels 1--4 are stable should the simulator predict full CompletionFormer/CSPN/NLSPN/DySPN latency and traffic. Report error at three granularities:

1. microkernel;
2. subgraph/module;
3. end-to-end accelerator core.

This prevents one over-estimated module from numerically compensating for an under-estimated one.

## Suggested engineering acceptance targets

These are project gates, not accuracy claims about the current uncalibrated model:

| Validation item | Initial gate |
|---|---:|
| SPN floating functional semantics | `max_abs <= 1e-5`, `RMSE <= 1e-6` for FP32 reference tests |
| Tensor microkernel cycles | <= 5% after same mapping/dataflow alignment |
| Conv/SRAM/DRAM element counts | exact for the same residency/mapping contract |
| SPN address/read/bank mapping | exact against RTL trace |
| Calibrated subgraph latency | <= 8% |
| Full accelerator latency | <= 10% initially, tighten toward 5% |

Every validation artifact should record the model config, external tool/version, workload shape, mapping/dataflow, and whether the number represents compute-only, accelerator-core, or software-inclusive latency.
