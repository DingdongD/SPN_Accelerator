# Calibration Experiment Plan

This document defines the calibration procedure for the SPN Accelerator CModel.
The goal is not to tune the simulator until one workload matches one measured
number. The goal is to establish a reproducible, hold-out-validated chain from
functional semantics to tensor-array timing, SPN memory behavior, RTL timing,
ACTSim, and board subgraphs.

## 0. Rules of the experiment

1. **Freeze the experiment identity.** Every reported result must record:
   - SPN_Accelerator git commit;
   - `configs/*.json` used by the CModel;
   - SCALE-Sim git commit/version;
   - RTL git commit and Verilator/simulator version, if used;
   - ACTSim architecture/config revision, if used;
   - board firmware/runtime/model-package revision, if used;
   - Python version and host machine.
2. **Separate calibration and hold-out cases.** Parameters may only be tuned on
   the calibration split. Accuracy claims use the hold-out split.
3. **Do not fit architectural facts.** Array dimensions, lane counts, number of
   SRAM banks/ports, bit widths, and clock frequency are architecture inputs,
   not free calibration variables.
4. **Calibrate one subsystem at a time.** Tensor compute is calibrated before
   DMA; SPN addresses before SPN cycles; accelerator-core timing before board
   software overhead.
5. **Do not compensate errors across levels.** A wrong tensor compute model must
   not be hidden by an inflated DMA latency, and a wrong SPN address stream must
   not be hidden by an average bandwidth correction.
6. **A skipped external stage is not a pass.** Run the suite with `--strict`
   before making a calibrated-accuracy claim.

## 1. Recommended environment

Create an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[validation]'
```

For PyTorch functional cross-checks, install a CPU or CUDA PyTorch build suitable
for the local machine.

For SCALE-Sim v3, clone the upstream repository separately and install it from
source:

```bash
git clone https://github.com/scalesim-project/SCALE-Sim.git ../SCALE-Sim
python -m pip install -e ../SCALE-Sim
python -c "import scalesim; print('SCALE-Sim import OK')"
```

SCALE-Sim's current upstream documentation supports installation from source and
invocation through `python -m scalesim.scale`.

## 2. One-command local suite

The repository provides an orchestrator:

```bash
PYTHONPATH=. python validation/run_calibration_suite.py
```

Without external goldens this performs:
- L0 unit/functional regressions;
- Python compile check;
- SCALE-Sim stages if SCALE-Sim is installed, otherwise records them as SKIP;
- deterministic 8x8 SPN RTL-vector export;
- summary generation.

Outputs:

```text
validation/out/calibration/
  summary.json
  summary.md
  L0_*.log
  L1a_*.log
  L1b_*.log
  L3a_*.log
  ...
```

Once all external goldens are available, use:

```bash
PYTHONPATH=. python validation/run_calibration_suite.py \
  --trace-npz <completionformer_trace.npz> \
  --offset-key offset \
  --affinity-key aff \
  --state-key pred_init \
  --golden-key pred \
  --steps 12 \
  --rtl-csv <rtl_vectors.csv> \
  --actsim-model-json <cmodel_actsim_metrics.json> \
  --actsim-golden-json <actsim_metrics.json> \
  --board-model-json <cmodel_board_metrics.json> \
  --board-golden-json <board_metrics.json> \
  --strict
```

`--strict` converts missing L1/L3/L4 external goldens into failures.

---

# 3. Level L0: functional semantics

## Purpose

Prove that the CModel and the framework agree on *what is computed* before
calibrating *how long it takes*.

## Experiments

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q spn_accel_cmodel validation tests examples
```

The existing regression suite covers:
- bilinear sampling against `torch.nn.functional.grid_sample`;
- zero padding and boundary coordinates;
- random 8-neighbor offset/affinity propagation;
- CompletionFormer-style 18-channel offset with center-pair removal;
- banked SRAM synthetic cases;
- tensor-engine scaling/traffic invariants;
- end-to-end small workload execution.

## Gate

Floating-point semantic golden:

```text
max_abs <= 1e-5
RMSE    <= 1e-6
```

For a future fixed-point SPN implementation, add a second integer functional
golden and require **bit-exact equality** for rounding, saturation, accumulator
width, and requantization.

No timing parameter may be tuned until L0 passes.

---

# 4. Level L1a: Tensor Engine compute-cycle calibration

## Purpose

Calibrate the systolic-array compute model independently of Conv traffic and
board/runtime overhead.

## Golden

SCALE-Sim v3, output-stationary (`os`) dataflow.

The current adapter uses GEMM topology in the upstream format:

```text
Layer Name, M, N, K
```

and reads:
- `COMPUTE_REPORT.csv`;
- `DETAILED_ACCESS_REPORT.csv`.

Run:

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py
```

## Initial microkernel matrix

The checked-in validator includes CompletionFormer-representative tile shapes:

| Case | M | N | K | Role |
|---|---:|---:|---:|---|
| m64_n16_k144 | 64 | 16 | 144 | small / under-filled |
| m128_n32_k288 | 128 | 32 | 288 | medium |
| m256_n32_k288 | 256 | 32 | 288 | dec/head-like |
| m256_n64_k576 | 256 | 64 | 576 | larger K/N |

After the first successful run, extend the matrix to at least 12 cases and
freeze the split **before tuning**:

```text
Calibration split: 8 cases
Hold-out split:     4 cases
```

The extension should cover:
- M smaller/equal/larger than array rows;
- N smaller/equal/larger than array columns;
- K = 144, 288, 576, 1440 or representative model values;
- partial M/N waves.

Repeat at array sizes 32x32, 64x64, and 128x128 if those are DSE targets.

## Parameters that may be calibrated here

Only compute-pipeline parameters:
- `tensor.pipeline_depth`;
- a documented fixed per-tile controller latency, if independent RTL evidence
  shows it is required.

Do **not** fit:
- array rows/columns;
- memory bandwidth;
- DMA setup;
- tensor bit widths.

## Metrics

For each hold-out case:

```text
relative_error = abs(C_cmodel - C_golden) / C_golden
```

Report:
- median relative error;
- mean absolute percentage error (MAPE);
- maximum relative error;
- utilization correlation.

## Gate

First calibrated target:

```text
hold-out median error <= 3%
hold-out max error    <= 5%
```

If this fails, inspect wave mapping and fill/drain accounting before adding any
empirical correction factor.

---

# 5. Level L1b: Conv tile cycle + traffic calibration

## Purpose

Validate actual Conv receptive-input/filter/output traffic without conflating it
with GEMM/im2col expansion.

Run:

```bash
PYTHONPATH=. python validation/validate_conv_scalesim.py
```

Current cases intentionally represent complete CModel tiles:
- 16x16 output, Cin=16, Cout=16;
- 16x16 output, Cin=32, Cout=32;
- 8x8 output, Cin=64, Cout=32.

For 3x3 stride-1 Conv, a 16x16 output tile uses an 18x18 receptive IFMAP tile.

## Metrics

Compare:
- compute cycles;
- DRAM IFMAP element count;
- DRAM filter element count;
- DRAM OFMAP element count.

## Gate

Traffic is a semantic/counting quantity and should be exact for an isolated
tile:

```text
IFMAP traffic error  = 0%
Filter traffic error = 0%
OFMAP traffic error  = 0%
```

Compute cycles:

```text
relative error <= 5%
```

If traffic differs, do not tune a bandwidth parameter. Fix tiling/reuse/counting
semantics first.

---

# 6. Level L2: DMA and on-chip SRAM calibration

## Purpose

Separate memory service behavior from Tensor Engine compute.

## L2a: banked SRAM

The CModel already has exact synthetic bank-service tests. Extend local testing
with patterns:

```text
sequential
stride-2
stride-4
stride-8
single-hot-bank
uniform-bank
random
```

For each pattern record:
- total scalar accesses;
- ideal service cycles;
- conflict stall cycles;
- effective reads/cycle.

When the SPN RTL SRAM/gather block exists, these synthetic traces become
CModel-vs-RTL exact tests.

## L2b: DMA

Use transfer-only or near-transfer-only ACTSim/RTL microbenchmarks over at least:

```text
1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB
```

Fit the model:

```text
cycles = setup_cycles + ceil(bytes / effective_bytes_per_cycle)
```

Use half the sizes for fitting and the others as hold-out.

Calibratable:
- `dma.setup_cycles`;
- `dma.bandwidth_bytes_per_cycle`.

Gate on hold-out:

```text
median error <= 5%
max error    <= 10%
```

Do not infer DMA parameters from full CompletionFormer E2E latency.

---

# 7. Level L3a: Real CompletionFormer SPN trace validation

## Required NPZ contract

Export one or more CompletionFormer frames containing:

```text
pred_init   initial depth state
offset      NLSPN offset tensor
aff         full 9-channel affinity tensor
pred        propagated output golden
```

Accepted offset layouts include the native 16-channel form and the
CompletionFormer-style 18-channel form with the zero center pair.

Run:

```bash
PYTHONPATH=. python validation/validate_spn_trace.py trace.npz \
  --offset-key offset \
  --affinity-key aff \
  --state-key pred_init \
  --golden-key pred \
  --steps 12
```

## Dataset split

Do not calibrate on one frame.

Recommended minimum:

```text
Calibration traces: 8 frames
Hold-out traces:     8 frames
```

Prefer a mix of:
- low-offset frames;
- large-offset/boundary-heavy frames;
- smooth depth;
- sparse-depth boundaries;
- high guidance variation.

## Metrics

Functional:
- max absolute error;
- mean absolute error;
- RMSE.

Timing/address statistics:
- cycles/iteration;
- gather reads;
- bank conflict stalls;
- bank conflict rate;
- AGU/gather/interp/affinity busy cycles.

The structural read-count invariant for 8 deformable neighbors is:

```text
reads = H * W * (1 + 4 * 8) * steps
      = H * W * 33 * steps
```

## Gate

Floating-point semantics:

```text
max_abs <= 1e-5
RMSE    <= 1e-6
```

Read count must be exact.

Bank-conflict rate has no target until an independent RTL trace is available.

---

# 8. Level L3b: SPN RTL exact-address and cycle calibration

Calibration must be two-stage.

## Stage A: exact address/bank contract

Generate CModel vectors:

```bash
PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --trace-npz trace.npz \
  --offset-key offset \
  --out validation/out/calibration/spn_cmodel_vectors.csv
```

RTL/Verilator must output the same schema:

```text
packet,issue_group,lane,address,bank
```

Compare:

```bash
PYTHONPATH=. python validation/compare_spn_rtl_trace.py \
  validation/out/calibration/spn_cmodel_vectors.csv \
  rtl_vectors.csv
```

Gate:

```text
row count: exact
address:   exact
bank:      exact
```

Do not proceed to cycle fitting if any address/bank mismatch exists.

## Stage B: pipeline timing

Once Stage A is exact, measure cycles for:
- zero offset;
- checker offset;
- random offset;
- real CompletionFormer traces;
- 8x8, 16x16, 32x32;
- multiple bank/gather-lane configurations if RTL supports them.

Calibratable timing parameters:
- `agu_latency`;
- `interp_latency`;
- `affinity_latency`;
- SRAM `read_latency`;
- any explicitly modeled queue/controller constant.

Architectural lane counts and SRAM bank counts are not fitting variables.

Gate on hold-out real traces:

```text
median cycle error <= 3%
max cycle error    <= 5%
bank-conflict cycles <= 2% error or exact if the RTL contract exposes them
```

---

# 9. Level L4: ACTSim and board subgraph calibration

## Target subgraphs

Start with isolated blocks before full CompletionFormer:

1. `dec2` high-resolution Conv path;
2. `dep_dec1/dep_dec0`;
3. `gd_dec1/gd_dec0`;
4. `cf_dec1/cf_dec0 Conv`;
5. SPN propagation once an accelerator implementation exists;
6. decoder + heads;
7. decoder + heads + SPN.

## Metric schema

Normalize both CModel and external golden to nested JSON:

```json
{
  "metrics": {
    "tensor.dec2.cycles": 0,
    "tensor.dec2.act_bytes": 0,
    "tensor.dec2.weight_bytes": 0,
    "tensor.dec2.output_bytes": 0,
    "spn.nlspn.cycles": 0,
    "spn.nlspn.gather_reads": 0
  }
}
```

Compare:

```bash
PYTHONPATH=. python validation/compare_external.py \
  cmodel.json actsim.json \
  --name dec2_actsim \
  --rel 0.10
```

## Accounting boundary

Board E2E must be decomposed into:

```text
accelerator core
DMA / transfer
package/model load
launch/runtime
CPU glue
quant/dequant
SPN CPU fallback
```

Only accelerator-core and modeled DMA components are used to calibrate this
CModel. Never tune tensor/SPN cycles to match package loading or CPU glue.

## Gate

Initial subgraph gate:

```text
accelerator-core latency error <= 10%
traffic error                  <= 5% (prefer exact where countable)
```

After microarchitecture calibration matures:

```text
subgraph latency error <= 5%
```

Board wall time may differ more because software overhead remains outside the
accelerator CModel.

---

# 10. Level L5: full-model validation

Only run L5 after L0-L4 pass independently.

Use at least:
- CompletionFormer;
- one CSPN model;
- one NLSPN model;
- DySPN when its propagation lowering is implemented.

For each model report:
- per-layer cycles;
- Tensor Engine utilization;
- SPN Engine utilization;
- DRAM/SRAM traffic;
- SPN bank conflict;
- accelerator-core E2E latency;
- separately reported runtime/software overhead.

Recommended first acceptance target:

```text
full accelerator-core model error <= 10%
```

Final target after calibration:

```text
median subgraph/full-model error <= 5%
```

---

# 11. Parameter-fitting order

Use this order so one subsystem cannot compensate for another:

```text
1. Functional semantics
2. Tensor array fill/drain / pipeline
3. Conv tile traffic
4. SRAM banking
5. DMA setup + effective bandwidth
6. SPN exact addresses/banks
7. SPN pipeline latencies
8. ACTSim subgraph residual constants
9. Full-model validation with NO new free parameters
```

After Step 8, freeze the architecture config. L5 is validation only.

---

# 12. What to save from each experiment

Recommended local layout:

```text
validation/local_goldens/
  scalesim/
    <scalesim_commit>/
      tensor/
      conv/
  rtl/
    <rtl_commit>/
      spn/
  actsim/
    <arch_revision>/
  board/
    <runtime_revision>/
```

Raw traces can be large and should normally remain local or be stored as release
artifacts. Commit only:
- small schemas/examples;
- summary JSON/Markdown when useful;
- calibration parameters and the source revision that produced them.

For every result preserve:

```text
experiment id
date/time
git revisions
hardware/config
command line
raw log
normalized JSON
pass/fail thresholds
```

---

# 13. First local session checklist

A practical first session is:

```bash
# 1. Checkout
git clone https://github.com/DingdongD/SPN_Accelerator.git
cd SPN_Accelerator
git checkout agent/add-architectural-cmodel

# 2. Environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[validation]'

# 3. Internal baseline
PYTHONPATH=. python validation/run_calibration_suite.py

# 4. Install SCALE-Sim
git clone https://github.com/scalesim-project/SCALE-Sim.git ../SCALE-Sim
python -m pip install -e ../SCALE-Sim

# 5. Tensor goldens
PYTHONPATH=. python validation/validate_tensor_scalesim.py
PYTHONPATH=. python validation/validate_conv_scalesim.py

# 6. Run suite again: L1 should no longer be SKIP
PYTHONPATH=. python validation/run_calibration_suite.py
```

Then provide the generated:

```text
validation/out/calibration/summary.json
validation/out/tensor_scalesim/summary.json
validation/out/conv_scalesim/summary.json
```

for the first model/golden discrepancy analysis.

The next session should add a real CompletionFormer trace and then an SPN RTL
golden.
