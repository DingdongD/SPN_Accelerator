# Calibration Experiment Plan

This plan defines how to turn the SPN Accelerator CModel from an architectural estimator into a hold-out-validated, cycle-level calibrated model. Calibration is performed subsystem-by-subsystem so one error cannot hide another.

## 0. Reproducibility rules

Every reported experiment must freeze:

- SPN_Accelerator git commit and `configs/*.json`;
- all `third_party/` submodule commits from `third_party/manifest.json`;
- RTL/Verilator revision when used;
- ACTSim architecture/compiler revision when used;
- board firmware/runtime/model-package revision when used;
- Python/compiler/CMake versions and host machine.

Never use `git submodule update --remote` for a reported experiment. Parameters are tuned only on a calibration split; all accuracy claims use hold-out cases.

Architectural facts are **not fitting variables**: array rows/columns, SPN lane counts, SRAM bank/port counts, bit widths and frequency remain fixed inputs.

## 1. One-shot environment setup

Recommended clone:

```bash
git clone --branch agent/add-architectural-cmodel --recurse-submodules \
  https://github.com/DingdongD/SPN_Accelerator.git
cd SPN_Accelerator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[validation]'
bash third_party/bootstrap.sh
python third_party/check.py
```

Or:

```bash
make setup
make check-third-party
```

The repository pins SCALE-Sim, Ramulator2, BookSim2, Accelergy, HWComponents, HWComponents-CACTI/CACTI and Timeloop. Timeloop is initialized but not built by the default bootstrap because its upstream build needs system ISL/Barvinok packages.

## 2. Calibration ladder

```text
L0 functional semantics
  -> L1a tensor-array compute
  -> L1b Conv traffic
  -> L2a SRAM banking
  -> L2b DMA / Ramulator memory
  -> L3a real SPN traces
  -> L3b SPN RTL exact address + cycles
  -> L4 ACTSim / board subgraphs
  -> L5 full models
```

Run locally available gates with:

```bash
make calibrate
```

When all independent goldens are available:

```bash
make calibrate-strict
```

A `SKIP` is never evidence of accuracy.

---

## L0 — Functional semantics

Run:

```bash
make test
```

The regression suite checks bilinear sampling against PyTorch `grid_sample`, boundary/zero-padding semantics, random 8-neighbor propagation, CompletionFormer 18-channel center-pair removal, banked SRAM invariants and Tensor Engine traffic/scaling invariants.

Gate for floating-point SPN semantics:

```text
max_abs <= 1e-5
RMSE    <= 1e-6
```

A future integer datapath must add a second fixed-point golden and require bit-exact rounding/saturation/accumulator behavior.

---

## L1a — Tensor Engine compute cycles vs SCALE-Sim

Run:

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py
```

The current representative GEMMs are:

| Case | M | N | K |
|---|---:|---:|---:|
| small | 64 | 16 | 144 |
| medium | 128 | 32 | 288 |
| dec/head-like | 256 | 32 | 288 |
| large-K/N | 256 | 64 | 576 |

Extend to at least 12 cases before tuning and freeze an **8 calibration / 4 hold-out** split. Cover partial M/N waves and M/N below, equal to, and above the physical array dimensions.

Allowed fitting parameters here:

- `tensor.pipeline_depth`;
- a documented fixed controller latency only if independent RTL evidence requires it.

Forbidden here: memory bandwidth, DMA setup, array dimensions and bit widths.

Hold-out gate:

```text
median cycle error <= 3%
max cycle error    <= 5%
```

If this fails, fix wave mapping/fill-drain accounting before adding empirical factors.

---

## L1b — Conv tile cycles and traffic vs SCALE-Sim

Run:

```bash
PYTHONPATH=. python validation/validate_conv_scalesim.py
```

Conv traffic is validated using true receptive IFMAP tiles rather than GEMM/im2col matrices. Compare:

- compute cycles;
- DRAM IFMAP element count;
- filter element count;
- OFMAP element count.

For an isolated tile, traffic is a counting invariant:

```text
IFMAP traffic error  = 0%
filter traffic error = 0%
OFMAP traffic error  = 0%
compute-cycle error <= 5%
```

Do not tune bandwidth to hide a traffic mismatch.

---

## L2a — SRAM bank service

Exercise sequential, stride-2/4/8, uniform-bank, single-hot-bank and random address traces. Record:

- scalar accesses;
- ideal cycles;
- bank-conflict stall cycles;
- effective reads/cycle.

When the SPN RTL SRAM/gather block exists, require exact bank assignment and exact service cycles for the synthetic traces.

---

## L2b — DMA and DRAM timing

Use transfer-only microbenchmarks over at least:

```text
1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB
```

Use half for fitting and half as hold-out. Fit only:

```text
cycles = setup_cycles + ceil(bytes / effective_bytes_per_cycle)
```

Calibratable: `dma.setup_cycles`, `dma.bandwidth_bytes_per_cycle`.

Hold-out target:

```text
median error <= 5%
max error    <= 10%
```

Then replace the bandwidth-only external-memory service with the pinned `third_party/ramulator2` backend and compare queue latency, row locality and achieved bandwidth. Do not infer DMA parameters from full CompletionFormer E2E time.

---

## L3a — Real CompletionFormer SPN traces

Export at least 16 representative frames with:

```text
pred_init
offset
aff
pred
```

Freeze:

```text
8 calibration frames
8 hold-out frames
```

Include low/high offset, boundary-heavy, smooth-depth and strong-depth-edge examples.

Run one trace with:

```bash
PYTHONPATH=. python validation/validate_spn_trace.py trace.npz \
  --offset-key offset --affinity-key aff \
  --state-key pred_init --golden-key pred --steps 12
```

Structural invariant for 8 deformable neighbors:

```text
reads = H * W * (1 + 4*8) * steps
      = H * W * 33 * steps
```

Functional gate remains `max_abs <= 1e-5`, `RMSE <= 1e-6`; read count must be exact.

---

## L3b — SPN RTL calibration

### Stage A: exact address/bank contract

```bash
PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --trace-npz trace.npz --offset-key offset \
  --out validation/out/spn_cmodel_vectors.csv

PYTHONPATH=. python validation/compare_spn_rtl_trace.py \
  validation/out/spn_cmodel_vectors.csv rtl_vectors.csv
```

RTL output schema:

```text
packet,issue_group,lane,address,bank
```

Gate: row count, address and bank are all exact. Do not tune timing until this passes.

### Stage B: pipeline cycles

Measure zero/checker/random offsets plus real CompletionFormer traces at 8x8, 16x16 and 32x32.

Allowed timing parameters:

- `agu_latency`;
- SRAM `read_latency`;
- `interp_latency`;
- `affinity_latency`;
- explicit queue/controller constants represented in the model.

Hold-out real-trace gate:

```text
median cycle error <= 3%
max cycle error    <= 5%
```

Architectural lane/bank counts remain fixed inputs.

---

## L4 — ACTSim and board subgraphs

Validate isolated blocks before E2E:

1. `dec2`;
2. depth head;
3. guidance head;
4. confidence head;
5. SPN once implemented on accelerator;
6. decoder + heads;
7. decoder + heads + SPN.

Normalize CModel and external results to the JSON schema used by `validation/compare_external.py`.

Initial gate:

```text
accelerator-core latency error <= 10%
traffic error                  <= 5% (exact where countable)
```

After microarchitecture calibration, target <=5% subgraph latency error.

Board accounting must remain decomposed into accelerator core, modeled DMA, package/model load, launch/runtime, CPU glue, Q/DQ and CPU fallback. Never tune MAC/SPN cycles to match package loading or CPU glue.

---

## L5 — Full-model hold-out

Only run after L0-L4 pass independently. Validate CompletionFormer plus at least one CSPN and one NLSPN configuration; add DySPN once its propagation lowering is implemented.

Report per-layer cycles, Tensor/SPN utilization, DRAM/SRAM traffic, bank conflicts and accelerator-core E2E latency. Runtime/software overhead is reported separately.

Targets:

```text
first full-model acceptance <= 10% accelerator-core error
mature calibrated target    <= 5% median subgraph/full-model error
```

## 3. Parameter fitting order

Use this order and never revisit an earlier subsystem merely to improve E2E fit:

```text
1 functional semantics
2 tensor compute pipeline
3 Conv traffic/reuse
4 SRAM banking
5 DMA setup/effective BW
6 Ramulator external memory timing
7 SPN exact address/bank behavior
8 SPN pipeline timing
9 ACTSim subgraph integration
10 board accelerator-core integration
```

Timeloop may then be used to cross-check/expand mapping search, BookSim2 for multi-engine/multi-core NoC contention, and Accelergy/HWComponents-CACTI for action-count energy/area. Their source revisions are already pinned in `third_party/` so these later experiments do not change the dependency baseline.
