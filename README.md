# SPN Accelerator CModel

A dependency-light architectural simulator for exploring **generic NPU tensor execution plus spatial propagation network (SPN) acceleration**. The first target is the CompletionFormer/NLSPN hot path, with CSPN and DySPN intended to share the same propagation abstraction.

## Why this simulator exists

Dense Conv/GEMM and SPN propagation stress different hardware resources:

- Dense operators are regular and map naturally to a systolic/tensor array with explicit DMA and scratchpad tiling.
- NLSPN-style propagation is metadata-driven: offsets generate irregular addresses, each deformable neighbor performs four-point bilinear sampling, and affinity-weighted reduction repeats over an on-chip depth state.

A single `MACs / peak` estimator cannot represent DMA overlap, array wavefronts, SRAM capacity, SPN bank conflicts, or iterative state residency. This CModel therefore uses **tile/packet-level event timing** and exposes replaceable backends for external validation tools.

## Modeled architecture

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

`*` VPU and cycle-accurate DRAM/NoC backends are extension points in the MVP. Tensor compute and SPN propagation are implemented now.

## Current timing models

### Tensor engine

`AnalyticalSystolicBackend` lowers a Conv tile to GEMM `(M=OH*OW, K=Kh*Kw*Cin, N=Cout)` and models array-sized M/N waves. Each wave includes K steady-state work plus systolic fill/drain. The system layer additionally models:

- activation/weight/output DMA bytes and setup cost;
- two-channel DMA contention;
- double-buffer reuse constraints;
- ABUF/WBUF/PBUF capacity checks;
- tile-level compute/DMA overlap;
- MAC utilization and traffic statistics.

The backend interface is deliberately replaceable by SCALE-Sim or a calibrated lookup table.

### SPN engine

`SPNEngine` packetizes pixels and pipelines:

```text
AGU -> banked state-SRAM gather -> bilinear interpolation -> affinity/reduction
```

For a deformable 3x3 NLSPN step, each pixel performs one center read plus `4 * 8` bilinear source reads. Offset metadata is reused across propagation iterations; depth state stays in ping-pong SRAM. Gather timing consumes an actual scalar address trace and keeps per-bank queues across issue cycles, so hot-bank conflicts become explicit stall cycles instead of an average bandwidth penalty.

The baseline 128x128 INT16 metadata/state footprint is exactly:

- offset: 512 KiB;
- eight affinities: 256 KiB;
- depth ping-pong: 64 KiB.

The default config therefore uses 768 KiB metadata SRAM plus 64 KiB state SRAM.

## Quick start

No third-party Python dependencies are required.

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/run_dec2_nlspn.py
PYTHONPATH=. python -m spn_accel_cmodel.cli \
  --config configs/baseline.json \
  --prop-steps 12 \
  --offset-pattern zero
```

SPN bank/gather sweep:

```bash
PYTHONPATH=. python examples/sweep_spn.py
```

## Validation quick start

The repository now contains an executable staged validation chain instead of placeholder hooks:

```text
PyTorch functional semantics
        -> real CompletionFormer offset/affinity trace
        -> CModel address/bank trace
        -> SPN RTL exact-vector comparison

Tensor CModel
        -> SCALE-Sim GEMM compute validation
        -> SCALE-Sim Conv cycle + traffic validation

Subgraph CModel
        -> normalized ACTSim/board JSON comparison
```

Install only the optional local validation dependency:

```bash
python -m pip install -e '.[validation]'
python -m unittest discover -s tests -v
```

Run the staged local calibration suite:

```bash
PYTHONPATH=. python validation/run_calibration_suite.py
```

The runner records the git revision/environment, executes all locally available gates, and marks external-golden stages as `SKIP` when SCALE-Sim/RTL/ACTSim/board data is absent. Re-run with `--strict` once all external goldens are present.

Tensor-array cross-checks (requires SCALE-Sim installed in that environment):

```bash
PYTHONPATH=. python validation/validate_tensor_scalesim.py
PYTHONPATH=. python validation/validate_conv_scalesim.py
```

Run a real CompletionFormer/NLSPN offset trace:

```bash
PYTHONPATH=. python validation/validate_spn_trace.py trace.npz \
  --offset-key offset --steps 12
```

Generate/compare the exact SPN RTL gather contract:

```bash
PYTHONPATH=. python validation/export_spn_rtl_vectors.py \
  --height 16 --width 16 --out validation/out/spn_vectors.csv
PYTHONPATH=. python validation/compare_spn_rtl_trace.py \
  validation/out/spn_vectors.csv rtl_vectors.csv
```

See [`CALIBRATION_PLAN.md`](CALIBRATION_PLAN.md) for the calibration/hold-out matrix, parameter-fitting order, and acceptance thresholds. See [`validation/README.md`](validation/README.md) for semantic assumptions, report schemas, and ACTSim/board calibration boundaries. The presence of these adapters does **not** mean SCALE-Sim/RTL/ACTSim accuracy has already been measured; an external golden must be supplied and the generated report must pass.

## CompletionFormer representative workload

`completionformer_dec2_nlspn_workload()` currently models:

1. `dec2` 3x3 Conv `160 -> 32` at 128x128;
2. two residual-block 3x3 Conv `32 -> 32` at 128x128;
3. deformable 8-neighbor NLSPN propagation for a configurable number of steps.

Resize is intentionally separated from the first dense model so future experiments can compare host/VPU resize against a fused Resize+Conv datapath without silently changing the Conv baseline.

## External backend roadmap

The repository does **not** vendor other simulators. It provides clean boundaries for:

- **SCALE-Sim**: tensor-tile cycle/traffic golden backend (`backends/scalesim.py`);
- **Ramulator2**: timestamped DRAM request/completion backend (`backends/ramulator2.py`);
- **BookSim2**: future multi-engine/multi-core NoC contention;
- **CACTI/Accelergy**: future SRAM action energy/area and action-count energy model;
- **Gemmini/NVDLA/Verilator**: selected RTL/SystemC validation points.

See [`CALIBRATION_PLAN.md`](CALIBRATION_PLAN.md) and [`validation/README.md`](validation/README.md) for the staged calibration plan.

## Important interpretation

The current cycle numbers are **architecture-model predictions**, not measured CompletionFormer board latency. They should be used for relative DSE only until tensor, memory, and SPN microbenchmarks are calibrated against external simulators/RTL/ACTSim. Board package loading, CPU glue, quant/dequant, and software launch overhead must remain separate from accelerator-core timing.
