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
| Ramulator2 | DDR/HBM cycle-level timing |
| BookSim2 | NoC contention |
| Accelergy | action-count energy framework |
| HWComponents | component-model stack |
| HWComponents-CACTI | CACTI-backed SRAM/cache/DRAM models; recursively pulls CACTI |
| Timeloop | optional mapper/tiling cross-validation |

Exact upstream commits are recorded in `third_party/manifest.json`. Reported experiments must not use `git submodule update --remote`.

## One-shot local setup

Clone the development branch with its pinned dependencies:

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

Equivalent Makefile path:

```bash
make setup
make check-third-party
```

If the repository was cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
bash third_party/bootstrap.sh
```

`bootstrap.sh` installs SCALE-Sim, Accelergy and HWComponents editable into the active venv, builds/installs Ramulator2, and builds BookSim2. Timeloop source is initialized but not built automatically because its upstream build additionally requires system ISL/Barvinok dependencies.

See `third_party/README.md` for `--init-only`, `--no-build`, pin verification and update policy.

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

Once real SPN/RTL/ACTSim/board goldens are supplied:

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

Do not calibrate Tensor/SPN cycles to board wall time. Keep these categories separate:

```text
accelerator core
DMA / modeled memory
package/model load
launch/runtime
CPU glue
quant/dequant
CPU fallback
```

Only components represented in the architectural CModel may be used to fit CModel parameters.

## Fidelity roadmap

```text
functional semantics
    -> SCALE-Sim tensor calibration
    -> SRAM/DMA/Ramulator calibration
    -> SPN exact RTL address contract
    -> SPN cycle calibration
    -> ACTSim subgraphs
    -> board subgraphs
    -> full CompletionFormer/CSPN/NLSPN/DySPN validation
    -> architecture DSE
```

Ramulator2, BookSim2, Accelergy/HWComponents-CACTI and Timeloop are already pinned in `third_party/` so later fidelity stages do not require changing the source-dependency baseline.
