# Validation plan

The simulator is split into replaceable timing backends so each layer can be calibrated independently instead of fitting end-to-end latency with one global scale factor.

## Level 0 — functional semantics

Use the CompletionFormer/CSPN/NLSPN PyTorch reference to validate quantization, interpolation, affinity normalization, propagation order, saturation, and rounding. The architectural CModel is performance-only; a bit-exact functional reference should remain separate.

## Level 1 — tensor-engine microkernels

Sweep GEMM/Conv shapes and compare the internal `AnalyticalSystolicBackend`, SCALE-Sim, and selected Gemmini/NVDLA RTL or SystemC measurements. Record compute cycles, MAC utilization, ABUF/WBUF/PBUF traffic, and external bytes. Do not calibrate tensor cycles against board end-to-end latency because board software, package switching, Q/DQ, and host glue are separate terms.

## Level 2 — memory subsystem

Use sequential, strided, uniform-bank, and hot-bank traces to validate bank arbitration and effective bandwidth. Replace analytical DMA timing with Ramulator2 when DRAM timing matters. Introduce NoC contention only when multiple engines or cores share the fabric.

## Level 3 — SPN microarchitecture

Validate `AGU -> banked gather -> bilinear interpolation -> affinity/reduction -> ping-pong state` independently. A small Verilator RTL model at 8x8/16x16/32x32 should compare exact generated addresses, scalar read count, bank conflicts, pipeline cycles, and output values. Use synthetic offset fields and real CompletionFormer/NLSPN traces.

## Level 4 — subgraphs

Validate `dec2`, full-resolution heads, and NLSPN independently against ACTSim and board traces. Keep launch/package/host overhead outside accelerator-core timing.

## Level 5 — full model

Only after Levels 1–4 are stable should the simulator predict full CompletionFormer/CSPN/NLSPN/DySPN latency and traffic. Report error per module as well as end-to-end error.

Suggested engineering targets, not external-tool claims: <=5% tensor microkernel cycle error after calibration, exact fixed-mapping byte counts, exact SPN address/read/bank-conflict counts against RTL trace, <=8% subgraph latency error, and <=10% initial full-accelerator error before tightening toward 5%.
