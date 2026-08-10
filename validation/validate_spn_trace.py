#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spn_accel_cmodel.config import NPUConfig, SPNEngineConfig, SRAMConfig
from spn_accel_cmodel.functional import error_metrics, propagate
from spn_accel_cmodel.spn_engine import SPNEngine
from spn_accel_cmodel.trace import ArrayOffsetProvider
from spn_accel_cmodel.workload import Prop2DOp


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate SPN timing and optional functional output with a recorded CompletionFormer trace")
    ap.add_argument("trace_npz")
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--offset-key", default="offset")
    ap.add_argument("--affinity-key", default="aff")
    ap.add_argument("--state-key", default="pred_init")
    ap.add_argument("--golden-key", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--output", default="validation/out/spn_real_trace.json")
    ap.add_argument("--max-abs-limit", type=float, default=1e-5)
    ap.add_argument("--rmse-limit", type=float, default=1e-6)
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy is required for recorded trace validation")
        return 2

    cfg = NPUConfig.load(args.config)
    offsets = ArrayOffsetProvider.from_npz(args.trace_npz, args.offset_key, neighbors=cfg.spn.neighbors)
    op = Prop2DOp(
        "nlspn_real_trace",
        offsets.height,
        offsets.width,
        neighbors=offsets.neighbors,
        steps=args.steps,
    )

    # Make recorded traces usable even if the chosen frame is not 128x128.
    meta_cap = max(cfg.spn.metadata_sram.capacity_bytes, op.metadata_bytes)
    state_cap = max(cfg.spn.state_sram.capacity_bytes, op.pingpong_bytes)
    spn_cfg = SPNEngineConfig(
        agu_lanes=cfg.spn.agu_lanes,
        gather_lanes=cfg.spn.gather_lanes,
        interp_lanes=cfg.spn.interp_lanes,
        affinity_lanes=cfg.spn.affinity_lanes,
        agu_latency=cfg.spn.agu_latency,
        interp_latency=cfg.spn.interp_latency,
        affinity_latency=cfg.spn.affinity_latency,
        packet_pixels=cfg.spn.packet_pixels,
        neighbors=cfg.spn.neighbors,
        metadata_sram=SRAMConfig(
            meta_cap,
            banks=cfg.spn.metadata_sram.banks,
            ports_per_bank=cfg.spn.metadata_sram.ports_per_bank,
            word_bytes=cfg.spn.metadata_sram.word_bytes,
            read_latency=cfg.spn.metadata_sram.read_latency,
            write_latency=cfg.spn.metadata_sram.write_latency,
        ),
        state_sram=SRAMConfig(
            state_cap,
            banks=cfg.spn.state_sram.banks,
            ports_per_bank=cfg.spn.state_sram.ports_per_bank,
            word_bytes=cfg.spn.state_sram.word_bytes,
            read_latency=cfg.spn.state_sram.read_latency,
            write_latency=cfg.spn.state_sram.write_latency,
        ),
    )
    timing = SPNEngine(spn_cfg).run(op, offset_provider=offsets)
    result = {
        "source": str(args.trace_npz),
        "offset_key": args.offset_key,
        "shape": [offsets.height, offsets.width],
        "neighbors": offsets.neighbors,
        "steps": args.steps,
        "timing": {
            "cycles": timing.cycles,
            "gather_reads": timing.gather_reads,
            "bank_conflict_stall_cycles": timing.bank_conflict_stall_cycles,
            "bank_conflict_rate": timing.bank_conflict_rate,
            "agu_busy_cycles": timing.agu_busy_cycles,
            "gather_busy_cycles": timing.gather_busy_cycles,
            "interp_busy_cycles": timing.interp_busy_cycles,
            "affinity_busy_cycles": timing.affinity_busy_cycles,
        },
    }

    functional_pass = True
    if args.golden_key:
        with np.load(args.trace_npz, allow_pickle=False) as data:
            for key in (args.state_key, args.affinity_key, args.golden_key):
                if key not in data.files:
                    raise KeyError(f"{key!r} not found; keys={data.files}")
            state = data[args.state_key]
            affinity = data[args.affinity_key]
            golden = data[args.golden_key]
        actual = propagate(state, offsets, affinity, args.steps, neighbors=offsets.neighbors)
        golden2 = golden
        while golden2.ndim > 2 and golden2.shape[0] == 1:
            golden2 = golden2[0]
        errors = error_metrics(actual, golden2)
        functional_pass = errors["max_abs"] <= args.max_abs_limit and errors["rmse"] <= args.rmse_limit
        result["functional"] = {
            **errors,
            "max_abs_limit": args.max_abs_limit,
            "rmse_limit": args.rmse_limit,
            "passed": functional_pass,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if functional_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
