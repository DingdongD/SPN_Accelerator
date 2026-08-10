#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from spn_accel_cmodel.config import NPUConfig
from spn_accel_cmodel.resources import BankedSRAM
from spn_accel_cmodel.spn_engine import OffsetPattern, SPNTraceGenerator
from spn_accel_cmodel.trace import ArrayOffsetProvider
from spn_accel_cmodel.workload import Prop2DOp


def main() -> int:
    ap = argparse.ArgumentParser(description="Export exact SPN gather address/bank vectors for RTL cross-check")
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--offset-pattern", choices=["zero", "random", "checker"], default="zero")
    ap.add_argument("--trace-npz")
    ap.add_argument("--offset-key", default="offset")
    ap.add_argument("--out", default="validation/out/spn_rtl_vectors.csv")
    args = ap.parse_args()

    cfg = NPUConfig.load(args.config)
    source = None
    h, w = args.height, args.width
    if args.trace_npz:
        source = ArrayOffsetProvider.from_npz(args.trace_npz, args.offset_key, cfg.spn.neighbors)
        h, w = source.height, source.width
    else:
        source = OffsetPattern(args.offset_pattern)
    op = Prop2DOp("rtl_vector", h, w, neighbors=cfg.spn.neighbors, steps=1)
    trace = SPNTraceGenerator(op, source)
    sram = BankedSRAM(cfg.spn.state_sram)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["packet", "issue_group", "lane", "address", "bank"])
        packet = 0
        for linear0 in range(0, op.pixels, cfg.spn.packet_pixels):
            count = min(cfg.spn.packet_pixels, op.pixels - linear0)
            y0, x0 = divmod(linear0, op.width)
            addresses, _ = trace.packet_addresses(y0, x0, count)
            for group_idx, base in enumerate(range(0, len(addresses), cfg.spn.gather_lanes)):
                group = addresses[base : base + cfg.spn.gather_lanes]
                for lane, address in enumerate(group):
                    writer.writerow([packet, group_idx, lane, address, sram.bank(address)])
            packet += 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
