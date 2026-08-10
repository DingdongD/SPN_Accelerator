#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spn_accel_cmodel.backends.scalesim import ScaleSimConvCase, ScaleSimRunner
from spn_accel_cmodel.config import NPUConfig
from spn_accel_cmodel.tensor_engine import ConvMapping, TensorEngine
from spn_accel_cmodel.validation import MetricTolerance, compare_metrics
from spn_accel_cmodel.workload import Conv2DOp


def cases() -> list[ScaleSimConvCase]:
    # Each case corresponds to one complete CModel tile.  IFMAP is the actual
    # receptive input tile, so traffic is comparable without GEMM/im2col expansion.
    return [
        ScaleSimConvCase("tile16_c16_o16", 18, 18, 3, 3, 16, 16),
        ScaleSimConvCase("tile16_c32_o32", 18, 18, 3, 3, 32, 32),
        ScaleSimConvCase("tile8_c64_o32", 10, 10, 3, 3, 64, 32),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-check Conv tile cycles and DRAM element counts against SCALE-Sim")
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--output-dir", default="validation/out/conv_scalesim")
    ap.add_argument("--cycle-rel-limit", type=float, default=0.05)
    ap.add_argument("--traffic-rel-limit", type=float, default=0.0)
    args = ap.parse_args()

    cfg = NPUConfig.load(args.config)
    runner = ScaleSimRunner(cfg.tensor, dataflow="os")
    if not runner.available():
        print("SCALE-Sim is not installed; install it to run Conv cycle/traffic cross-validation.")
        return 2
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    golden = runner.run_conv(cases(), output_root=out / "scalesim_inputs")

    rows = []
    overall = True
    for item in golden:
        c = item.case
        assert isinstance(c, ScaleSimConvCase)
        op = Conv2DOp(c.name, c.out_h, c.out_w, c.channels, c.filters, c.filter_h, c.stride, 0)
        mapping = ConvMapping(c.out_h, c.out_w, c.channels, c.filters)
        model = TensorEngine(cfg.tensor, cfg.dma).run_conv(op, mapping)
        if item.access is None:
            raise RuntimeError("SCALE-Sim DETAILED_ACCESS_REPORT.csv is required")
        model_metrics = {
            "compute_cycles": model.compute_cycles,
            "ifmap_elements": model.act_bytes / cfg.tensor.act_bytes,
            "filter_elements": model.weight_bytes / cfg.tensor.weight_bytes,
            "ofmap_elements": model.output_bytes / cfg.tensor.act_bytes,
        }
        golden_metrics = {
            "compute_cycles": item.compute.total_cycles,
            "ifmap_elements": item.access.dram_ifmap_reads,
            "filter_elements": item.access.dram_filter_reads,
            "ofmap_elements": item.access.dram_ofmap_writes,
        }
        report = compare_metrics(
            c.name,
            model_metrics,
            golden_metrics,
            tolerances={
                "compute_cycles": MetricTolerance(relative=args.cycle_rel_limit, absolute=1),
                "ifmap_elements": MetricTolerance(relative=args.traffic_rel_limit, absolute=0),
                "filter_elements": MetricTolerance(relative=args.traffic_rel_limit, absolute=0),
                "ofmap_elements": MetricTolerance(relative=args.traffic_rel_limit, absolute=0),
            },
        )
        overall &= report.passed
        rows.append(report.to_dict())

    summary = {"name": "conv_tile_vs_scalesim", "passed": overall, "cases": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
