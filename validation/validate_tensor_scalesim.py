#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spn_accel_cmodel.backends.scalesim import (
    ScaleSimCase,
    ScaleSimRunner,
    parse_compute_report,
    parse_detailed_access_report,
)
from spn_accel_cmodel.config import NPUConfig
from spn_accel_cmodel.tensor_engine import AnalyticalSystolicBackend
from spn_accel_cmodel.validation import MetricTolerance, ValidationReport, compare_metrics


def representative_cases() -> list[ScaleSimCase]:
    # Shapes cover the tile regime used by CompletionFormer dec2/heads.
    return [
        ScaleSimCase("m64_n16_k144", 64, 16, 144),
        ScaleSimCase("m128_n32_k288", 128, 32, 288),
        ScaleSimCase("m256_n32_k288", 256, 32, 288),
        ScaleSimCase("m256_n64_k576", 256, 64, 576),
    ]


def load_existing(report_dir: Path, cases: list[ScaleSimCase]):
    compute = parse_compute_report(report_dir / "COMPUTE_REPORT.csv")
    access_path = report_dir / "DETAILED_ACCESS_REPORT.csv"
    access = parse_detailed_access_report(access_path) if access_path.exists() else []
    by_access = {x.layer_id: x for x in access}
    from spn_accel_cmodel.backends.scalesim import ScaleSimCaseResult

    if len(compute) < len(cases):
        raise RuntimeError(f"report has {len(compute)} layers but {len(cases)} cases are expected")
    return [ScaleSimCaseResult(cases[i], compute[i], by_access.get(i)) for i in range(len(cases))]


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-check tensor microkernels against SCALE-Sim v3")
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--output-dir", default="validation/out/tensor_scalesim")
    ap.add_argument("--report-dir", help="Parse an existing SCALE-Sim run directory instead of invoking SCALE-Sim")
    ap.add_argument("--cycle-rel-limit", type=float, default=0.05)
    args = ap.parse_args()

    cfg = NPUConfig.load(args.config)
    cases = representative_cases()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.report_dir:
        results = load_existing(Path(args.report_dir), cases)
    else:
        runner = ScaleSimRunner(cfg.tensor, dataflow="os")
        if not runner.available():
            print("SCALE-Sim is not installed. Install it or pass --report-dir with an existing run.")
            return 2
        results = runner.run(cases, output_root=out / "scalesim_inputs")

    backend = AnalyticalSystolicBackend(cfg.tensor)
    all_reports: list[ValidationReport] = []
    rows = []
    for result in results:
        case = result.case
        model_cycles = backend.conv_compute_cycles(case.m, case.n, case.k)
        golden_cycles = result.compute.total_cycles
        report = compare_metrics(
            case.name,
            {"cycles": model_cycles},
            {"cycles": golden_cycles},
            tolerances={"cycles": MetricTolerance(relative=args.cycle_rel_limit, absolute=1)},
            metadata={
                "m": case.m,
                "n": case.n,
                "k": case.k,
                "scalesim_stall_cycles": result.compute.stall_cycles,
                "scalesim_overall_util_pct": result.compute.overall_util_pct,
                "scalesim_mapping_efficiency_pct": result.compute.mapping_efficiency_pct,
                "scalesim_compute_util_pct": result.compute.compute_util_pct,
            },
        )
        all_reports.append(report)
        cmp = report.comparisons[0]
        rows.append({
            "case": case.name,
            "m": case.m,
            "n": case.n,
            "k": case.k,
            "model_cycles": model_cycles,
            "golden_cycles": golden_cycles,
            "rel_error": cmp.rel_error,
            "pass": cmp.passed,
        })

    summary = {
        "name": "tensor_vs_scalesim",
        "passed": all(r.passed for r in all_reports),
        "cycle_rel_limit": args.cycle_rel_limit,
        "cases": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Tensor Engine vs SCALE-Sim",
        "",
        f"Overall: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Case | M | N | K | CModel cycles | SCALE-Sim cycles | Rel. error | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['case']} | {r['m']} | {r['n']} | {r['k']} | {r['model_cycles']} | "
            f"{r['golden_cycles']} | {r['rel_error']:.2%} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
