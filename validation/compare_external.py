#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spn_accel_cmodel.validation import MetricTolerance, compare_metrics, load_json_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CModel JSON metrics against RTL/ACTSim/board golden JSON")
    ap.add_argument("model_json")
    ap.add_argument("golden_json")
    ap.add_argument("--name", default="external_validation")
    ap.add_argument("--rel", type=float, default=0.10)
    ap.add_argument("--abs", dest="abs_tol", type=float, default=0.0)
    ap.add_argument("--out", default="validation/out/external_validation")
    args = ap.parse_args()

    model = load_json_metrics(args.model_json)
    golden = load_json_metrics(args.golden_json)
    report = compare_metrics(
        args.name,
        model,
        golden,
        default_tolerance=MetricTolerance(relative=args.rel, absolute=args.abs_tol),
        metadata={"model": args.model_json, "golden": args.golden_json},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(out.with_suffix(".json"))
    report.write_markdown(out.with_suffix(".md"))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
