#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (int(r["packet"]), int(r["issue_group"]), int(r["lane"]), int(r["address"]), int(r["bank"]))
            for r in csv.DictReader(f)
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact compare of CModel SPN gather vectors and RTL trace")
    ap.add_argument("cmodel_csv")
    ap.add_argument("rtl_csv")
    ap.add_argument("--out", default="validation/out/spn_rtl_compare.json")
    args = ap.parse_args()
    cmodel = load(args.cmodel_csv)
    rtl = load(args.rtl_csv)
    mismatches = []
    for i in range(max(len(cmodel), len(rtl))):
        a = cmodel[i] if i < len(cmodel) else None
        b = rtl[i] if i < len(rtl) else None
        if a != b:
            mismatches.append({"index": i, "cmodel": a, "rtl": b})
            if len(mismatches) >= 32:
                break
    result = {
        "passed": len(cmodel) == len(rtl) and not mismatches,
        "cmodel_rows": len(cmodel),
        "rtl_rows": len(rtl),
        "first_mismatches": mismatches,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
