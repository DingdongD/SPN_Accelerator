#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "third_party" / "manifest.json"


def git_head(path: Path) -> str:
    p = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify pinned third-party simulators and local installs")
    ap.add_argument("--pins-only", action="store_true")
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    print("Third-party pin check")
    print("=" * 80)
    for item in data["modules"]:
        path = ROOT / item["path"]
        actual = git_head(path)
        expected = item["commit"]
        ok = actual == expected
        print(f"{item['name']:22s} {'OK' if ok else 'FAIL':4s} {actual or '<not initialized>'}")
        if not ok:
            failures.append(f"{item['name']}: expected {expected}, got {actual or 'not initialized'}")
        nested = item.get("nested")
        if nested and path.exists():
            nested_path = path / nested["path"]
            nested_actual = git_head(nested_path)
            nested_ok = nested_actual == nested["commit"]
            print(f"  nested CACTI          {'OK' if nested_ok else 'FAIL':4s} {nested_actual or '<not initialized>'}")
            if not nested_ok:
                failures.append(
                    f"{item['name']} nested CACTI: expected {nested['commit']}, got {nested_actual or 'not initialized'}"
                )

    if not args.pins_only:
        print("\nRuntime/tool check")
        print("=" * 80)
        checks = [
            ("SCALE-Sim import", module_available("scalesim")),
            ("Ramulator2 import", module_available("ramulator")),
            ("Accelergy import", module_available("accelergy")),
            ("HWComponents import", module_available("hwcomponents")),
            ("HWComponents-CACTI import", module_available("hwcomponents_cacti")),
            ("BookSim2 binary", (ROOT / "third_party" / "booksim2" / "src" / "booksim").exists()),
            ("cmake", shutil.which("cmake") is not None),
            ("make", shutil.which("make") is not None),
        ]
        for name, ok in checks:
            print(f"{name:28s} {'OK' if ok else 'MISSING'}")
        # Timeloop is intentionally source-only in the default bootstrap.
        required = checks[:6]
        for name, ok in required:
            if not ok:
                failures.append(f"runtime: {name} missing")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
