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
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify pinned third-party simulators and local installs")
    ap.add_argument("--pins-only", action="store_true")
    ap.add_argument("--core", action="store_true", help="require only the timing stack at runtime")
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    print("Third-party pin check")
    print("=" * 88)
    for item in data["modules"]:
        path = ROOT / item["path"]
        actual = git_head(path)
        expected = item["commit"]
        ok = actual == expected
        print(f"{item['name']:28s} {'OK' if ok else 'FAIL':4s} {actual or '<not initialized>'}")
        if not ok:
            failures.append(f"{item['name']}: expected {expected}, got {actual or 'not initialized'}")

        nested_items = item.get("nested", [])
        if isinstance(nested_items, dict):
            nested_items = [nested_items]
        for nested in nested_items:
            nested_path = path / nested["path"]
            nested_actual = git_head(nested_path)
            nested_ok = nested_actual == nested["commit"]
            label = f"  {nested.get('name', 'nested')}"
            print(f"{label:28s} {'OK' if nested_ok else 'FAIL':4s} {nested_actual or '<not initialized>'}")
            if not nested_ok:
                failures.append(
                    f"{item['name']} / {nested.get('name', nested['path'])}: "
                    f"expected {nested['commit']}, got {nested_actual or 'not initialized'}"
                )

    if not args.pins_only:
        print("\nRuntime/tool check")
        print("=" * 88)
        core_checks = [
            ("SCALE-Sim import", module_available("scalesim")),
            ("Ramulator2 import", module_available("ramulator")),
            ("BookSim2 binary", (ROOT / "third_party" / "booksim2" / "src" / "booksim").exists()),
        ]
        energy_checks = [
            ("Accelergy import", module_available("accelergy")),
            ("HWComponents import", module_available("hwcomponents")),
            ("HWComponents-CACTI import", module_available("hwcomponents_cacti")),
        ]
        tool_checks = [
            ("cmake", shutil.which("cmake") is not None),
            ("make", shutil.which("make") is not None),
            ("flex", shutil.which("flex") is not None),
            ("bison", shutil.which("bison") is not None),
        ]
        checks = core_checks + ([] if args.core else energy_checks) + tool_checks
        for name, ok in checks:
            print(f"{name:32s} {'OK' if ok else 'MISSING'}")

        required = core_checks + ([] if args.core else energy_checks)
        for name, ok in required:
            if not ok:
                failures.append(f"runtime: {name} missing")

        if not args.core and sys.version_info < (3, 12):
            failures.append(
                f"full energy stack requires Python >=3.12; current={sys.version.split()[0]}"
            )

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
