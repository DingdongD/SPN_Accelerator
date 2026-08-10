#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass
class StepResult:
    level: str
    name: str
    status: str
    command: list[str]
    returncode: int | None
    log: str
    note: str = ""


def run_step(
    level: str,
    name: str,
    command: Sequence[str],
    out_dir: Path,
    *,
    optional: bool = False,
    skip_returncodes: tuple[int, ...] = (),
) -> StepResult:
    log_path = out_dir / f"{level}_{name}.log"
    cmd = [str(x) for x in command]
    try:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
    except FileNotFoundError as exc:
        status = "SKIP" if optional else "FAIL"
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return StepResult(level, name, status, cmd, None, str(log_path), str(exc))

    if proc.returncode == 0:
        status = "PASS"
        note = ""
    elif proc.returncode in skip_returncodes or optional and proc.returncode == 2:
        status = "SKIP"
        note = f"optional dependency/golden unavailable (return code {proc.returncode})"
    else:
        status = "FAIL"
        note = f"return code {proc.returncode}"
    return StepResult(level, name, status, cmd, proc.returncode, str(log_path), note)


def git_output(*args: str) -> str:
    if shutil.which("git") is None:
        return ""
    proc = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the staged SPN Accelerator CModel calibration/validation suite."
    )
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--out-dir", default="validation/out/calibration")
    ap.add_argument("--trace-npz")
    ap.add_argument("--offset-key", default="offset")
    ap.add_argument("--affinity-key", default="aff")
    ap.add_argument("--state-key", default="pred_init")
    ap.add_argument("--golden-key", default="")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--rtl-csv", help="RTL-produced gather-vector CSV using the exact CModel schema")
    ap.add_argument("--actsim-model-json")
    ap.add_argument("--actsim-golden-json")
    ap.add_argument("--board-model-json")
    ap.add_argument("--board-golden-json")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat skipped external-golden stages as failures.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    steps: list[StepResult] = []

    steps.append(
        run_step(
            "L0",
            "unit_tests",
            [py, "-m", "unittest", "discover", "-s", "tests", "-v"],
            out,
        )
    )
    steps.append(
        run_step(
            "L0",
            "compileall",
            [py, "-m", "compileall", "-q", "spn_accel_cmodel", "validation", "tests", "examples"],
            out,
        )
    )

    # Tensor external golden. Existing validators return 2 if SCALE-Sim is absent.
    steps.append(
        run_step(
            "L1a",
            "tensor_scalesim",
            [py, "validation/validate_tensor_scalesim.py", "--config", args.config,
             "--output-dir", str(out / "tensor_scalesim")],
            out,
            optional=True,
            skip_returncodes=(2,),
        )
    )
    steps.append(
        run_step(
            "L1b",
            "conv_scalesim",
            [py, "validation/validate_conv_scalesim.py", "--config", args.config,
             "--output-dir", str(out / "conv_scalesim")],
            out,
            optional=True,
            skip_returncodes=(2,),
        )
    )

    # Always export a deterministic CModel RTL contract. If a real trace is
    # supplied, it is used; otherwise an 8x8 zero-offset microcase is exported.
    cmodel_rtl_csv = out / "spn_cmodel_vectors.csv"
    export_cmd = [
        py,
        "validation/export_spn_rtl_vectors.py",
        "--config",
        args.config,
        "--out",
        str(cmodel_rtl_csv),
    ]
    if args.trace_npz:
        export_cmd += ["--trace-npz", args.trace_npz, "--offset-key", args.offset_key]
    else:
        export_cmd += ["--height", "8", "--width", "8", "--offset-pattern", "zero"]
    steps.append(run_step("L3a", "export_spn_vectors", export_cmd, out))

    if args.trace_npz:
        trace_cmd = [
            py,
            "validation/validate_spn_trace.py",
            args.trace_npz,
            "--config",
            args.config,
            "--offset-key",
            args.offset_key,
            "--affinity-key",
            args.affinity_key,
            "--state-key",
            args.state_key,
            "--steps",
            str(args.steps),
            "--output",
            str(out / "spn_real_trace.json"),
        ]
        if args.golden_key:
            trace_cmd += ["--golden-key", args.golden_key]
        steps.append(run_step("L3a", "real_spn_trace", trace_cmd, out))
    else:
        steps.append(
            StepResult(
                "L3a",
                "real_spn_trace",
                "SKIP",
                [],
                None,
                "",
                "no --trace-npz supplied",
            )
        )

    if args.rtl_csv:
        steps.append(
            run_step(
                "L3b",
                "rtl_exact_trace",
                [
                    py,
                    "validation/compare_spn_rtl_trace.py",
                    str(cmodel_rtl_csv),
                    args.rtl_csv,
                    "--out",
                    str(out / "spn_rtl_compare.json"),
                ],
                out,
            )
        )
    else:
        steps.append(
            StepResult("L3b", "rtl_exact_trace", "SKIP", [], None, "", "no --rtl-csv supplied")
        )

    for name, model_json, golden_json in (
        ("actsim", args.actsim_model_json, args.actsim_golden_json),
        ("board", args.board_model_json, args.board_golden_json),
    ):
        if model_json and golden_json:
            steps.append(
                run_step(
                    "L4",
                    name,
                    [
                        py,
                        "validation/compare_external.py",
                        model_json,
                        golden_json,
                        "--name",
                        f"{name}_subgraph",
                        "--rel",
                        "0.10",
                        "--out",
                        str(out / f"{name}_compare"),
                    ],
                    out,
                )
            )
        else:
            steps.append(
                StepResult(
                    "L4",
                    name,
                    "SKIP",
                    [],
                    None,
                    "",
                    f"no complete --{name}-model-json/--{name}-golden-json pair supplied",
                )
            )

    scalesim_importable = subprocess.run(
        [
            py,
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('scalesim') else 1)",
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "config": args.config,
        "cwd": str(root),
        "scalesim_importable": scalesim_importable,
    }

    if args.strict:
        for s in steps:
            if s.status == "SKIP" and (
                s.level in {"L1a", "L1b", "L3b", "L4"} or s.name == "real_spn_trace"
            ):
                s.status = "FAIL"
                s.note = "strict mode: required external golden/dependency is missing"

    required_failed = any(s.status == "FAIL" for s in steps)
    summary = {
        "passed": not required_failed,
        "strict": args.strict,
        "environment": environment,
        "steps": [asdict(s) for s in steps],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Calibration suite summary",
        "",
        f"Overall: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        f"- Git commit: `{environment['git_commit'] or 'unknown'}`",
        f"- Python: `{platform.python_version()}`",
        f"- SCALE-Sim importable: `{environment['scalesim_importable']}`",
        f"- Config: `{args.config}`",
        "",
        "| Level | Step | Status | Note |",
        "|---|---|---|---|",
    ]
    for s in steps:
        md.append(f"| {s.level} | {s.name} | **{s.status}** | {s.note} |")
    md += [
        "",
        "A skipped external stage is not evidence of accuracy. Re-run with `--strict`",
        "after SCALE-Sim/RTL/ACTSim/board goldens are available.",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
