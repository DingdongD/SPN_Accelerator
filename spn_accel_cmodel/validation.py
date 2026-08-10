from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class MetricTolerance:
    relative: float = 0.05
    absolute: float = 0.0


@dataclass(frozen=True)
class MetricComparison:
    name: str
    model: float
    golden: float
    abs_error: float
    rel_error: float
    tolerance_relative: float
    tolerance_absolute: float
    passed: bool


@dataclass
class ValidationReport:
    name: str
    comparisons: list[MetricComparison] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(x.passed for x in self.comparisons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metadata": self.metadata,
            "comparisons": [asdict(x) for x in self.comparisons],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_markdown(self, path: str | Path) -> None:
        lines = [
            f"# Validation: {self.name}",
            "",
            f"Status: **{'PASS' if self.passed else 'FAIL'}**",
            "",
            "| Metric | Model | Golden | Abs err | Rel err | Limit | Status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for x in self.comparisons:
            limit = f"rel<={x.tolerance_relative:.4g}, abs<={x.tolerance_absolute:.4g}"
            lines.append(
                f"| `{x.name}` | {x.model:.6g} | {x.golden:.6g} | {x.abs_error:.6g} | "
                f"{x.rel_error:.3%} | {limit} | {'PASS' if x.passed else 'FAIL'} |"
            )
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_error(model: float, golden: float) -> float:
    if golden == 0:
        return 0.0 if model == 0 else math.inf
    return abs(model - golden) / abs(golden)


def compare_metrics(
    name: str,
    model: Mapping[str, float | int],
    golden: Mapping[str, float | int],
    tolerances: Mapping[str, MetricTolerance] | None = None,
    default_tolerance: MetricTolerance = MetricTolerance(),
    metadata: Mapping[str, Any] | None = None,
) -> ValidationReport:
    tolerances = tolerances or {}
    comparisons: list[MetricComparison] = []
    common = sorted(set(model) & set(golden))
    for metric in common:
        m = float(model[metric])
        g = float(golden[metric])
        tol = tolerances.get(metric, default_tolerance)
        abs_err = abs(m - g)
        rel_err = relative_error(m, g)
        # Accept if either an absolute near-zero guard or the relative bound passes.
        passed = abs_err <= tol.absolute or rel_err <= tol.relative
        comparisons.append(
            MetricComparison(
                name=metric,
                model=m,
                golden=g,
                abs_error=abs_err,
                rel_error=rel_err,
                tolerance_relative=tol.relative,
                tolerance_absolute=tol.absolute,
                passed=passed,
            )
        )
    if not comparisons:
        raise ValueError("model and golden metric sets have no common keys")
    return ValidationReport(name=name, comparisons=comparisons, metadata=dict(metadata or {}))


def flatten_metrics(data: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def load_json_metrics(path: str | Path) -> Dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metrics" in data and isinstance(data["metrics"], Mapping):
        data = data["metrics"]
    return flatten_metrics(data)
