"""Regression checks: turn a comparison into a pass/fail verdict.

For pipelines the question is rarely "how similar are these" but "did this
drift past what we allow". A :class:`Check` is one rule against a report —
a floor or ceiling on a scalar metric, a cap on a status count, or a bound on
how far a metric moved from a stored baseline report — and :func:`run_checks`
evaluates a list of them into a :class:`CheckReport` whose ``ok`` decides the
exit code.

Rules are written as strings so they fit on a command line::

    jaccard_typed_edges>=0.95     metric floor
    ged_normalized_distance<=0.1  metric ceiling
    nodes.A_ONLY<=0               status-count cap (nodes.* / edges.*)
    significance.max<=20          cap on the top significance score
    similarity.shared>=0.9        the shared-subgraph variant of a metric
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .report import SCALAR_METRICS, ComparisonReport

__all__ = ["Check", "CheckReport", "CheckResult", "load_baseline", "parse_check", "run_checks"]

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}
_RULE = re.compile(
    r"^\s*([A-Za-z_][\w.]*)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)

_ALIASES = {"similarity": "ged_similarity", "distance": "ged_normalized_distance"}


@dataclass(frozen=True)
class Check:
    """One rule: ``quantity <op> threshold``."""

    quantity: str
    op: str
    threshold: float

    def __str__(self) -> str:
        return f"{self.quantity} {self.op} {self.threshold:g}"


@dataclass
class CheckResult:
    """Outcome of one :class:`Check` against a report."""

    check: Check
    value: float | None
    ok: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": str(self.check),
            "quantity": self.check.quantity,
            "op": self.check.op,
            "threshold": self.check.threshold,
            "value": self.value,
            "ok": self.ok,
            "note": self.note,
        }


@dataclass
class CheckReport:
    """All results plus the overall verdict."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [r.to_dict() for r in self.results]}

    def summary(self) -> str:
        """One line per check, ``PASS``/``FAIL`` first, suitable for a log."""
        lines = []
        for r in self.results:
            shown = "n/a" if r.value is None else f"{r.value:.6g}"
            lines.append(f"{'PASS' if r.ok else 'FAIL'}  {r.check}   (actual {shown}){r.note}")
        verdict = "all checks passed" if self.ok else f"{len(self.failures)} check(s) failed"
        lines.append(verdict)
        return "\n".join(lines)


def parse_check(rule: str) -> Check:
    """Parse ``"jaccard_typed_edges>=0.95"`` into a :class:`Check`."""
    m = _RULE.match(rule)
    if m is None:
        raise ValueError(
            f"cannot parse check {rule!r}; expected <quantity><op><number>, "
            f"e.g. jaccard_typed_edges>=0.95"
        )
    quantity, op, number = m.groups()
    quantity = _ALIASES.get(quantity, quantity)
    return Check(quantity=quantity, op=op, threshold=float(number))


def _quantity(report: ComparisonReport, name: str) -> float | None:
    """Resolve a quantity name against a report. ``None`` when unavailable."""
    base, _, suffix = name.partition(".")
    base = _ALIASES.get(base, base)
    if base in ("nodes", "edges"):
        counts = report.node_status_counts if base == "nodes" else report.edge_status_counts
        if suffix == "":
            return float(sum(counts.values()))
        if suffix.upper() in counts:
            return float(counts[suffix.upper()])
        if suffix == "changed_any":
            return float(sum(v for k, v in counts.items() if k != "SHARED"))
        raise KeyError(f"unknown status {suffix!r} for {base}; expected one of {list(counts)}")
    if base == "significance":
        if report.significance is None:
            return None
        table = report.significance.table
        if suffix in ("", "max"):
            return float(table["significance"].max()) if len(table) else 0.0
        if suffix == "mean":
            return float(table["significance"].mean()) if len(table) else 0.0
        if suffix.startswith("n_over_"):  # significance.n_over_10 → count of nodes above 10
            cut = float(suffix.removeprefix("n_over_"))
            return float((table["significance"] > cut).sum())
        raise KeyError(f"unknown significance quantity {suffix!r}")
    if base in SCALAR_METRICS:
        variant = suffix or "raw"
        if variant not in ("raw", "shared"):
            raise KeyError(f"unknown variant {suffix!r} for {base}; use .raw or .shared")
        return report.score(base, shared_subgraph=variant == "shared")
    raise KeyError(
        f"unknown quantity {name!r}; expected a metric ({', '.join(SCALAR_METRICS)}), "
        f"nodes.<STATUS>, edges.<STATUS>, or significance[.max|.mean|.n_over_X]"
    )


def load_baseline(path: str | Path) -> dict[str, float | None]:
    """Flatten a stored report JSON to ``{quantity: value}`` for drift checks."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, float | None] = {}
    for name, both in payload.get("scalar_scores", {}).items():
        out[name] = both.get("raw")
        out[f"{name}.raw"] = both.get("raw")
        out[f"{name}.shared"] = both.get("shared")
    for base, key in (("nodes", "node_status_counts"), ("edges", "edge_status_counts")):
        counts = payload.get(key, {})
        for status, v in counts.items():
            out[f"{base}.{status}"] = float(v)
        out[base] = float(sum(counts.values()))
    sig = payload.get("significance")
    if sig and sig.get("top"):
        out["significance"] = float(sig["top"][0]["significance"])
        out["significance.max"] = out["significance"]
    return out


def run_checks(
    report: ComparisonReport,
    checks: Iterable[Check | str],
    *,
    baseline: dict[str, float | None] | None = None,
    tolerance: float = 0.0,
    drift: Iterable[str] = (),
) -> CheckReport:
    """Evaluate every check (and optional drift-from-baseline rules) on a report.

    Parameters
    ----------
    report:
        The comparison to judge.
    checks:
        :class:`Check` objects or rule strings (see module docstring).
    baseline:
        Flattened values from :func:`load_baseline`. Required when ``drift``
        is non-empty.
    tolerance:
        Allowed absolute movement for each quantity in ``drift``.
    drift:
        Quantities whose current value must stay within ``tolerance`` of the
        baseline. Empty by default; pass ``SCALAR_METRICS`` to guard them all.
    """
    out = CheckReport()
    for c in checks:
        check = parse_check(c) if isinstance(c, str) else c
        value = _quantity(report, check.quantity)
        if value is None or value != value:  # None or NaN
            out.results.append(
                CheckResult(check, None, False, "  ← quantity unavailable for this comparison")
            )
            continue
        out.results.append(CheckResult(check, value, _OPS[check.op](value, check.threshold)))

    drift = list(drift)
    if drift:
        if baseline is None:
            raise ValueError("drift checks need a baseline")
        for name in drift:
            name = _ALIASES.get(name, name)
            current = _quantity(report, name)
            previous = baseline.get(name)
            check = Check(quantity=f"drift:{name}", op="<=", threshold=tolerance)
            if current is None or previous is None:
                out.results.append(
                    CheckResult(check, None, False, "  ← not present in both report and baseline")
                )
                continue
            moved = abs(current - previous)
            out.results.append(
                CheckResult(
                    check,
                    moved,
                    moved <= tolerance,
                    f"  (baseline {previous:.6g} → now {current:.6g})",
                )
            )
    return out
