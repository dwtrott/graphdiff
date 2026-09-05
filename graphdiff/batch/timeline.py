"""Timeline: comparing a *sequence* of snapshots, not just a pair.

Two graphs answer "what changed". A series answers the questions that usually
come next: *when* did it change, was it one event or a steady drift, which
nodes keep changing, and which changes stuck versus flickered back. This
module runs the pairwise comparison over consecutive snapshots and lifts the
results one level:

* :attr:`TimelineReport.metrics` — every scalar metric per step, long format,
  so a churn curve is one ``pivot``.
* :attr:`TimelineReport.node_history` — significance per node per step for
  the nodes that were ever notable; the heatmap of *where* and *when*.
* :attr:`TimelineReport.presence` — which snapshots each node is in, which
  makes "arrived in step 3", "left in step 5" and "flickered" plain lookups.
* :attr:`TimelineReport.findings` — the sentences: the biggest step, the
  steady drift, the recurrent nodes, the flickers.

Each step's :class:`~graphdiff.report.report.ComparisonReport` is kept, so
any single transition can be opened in the viewer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._types import LABEL, STATUS
from ..api import GraphLike, _coerce, compare
from ..report.findings import Finding
from ..report.report import ComparisonReport

__all__ = ["TimelineReport", "compare_sequence"]


@dataclass
class TimelineReport:
    """Consecutive comparisons over a series of graphs and what they add up to."""

    names: list[str]
    steps: list[ComparisonReport]
    metrics: pd.DataFrame
    node_history: pd.DataFrame
    presence: pd.DataFrame
    findings: list[Finding] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ views

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def step_labels(self) -> list[str]:
        """``"a → b"`` per step, for axes and tables."""
        return [f"{self.names[i]} → {self.names[i + 1]}" for i in range(self.n_steps)]

    def metric(self, name: str, *, shared: bool = False) -> pd.Series:
        """One metric across steps, indexed by step number."""
        sub = self.metrics[self.metrics["metric"] == name]
        col = "shared_subgraph_score" if shared else "raw_score"
        return pd.Series(sub[col].to_numpy(), index=sub["step"].to_numpy(), name=name)

    def churn(self) -> pd.DataFrame:
        """Per step: nodes and edges removed / added / reweighted, plus similarity."""
        rows = []
        for k, rep in enumerate(self.steps):
            nc, ec = rep.node_status_counts, rep.edge_status_counts
            rows.append(
                {
                    "step": k,
                    "from": self.names[k],
                    "to": self.names[k + 1],
                    "nodes_removed": nc["A_ONLY"],
                    "nodes_added": nc["B_ONLY"],
                    "edges_removed": ec["A_ONLY"],
                    "edges_added": ec["B_ONLY"],
                    "edges_reweighted": ec["CHANGED"],
                    "similarity": rep.score("ged_similarity"),
                }
            )
        return pd.DataFrame(rows)

    def recurrent_nodes(self, *, min_steps: int = 2, min_significance: float = 0.0) -> pd.DataFrame:
        """Nodes notable in at least ``min_steps`` steps, most recurrent first."""
        hist = self.node_history
        if hist.empty:
            return pd.DataFrame(columns=["label", "steps_notable", "total_significance"])
        notable = (hist > min_significance).sum(axis=1)
        keep = notable[notable >= min_steps]
        out = pd.DataFrame(
            {
                "label": keep.index,
                "steps_notable": keep.to_numpy(),
                "total_significance": hist.loc[keep.index].sum(axis=1).to_numpy(),
            }
        )
        return out.sort_values(
            ["steps_notable", "total_significance"], ascending=[False, False]
        ).reset_index(drop=True)

    def flickers(self) -> pd.DataFrame:
        """Nodes that left and came back (or arrived and left) within the series."""
        pres = self.presence.to_numpy().astype(int)
        changes = np.abs(np.diff(pres, axis=1)).sum(axis=1)
        mask = changes >= 2
        return (
            pd.DataFrame(
                {
                    "label": self.presence.index[mask],
                    "transitions": changes[mask],
                    "pattern": ["".join("●" if v else "·" for v in row) for row in pres[mask]],
                }
            )
            .sort_values("transitions", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------ export

    def to_dict(self, *, max_nodes: int = 50) -> dict[str, Any]:
        hist = self.node_history.head(max_nodes)
        return {
            "names": self.names,
            "step_labels": self.step_labels(),
            "parameters": self.parameters,
            "findings": [f.to_dict() for f in self.findings],
            "churn": self.churn().replace({np.nan: None}).to_dict("records"),
            "metrics": self.metrics.replace({np.nan: None}).to_dict("records"),
            "node_history": {
                "labels": [str(x) for x in hist.index],
                "steps": [int(c) for c in hist.columns],
                "significance": [[float(v) for v in row] for row in hist.to_numpy()],
            },
            "recurrent": self.recurrent_nodes().head(max_nodes).to_dict("records"),
            "flickers": self.flickers().head(max_nodes).to_dict("records"),
            "steps": [s.to_dict(include_top_changed=10) for s in self.steps],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, *, top_n: int = 10) -> str:
        lines = [f"# graphdiff timeline: {' → '.join(self.names)}", ""]
        if self.findings:
            lines += ["## Findings", ""]
            lines += [f"- **{f.severity}** — {f.text}" for f in self.findings]
            lines += [""]
        lines += [
            "## Churn per step",
            "",
            "| Step | Similarity | Nodes -/+ | Edges -/+ | Reweighted |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in self.churn().to_dict("records"):
            sim = (
                "—"
                if r["similarity"] is None or pd.isna(r["similarity"])
                else f"{r['similarity']:.3f}"
            )
            lines.append(
                f"| {r['from']} → {r['to']} | {sim} | "
                f"{r['nodes_removed']:,} / {r['nodes_added']:,} | "
                f"{r['edges_removed']:,} / {r['edges_added']:,} | {r['edges_reweighted']:,} |"
            )
        rec = self.recurrent_nodes().head(top_n)
        if len(rec):
            lines += [
                "",
                f"## Recurrently changing nodes (top {len(rec)})",
                "",
                "| Node | Steps notable | Total significance |",
                "| --- | ---: | ---: |",
            ]
            lines += [
                f"| {r.label} | {r.steps_notable} of {self.n_steps} | {r.total_significance:.1f} |"
                for r in rec.itertuples(index=False)
            ]
        fl = self.flickers().head(top_n)
        if len(fl):
            lines += [
                "",
                f"## Flickering nodes (top {len(fl)})",
                "",
                "| Node | Presence | |",
                "| --- | --- | --- |",
            ]
            lines += [
                f"| {r.label} | `{r.pattern}` | {r.transitions} transitions |"
                for r in fl.itertuples(index=False)
            ]
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path, *, top_n: int = 10) -> Path:
        path = Path(path)
        path.write_text(self.to_markdown(top_n=top_n), encoding="utf-8")
        return path


def _timeline_findings(tl: TimelineReport, *, max_nodes: int = 3) -> list[Finding]:
    out: list[Finding] = []
    churn = tl.churn()
    sims = churn["similarity"].astype(float)
    labels = tl.step_labels()
    if len(churn) == 0:
        return out

    total = (
        churn["nodes_removed"]
        + churn["nodes_added"]
        + churn["edges_removed"]
        + churn["edges_added"]
    ).astype(float)
    share = total / max(total.sum(), 1.0)
    k = int(np.argmax(total.to_numpy()))
    sim_text = f"similarity {sims.iloc[k]:.2f}" if not pd.isna(sims.iloc[k]) else "similarity n/a"
    if len(churn) > 1 and share.iloc[k] >= 0.5:
        out.append(
            Finding(
                "step",
                f"One step dominates: {labels[k]} carries {share.iloc[k] * 100:.0f}% of all change "
                f"across the series ({sim_text}). The other {len(churn) - 1} steps are "
                f"comparatively quiet.",
                severity="high",
                view="timeline",
                target=str(k),
                evidence={
                    "step": k,
                    "share": float(share.iloc[k]),
                    "similarity": float(sims.iloc[k]) if not pd.isna(sims.iloc[k]) else None,
                },
            )
        )
    elif len(churn) > 1:
        cv = float(total.std() / total.mean()) if total.mean() > 0 else 0.0
        out.append(
            Finding(
                "step",
                f"Change is spread across the series: the busiest step ({labels[k]}) carries "
                f"{share.iloc[k] * 100:.0f}% of it. "
                + (
                    "Steady drift rather than an event."
                    if cv < 0.5
                    else "Uneven, but no single event."
                ),
                severity="medium",
                view="timeline",
                target=str(k),
                evidence={"step": k, "share": float(share.iloc[k]), "cv": cv},
            )
        )
    else:
        out.append(
            Finding(
                "step",
                f"One step, {labels[0]}: {sim_text}.",
                severity="low",
                view="timeline",
                target="0",
            )
        )

    # Trend in similarity.
    if len(sims.dropna()) >= 3:
        x = np.arange(len(sims))
        ok = ~sims.isna().to_numpy()
        slope = float(np.polyfit(x[ok], sims.to_numpy()[ok], 1)[0])
        if abs(slope) >= 0.02:
            out.append(
                Finding(
                    "trend",
                    f"Step-to-step similarity is {'falling' if slope < 0 else 'rising'} "
                    f"({slope * 100:+.1f} points per step): the series is "
                    f"{'destabilizing' if slope < 0 else 'settling'}.",
                    severity="medium" if slope < 0 else "low",
                    view="timeline",
                    evidence={"slope": slope},
                )
            )

    fl = tl.flickers()
    half = max(2, -(-tl.n_steps // 2))  # ceil(n/2): notable in at least half the steps
    rec = tl.recurrent_nodes(min_steps=half)
    rec = rec[~rec["label"].isin(set(fl["label"]))]  # flickers are their own story
    if len(rec):
        top = rec.head(max_nodes)
        names = ", ".join(
            f"{r.label} ({r.steps_notable} of {tl.n_steps})" for r in top.itertuples()
        )
        out.append(
            Finding(
                "recurrent",
                f"{len(rec):,} node{'s' if len(rec) != 1 else ''} changed notably in at least "
                f"{half} of {tl.n_steps} steps — a hotspot rather than a one-off: {names}.",
                severity="high"
                if int(top.iloc[0]["steps_notable"]) >= max(3, tl.n_steps // 2)
                else "medium",
                view="timeline",
                target=str(top.iloc[0]["label"]),
                evidence={"n": len(rec), "top": top.to_dict("records")},
            )
        )
    if len(fl):
        names = ", ".join(f"{r.label} {r.pattern}" for r in fl.head(max_nodes).itertuples())
        out.append(
            Finding(
                "flicker",
                f"{len(fl):,} node{'s' if len(fl) != 1 else ''} left and came back (or arrived and "
                f"left) within the series — worth checking for an unstable source rather than a "
                f"real change: {names}.",
                severity="medium",
                view="timeline",
                target=str(fl.iloc[0]["label"]),
                evidence={"n": len(fl)},
            )
        )
    return out


def compare_sequence(
    graphs: Sequence[GraphLike],
    *,
    names: Sequence[str] | None = None,
    top_nodes: int = 200,
    findings: bool = True,
    **compare_kwargs: Any,
) -> TimelineReport:
    """Compare consecutive snapshots and summarize the series.

    Parameters
    ----------
    graphs:
        Two or more graphs (objects or paths), in order.
    names:
        Display names; default to each graph's own name or ``g0``, ``g1``, …
    top_nodes:
        How many nodes to keep in :attr:`TimelineReport.node_history` — the
        union of each step's top significance rows, capped.
    findings:
        Generate timeline-level findings (per-step findings are always on).
    **compare_kwargs:
        Passed to :func:`~graphdiff.compare` for every step (``align``,
        ``weight_attribute``, ``cluster_by``, …).
    """
    if len(graphs) < 2:
        raise ValueError("a timeline needs at least two graphs")
    loaded = [_coerce(g) for g in graphs]
    labels = list(names) if names is not None else [g.name or f"g{i}" for i, g in enumerate(loaded)]
    if len(labels) != len(loaded):
        raise ValueError("names must match graphs one-to-one")
    if len(set(labels)) != len(labels):
        labels = [f"{n} [{i}]" if labels.count(n) > 1 else n for i, n in enumerate(labels)]

    steps: list[ComparisonReport] = []
    rows: list[dict[str, Any]] = []
    hist: dict[str, dict[int, float]] = {}
    for k in range(len(loaded) - 1):
        rep = compare(
            loaded[k], loaded[k + 1], name_a=labels[k], name_b=labels[k + 1], **compare_kwargs
        )
        steps.append(rep)
        for name, values in rep.scalar_scores().items():
            rows.append(
                {
                    "step": k,
                    "from": labels[k],
                    "to": labels[k + 1],
                    "metric": name,
                    "raw_score": values["raw"],
                    "shared_subgraph_score": values["shared"],
                }
            )
        if rep.significance is not None:
            top = rep.significance.top(top_nodes)
            top = top[top["significance"] > 0]
            for lab, sig in zip(top[LABEL], top["significance"], strict=True):
                hist.setdefault(str(lab), {})[k] = float(sig)

    n_steps = len(steps)
    node_history = pd.DataFrame(
        {lab: [vals.get(k, 0.0) for k in range(n_steps)] for lab, vals in hist.items()},
        index=pd.RangeIndex(n_steps, name="step"),
    ).T
    node_history.index.name = LABEL
    if len(node_history):
        node_history = node_history.loc[node_history.sum(axis=1).sort_values(ascending=False).index]
        node_history = node_history.head(top_nodes)

    all_labels = pd.Index(sorted(set().union(*(set(g.nodes.index) for g in loaded))), name=LABEL)
    presence = pd.DataFrame(
        {labels[i]: all_labels.isin(g.nodes.index) for i, g in enumerate(loaded)}, index=all_labels
    )
    # With fuzzy alignment, a renamed node is "present" under A's label in the
    # union; reflect that so a rename does not read as leave-and-arrive.
    for k, rep in enumerate(steps):
        if rep.union is not None and "label_b" in rep.union.nodes.columns:
            nodes = rep.union.nodes
            matched = nodes[
                (nodes[STATUS].astype(str) != "A_ONLY")
                & (nodes["match_method"] != "exact")
                & (nodes["match_method"] != "")
            ]
            for lab_a, lab_b in zip(matched.index, matched["label_b"], strict=True):
                if lab_b in presence.index and lab_a in presence.index:
                    presence.loc[lab_a, labels[k + 1]] = True
                    presence.loc[lab_b, labels[k + 1]] = False
    presence = presence[presence.any(axis=1)]

    tl = TimelineReport(
        names=labels,
        steps=steps,
        metrics=pd.DataFrame(
            rows, columns=["step", "from", "to", "metric", "raw_score", "shared_subgraph_score"]
        ),
        node_history=node_history,
        presence=presence,
        parameters={
            "top_nodes": top_nodes,
            **{
                k: v
                for k, v in compare_kwargs.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            },
        },
    )
    if findings:
        tl.findings = _timeline_findings(tl)
    return tl
