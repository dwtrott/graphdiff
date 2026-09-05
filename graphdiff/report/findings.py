"""Findings: the comparison as a short list of plain-language statements.

A viewer with five lenses and no narrative leaves the reader to work out what
matters. This module writes the narrative. Each finding is one sentence a
person could read aloud in a briefing, carries the evidence behind it, and
points at the view and target where it can be seen.

Findings are ordered by how much they should change what the reader does next:
concentration first (is the change localized or diffuse?), then the specific
places and nodes, then the background rate everything else is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._types import Status
from ..core.union import UnionDiffGraph
from ..metrics.cluster import ClusterMap
from ..metrics.significance import SignificanceResult

__all__ = ["Finding", "generate_findings"]


@dataclass
class Finding:
    """One statement, its evidence, and where to look.

    Attributes
    ----------
    kind:
        Machine-readable category: ``headline``, ``concentration``, ``cluster``,
        ``node``, ``removed_hub``, ``added_hub``, ``weights``, ``background``.
    text:
        The sentence.
    severity:
        ``high`` / ``medium`` / ``low`` — how much it should draw the eye.
    view / target:
        Where in the viewer to go, and what to select there.
    evidence:
        Numbers the sentence was built from, for the report and for tooltips.
    """

    kind: str
    text: str
    severity: str = "medium"
    view: str | None = None
    target: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain containers for JSON."""
        return {
            "kind": self.kind,
            "text": self.text,
            "severity": self.severity,
            "view": self.view,
            "target": self.target,
            "evidence": {k: _plain(v) for k, v in self.evidence.items()},
        }


def _plain(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _what_happened(row: pd.Series) -> str:
    parts = []
    if row["n_removed"]:
        parts.append(f"lost {int(row['n_removed'])}")
    if row["n_added"]:
        parts.append(f"gained {int(row['n_added'])}")
    if row["n_changed"]:
        parts.append(f"{int(row['n_changed'])} reweighted")
    return ", ".join(parts) if parts else "unchanged"


def generate_findings(
    union: UnionDiffGraph,
    significance: SignificanceResult,
    clusters: ClusterMap | None,
    *,
    similarity: float | None,
    weight_spearman: float | None,
    max_nodes: int = 5,
    max_clusters: int = 3,
) -> list[Finding]:
    """Build the findings list for one comparison.

    Parameters
    ----------
    union:
        The union diff graph.
    significance:
        Per-node significance ranking.
    clusters:
        The cluster map, or ``None`` to skip concentration findings.
    similarity:
        Headline similarity (normalized edit distance), if available.
    weight_spearman:
        Weight agreement over shared edges, if available.
    max_nodes / max_clusters:
        How many specific nodes and clusters to call out.
    """
    out: list[Finding] = []
    name_a, name_b = union.name_a, union.name_b
    nc, ec = union.node_status_counts(), union.edge_status_counts()
    n_union, e_union = union.n_nodes, union.n_edges
    n_diff_nodes = nc["A_ONLY"] + nc["B_ONLY"] + nc["CHANGED"]
    n_diff_edges = ec["A_ONLY"] + ec["B_ONLY"] + ec["CHANGED"]

    # ---- headline -------------------------------------------------------------
    if n_diff_nodes == 0 and n_diff_edges == 0:
        out.append(
            Finding(
                "headline",
                f"{name_a} and {name_b} are identical: every node and edge is shared.",
                severity="low",
                evidence={"n_nodes": n_union, "n_edges": e_union},
            )
        )
        return out

    sim_text = f"{_pct(similarity)} similar by edit distance. " if similarity is not None else ""
    out.append(
        Finding(
            "headline",
            f"{sim_text}{n_diff_nodes:,} of {n_union:,} nodes and {n_diff_edges:,} of "
            f"{e_union:,} edges differ — {nc['B_ONLY']:,} nodes and {ec['B_ONLY']:,} edges only in "
            f"{name_b}, {nc['A_ONLY']:,} nodes and {ec['A_ONLY']:,} edges only in {name_a}, "
            f"{ec['CHANGED']:,} edges reweighted.",
            severity="high" if (similarity is not None and similarity < 0.8) else "medium",
            view="overview",
            evidence={"similarity": similarity, "node_counts": nc, "edge_counts": ec},
        )
    )

    # ---- concentration ------------------------------------------------------
    if clusters is not None and len(clusters.clusters) > 1:
        changed_per = np.array([c.n_changed_nodes + c.n_changed_edges for c in clusters.clusters])
        total_changed = changed_per.sum()
        if total_changed > 0:
            share = np.sort(changed_per)[::-1] / total_changed
            cum = np.cumsum(share)
            k80 = int(np.searchsorted(cum, 0.8) + 1)
            n_cl = len(clusters.clusters)
            top_share = share[:max_clusters].sum()
            hot = [c for c in clusters.clusters if c.change_density >= 0.25]
            if k80 <= max(2, n_cl // 5):
                out.append(
                    Finding(
                        "concentration",
                        f"The change is concentrated: {k80} of {n_cl} clusters account for "
                        f"{_pct(cum[k80 - 1])} of everything that differs. The rest of the graph "
                        f"is close to unchanged.",
                        severity="high",
                        view="clusters",
                        evidence={"k80": k80, "n_clusters": n_cl, "top_share": float(top_share)},
                    )
                )
            elif k80 >= max(3, int(n_cl * 0.6)):
                out.append(
                    Finding(
                        "concentration",
                        f"The change is diffuse: it takes {k80} of {n_cl} clusters to cover 80% "
                        f"of what differs. No single region explains it — look for a "
                        f"graph-wide cause rather than a local one.",
                        severity="medium",
                        view="clusters",
                        evidence={"k80": k80, "n_clusters": n_cl},
                    )
                )
            for c in hot[:max_clusters]:
                out.append(
                    Finding(
                        "cluster",
                        f"Cluster around {c.name} ({c.size:,} nodes) had {_pct(c.change_density)} "
                        f"of its own structure change: {c.node_counts.get('A_ONLY', 0)} nodes "
                        f"left, {c.node_counts.get('B_ONLY', 0)} arrived, "
                        f"{c.edge_counts.get('A_ONLY', 0)} internal edges removed and "
                        f"{c.edge_counts.get('B_ONLY', 0)} added.",
                        severity="high" if c.change_density >= 0.5 else "medium",
                        view="clusters",
                        target=c.key,
                        evidence={
                            "cluster": c.key,
                            "size": c.size,
                            "change_density": c.change_density,
                        },
                    )
                )

    # ---- removed / added hubs -------------------------------------------------
    table = significance.table
    gone = table[(table["status"] == Status.A_ONLY.value) & (table["degree_a"] >= 5)]
    if len(gone):
        top_gone = gone.head(3)
        names = ", ".join(f"{r['label']} ({int(r['degree_a'])})" for _, r in top_gone.iterrows())
        out.append(
            Finding(
                "removed_hub",
                f"{len(gone)} well-connected node{'s' if len(gone) != 1 else ''} "
                f"(degree ≥ 5) disappeared from {name_a}"
                f"{' — most connected: ' + names if len(gone) else ''}.",
                severity="high" if gone["degree_a"].max() >= 10 else "medium",
                view="overview",
                target=str(top_gone.iloc[0]["label"]),
                evidence={"n": len(gone), "max_degree": int(gone["degree_a"].max())},
            )
        )
    new = table[(table["status"] == Status.B_ONLY.value) & (table["degree_b"] >= 5)]
    if len(new):
        top_new = new.head(3)
        names = ", ".join(f"{r['label']} ({int(r['degree_b'])})" for _, r in top_new.iterrows())
        out.append(
            Finding(
                "added_hub",
                f"{len(new)} well-connected node{'s' if len(new) != 1 else ''} "
                f"(degree ≥ 5) appeared in {name_b} — most connected: {names}.",
                severity="high" if new["degree_b"].max() >= 10 else "medium",
                view="overview",
                target=str(top_new.iloc[0]["label"]),
                evidence={"n": len(new), "max_degree": int(new["degree_b"].max())},
            )
        )

    # ---- most significant individual nodes --------------------------------------
    shared_top = table[table["status"].isin({Status.SHARED.value, Status.CHANGED.value})].head(
        max_nodes
    )
    for _, row in shared_top.iterrows():
        if row["surprise"] < 2:  # p > 0.01: not worth a sentence
            break
        out.append(
            Finding(
                "node",
                f"{row['label']} {_what_happened(row)} connections "
                f"(had {int(row['degree_a'])}, now {int(row['degree_b'])}); "
                f"that much change on one node is unlikely by chance "
                f"(p ≈ 1e-{max(1, round(float(row['surprise'])))}).",
                severity="high" if row["surprise"] >= 6 else "medium",
                view="cards",
                target=str(row["label"]),
                evidence={
                    "significance": float(row["significance"]),
                    "surprise": float(row["surprise"]),
                    "n_removed": int(row["n_removed"]),
                    "n_added": int(row["n_added"]),
                    "n_changed": int(row["n_changed"]),
                },
            )
        )

    # ---- centrality shifts ----------------------------------------------------
    shifts = table.dropna(subset=["centrality_shift"])
    if len(shifts):
        big = shifts.reindex(shifts["centrality_shift"].abs().sort_values(ascending=False).index)
        big = big[big["centrality_shift"].abs() > 0]
        if len(big):
            r = big.iloc[0]
            pr_a = float(r["pagerank"] - r["centrality_shift"] / 2)
            rel = abs(float(r["centrality_shift"])) / max(pr_a, 1e-12)
            if rel >= 0.5:
                direction = "more" if r["centrality_shift"] > 0 else "less"
                out.append(
                    Finding(
                        "centrality",
                        f"{r['label']} became markedly {direction} central "
                        f"(PageRank moved {'+' if r['centrality_shift'] > 0 else '-'}"
                        f"{abs(float(r['centrality_shift'])) * 100:.2f} points).",
                        severity="medium",
                        view="cards",
                        target=str(r["label"]),
                        evidence={"centrality_shift": float(r["centrality_shift"])},
                    )
                )

    # ---- weights -----------------------------------------------------------
    if weight_spearman is not None and ec["CHANGED"] > 0 and weight_spearman < 0.5:
        out.append(
            Finding(
                "weights",
                f"Edge weights disagree substantially on shared edges "
                f"(Spearman {weight_spearman:.2f}); the graphs may rank relationships "
                f"differently even where the structure matches.",
                severity="medium",
                view="overview",
                evidence={"spearman": weight_spearman},
            )
        )

    # ---- background ---------------------------------------------------------
    rates = significance.rates
    out.append(
        Finding(
            "background",
            f"Background rates: {_pct(rates.removal)} of {name_a}'s edges were removed, "
            f"{_pct(rates.addition)} of {name_b}'s edges are new, {_pct(rates.reweight)} of "
            f"shared edges were reweighted. Significance above is measured against these.",
            severity="low",
            evidence={
                "removal": rates.removal,
                "addition": rates.addition,
                "reweight": rates.reweight,
            },
        )
    )
    return out
