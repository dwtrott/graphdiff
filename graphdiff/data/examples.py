"""Bundled example graphs, so the demo and the tests have something real to chew on.

The pair models the common case: a knowledge graph rebuilt from a later
snapshot of the same corpus. Most entities persist, a few appear, a few drop
out, some relationships are rewired, and edge confidences drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.graph import PropertyGraph

__all__ = ["ORGS", "PEOPLE", "TOPICS", "example_pair", "large_example_pair"]

PEOPLE: tuple[str, ...] = (
    "Alvarez",
    "Bennett",
    "Chowdhury",
    "Delacroix",
    "Eriksen",
    "Farrow",
    "Gadde",
    "Halloran",
    "Ibarra",
    "Jankowski",
    "Keller",
    "Lindqvist",
    "Moreau",
    "Nakamura",
    "Okonkwo",
    "Pereira",
    "Quintero",
    "Rasmussen",
    "Sandoval",
    "Takahashi",
    "Ustinov",
    "Vance",
    "Whitfield",
    "Xu",
    "Yeboah",
    "Zieliński",
    "Abadi",
    "Brennan",
    "Castellanos",
    "Dumitrescu",
    "Espinoza",
    "Fitzgerald",
)

ORGS: tuple[str, ...] = (
    "Northgate Institute",
    "Halcyon Labs",
    "Meridian University",
    "Cobalt Research",
    "Vantage Analytics",
    "Orchard Foundation",
    "Ferrous Systems Group",
    "Lumen Collective",
)

TOPICS: tuple[str, ...] = (
    "graph alignment",
    "spectral methods",
    "entity resolution",
    "provenance",
    "streaming ingest",
    "schema drift",
    "embedding search",
    "record linkage",
    "anomaly scoring",
    "temporal joins",
    "federated indexing",
    "coreference",
)


def _edges_v1(rng: np.random.Generator) -> list[tuple[str, str, str, float]]:
    """Edges with real community structure: people cluster into labs.

    A uniformly random graph lays out as a hairball no matter how good the
    layout engine is, which makes it useless for judging a visual diff. Real
    knowledge graphs have blocks, so the example has blocks: each person belongs
    to one organization, coauthors mostly within it, and works on the topics
    that organization funds.
    """
    rows: list[tuple[str, str, str, float]] = []
    w = lambda lo, hi: round(float(rng.uniform(lo, hi)), 3)  # noqa: E731

    # Each org owns a slice of the topic space; each person belongs to one org.
    topics_by_org = {org: list(TOPICS[i :: len(ORGS)]) for i, org in enumerate(ORGS)}
    for org, owned in topics_by_org.items():
        if not owned:  # pragma: no cover - only if TOPICS < ORGS
            topics_by_org[org] = [TOPICS[0]]

    org_of = {p: ORGS[i % len(ORGS)] for i, p in enumerate(PEOPLE)}
    members: dict[str, list[str]] = {org: [] for org in ORGS}
    for person, org in org_of.items():
        members[org].append(person)

    for person, org in org_of.items():
        rows.append((person, "affiliated_with", org, w(0.6, 1.0)))
        own = topics_by_org[org]
        for topic in rng.choice(own, size=min(2, len(own)), replace=False):
            rows.append((person, "researches", str(topic), w(0.5, 1.0)))
        if rng.random() < 0.35:  # occasional cross-disciplinary interest
            rows.append((person, "researches", str(rng.choice(TOPICS)), w(0.2, 0.6)))

    # Coauthorship: dense inside a lab, sparse across labs.
    for group in members.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if rng.random() < 0.55:
                    rows.append((a, "coauthored", b, w(0.3, 1.0)))
    for _ in range(14):
        a, b = rng.choice(PEOPLE, size=2, replace=False)
        rows.append((str(a), "coauthored", str(b), w(0.1, 0.5)))

    # Topics within one org's remit are related; a few bridges across orgs.
    for owned in topics_by_org.values():
        for i, a in enumerate(owned):
            for b in owned[i + 1 :]:
                rows.append((a, "related_to", b, w(0.4, 0.9)))
    for _ in range(6):
        a, b = rng.choice(TOPICS, size=2, replace=False)
        rows.append((str(a), "related_to", str(b), w(0.2, 0.5)))

    for org, owned in topics_by_org.items():
        for topic in owned:
            rows.append((org, "funds", str(topic), w(0.4, 1.0)))

    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str, float]] = []
    for s, t, d, weight in rows:
        if s == d or (s, t, d) in seen:
            continue
        seen.add((s, t, d))
        unique.append((s, t, d, weight))
    return unique


def example_pair(seed: int = 11) -> tuple[PropertyGraph, PropertyGraph]:
    """Two versions of a small named-entity knowledge graph.

    Returns
    -------
    (PropertyGraph, PropertyGraph)
        ``snapshot_2024`` and ``snapshot_2025``. The second drops two people,
        adds three, removes about a tenth of the edges, adds a handful of new
        relationships, and nudges roughly a fifth of the edge weights — so every
        status (``SHARED``, ``A_ONLY``, ``B_ONLY``, ``CHANGED``) is represented.

    Examples
    --------
    >>> from graphdiff.data.examples import example_pair
    >>> a, b = example_pair()
    >>> a.n_nodes > 0 and b.n_nodes > 0
    True
    """
    rng = np.random.default_rng(seed)
    rows = _edges_v1(rng)
    frame_a = pd.DataFrame(rows, columns=["source", "type", "target", "weight"])
    graph_a = PropertyGraph.from_edges(frame_a, name="snapshot_2024")

    # --- build the later snapshot -----------------------------------------
    departed = {"Ustinov", "Quintero"}
    arrived = ("Novak", "Osei", "Petrov")

    kept = frame_a[
        ~frame_a["source"].isin(departed) & ~frame_a["target"].isin(departed)
    ].reset_index(drop=True)

    drop = rng.random(len(kept)) < 0.10
    later = kept.loc[~drop].copy().reset_index(drop=True)

    bump = rng.random(len(later)) < 0.20
    later.loc[bump, "weight"] = (later.loc[bump, "weight"] * 1.35).round(3).clip(upper=1.0)

    new_rows: list[tuple[str, str, str, float]] = []
    for person in arrived:
        org = ORGS[rng.integers(0, len(ORGS))]
        new_rows.append((person, "affiliated_with", org, round(float(rng.uniform(0.6, 1.0)), 3)))
        for topic in rng.choice(TOPICS, size=2, replace=False):
            new_rows.append(
                (person, "researches", str(topic), round(float(rng.uniform(0.3, 1.0)), 3))
            )
        collaborator = str(rng.choice([p for p in PEOPLE if p not in departed]))
        new_rows.append(
            (person, "coauthored", collaborator, round(float(rng.uniform(0.2, 0.9)), 3))
        )

    survivors = [p for p in PEOPLE if p not in departed]
    for _ in range(18):
        a, b = rng.choice(survivors, size=2, replace=False)
        new_rows.append((str(a), "coauthored", str(b), round(float(rng.uniform(0.1, 1.0)), 3)))

    frame_b = pd.concat(
        [later, pd.DataFrame(new_rows, columns=["source", "type", "target", "weight"])],
        ignore_index=True,
    )
    frame_b = frame_b[frame_b["source"] != frame_b["target"]]
    frame_b = frame_b.drop_duplicates(subset=["source", "type", "target"], ignore_index=True)

    graph_b = PropertyGraph.from_edges(frame_b, name="snapshot_2025")
    return graph_a, graph_b


def large_example_pair(
    *,
    n_communities: int = 28,
    community_size: int = 110,
    seed: int = 5,
    changed_communities: int = 3,
) -> tuple[PropertyGraph, PropertyGraph]:
    """A pair large enough that the overview is a hairball and the other views matter.

    A stochastic block model: ``n_communities`` dense blocks with sparse links
    between them, giving roughly 3,000 nodes and 9,000 edges by default. The
    second snapshot leaves most blocks untouched and concentrates the change in
    ``changed_communities`` of them — members leave, newcomers arrive, ties are
    rewired and reweighted — plus a light scatter of noise everywhere. That is
    the shape real drift usually has, and it is the case the cluster view exists
    for: a handful of dark marks among many pale ones.

    Returns
    -------
    (PropertyGraph, PropertyGraph)
        ``baseline`` and ``rebuilt``.
    """
    rng = np.random.default_rng(seed)
    members: list[list[str]] = []
    rows: list[tuple[str, str, str, float]] = []
    w = lambda lo, hi: round(float(rng.uniform(lo, hi)), 3)  # noqa: E731
    types = ("links_to", "cites", "mentions")

    for c in range(n_communities):
        block = [f"c{c:02d}-n{i:03d}" for i in range(community_size)]
        members.append(block)
        # Dense inside: each node reaches ~6 others in its block.
        for a in block:
            for _ in range(3):
                b = block[int(rng.integers(0, community_size))]
                if a != b:
                    rows.append((a, str(rng.choice(types)), b, w(0.2, 1.0)))
    # Sparse between blocks.
    flat = [n for block in members for n in block]
    for _ in range(n_communities * 25):
        a, b = rng.choice(flat, size=2, replace=False)
        rows.append((str(a), "links_to", str(b), w(0.05, 0.4)))

    seen: set[tuple[str, str, str]] = set()
    base_rows = []
    for s, t, d, weight in rows:
        if s == d or (s, t, d) in seen:
            continue
        seen.add((s, t, d))
        base_rows.append((s, t, d, weight))
    frame_a = pd.DataFrame(base_rows, columns=["source", "type", "target", "weight"])
    graph_a = PropertyGraph.from_edges(frame_a, name="baseline")

    # ---- the rebuilt snapshot ---------------------------------------------
    hot = set(rng.choice(n_communities, size=changed_communities, replace=False).tolist())
    departed: set[str] = set()
    for c in hot:
        departed.update(rng.choice(members[c], size=community_size // 5, replace=False))

    later = frame_a[~frame_a["source"].isin(departed) & ~frame_a["target"].isin(departed)].copy()

    in_hot = later["source"].str[:3].isin({f"c{c:02d}" for c in hot})
    drop = (rng.random(len(later)) < 0.35) & in_hot
    drop |= rng.random(len(later)) < 0.01  # background noise everywhere
    later = later.loc[~drop].reset_index(drop=True)

    bump = (rng.random(len(later)) < 0.30) & later["source"].str[:3].isin(
        {f"c{c:02d}" for c in hot}
    )
    later.loc[bump, "weight"] = (later.loc[bump, "weight"] * 1.5).round(3).clip(upper=1.0)

    new_rows: list[tuple[str, str, str, float]] = []
    for c in hot:
        survivors = [n for n in members[c] if n not in departed]
        newcomers = [f"c{c:02d}-new{i:02d}" for i in range(community_size // 6)]
        for a in newcomers:
            for _ in range(4):
                b = str(rng.choice(survivors))
                new_rows.append((a, str(rng.choice(types)), b, w(0.3, 1.0)))
        for _ in range(community_size):
            a, b = rng.choice(survivors, size=2, replace=False)
            new_rows.append((str(a), str(rng.choice(types)), str(b), w(0.2, 1.0)))

    frame_b = pd.concat(
        [later, pd.DataFrame(new_rows, columns=["source", "type", "target", "weight"])],
        ignore_index=True,
    )
    frame_b = frame_b[frame_b["source"] != frame_b["target"]]
    frame_b = frame_b.drop_duplicates(subset=["source", "type", "target"], ignore_index=True)
    graph_b = PropertyGraph.from_edges(frame_b, name="rebuilt")
    return graph_a, graph_b
