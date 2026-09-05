"""Fuzzy node alignment: matching nodes whose labels are not identical.

The exact join on label is the right default — it is fast, deterministic and
never wrong. It is also brittle: ``"Acme Corp"`` and ``"ACME Corp."``, a
renamed account, a typo fixed between two snapshots. Each of those shows up as
one node removed and one added, and a diff full of such pairs hides the real
changes behind bookkeeping.

:func:`align_graphs` recovers those pairs in three passes, each cheaper than
the next is precise:

1. **Exact** — identical labels (the ordinary join).
2. **Normalized** — identical after lower-casing, trimming, collapsing
   whitespace and stripping punctuation.
3. **Fuzzy** — for what is still unmatched, candidate pairs by character
   trigram overlap (an inverted index, so it scales to 10^5 nodes without an
   all-pairs comparison), each scored on two kinds of evidence:

   * *label similarity* — character-trigram similarity of the two labels
     (mean of Jaccard and overlap coefficient, so a suffix or prefix does
     not sink a short name);
   * *structural similarity* — Jaccard of the two nodes' neighbourhoods,
     expressed through neighbours that are already aligned. Two nodes that
     look alike **and** connect to the same things are almost certainly the
     same thing; two that merely look alike are not.

   The score blends the two; one-to-one assignment is greedy by score with a
   floor, and it runs twice so first-round matches become anchors for the
   second.

Every fuzzy match carries a **confidence** in ``[0, 1]``: the score, discounted
by how close the runner-up candidate was. A node whose best match is barely
better than its second-best is genuinely ambiguous, and the viewer and report
say so rather than hiding it.

Alignment never edits the inputs. The result can relabel B onto A's labels
(:meth:`AlignmentResult.relabel_b`) so the union diff graph is built as usual,
with the match method and confidence attached to each union node.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from .._types import LABEL
from .graph import PropertyGraph

__all__ = ["AlignmentResult", "align_graphs", "normalize_label"]

AlignMethod = Literal["exact", "normalized", "fuzzy"]

#: Column names on :attr:`AlignmentResult.table`.
ALIGN_COLUMNS = (
    "label_a",
    "label_b",
    "method",
    "label_similarity",
    "structural_similarity",
    "score",
    "confidence",
    "runner_up",
)

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    """Case-fold, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(label))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT.sub(" ", text.casefold())
    return _WS.sub(" ", text).strip()


def _trigrams(text: str) -> list[str]:
    padded = f"  {text} "
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


def _trigram_matrix(texts: list[str], vocab: dict[str, int], *, grow: bool) -> sparse.csr_matrix:
    """Binary document x trigram matrix. ``grow`` adds unseen trigrams to ``vocab``."""
    indptr = [0]
    indices: list[int] = []
    for text in texts:
        seen: set[int] = set()
        for gram in _trigrams(text):
            idx = vocab.get(gram)
            if idx is None:
                if not grow:
                    continue
                idx = len(vocab)
                vocab[gram] = idx
            seen.add(idx)
        indices.extend(sorted(seen))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
        shape=(len(texts), max(len(vocab), 1)),
    )


def _label_candidates(
    a_texts: list[str],
    b_texts: list[str],
    *,
    floor: float,
    max_candidates: int,
    max_work: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate (i, j, trigram_jaccard) triples above ``floor``, top-k per A row.

    The inverted index is built on every trigram unless the implied pairwise
    work (``sum(df_a * df_b)``) exceeds ``max_work``; then the most common
    trigrams are left out of the *index* — they pair everything with
    everything — but stay in the Jaccard denominators, so the similarity is a
    slight underestimate for labels made mostly of very common trigrams.
    """
    if not a_texts or not b_texts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    vocab: dict[str, int] = {}
    ma = _trigram_matrix(a_texts, vocab, grow=True)
    mb = _trigram_matrix(b_texts, vocab, grow=True)
    ma.resize((ma.shape[0], len(vocab)))
    mb.resize((mb.shape[0], len(vocab)))
    size_a = np.asarray(ma.sum(axis=1)).ravel()
    size_b = np.asarray(mb.sum(axis=1)).ravel()

    # Blocking budget: the product's work is sum(df_a * df_b) over trigrams.
    # Drop the most promiscuous trigrams from the *index* until it fits, so a
    # namespace of near-identical IDs ("n00" in every label) still indexes on
    # the trigrams that actually discriminate.
    df_a = np.asarray(ma.sum(axis=0)).ravel()
    df_b = np.asarray(mb.sum(axis=0)).ravel()
    work = df_a * df_b
    order = np.argsort(work)
    allowed = np.cumsum(work[order]) <= max_work
    keep_mask = np.zeros(len(work), dtype=bool)
    keep_mask[order[allowed]] = True
    keep = sparse.diags(keep_mask.astype(np.float32))
    shared = (ma @ keep) @ (mb @ keep).T  # (n_a, n_b) sparse, entries = shared trigrams
    shared = shared.tocsr()
    shared.eliminate_zeros()
    if shared.nnz == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

    coo = shared.tocoo()
    i, j, s = coo.row, coo.col, coo.data.astype(np.float64)
    # Jaccard alone punishes a short label that gained a suffix ("Gadde" vs
    # "Gadde Inc." is 0.6); the overlap coefficient forgives it (1.0) but is
    # blind to how much was added. Their mean rewards both containment and
    # closeness in length.
    jaccard = s / np.maximum(size_a[i] + size_b[j] - s, 1.0)
    overlap = s / np.maximum(np.minimum(size_a[i], size_b[j]), 1.0)
    jac = 0.5 * jaccard + 0.5 * overlap
    ok = jac >= floor
    i, j, jac = i[ok], j[ok], jac[ok]
    if not len(i):
        return i.astype(np.int64), j.astype(np.int64), jac

    # Top-k per A row.
    order = np.lexsort((-jac, i))
    i, j, jac = i[order], j[order], jac[order]
    first = np.r_[0, np.flatnonzero(np.diff(i)) + 1]
    rank = np.arange(len(i)) - np.repeat(first, np.diff(np.r_[first, len(i)]))
    top = rank < max_candidates
    return i[top].astype(np.int64), j[top].astype(np.int64), jac[top]


def _anchor_adjacency(
    graph: PropertyGraph, rows: pd.Index, anchor_position: pd.Series
) -> sparse.csr_matrix:
    """Binary (rows x anchors) matrix: which anchor nodes each row is adjacent to.

    ``anchor_position`` maps a label *in this graph's namespace* to an anchor
    index; neighbours that are not anchors are ignored.
    """
    edges = graph.edges
    row_pos = pd.Series(np.arange(len(rows)), index=rows)
    src = edges["source"].to_numpy()
    dst = edges["target"].to_numpy()
    r = np.concatenate([row_pos.reindex(src).to_numpy(), row_pos.reindex(dst).to_numpy()])
    c = np.concatenate(
        [anchor_position.reindex(dst).to_numpy(), anchor_position.reindex(src).to_numpy()]
    )
    ok = ~(pd.isna(r) | pd.isna(c))
    mat = sparse.csr_matrix(
        (
            np.ones(int(ok.sum()), dtype=np.float32),
            (r[ok].astype(np.int64), c[ok].astype(np.int64)),
        ),
        shape=(len(rows), len(anchor_position)),
    )
    mat.data[:] = 1.0
    return mat


@dataclass
class AlignmentResult:
    """Node correspondences between A and B and how sure each one is.

    Attributes
    ----------
    table:
        One row per matched pair: ``label_a``, ``label_b``, ``method``
        (``exact`` / ``normalized`` / ``fuzzy``), ``label_similarity``,
        ``structural_similarity`` (NaN when neither node had an aligned
        neighbour to judge by), ``score``, ``confidence``, ``runner_up``.
    unmatched_a / unmatched_b:
        Labels that found no counterpart; they stay ``A_ONLY`` / ``B_ONLY``.
    threshold:
        Minimum score a fuzzy match needed.
    """

    table: pd.DataFrame
    unmatched_a: pd.Index
    unmatched_b: pd.Index
    threshold: float
    parameters: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ summaries

    @property
    def counts(self) -> dict[str, int]:
        """Matches per method, plus unmatched on each side."""
        vc = self.table["method"].value_counts()
        return {
            "exact": int(vc.get("exact", 0)),
            "normalized": int(vc.get("normalized", 0)),
            "fuzzy": int(vc.get("fuzzy", 0)),
            "unmatched_a": len(self.unmatched_a),
            "unmatched_b": len(self.unmatched_b),
        }

    @property
    def fuzzy(self) -> pd.DataFrame:
        """Only the inexact matches, least confident first."""
        sub = self.table[self.table["method"] != "exact"]
        return sub.sort_values("confidence").reset_index(drop=True)

    def mapping(self) -> pd.Series:
        """``label_b -> label_a`` for every inexact match (exact ones map to themselves)."""
        sub = self.table[self.table["method"] != "exact"]
        return pd.Series(sub["label_a"].to_numpy(), index=pd.Index(sub["label_b"], name=LABEL))

    def relabel_b(self, b: PropertyGraph) -> PropertyGraph:
        """Return ``b`` with matched nodes renamed onto A's labels.

        Nothing else changes: attributes, edges and directedness are carried
        over, and unmatched nodes keep their own labels.
        """
        mapping = self.mapping()
        if mapping.empty:
            return b
        lookup = mapping.to_dict()
        nodes = b.nodes.copy()
        nodes.index = pd.Index([lookup.get(x, x) for x in nodes.index], name=LABEL)
        edges = b.edges.copy()
        edges["source"] = edges["source"].map(lambda x: lookup.get(x, x))
        edges["target"] = edges["target"].map(lambda x: lookup.get(x, x))
        return PropertyGraph(
            nodes=nodes,
            edges=edges,
            directed=b.directed,
            name=b.name,
            on_duplicate_edge="first",  # two B nodes cannot map to one A label, but be safe
        )

    def to_dict(self, *, max_rows: int = 50) -> dict[str, Any]:
        """Counts, parameters and the least-confident matches, JSON-safe."""
        fuzzy = self.fuzzy.head(max_rows)
        return {
            "counts": self.counts,
            "threshold": self.threshold,
            "parameters": self.parameters,
            "mean_confidence": float(
                self.table.loc[self.table["method"] == "fuzzy", "confidence"].mean()
            )
            if (self.table["method"] == "fuzzy").any()
            else None,
            "least_confident": fuzzy.replace({np.nan: None}).to_dict("records"),
        }


def _exact_and_normalized(
    a: PropertyGraph, b: PropertyGraph, *, normalize: bool
) -> tuple[list[dict[str, Any]], pd.Index, pd.Index]:
    rows: list[dict[str, Any]] = []
    exact = a.nodes.index.intersection(b.nodes.index)
    for lab in exact:
        rows.append(
            {
                "label_a": lab,
                "label_b": lab,
                "method": "exact",
                "label_similarity": 1.0,
                "structural_similarity": np.nan,
                "score": 1.0,
                "confidence": 1.0,
                "runner_up": np.nan,
            }
        )
    rest_a = a.nodes.index.difference(exact)
    rest_b = b.nodes.index.difference(exact)
    if not normalize or not len(rest_a) or not len(rest_b):
        return rows, rest_a, rest_b

    norm_a = pd.Series([normalize_label(x) for x in rest_a], index=rest_a)
    norm_b = pd.Series([normalize_label(x) for x in rest_b], index=rest_b)
    # Only unambiguous normalized forms: one node on each side.
    ua = norm_a[~norm_a.duplicated(keep=False) & (norm_a != "")]
    ub = norm_b[~norm_b.duplicated(keep=False) & (norm_b != "")]
    inv_b = pd.Series(ub.index, index=ub.to_numpy())
    hit = ua[ua.isin(inv_b.index)]
    matched_a, matched_b = [], []
    for lab_a, form in hit.items():
        lab_b = inv_b[form]
        rows.append(
            {
                "label_a": lab_a,
                "label_b": lab_b,
                "method": "normalized",
                "label_similarity": 1.0,
                "structural_similarity": np.nan,
                "score": 1.0,
                "confidence": 1.0,
                "runner_up": np.nan,
            }
        )
        matched_a.append(lab_a)
        matched_b.append(lab_b)
    return rows, rest_a.difference(pd.Index(matched_a)), rest_b.difference(pd.Index(matched_b))


def align_graphs(
    a: PropertyGraph,
    b: PropertyGraph,
    *,
    method: Literal["exact", "normalized", "fuzzy"] = "fuzzy",
    threshold: float = 0.6,
    label_weight: float = 0.6,
    trigram_floor: float = 0.25,
    max_candidates: int = 10,
    rounds: int = 2,
    max_work: float = 2e7,
) -> AlignmentResult:
    """Find node correspondences between ``a`` and ``b``.

    Parameters
    ----------
    a, b:
        The graphs. Directedness must agree (checked later by the union build).
    method:
        ``"exact"`` reproduces the plain label join; ``"normalized"`` adds
        case/punctuation-insensitive matching; ``"fuzzy"`` adds trigram +
        structure scoring for the rest.
    threshold:
        Minimum blended score for a fuzzy match. ``0.6`` is conservative;
        lower it for noisier labels and watch the confidences.
    label_weight:
        Share of the score that comes from label similarity; the remainder is
        structural. When a pair has no aligned neighbours to judge by, the
        score is the label similarity alone.
    trigram_floor:
        Minimum trigram Jaccard for a pair to become a candidate at all.
    max_candidates:
        Candidates kept per A node.
    rounds:
        Assignment passes; matches from one pass become structural anchors for
        the next.
    max_work:
        Budget for candidate generation, as the number of (label, label)
        trigram co-occurrences the inverted index may produce. The most common
        trigrams are dropped from the index until the budget fits; ``2e7`` is
        a couple of seconds and covers 10^5 unmatched labels a side.
    """
    rows, rest_a, rest_b = _exact_and_normalized(a, b, normalize=method != "exact")
    params = {
        "method": method,
        "threshold": threshold,
        "label_weight": label_weight,
        "trigram_floor": trigram_floor,
        "max_candidates": max_candidates,
        "rounds": rounds,
        "max_work": max_work,
    }
    if method != "fuzzy" or not len(rest_a) or not len(rest_b):
        return AlignmentResult(_frame(rows), rest_a, rest_b, threshold, params)

    a_texts = [normalize_label(x) for x in rest_a]
    b_texts = [normalize_label(x) for x in rest_b]
    ci, cj, label_sim = _label_candidates(
        a_texts, b_texts, floor=trigram_floor, max_candidates=max_candidates, max_work=max_work
    )
    if not len(ci):
        return AlignmentResult(_frame(rows), rest_a, rest_b, threshold, params)

    # Anchors: (label_a, label_b) pairs already matched, in a shared index space.
    anchor_a = [r["label_a"] for r in rows]
    anchor_b = [r["label_b"] for r in rows]
    taken_a = np.zeros(len(rest_a), dtype=bool)
    taken_b = np.zeros(len(rest_b), dtype=bool)
    accepted: list[dict[str, Any]] = []

    for _round in range(max(1, rounds)):
        if len(anchor_a):
            pos_a = pd.Series(np.arange(len(anchor_a)), index=pd.Index(anchor_a))
            pos_b = pd.Series(np.arange(len(anchor_b)), index=pd.Index(anchor_b))
            adj_a = _anchor_adjacency(a, rest_a, pos_a)
            adj_b = _anchor_adjacency(b, rest_b, pos_b)
            deg_a = np.asarray(adj_a.sum(axis=1)).ravel()
            deg_b = np.asarray(adj_b.sum(axis=1)).ravel()
            shared = np.asarray(adj_a[ci].multiply(adj_b[cj]).sum(axis=1)).ravel()
            denom = deg_a[ci] + deg_b[cj] - shared
            with np.errstate(divide="ignore", invalid="ignore"):
                struct = np.where(denom > 0, shared / np.maximum(denom, 1), np.nan)
        else:
            struct = np.full(len(ci), np.nan)

        score = np.where(
            np.isnan(struct),
            label_sim,
            label_weight * label_sim + (1 - label_weight) * struct,
        )
        live = ~taken_a[ci] & ~taken_b[cj]
        # Runner-up per A node among live candidates, for the confidence discount.
        order = np.lexsort((-score, ci))
        runner_up = np.full(len(ci), np.nan)
        best_idx: dict[int, int] = {}
        for k in order:
            if not live[k]:
                continue
            i = int(ci[k])
            if i not in best_idx:
                best_idx[i] = k
            elif np.isnan(runner_up[best_idx[i]]):
                runner_up[best_idx[i]] = score[k]

        new_matches = 0
        for k in np.argsort(-score, kind="stable"):
            if not live[k] or score[k] < threshold:
                continue
            i, j = int(ci[k]), int(cj[k])
            if taken_a[i] or taken_b[j]:
                continue
            taken_a[i] = taken_b[j] = True
            ru = runner_up[k]
            conf = (
                float(score[k] * (1.0 - 0.5 * (ru / score[k])))
                if not np.isnan(ru)
                else float(score[k])
            )
            accepted.append(
                {
                    "label_a": rest_a[i],
                    "label_b": rest_b[j],
                    "method": "fuzzy",
                    "label_similarity": float(label_sim[k]),
                    "structural_similarity": float(struct[k])
                    if not np.isnan(struct[k])
                    else np.nan,
                    "score": float(score[k]),
                    "confidence": conf,
                    "runner_up": float(ru) if not np.isnan(ru) else np.nan,
                }
            )
            anchor_a.append(rest_a[i])
            anchor_b.append(rest_b[j])
            new_matches += 1
        if new_matches == 0:
            break

    rows.extend(accepted)
    return AlignmentResult(
        _frame(rows),
        rest_a[~taken_a],
        rest_b[~taken_b],
        threshold,
        params,
    )


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in ALIGN_COLUMNS})
    frame = pd.DataFrame(rows, columns=list(ALIGN_COLUMNS))
    for col in ("label_similarity", "structural_similarity", "score", "confidence", "runner_up"):
        frame[col] = frame[col].astype("float64")
    return frame
