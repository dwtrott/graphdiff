"""Metric 3 — agreement of edge weights across the shared edge set.

Spearman (rank) correlation is the headline figure: it answers "do the two
graphs agree about which edges are strong?" without assuming the two weight
scales are commensurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .._types import A_PREFIX, B_PREFIX, STATUS, Status
from ..core.union import UnionDiffGraph

__all__ = ["WeightAgreementResult", "weight_agreement"]


@dataclass
class WeightAgreementResult:
    """Correlation of edge weights over edges present in both graphs.

    ``spearman`` is ``None`` when it is undefined: fewer than two shared edges
    carry weights in both graphs, or one side's weights are constant.
    """

    spearman: float | None
    pvalue: float | None
    pearson: float | None
    mean_abs_diff: float | None
    n_shared_edges: int
    n_compared: int
    weight_attribute: str


def weight_agreement(
    union: UnionDiffGraph, weight_attribute: str = "weight"
) -> WeightAgreementResult:
    """Correlate ``weight_attribute`` between A and B over shared edges.

    Parameters
    ----------
    union:
        Union diff graph.
    weight_attribute:
        Name of the numeric edge attribute to compare. Both ``a_<attr>`` and
        ``b_<attr>`` must exist in the union edge table.

    Returns
    -------
    WeightAgreementResult
        Spearman correlation with its p-value, Pearson correlation, and the
        mean absolute difference, alongside the counts they rest on.
    """
    shared_mask = union.edges[STATUS].isin({Status.SHARED.value, Status.CHANGED.value})
    n_shared = int(shared_mask.sum())

    a_col, b_col = f"{A_PREFIX}{weight_attribute}", f"{B_PREFIX}{weight_attribute}"
    if a_col not in union.edges.columns or b_col not in union.edges.columns or n_shared == 0:
        return WeightAgreementResult(
            spearman=None,
            pvalue=None,
            pearson=None,
            mean_abs_diff=None,
            n_shared_edges=n_shared,
            n_compared=0,
            weight_attribute=weight_attribute,
        )

    shared = union.edges.loc[shared_mask, [a_col, b_col]]
    a_vals = pd.to_numeric(shared[a_col], errors="coerce").to_numpy(dtype="float64")
    b_vals = pd.to_numeric(shared[b_col], errors="coerce").to_numpy(dtype="float64")
    valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
    a_vals, b_vals = a_vals[valid], b_vals[valid]
    n_compared = int(a_vals.size)

    mean_abs_diff = float(np.abs(a_vals - b_vals).mean()) if n_compared else None

    spearman = pvalue = pearson = None
    if n_compared >= 2 and np.ptp(a_vals) > 0 and np.ptp(b_vals) > 0:
        rho = stats.spearmanr(a_vals, b_vals)
        spearman = None if np.isnan(rho.statistic) else float(rho.statistic)
        pvalue = None if np.isnan(rho.pvalue) else float(rho.pvalue)
        r = stats.pearsonr(a_vals, b_vals)
        pearson = None if np.isnan(r.statistic) else float(r.statistic)
    elif n_compared >= 2 and np.array_equal(a_vals, b_vals):
        # Constant and identical on both sides: perfect agreement, undefined rank
        # correlation. Report the agreement rather than a misleading None.
        spearman = 1.0
        pearson = 1.0

    return WeightAgreementResult(
        spearman=spearman,
        pvalue=pvalue,
        pearson=pearson,
        mean_abs_diff=mean_abs_diff,
        n_shared_edges=n_shared,
        n_compared=n_compared,
        weight_attribute=weight_attribute,
    )
