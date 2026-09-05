"""All-pairs comparison over a directory of graphs.

Pairs are independent, so they fan out over a :mod:`multiprocessing` pool. Each
worker caches the graphs it has loaded, so a graph read once in a process is
not re-parsed for every pair it takes part in.
"""

from __future__ import annotations

import itertools
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..api import compare
from ..io import detect_format
from ..report.report import SCALAR_METRICS

__all__ = ["PairResult", "all_pairs", "discover_graphs"]

PairResult = tuple[str, str, str, float | None, float | None]

_GRAPH_SUFFIXES = {".graphml", ".xml", ".gml", ".json", ".csv", ".tsv", ".parquet", ".pq"}


def discover_graphs(directory: str | Path) -> list[Path]:
    """Graph files directly inside ``directory``, sorted by name."""
    directory = Path(directory)
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_dir() and (path / "edges.parquet").exists():
            found.append(path)
        elif path.is_file() and path.suffix.lower() in _GRAPH_SUFFIXES:
            try:
                detect_format(path)
            except ValueError:
                continue
            found.append(path)
    return found


@lru_cache(maxsize=64)
def _load(path: str):  # type: ignore[no-untyped-def]
    from ..io import read_graph

    return read_graph(path)


def _score_pair(job: tuple[str, str, tuple[str, ...]]) -> list[PairResult]:
    path_a, path_b, metrics = job
    report = compare(_load(path_a), _load(path_b), keep_union=False)
    rows: list[PairResult] = []
    for metric in metrics:
        rows.append(
            (
                Path(path_a).name,
                Path(path_b).name,
                metric,
                report.score(metric),
                report.score(metric, shared_subgraph=True),
            )
        )
    return rows


def all_pairs(
    paths: Sequence[str | Path],
    *,
    metrics: Iterable[str] = ("jaccard_typed_edges",),
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    include_self: bool = False,
) -> pd.DataFrame:
    """Score every unordered pair of graphs.

    Parameters
    ----------
    paths:
        Graph files, any supported format.
    metrics:
        Canonical scalar metric names (see :data:`~graphdiff.SCALAR_METRICS`).
    workers:
        Pool size; ``None`` uses every CPU, ``1`` runs in-process (useful for
        debugging and for tiny inputs where the pool costs more than it saves).
    progress:
        Called with ``(done, total)`` after each pair.
    include_self:
        Also score each graph against itself (always 1.0 / 0 — a sanity row).

    Returns
    -------
    pandas.DataFrame
        Tidy long format: ``graph_a, graph_b, metric, raw_score,
        shared_subgraph_score``.
    """
    metric_tuple = tuple(metrics)
    unknown = [m for m in metric_tuple if m not in SCALAR_METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; choose from {', '.join(SCALAR_METRICS)}")

    names = [str(Path(p)) for p in paths]
    pairs = list(itertools.combinations(names, 2))
    if include_self:
        pairs = [(n, n) for n in names] + pairs
    jobs = [(a, b, metric_tuple) for a, b in pairs]

    rows: list[PairResult] = []
    total = len(jobs)
    if total == 0:
        return pd.DataFrame(
            columns=["graph_a", "graph_b", "metric", "raw_score", "shared_subgraph_score"]
        )

    if workers == 1 or total == 1:
        for i, job in enumerate(jobs, 1):
            rows.extend(_score_pair(job))
            if progress:
                progress(i, total)
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for i, chunk in enumerate(pool.imap_unordered(_score_pair, jobs), 1):
                rows.extend(chunk)
                if progress:
                    progress(i, total)

    frame = pd.DataFrame(
        rows, columns=["graph_a", "graph_b", "metric", "raw_score", "shared_subgraph_score"]
    )
    return frame.sort_values(["graph_a", "graph_b", "metric"], ignore_index=True)


def _stderr_progress(done: int, total: int) -> None:  # pragma: no cover - console only
    width = 28
    filled = int(width * done / total) if total else width
    sys.stderr.write(f"\r[{'#' * filled}{'.' * (width - filled)}] {done}/{total} pairs")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()
