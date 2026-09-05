# graphdiff

Graph similarity, alignment, and structural diffing for large property graphs.

`graphdiff` compares property graphs with **uniquely labeled nodes** and **typed,
optionally weighted edges**, at the scale of 10^5–10^6 nodes and edges. It is
built for **air-gapped deployment**: no network access at runtime, no CDN assets,
no lazy downloads, no telemetry.

## The one idea

Align A and B by node label, then materialize the result as a single **union diff
graph** in which every node and edge carries a `status`:

| status | meaning |
| --- | --- |
| `SHARED` | present in both, compared attributes equal |
| `A_ONLY` / `B_ONLY` | present in exactly one graph |
| `CHANGED` | same identity, differing attributes (old/new recorded) |

Every metric, report, and visualization is derived from that one structure.
Because node labels are unique, alignment is an exact join — nodes on `label`,
edges on `(source, type, target)` — so it is `O(n + m)` with no heuristic
matching and no combinatorial search.

## Install

```bash
pip install -e .            # development
pip install -e ".[plot]"    # + matplotlib helpers
```

Offline install from a prebuilt bundle:

```bash
pip install --no-index --find-links wheelhouse graphdiff
```

Every dependency ships a manylinux wheel; nothing compiles at install time.

## Library

```python
import graphdiff as gd

report = gd.compare("v1.graphml", "v2.graphml")

report.score("jaccard_edges")                  # 0.8185
report.score("jaccard_edges", shared_subgraph=True)
report.edge_status_counts                      # {'SHARED': ..., 'CHANGED': ...}
report.neighborhood.top_changed(10)            # ranked most-changed nodes

report.to_json("report.json")
report.to_parquet("scores.parquet")            # tidy long format
report.write_markdown("summary.md")
```

In Jupyter, a `ComparisonReport` renders as a compact summary table
(`_repr_html_`). The union diff graph is available for your own analysis:

```python
nodes, edges = report.union.to_dataframes()
rx_graph = report.union.to_rustworkx()
ig_graph = report.union.to_igraph()
```

### Raw vs shared-subgraph scores

Every metric is reported twice: over the full union (`raw`), and over the
subgraph induced on nodes present in **both** graphs (`shared_subgraph`).
Comparing the two separates *"these graphs differ in size"* from *"the
overlapping part differs in structure"* — essential when A and B are of
different scales.

## Metrics

| Metric | What it answers |
| --- | --- |
| Set-theoretic — Jaccard, overlap, Dice on node / edge / typed-edge sets | how much of each element kind is common? |
| Exact normalized graph edit distance | what does it cost to turn A into B? |
| Weighted-edge agreement (Spearman over shared edges) | do the graphs agree on which edges are strong? |
| Per-node neighborhood delta | **which entities changed the most?** |

Set-theoretic scores are reported **separately** per element kind and never
blended into one number by default: a pair can share every node while sharing no
edges, and collapsing that into a single figure hides exactly what you are
looking for.

Graph edit distance accepts per-operation and per-edge-type costs:

```python
from graphdiff import GEDCosts

costs = GEDCosts(edge_type_costs={"owns": 10.0, "mentions": 0.5})
report = gd.compare(a, b, ged_costs=costs)
```

## Supported formats

GraphML, node-link JSON, edge-list CSV/TSV, and Parquet, with auto-detection by
extension and then by content:

```python
graph = gd.read_graph("graph.graphml")   # detected
graph = gd.read_graph("edges.csv", directed=False)
gd.write_graph(graph, "out.parquet")
```

GraphML is parsed directly (streaming `iterparse`, no networkx). Edge-list
headers accept the common aliases (`src`/`dst`, `from`/`to`, `relation`).

## Example data

```python
from graphdiff.data import example_pair

a, b = example_pair()   # two snapshots of a named-entity knowledge graph
```

52 and 53 nodes of people, organizations and topics with community structure,
differing by departed and arrived entities, rewired relationships and drifted
edge weights — every status is represented.

## CLI

*Not yet implemented.* Planned surface:

```bash
graphdiff compare A.graphml B.graphml --out report.json [--markdown summary.md]
graphdiff matrix ./graphs/ --metric jaccard_edges --out scores.parquet --workers 8
graphdiff inspect A.graphml
graphdiff serve report.json --port 8080
```

## Visual diff

```python
import graphdiff as gd
from graphdiff.viewer import write_html

report = gd.compare("v1.graphml", "v2.graphml")
write_html(report, "diff.html")      # open it in any browser
```

The output is **one self-contained HTML file**. Layout is computed here in
Python and the coordinates are baked into the page, so it ships no layout
library, loads no fonts, scripts or stylesheets, and makes no network requests
of any kind. Copy it to an air-gapped machine and double-click it.

**Status is encoded twice — hue and node shape** — so nothing depends on colour
alone and the picture survives greyscale printing and colour-vision deficiency:

| Status | Shape | Light | Dark | Edge |
| --- | --- | --- | --- | --- |
| In both | circle | grey `#a8afb8` | `#6b7079` | hairline, faded |
| Only in A | square | blue `#2a78d6` | `#3987e5` | solid |
| Only in B | triangle | orange `#eb6834` | `#d95926` | solid |
| Changed | diamond | green `#199e70` | `#199e70` | dashed |

The legend names the graphs rather than the status codes — "Only in
snapshot_2024", not "A_ONLY". The palette is validated rather than eyeballed:
all-pairs CVD ΔE 8.4 light / 9.4 dark, normal-vision ΔE 21.6 / 20.9, every hue
≥ 3:1 against its surface. (A magenta `CHANGED` was tried first and rejected —
ΔE 12.9 against orange for normal vision, below the 15 floor.)

Interaction: hover or click any node and everything unrelated dims, leaving that
node's neighbourhood; the side panel then names its added, removed and changed
relationships. **Differences only** hides the shared scaffolding. Search by
label, toggle any status, click a row of the most-changed table to fly to that
node, `Esc` to clear. The theme follows your OS and has a manual toggle.

### Three views

The page opens on **Overview** — the whole union graph — but that view answers
"how much changed", not "where". Two more views answer *where*:

**Changes** is the one to use first. One small card per most-changed node, in the
same rank order as the analysis, each showing that node's neighbourhood.
Neighbours sit in fixed angular sectors — *removed left, added right, reweighted
below, unchanged above* — so the same kind of change lands in the same place on
every card, and a wall of cards is scannable instead of forty separate puzzles.
Cards are built from the whole union, never the drawn subset, so one is never
missing a neighbour the overview happened to cap away.

**Compare** draws both graphs on *identical coordinates*. Nothing moves between
them, so anything that appears or vanishes is unmissable. A slider crossfades A
into B; **Flicker** alternates them automatically, which turns the diff into
motion — a far stronger perceptual channel than colour. **Split** puts them side
by side with linked pan and zoom.

```python
write_html(report, "diff.html", max_cards=60, max_card_neighbors=30)
```

At scale the page draws a **focus subgraph** rather than everything: all
differing elements, plus a ring of unchanged context (`context_hops`), capped at
`max_nodes` with differences kept ahead of context. The header always states
how much of the union is actually on screen.

```python
write_html(report, "diff.html", max_nodes=4000, context_hops=2)
write_html(report, "changes-only.html", context_hops=0)
```

`tests/test_viewer.py` enforces the offline guarantee: the build fails if any
external URL, remote-resource tag, or network primitive appears in the output.

### Interactive server viewer

*Not yet built.* The Vite + React + Cosmograph + FastAPI viewer from the spec
remains an option for live exploration; the static export above covers reviewing
and sharing a single comparison, which is the common case.

## Development

```bash
pytest              # full suite
pytest -m "not slow"   # skip the 500k-edge performance test
ruff check . && ruff format --check .
mypy graphdiff
```

### Performance

`compare()` on two ~500k-edge graphs sharing 200k node labels runs in about
**11 s** on a development machine. `tests/test_performance.py` guards this with
a deliberately loose 90 s budget — it exists to catch an accidental quadratic or
a per-row Python loop, not to benchmark.

The hot paths deliberately avoid two pandas traps at this scale: `Series.isin`
on Arrow-backed string columns (which falls back to a Python listcomp — see
`isin_labels`), and round-tripping label columns between Arrow strings and numpy
object arrays.

## Design notes

- **Duplicate edges are an error by default.** A repeated `(source, type,
  target)` triple means the input does not match the model this tool assumes;
  silently dropping rows in a diff tool hides real differences. Pass
  `on_duplicate_edge="first"` to override.
- **Undirected graphs** canonicalize endpoints so `source <= target`; a graph
  cannot be compared with one of different directedness.
- **Float attributes** compare with `numpy.isclose` tolerances (configurable via
  `AttributeComparison`) so serialization round-trips do not read as changes.
- **`CHANGED` requires presence on both sides.** A one-sided element is absent,
  not different, and carries no `changed_attrs`.
