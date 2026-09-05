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

for f in report.findings:                      # plain-language, most important first
    print(f.severity, f.text)

report.score("jaccard_edges")                  # 0.8185
report.score("jaccard_edges", shared_subgraph=True)
report.edge_status_counts                      # {'SHARED': ..., 'CHANGED': ...}
report.top_changed(10)                         # most *significant* nodes (see below)
report.significance.table                      # every node, ranked, with the evidence
report.clusters                                # where the change is, by region
report.provenance                              # input hashes, parameters, versions

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
| Per-node neighborhood delta | how much of each node's neighbourhood turned over? |
| **Per-node significance** (binomial surprise) | **which changes actually matter?** |
| Cluster change density | *where* is the change — concentrated or diffuse? |

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

### Significance: which changes matter

Neighbourhood Jaccard is a *proportion*, and proportions rank a two-neighbour
node that lost one edge (0.5) above a 130-edge hub that lost thirty (0.77). For
"where are the significant changes" that is backwards, so ranking is by
**statistical surprise** instead: given the graph-wide rate of removals,
additions and reweightings, how unlikely is what happened to *this* node?

- A leaf losing its one edge when 7% of edges were removed is unremarkable.
- A hub losing thirty of 130 under that rate has a binomial tail around 1e-12.
- A removed hub — every edge gone — scores as the extreme event it is.

The background rate for each node is **leave-one-out** (its own edges excluded,
so a hub that lost everything in an otherwise static graph is not its own
baseline), and the score is `(1 + surprise) · log2(2 + magnitude)` where
`surprise = -log10 p` summed across the three tests, so a large change still
outranks a small one when neither is statistically unusual. Alongside it:
**impact** (edge changes weighted by PageRank in the union) and **centrality
shift** (PageRank in B minus A) for the node that quietly became, or stopped
being, a hub.

### Findings

Every report carries a short list of sentences a person could read aloud in a
briefing, each with its evidence and a pointer to where in the viewer to look:

> 94% similar by edit distance. 120 of 3,134 nodes and 1,413 of 10,358 edges differ …
> The change is concentrated: 3 of 28 clusters account for 94% of everything that differs …
> Cluster around c16-n014 (129 nodes) had 72% of its own structure change …
> 61 well-connected nodes (degree ≥ 5) disappeared from baseline — most connected: …
> Background rates: 7% of baseline's edges were removed, 6% of rebuilt's edges are new …

Findings lead the markdown summary, the JSON report, the notebook rendering and
the viewer. `compare(..., findings=False)` skips them and the clustering pass.

### Provenance

`report.provenance` records the SHA-256 and size of each input file, every
parameter the comparison used, and the versions of graphdiff, Python, numpy and
pandas, so a report can be tied back to exactly what produced it. It is in the
JSON and at the foot of the markdown.

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
edge weights — every status is represented. Small enough to read whole.

```python
from graphdiff.data import large_example_pair

a, b = large_example_pair()   # ~3,000 nodes, ~10,000 edges, 28 communities
```

A stochastic block model where the change is concentrated in three of the 28
communities plus a light scatter of noise — the shape real drift usually has,
and the case the Clusters view exists for. Use this one to judge the viewer; the
small pair fits in the Overview and makes every other view look redundant.

## CLI

```bash
graphdiff inspect A.graphml                       # counts, types, density, degree stats
graphdiff compare A.graphml B.graphml             # markdown summary to stdout
graphdiff compare A.graphml B.graphml --out report.json --markdown summary.md \
                                      --parquet scores.parquet --html diff.html --union-dir union/
graphdiff matrix ./graphs/ --metric jaccard_typed_edges --metric ged_similarity \
                           --out scores.parquet --workers 8
graphdiff render A.graphml B.graphml --out diff.html [--cluster-by kind]
graphdiff plot A.graphml B.graphml --out diff.png --kind dashboard   # static PNG/SVG/PDF
graphdiff serve diff.html --port 8080             # localhost only; nothing outbound

# regression gate: exit 1 when a rule fails
graphdiff check A.graphml B.graphml -r "jaccard_typed_edges>=0.95" -r "nodes.A_ONLY<=0"
graphdiff check A.graphml B.graphml --baseline last-report.json --drift all --tolerance 0.02
```

`check` turns a comparison into a pass/fail verdict for CI. Rules are
`quantity<op>number` where the quantity is any scalar metric (optionally
`.shared`), `nodes.<STATUS>` / `edges.<STATUS>` counts, or
`significance.max` / `.mean` / `.n_over_X`; `--drift` guards named quantities
(or `all`) against a stored report within `--tolerance`. `--json` emits the
verdict as data. The same thing is available in Python as
`graphdiff.report.check.run_checks`.

`matrix` scores every pair in a directory over a `multiprocessing` pool, with
progress on stderr, into the same tidy long format `compare --parquet` writes
(`graph_a, graph_b, metric, raw_score, shared_subgraph_score`). Each worker
caches the graphs it has loaded. `serve` is a stdlib HTTP server bound to
`127.0.0.1` serving a file that was already rendered with no external
references — it exists so a colleague can open the diff without a file share.

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

### The views

The page opens on the **Map** with the **Findings** panel beside it. The Map
draws the differences with node size and a halo proportional to significance
and labels only the most significant, so the eye lands on what matters before
anything is clicked. Its show-mode switch narrows from *everything* to
*changes only* to *significant only*; every finding is a link that switches to
the right view and selects the node or cluster it talks about.

**Regions** is the one that scales. The union is partitioned (Louvain by
default, or by any node attribute you name), and each partition is drawn as one
mark sized by membership and shaded by **change density** — the share of its own
nodes and internal edges that differ. A million nodes becomes twenty blobs, the
dark ones are where the change is, and clicking one drills the Overview down to
just that cluster. The shading is a sequential violet ramp, deliberately outside
the categorical blue/orange/green so an aggregate mark is never misread as a
status.

```python
write_html(report, "diff.html", cluster_by="kind")   # partition by an attribute
```


**Top changes** is the one to use when you already know roughly where to look. One small card per most-changed node, in the
same rank order as the analysis, each showing that node's neighbourhood.
Neighbours sit in fixed angular sectors — *removed left, added right, reweighted
below, unchanged above* — so the same kind of change lands in the same place on
every card, and a wall of cards is scannable instead of forty separate puzzles.
Cards are built from the whole union, never the drawn subset, so one is never
missing a neighbour the overview happened to cap away.

**Before / After** draws both graphs on *identical coordinates*. Nothing moves between
them, so anything that appears or vanishes is unmissable. A slider crossfades A
into B; **Flicker** alternates them automatically, which turns the diff into
motion — a far stronger perceptual channel than colour. **Split** puts them side
by side with linked pan and zoom.

**3D** is a separate three-dimensional force layout (depth is real, not a
random z), drawn with a plain perspective projection onto the canvas — no WebGL,
no library, nothing to bundle — with depth cueing, drag-to-rotate, auto-rotate,
and the same significance sizing, halos, labels and show modes as the Map, so
nothing is lost by switching. Dense graphs that are a hairball flat often
separate in depth.

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

### Static figures

With the `[plot]` extra, `graphdiff.plot` renders the same encoding to
PNG / SVG / PDF for slides, papers and CI artifacts: `plot_overview`,
`plot_clusters`, `plot_top_changed`, `plot_degree_distributions`,
`plot_similarity_matrix` (from a `matrix` table) and `plot_dashboard`, which
puts the map, the regions, the top-changed bars and the findings on one page.

```python
from graphdiff import plot
plot.save(plot.plot_dashboard(report), "diff.png")
```

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
