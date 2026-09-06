# graphdiff — quickstart

Five minutes from a fresh clone to a comparison on screen. Everything runs on
your own machine; nothing is downloaded after the install.

## 0. Where things are

| what | where |
| --- | --- |
| the code | this folder (the git repo; `graphdiff/` inside it is the Python package) |
| the web app | `graphdiff app` — a local page at http://127.0.0.1:8765/ |
| example datasets | `graphdiff demo` writes them to `./graphdiff-demo/` (or `graphdiff app --demo` does it for you) |
| your own graphs | any folder; point `graphdiff app --workspace <folder>` at it, or drop files onto the page |
| results | inside the workspace, under `.graphdiff/jobs/<id>/` — viewer page, JSON, markdown, PNG |

## 1. The no-terminal way (Windows)

Double-click **`run-graphdiff.bat`** in this folder. The first time it finds a
Python 3.11+ on your machine — a python.org install, or **Anaconda /
Miniconda** — builds a private environment in `.venv` next to it and installs
graphdiff (a few minutes, one time). Every time after that it just starts the
app on the demo datasets and opens your browser. Close the window to stop.

To run it on your own folder of graphs, drag the folder onto the `.bat`, or in
a terminal: `run-graphdiff.bat C:\path\to\graphs`. macOS/Linux/Git Bash:
`./run-graphdiff.sh`.

If the window ends with a red `[!]` line, screenshot it and send it to me —
that is the fastest way to get unstuck. The rest of this section is the same
setup done by hand.

## 1b. Install by hand

You need **Python 3.11 or newer**. On an **Anaconda** machine, do this from
an *Anaconda Prompt* (Start menu), and use `conda create -p .venv python=3.11 pip`
then `conda activate .\.venv` in place of the `venv`/activate lines below. Check with `python --version` (on Windows,
`py --version` also works). If it is older or missing, install from
https://www.python.org/downloads/ and tick *Add python.exe to PATH*.

Open a terminal **in this folder** (PowerShell, Command Prompt, or Git Bash —
in File Explorer, right-click the folder → *Open in Terminal*), then:

```bash
python -m venv .venv
```

Activate the environment — this is the one step that differs by shell:

```powershell
.venv\Scripts\Activate.ps1        # PowerShell
```
```bat
.venv\Scripts\activate.bat        # Command Prompt
```
```bash
source .venv/Scripts/activate     # Git Bash on Windows
source .venv/bin/activate         # macOS / Linux
```

(If PowerShell refuses with an *execution policy* error, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or just use Git
Bash.) Then install graphdiff with the web app and plotting extras:

```bash
pip install -e ".[viewer,plot,dev]"
graphdiff --version
```

## 2. Run it

```bash
graphdiff app --demo
```

That writes the example datasets to `./graphdiff-demo/`, starts the app, and
opens your browser. In the page:

1. In **Graphs**, click **A** next to `baseline.graphml` and **B** next to
   `rebuilt.graphml`, then **Compare A → B**. Wait for the stage line to
   finish (about 20–30 s for this size, mostly the viewer layout).
2. Read the **Findings** on the right of the viewer, then click a finding —
   it jumps to the place it talks about.
3. Try **A** = `baseline`, **B** = `rebuilt-renamed`, and set *Node matching*
   to **fuzzy**: a tenth of the labels differ, and the matcher recovers them
   with a confidence on each (dashed rings in the map).
4. Click **T** on `t0` … `t5` in order and press **Build timeline** to see when
   the change happened, where it keeps happening, and what flickered.

Stop the app with Ctrl+C in the terminal. Next time, just
`graphdiff app --demo` (or `--workspace <your folder>`) again after
activating the environment.

## 3. Use your own graphs

Drop files onto the **Graphs** panel, or put them in a folder and start with
`graphdiff app --workspace C:\path\to\that\folder`. Accepted formats:

- **GraphML** (`.graphml`)
- **node-link JSON** (`{"nodes": [{"id": …}], "links": [{"source": …, "target": …}]}`)
- **edge list** CSV/TSV with a header: `source,target[,type][,weight][,…]`
- **Parquet**: a folder with `edges.parquet` (and optionally `nodes.parquet`)

The one requirement is that node labels are unique within a graph. If the
same entity is named slightly differently in the two files, use *fuzzy*
matching.

If you do not have real data yet, some public graph collections that export
to edge lists: SNAP (https://snap.stanford.edu/data/), Network Repository
(https://networkrepository.com/), and the KONECT collection — pick two
snapshots of the same network, or one network and a perturbed copy.

## 4. Without the browser

```bash
graphdiff compare A.graphml B.graphml                       # markdown summary to the terminal
graphdiff compare A.graphml B.graphml --html diff.html      # the viewer as one file you can send
graphdiff timeline t0.graphml t1.graphml t2.graphml --html timeline.html
graphdiff check A.graphml B.graphml -r "ged_similarity>=0.9"  # exit 1 if it fails (CI)
```

And in Python:

```python
import graphdiff as gd
report = gd.compare("A.graphml", "B.graphml", align="fuzzy")
for f in report.findings: print(f.text)
report.top_changed(10)
```

`examples/graphdiff_demo.ipynb` is the same tour as a notebook.

## 5. Checking the code works

```bash
pytest -m "not slow"     # ~2 minutes
```

## Troubleshooting

- **`graphdiff` is not recognized** — the virtual environment is not active in
  this terminal; run the activate line from step 1 again.
- **`No module named fastapi`** — install the extras: `pip install -e ".[viewer,plot]"`.
- **The page opens but a job stays "running" for minutes** — large graphs take
  time to lay out; the stage line says what it is doing. Lower *Max nodes
  drawn* under *Options → more* for a faster picture.
- **Port already in use** — `graphdiff app --demo --port 8800`.
