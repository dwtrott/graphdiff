#!/usr/bin/env bash
# graphdiff launcher for macOS / Linux / Git Bash. Run:  ./run-graphdiff.sh [folder-of-graphs]
# First run creates .venv next to this file and installs graphdiff; every run starts the app.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python && ! -x .venv/Scripts/python.exe ]]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3 python py; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then PY="$c"; break; fi
  done
  if [[ -z "$PY" ]]; then echo "No Python 3.11+ found; install it from https://www.python.org/downloads/"; exit 1; fi
  echo "using $PY; creating .venv and installing graphdiff (one-time)..."
  "$PY" -m venv .venv
fi
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe
"$PY" -c 'import graphdiff, fastapi, uvicorn' 2>/dev/null || "$PY" -m pip install -e ".[viewer,plot]" --quiet --disable-pip-version-check
if [[ $# -gt 0 ]]; then exec "$PY" -m graphdiff.cli app --workspace "$1"; else exec "$PY" -m graphdiff.cli app --demo; fi
