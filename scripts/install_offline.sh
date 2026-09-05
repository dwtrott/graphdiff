#!/usr/bin/env bash
# Install graphdiff from an offline bundle. Run on the air-gapped host:
#
#   tar -xzf graphdiff-offline-<version>.tar.gz
#   cd graphdiff-offline-<version>
#   ./install_offline.sh [python-executable]
#
# Uses only the wheels in ./wheelhouse; never touches the network.
set -euo pipefail
PY="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> verifying wheel checksums"
( cd wheelhouse && sha256sum --check --quiet ../SHA256SUMS )

echo "==> installing with $("$PY" --version)"
"$PY" -m pip install --no-index --find-links wheelhouse --requirement requirements.txt --quiet
"$PY" -m pip install --no-index --find-links wheelhouse --no-deps graphdiff --quiet

echo "==> smoke test (offline)"
"$PY" - <<'PYEOF'
import graphdiff as gd
from graphdiff.data import example_pair
a, b = example_pair()
r = gd.compare(a, b)
assert r.findings and r.score("ged_similarity") is not None
print(f"graphdiff {gd.__version__} OK — similarity {r.score('ged_similarity'):.3f}")
PYEOF
echo "done. try:  graphdiff --help"
