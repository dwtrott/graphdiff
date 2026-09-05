#!/usr/bin/env bash
# Build a self-contained wheelhouse for installing graphdiff on an air-gapped host.
#
#   scripts/build_offline_bundle.sh [--python 3.11] [--platform manylinux2014_x86_64] [--extras plot,viewer]
#
# Run this on a machine WITH network access. It produces
#   dist/graphdiff-offline-<version>.tar.gz
# containing every wheel (graphdiff itself plus all dependencies, resolved for
# the target platform and Python), a pinned requirements file, the install
# script, the README and the example notebook. Copy the tarball across the
# air gap and run scripts/install_offline.sh there.
#
# Every dependency ships manylinux wheels, so nothing needs a compiler on the
# target; pass --platform for a different target (e.g. win_amd64) and pip will
# fetch that platform's wheels.
set -euo pipefail

PY_VERSION="3.11"
PLATFORM="manylinux2014_x86_64"
EXTRAS="plot,viewer"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)   PY_VERSION="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --extras)   EXTRAS="$2"; shift 2 ;;
    -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VERSION="$(python -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("graphdiff/__init__.py").read_text()).group(1))')"
STAGE="$(mktemp -d)"
BUNDLE="graphdiff-offline-${VERSION}"
WHEELHOUSE="${STAGE}/${BUNDLE}/wheelhouse"
mkdir -p "$WHEELHOUSE" dist

echo "==> building graphdiff ${VERSION} wheel"
python -m pip wheel . --no-deps --wheel-dir "$WHEELHOUSE" --quiet

echo "==> resolving dependencies for python ${PY_VERSION} / ${PLATFORM} (extras: ${EXTRAS})"
SPEC=".[${EXTRAS}]"
[[ -z "$EXTRAS" ]] && SPEC="."
python -m pip download "$SPEC" \
  --dest "$WHEELHOUSE" \
  --only-binary=:all: \
  --python-version "$PY_VERSION" \
  --platform "$PLATFORM" \
  --platform "any" \
  --implementation cp \
  --abi "cp${PY_VERSION/./}" \
  --abi abi3 \
  --abi none \
  --quiet
# pip download of "." also re-downloads graphdiff's build deps; drop non-wheels.
find "$WHEELHOUSE" -type f ! -name '*.whl' -delete

echo "==> writing pinned requirements"
( cd "$WHEELHOUSE" && ls *.whl | sed -E 's/^([A-Za-z0-9_.]+)-([0-9][^-]*)-.*\.whl$/\1==\2/' | sort -u ) \
  > "${STAGE}/${BUNDLE}/requirements.txt"

cp scripts/install_offline.sh README.md "${STAGE}/${BUNDLE}/"
mkdir -p "${STAGE}/${BUNDLE}/examples"
cp examples/graphdiff_demo.ipynb "${STAGE}/${BUNDLE}/examples/"
( cd "$WHEELHOUSE" && sha256sum *.whl ) > "${STAGE}/${BUNDLE}/SHA256SUMS"

echo "==> packing"
tar -C "$STAGE" -czf "dist/${BUNDLE}.tar.gz" "$BUNDLE"
rm -rf "$STAGE"
echo "wrote dist/${BUNDLE}.tar.gz ($(du -h "dist/${BUNDLE}.tar.gz" | cut -f1)) with $(ls dist >/dev/null; tar -tzf "dist/${BUNDLE}.tar.gz" | grep -c '\.whl$') wheels"
