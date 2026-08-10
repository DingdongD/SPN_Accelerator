#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-$(python - <<'PY'
import os
print(max(1, os.cpu_count() or 1))
PY
)}"
INIT_ONLY=0
NO_BUILD=0

for arg in "$@"; do
  case "$arg" in
    --init-only) INIT_ONLY=1 ;;
    --no-build) NO_BUILD=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash third_party/bootstrap.sh [--init-only] [--no-build]

  --init-only  fetch all pinned submodules recursively and stop
  --no-build   fetch sources and install Python-only tools; skip C/C++ builds
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }

echo "[third_party] syncing pinned submodules"
git submodule sync --recursive
git submodule update --init --recursive

if [[ "$INIT_ONLY" == 1 ]]; then
  python third_party/check.py --pins-only
  exit 0
fi

echo "[third_party] installing Python simulator packages into: $(python -c 'import sys; print(sys.executable)')"
python -m pip install -e third_party/SCALE-Sim
python -m pip install -e third_party/accelergy
python -m pip install -e third_party/hwcomponents
python -m pip install -e third_party/hwcomponents-cacti

if [[ "$NO_BUILD" == 0 ]]; then
  command -v cmake >/dev/null || { echo "cmake is required for Ramulator2" >&2; exit 2; }
  command -v make >/dev/null || { echo "make is required for BookSim2" >&2; exit 2; }

  echo "[third_party] building Ramulator2"
  cmake -S third_party/ramulator2 -B third_party/ramulator2/build -DCMAKE_BUILD_TYPE=Release
  cmake --build third_party/ramulator2/build -j "$JOBS"
  python -m pip install -e third_party/ramulator2

  echo "[third_party] building BookSim2"
  make -C third_party/booksim2/src -j "$JOBS"
else
  echo "[third_party] --no-build: skipping Ramulator2 and BookSim2 compilation"
fi

cat <<'EOF'
[third_party] Timeloop source is pinned and initialized, but is not built by default.
              Its upstream build requires system ISL/Barvinok dependencies.
EOF

python third_party/check.py

echo "[third_party] setup complete"
