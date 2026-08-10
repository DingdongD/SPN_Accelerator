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
CORE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --init-only) INIT_ONLY=1 ;;
    --no-build) NO_BUILD=1 ;;
    --core) CORE_ONLY=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash third_party/bootstrap.sh [--init-only] [--no-build] [--core]

  --init-only  fetch every pinned submodule recursively and stop
  --no-build   fetch + Python installs but skip Ramulator2/BookSim2 C/C++ builds
  --core       install/build only the timing stack (SCALE-Sim, Ramulator2, BookSim2);
               energy sources remain pinned/initialized but HWComponents are not installed

Default mode prepares the complete stack and therefore requires Python >= 3.12.
Core mode supports Python >= 3.10.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }

python - "$CORE_ONLY" <<'PY'
import sys
core = bool(int(sys.argv[1]))
minimum = (3, 10) if core else (3, 12)
if sys.version_info < minimum:
    mode = "--core" if core else "full"
    raise SystemExit(
        f"{mode} third-party setup requires Python >= {minimum[0]}.{minimum[1]}; "
        f"current={sys.version.split()[0]}. Use a Python 3.12 venv for the complete stack, "
        "or pass --core on Python 3.10/3.11."
    )
PY

echo "[third_party] syncing pinned submodules"
git submodule sync --recursive
git submodule update --init --recursive

if [[ "$INIT_ONLY" == 1 ]]; then
  python third_party/check.py --pins-only
  exit 0
fi

echo "[third_party] Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "[third_party] installing SCALE-Sim from pinned source"
python -m pip install -e third_party/SCALE-Sim

if [[ "$CORE_ONLY" == 0 ]]; then
  echo "[third_party] installing pinned energy/component stack"
  python -m pip install -e third_party/accelergy
  python -m pip install -e third_party/hwcomponents
  # Avoid build isolation resolving an unpinned PyPI hwcomponents while the pinned
  # editable checkout is already installed in this environment.
  python -m pip install --no-build-isolation -e third_party/hwcomponents-cacti
else
  echo "[third_party] --core: energy packages are source-pinned but not installed"
fi

if [[ "$NO_BUILD" == 0 ]]; then
  command -v cmake >/dev/null || { echo "cmake >=3.14 is required for Ramulator2" >&2; exit 2; }
  command -v make >/dev/null || { echo "make is required" >&2; exit 2; }
  command -v "${CXX:-c++}" >/dev/null || { echo "a C++20-capable compiler is required for Ramulator2" >&2; exit 2; }
  command -v flex >/dev/null || { echo "flex is required to build BookSim2" >&2; exit 2; }
  command -v bison >/dev/null || { echo "bison is required to build BookSim2" >&2; exit 2; }

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
[third_party] SCALE-Sim's nested legacy Ramulator is pinned only for SCALE-Sim's
              own integration; third_party/ramulator2 is the standalone CModel backend.
EOF

if [[ "$CORE_ONLY" == 1 ]]; then
  python third_party/check.py --core
else
  python third_party/check.py
fi

echo "[third_party] setup complete"
