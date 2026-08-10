# Third-party simulators

This directory is the reproducibility boundary for external architecture tools used by the CModel. Sources are included as **git submodules pinned to exact commits**; calibration scripts must not silently follow upstream branches.

## Included modules

| Module | Purpose | Bootstrap |
|---|---|---|
| SCALE-Sim | Tensor-array cycle/traffic golden | installed editable |
| Ramulator2 | DDR/HBM cycle-level timing | CMake build + editable Python install |
| BookSim2 | NoC contention | `make` in `src/` |
| Accelergy | action-count energy framework | installed editable |
| HWComponents | component-model stack | installed editable |
| HWComponents-CACTI | SRAM/cache/DRAM CACTI models | installed editable; nested CACTI pulled recursively |
| Timeloop | optional mapper/tiling cross-check | source initialized; build is opt-in because system ISL/Barvinok dependencies are required |

Exact revisions are recorded in `manifest.json` and by the gitlink entries in the SPN_Accelerator commit.

## Clone correctly

Preferred:

```bash
git clone --recurse-submodules https://github.com/DingdongD/SPN_Accelerator.git
cd SPN_Accelerator
git checkout agent/add-architectural-cmodel
git submodule update --init --recursive
```

If the repository was cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Bootstrap

Activate the Python environment that should own the editable installs, then run:

```bash
bash third_party/bootstrap.sh
```

The default bootstrap initializes every pinned source and builds/installs the modules needed by the current calibration chain. Timeloop source is initialized but is not compiled automatically.

Useful modes:

```bash
bash third_party/bootstrap.sh --init-only   # only fetch pinned sources
bash third_party/bootstrap.sh --no-build    # fetch + Python installs, skip C/C++ builds
python third_party/check.py                 # verify pins and available tools
```

System prerequisites for the default build are a C/C++ compiler, `make`, CMake >= 3.14, Git, and Python >= 3.10. Ramulator2 2.1 requires a C++20-capable compiler. Timeloop additionally requires system ISL/Barvinok dependencies and is therefore deliberately excluded from the default build.

## Updating a dependency

Dependency updates are explicit experiments, not routine setup:

1. update the submodule checkout to the desired upstream commit;
2. update `manifest.json`;
3. rerun `python third_party/check.py`;
4. rerun the calibration hold-out suite;
5. commit the gitlink + manifest change together.

Do not use `git submodule update --remote` for reported experiments.
