# Third-party simulators

`third_party/` is the reproducibility boundary for external architecture tools. Every source tree is a git submodule pinned to an exact commit; nested dependencies are initialized recursively and important nested pins are also recorded in `manifest.json`.

## Stack

| Module | Purpose | Default action |
|---|---|---|
| SCALE-Sim v3 | Tensor-array cycle/traffic golden | editable install |
| Ramulator2 | standalone modern DDR/HBM timing backend | CMake build + editable Python install |
| BookSim2 | cycle-accurate NoC contention | build `src/booksim` |
| Accelergy | action-count energy framework | editable install |
| HWComponents | hardware component models | editable install |
| HWComponents-CACTI | CACTI-backed SRAM/cache/DRAM models | editable install + nested CACTI build |
| Timeloop | optional mapper/tiling cross-check | source only by default |

SCALE-Sim's pinned source also contains its own legacy `CMU-SAFARI/ramulator` submodule for SCALE-Sim's internal integration. That is distinct from the top-level `third_party/ramulator2`, which is the standalone modern memory backend intended for the SPN CModel.

## Python versions

- CModel + core timing stack: Python >= 3.10.
- Complete stack including current HWComponents/HWComponents-CACTI pins: **Python >= 3.12**.

For the simplest reproducible setup, use Python 3.12 for the whole project.

## Clone

```bash
git clone --branch agent/add-architectural-cmodel --recurse-submodules \
  https://github.com/DingdongD/SPN_Accelerator.git
cd SPN_Accelerator
```

If already cloned:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Bootstrap

Complete stack, recommended with Python 3.12:

```bash
bash third_party/bootstrap.sh
python third_party/check.py
```

Core timing stack only, usable on Python 3.10/3.11:

```bash
bash third_party/bootstrap.sh --core
python third_party/check.py --core
```

Other modes:

```bash
bash third_party/bootstrap.sh --init-only
bash third_party/bootstrap.sh --no-build
```

The default build requires Git, `make`, CMake >=3.14, a C++20-capable compiler, `flex`, and `bison`. Ramulator2 is built through CMake. BookSim2 uses `flex`/`bison`. HWComponents-CACTI compiles the recursively pinned CACTI source during installation. Timeloop is initialized but not compiled automatically because upstream additionally requires system ISL/Barvinok.

## Updating pins

Dependency updates are explicit experiments:

1. move the submodule to the desired commit;
2. update `manifest.json`;
3. run `python third_party/check.py --pins-only`;
4. rerun calibration + hold-out validation;
5. commit the gitlink and manifest change together.

Never use `git submodule update --remote` for a reported experiment.
