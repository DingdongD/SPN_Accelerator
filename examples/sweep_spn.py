from dataclasses import replace
from pathlib import Path

from spn_accel_cmodel.config import NPUConfig
from spn_accel_cmodel.spn_engine import OffsetPattern, SPNEngine
from spn_accel_cmodel.workload import Prop2DOp

ROOT = Path(__file__).resolve().parents[1]
base = NPUConfig.load(ROOT / "configs" / "baseline.json")
op = Prop2DOp("nlspn", 128, 128, neighbors=8, steps=1)

print("banks,gather_lanes,cycles,bank_conflict_rate")
for banks in (4, 8, 16, 32):
    for lanes in (8, 16, 32):
        state_cfg = replace(base.spn.state_sram, banks=banks)
        spn_cfg = replace(base.spn, gather_lanes=lanes, state_sram=state_cfg)
        stats = SPNEngine(spn_cfg).run(op, OffsetPattern("zero"))
        print(f"{banks},{lanes},{stats.cycles},{stats.bank_conflict_rate:.6f}")
