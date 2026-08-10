from pathlib import Path

from spn_accel_cmodel import ConvMapping, NPUConfig, NPUSimulator, OffsetPattern, completionformer_dec2_nlspn_workload

ROOT = Path(__file__).resolve().parents[1]
config = NPUConfig.load(ROOT / "configs" / "baseline.json")
sim = NPUSimulator(config)
stats = sim.run(
    completionformer_dec2_nlspn_workload(prop_steps=12),
    ConvMapping(tile_oh=16, tile_ow=16, tile_cin=32, tile_cout=32),
    OffsetPattern("zero"),
)

print(f"total cycles: {stats.total_cycles:,}")
print(f"latency @ 1GHz: {stats.metadata['latency_ms']:.3f} ms")
for name, value in stats.tensor.items():
    print(f"{name:30s} {value.cycles:10,d} cycles util={value.utilization:.3f}")
for name, value in stats.spn.items():
    print(f"{name:30s} {value.cycles:10,d} cycles bank-conflict={value.bank_conflict_rate:.3f}")
