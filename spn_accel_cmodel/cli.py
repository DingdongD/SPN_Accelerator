from __future__ import annotations

import argparse
import json

from .config import NPUConfig
from .simulator import NPUSimulator, completionformer_dec2_nlspn_workload
from .spn_engine import OffsetPattern
from .tensor_engine import ConvMapping


def main() -> None:
    p = argparse.ArgumentParser(description="SPN Accelerator architectural CModel")
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--prop-steps", type=int, default=12)
    p.add_argument("--offset-pattern", choices=("zero", "random", "checker"), default="zero")
    p.add_argument("--offset-amplitude", type=float, default=0.75)
    p.add_argument("--tile-oh", type=int, default=16)
    p.add_argument("--tile-ow", type=int, default=16)
    p.add_argument("--tile-cin", type=int, default=32)
    p.add_argument("--tile-cout", type=int, default=32)
    args = p.parse_args()

    config = NPUConfig.load(args.config)
    simulator = NPUSimulator(config)
    stats = simulator.run(
        completionformer_dec2_nlspn_workload(args.prop_steps),
        ConvMapping(args.tile_oh, args.tile_ow, args.tile_cin, args.tile_cout),
        OffsetPattern(args.offset_pattern, amplitude=args.offset_amplitude),
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
