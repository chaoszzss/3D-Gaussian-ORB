#!/usr/bin/env python3
"""Print a reviewed training command; --execute tests CUDA/gsplat then trains."""

import argparse
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orbvi_gaussian.training import launch_splatfacto, splatfacto_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ns-train", type=Path)
    p.add_argument("--iterations", type=int, default=7000)
    p.add_argument("--allow-random-init", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-candidate-calibration", action="store_true")
    args = p.parse_args()
    command = splatfacto_command(args.dataset, args.output, args.ns_train,
                                 args.iterations, args.allow_random_init, args.allow_candidate_calibration)
    print(shlex.join(command), flush=True)
    return launch_splatfacto(command) if args.execute else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
