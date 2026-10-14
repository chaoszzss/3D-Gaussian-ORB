#!/usr/bin/env python3
"""Train a real prepared block with a reproducible, memory-bounded profile."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orbvi_gaussian.training import confined_environment, splatfacto_command
from orbvi_gaussian.io import new_output_dir, output_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--profile", choices=["pilot", "quality"], default="pilot")
    p.add_argument("--allow-candidate-calibration", action="store_true")
    args = p.parse_args()
    command = splatfacto_command(args.dataset, args.output, iterations=args.iterations,
                                 allow_candidate_calibration=args.allow_candidate_calibration)
    marker = command.index("nerfstudio-data")
    extra = ["--viewer.quit-on-train-completion", "True", "--steps-per-save", "1000",
             "--steps-per-eval-image", "250", "--steps-per-eval-all-images", "1000",
             "--pipeline.model.num-downscales", "1", "--pipeline.model.resolution-schedule", "500" if args.profile == "pilot" else "1500",
             "--pipeline.model.stop-split-at", str(max(1000, args.iterations - 1000))]
    command[marker:marker] = extra
    out = new_output_dir(args.output)
    (out / "launch.json").write_text(json.dumps({"command": command, "profile": args.profile,
                                               "dataset": str(args.dataset.resolve()), "iterations": args.iterations,
                                               "calibration_experimental": args.allow_candidate_calibration}, indent=2))
    env = confined_environment()
    print("Training %s: %d steps, log %s" % (args.dataset, args.iterations, out / "train.log"), flush=True)
    with (out / "train.log").open("x") as log:
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    configs = [str(x.relative_to(ROOT)) for x in out.rglob("config.yml")]
    report = {"returncode": result.returncode, "status": "trained" if result.returncode == 0 else "failed",
              "configs": configs, "iterations_requested": args.iterations,
              "note": "Training completion is not map quality acceptance; evaluate held-out views."}
    (out / "training_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
