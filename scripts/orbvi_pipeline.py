#!/usr/bin/env python3
"""ORBVI reconstruction input CLI. ROS stages use system Python, no roscore needed."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbvi_gaussian.io import PROJECT_ROOT, extract_bag, inspect_bag, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_bag = PROJECT_ROOT.parent / "orbvi_4vl_data_czy_all.bag"
    inspect = sub.add_parser("inspect", help="Fast connection/first-payload audit")
    inspect.add_argument("--bag", type=Path, default=default_bag)
    inspect.add_argument("--output", type=Path)
    extract = sub.add_parser("extract", help="Stream images + complete sensor/pose indexes")
    extract.add_argument("--bag", type=Path, default=default_bag)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--camera-hz", type=float, default=2.0)
    extract.add_argument("--ipm-hz", type=float, default=0.2)
    extract.add_argument("--cloud-hz", type=float, default=0.5)
    extract.add_argument("--start-s", type=float, default=0.0)
    extract.add_argument("--duration-s", type=float)
    extract.add_argument("--include-clouds", action="store_true")
    candidate = sub.add_parser("import-calibration", help="Import TripleSphere board candidates, leave unverified")
    candidate.add_argument("--camera-config", required=True, type=Path)
    candidate.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("check-calibration", help="Validate all required geometry/time contracts")
    check.add_argument("--calibration", required=True, type=Path)
    preview = sub.add_parser("rectify-preview", help="Diagnostic views from candidate intrinsics; no poses")
    preview.add_argument("--extracted", required=True, type=Path)
    preview.add_argument("--calibration", required=True, type=Path)
    preview.add_argument("--output", required=True, type=Path)
    quality = sub.add_parser("trajectory-qa", help="Reject implausible motion and numerical VIO divergence")
    quality.add_argument("--pose-csv", required=True, type=Path)
    quality.add_argument("--output", type=Path)
    poses = sub.add_parser("export-poses", help="Import matching native LIO Odometry from an evidence bag")
    poses.add_argument("--pose-bag", required=True, type=Path)
    poses.add_argument("--pose-topic", required=True)
    poses.add_argument("--extracted", required=True, type=Path)
    poses.add_argument("--output", required=True, type=Path)
    blocks = sub.add_parser("plan-blocks", help="Provisional arclength/time blocks; no inferred floor labels")
    blocks.add_argument("--extracted", required=True, type=Path)
    blocks.add_argument("--output", required=True, type=Path)
    blocks.add_argument("--pose-csv", type=Path)
    blocks.add_argument("--length-m", type=float, default=30.0)
    blocks.add_argument("--overlap-m", type=float, default=5.0)
    blocks.add_argument("--max-duration-s", type=float, default=90.0)
    prepare = sub.add_parser("prepare", help="Calibrated virtual images + metric COLMAP/Nerfstudio")
    for name in ["extracted", "calibration", "output"]:
        prepare.add_argument("--" + name, required=True, type=Path)
    prepare.add_argument("--pose-csv", type=Path)
    prepare.add_argument("--sync-ms", type=float, default=10.0)
    prepare.add_argument("--max-gap-ms", type=float, default=250.0)
    prepare.add_argument("--holdout-every", type=int, default=8)
    prepare.add_argument("--max-groups", type=int, default=160, help="0 = all groups; start with 160 on 8 GB GPU")
    prepare.add_argument("--min-sharpness", type=float, default=0.0)
    prepare.add_argument("--block-plan", type=Path)
    prepare.add_argument("--block-id")
    prepare.add_argument("--initialization-map", type=Path)
    prepare.add_argument("--map-world-frame")
    prepare.add_argument("--map-alignment-verified", action="store_true")
    prepare.add_argument("--map-voxel-m", type=float, default=0.1)
    prepare.add_argument("--max-points", type=int, default=100000)
    prepare.add_argument("--save-depth", action="store_true")
    prepare.add_argument("--map-rgb-verified", action="store_true", help="RGB field is measured image color, not display pseudocolor")
    prepare.add_argument("--colorize-from-images", action="store_true", help="Color visible initialization points using training groups only")
    prepare.add_argument("--allow-candidate-calibration", action="store_true", help="Explicit experimental dataset; never relabels calibration as verified")
    prepare.add_argument("--separate-test", action="store_true", help="Divide held-out groups into dev and test, preserving training-only initializer colors")
    prepare.add_argument("--global-split", action="store_true", help="Use one assignment for all source groups, including overlapping blocks")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    options = vars(args).copy()
    command = options.pop("command")
    if command == "inspect":
        result = inspect_bag(options["bag"])
        if options["output"]:
            write_json(options["output"], result)
    elif command == "extract":
        options["path"] = options.pop("bag")
        result = extract_bag(**options)
    elif command == "import-calibration":
        from orbvi_gaussian.calibration import import_opencv_candidate
        result = import_opencv_candidate(options["camera_config"], options["output"])
    elif command == "check-calibration":
        from orbvi_gaussian.calibration import load_calibration
        c = load_calibration(options["calibration"])
        result = {"status": "verified_schema", "calibration_id": c["calibration_id"]}
    elif command == "trajectory-qa":
        from orbvi_gaussian.workflow import load_trajectory
        result = load_trajectory(options["pose_csv"], require_plausible=False).quality_report
        if options["output"]:
            write_json(options["output"], result)
    elif command == "export-poses":
        from orbvi_gaussian.trajectory import export_pose_bag
        result = export_pose_bag(**options)
    else:
        from orbvi_gaussian.workflow import plan_blocks, prepare_dataset, rectify_preview
        function = {"plan-blocks": plan_blocks, "prepare": prepare_dataset, "rectify-preview": rectify_preview}[command]
        result = function(**options)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
