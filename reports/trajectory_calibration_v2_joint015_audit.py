"""CPU audit of the requested v3 joint015 inputs/output; never writes datasets."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-4:]))
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
import prepare_joint_blocks as joint
cv2.setNumThreads(4)

SOURCE = ROOT / "data/garage_fidelity_depthv3_overlap015"
OUTPUT = ROOT / "data/garage_fidelity_depthv3_joint/core_015"
PREFIX = ROOT / "reports/trajectory_calibration_v2_joint015"
IDS = ["block_014", "block_015", "block_016"]
SPLITS = joint.SPLITS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(suffix, data):
    path = Path(str(PREFIX) + suffix + ".json")
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(frame):
    return str(frame["group_id"]), frame["camera_id"]


def nearest_unknown(reference, frame):
    """Independent two-minimum oracle; no call to either teacher projector.

    Measured camera-Z uses the approved float32 z-buffer tie convention. A
    nearest pixel is trusted if ANY equally-near valid-normal return exists;
    using the first input row would falsely flag ties between overlapping maps.
    """
    w, h = frame["w"], frame["h"]
    T = np.asarray(frame["transform_matrix"], float) @ np.diag([1., -1., -1., 1.])
    nearest = np.full(w*h, np.inf, np.float32)
    trusted = np.full(w*h, np.inf, np.float32)
    for start in range(0, len(reference["xyz"]), 65536):
        xyz = reference["xyz"][start:start+65536].astype(np.float64)
        normals = reference["normals"][start:start+65536].astype(np.float64)
        normal_ok = (reference["normal_valid"][start:start+65536]
                     & np.isfinite(normals).all(1) & (np.linalg.norm(normals, axis=1) > 1e-8))
        camera = (xyz-T[:3, 3]) @ T[:3, :3]
        good = np.isfinite(camera).all(1) & (camera[:, 2] > 0)
        camera, normal_ok = camera[good], normal_ok[good]
        # Use the producer's half-integer NS -> integer-centre CV conversion.
        K = np.array([[frame["fl_x"], 0., frame["cx"]-.5],
                      [0., frame["fl_y"], frame["cy"]-.5], [0., 0., 1.]])
        projected = camera @ K.T
        uv = projected[:, :2] / projected[:, 2:3]
        inside = (np.isfinite(uv).all(1) & (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5)
                  & (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5))
        camera, uv, normal_ok = camera[inside], uv[inside], normal_ok[inside]
        ij = np.floor(uv+.5).astype(np.int64)
        pid = ij[:, 1]*w + ij[:, 0]
        z = camera[:, 2].astype(np.float32)
        np.minimum.at(nearest, pid, z)
        valid_normal = normal_ok & (camera[:, 2] > .1)
        np.minimum.at(trusted, pid[valid_normal], z[valid_normal])
    measured = np.isfinite(nearest)
    return measured.reshape(h, w), (measured & (nearest != trusted)).reshape(h, w)


def main():
    started = time.monotonic()
    require(not (OUTPUT / "INCOMPLETE.json").exists(), "Joint dataset is incomplete")
    prep = read(OUTPUT / "preparation.json")
    meta = read(OUTPUT / "transforms.json")
    require(prep["status"] == "prepared_experimental", "Joint status is not prepared")
    hashes = {}
    policy, approval = joint.validate_depth_approval(
        ROOT / "reports/resolution_pilot_v2_depthv3_overlap015_evidence_archive.json", hashes)
    require(joint.canonical(approval["policy"]) == joint.canonical(prep["teacher_provenance"]["policy"]), "Joint policy differs")
    require(prep["reference_voxel_m"] is None, "V3 reference lost unknown-normal occluders to voxels")
    files = {str(OUTPUT / name): sha(OUTPUT / name) for name in ("preparation.json", "transforms.json", "lidar_reference.npz")}
    for path, expected in prep["source_hashes"].items():
        require(sha(ROOT / path) == expected, "Pinned generation input changed: " + path)
    for name, expected in prep["generated_hashes"].items():
        require(sha(OUTPUT / name) == expected, "Generated output changed: " + name)
    frames = {key(f): f for f in meta["frames"]}
    require(len(frames) == len(meta["frames"]), "Joint contains duplicate camera/group")
    memberships, input_memberships, observations = {}, {}, {}
    for split in SPLITS:
        for name in meta[split]:
            require(name not in memberships, "Joint split leakage")
            memberships[name] = split
    with np.load(OUTPUT / "lidar_reference.npz", allow_pickle=False) as archive:
        ref = {k: archive[k] for k in archive.files}
    source_counts, offset = {}, 0
    for block_index, bid in enumerate(IDS):
        base = SOURCE / bid
        original = read(base / "transforms.json")
        for split in SPLITS:
            for name in original[split]:
                require(memberships.get(name) == split, "Source global split changed")
                require(name not in input_memberships or input_memberships[name] == split, "Input split conflict")
                input_memberships[name] = split
        for f in original["frames"]:
            actual = frames[key(f)]
            require(actual == f, "Input frame metadata/pose/timestamp changed")
            for asset in ("file_path", "mask_path"):
                require(sha(base / f[asset]) == sha(OUTPUT / actual[asset]), "RGB/mask bytes changed")
            observations.setdefault(key(f), []).append(bid)
        with np.load(base / "lidar_reference.npz", allow_pickle=False) as source_ref:
            count = len(source_ref["xyz"])
            for field in ("xyz", "normals", "normal_valid"):
                require(np.array_equal(source_ref[field], ref[field][offset:offset+count]), "Merged reference changed input record")
            require(np.all(ref["source_block_index"][offset:offset+count] == block_index), "Reference origin block differs")
            require(np.array_equal(ref["source_row_index"][offset:offset+count], np.arange(count)), "Reference origin rows differ")
        offset += count
        source_counts[bid] = dict(frames=len(original["frames"]), reference_points=count,
                                  splits={s: len(original[s]) for s in SPLITS})
    require(offset == len(ref["xyz"]), "Reference contains missing/extra rows")
    require(input_memberships == memberships, "Joint split union differs")
    fixed = ROOT / "data/garage_fidelity_depthv3/block_015"
    fixed_meta = read(fixed / "transforms.json")
    fixed_lookup = {f["file_path"]: f for f in fixed_meta["frames"]}
    fixed_dev = [fixed_lookup[name] for name in fixed_meta["val_filenames"]]
    for f in fixed_dev:
        require(frames[key(f)] == f and memberships[f["file_path"]] == "val_filenames", "Fixed highres015 DEV changed")
        for field in ("file_path", "mask_path"):
            require(sha(fixed / f[field]) == sha(OUTPUT / f[field]), "Fixed highres015 DEV image/mask changed")
    plan_path = ROOT / "reports/acceptance_plan_v1.json"
    plan = read(plan_path)
    old_dev = [f for f in plan["frames"] if f["block_id"] == "block_015"]
    require({key(f) for f in fixed_dev} == {key(f) for f in old_dev}, "Original frozen640 DEV identities differ")
    for old in old_dev:
        f, camera = frames[key(old)], old["camera"]
        require(f["transform_matrix"][:3] == camera["camera_to_worlds_ns"], "Frozen640 DEV pose differs")
        for name in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
            require(f[name] == 2*camera[name], "1280 camera is not exact 2x NS intrinsics")
    shared = [{"group_id": k[0], "camera_id": k[1], "source_blocks": v}
              for k, v in observations.items() if len(v) > 1]
    closure = dict(status="PASS", source_counts=source_counts, joint_frames=len(frames),
        joint_splits={s: len(meta[s]) for s in SPLITS}, global_split_union_exact=True,
        frame_metadata_poses_timestamps_exact=True, shared_rgb_mask_bytes_exact=True,
        shared_frames=len(shared), duplicate_observations_removed=sum(len(v)-1 for v in observations.values()),
        shared_frame_sources=shared, fixed_highres015_dev_frames=len(fixed_dev),
        fixed_highres015_dev_metadata_rgb_mask_exact=True,
        frozen_original640_dev_identity_pose_exact=True, frozen_original640_camera_exact_scale=2,
        fixed_original_plan_id=plan["plan_id"], merged_reference_points=len(ref["xyz"]),
        all_reference_records_preserved=True, independent_geometry_ground_truth=False,
        scoring_instruction="Evaluate the joint checkpoint with dataset_override to the chosen fixed original015 scoring dataset; joint DEV has additional frames",
        source_sha256=files, gpu_used=False, cpu_affinity=sorted(os.sched_getaffinity(0)))
    closure_path = save("_closure", closure)
    print("Closure PASS:", len(frames), "frames;", len(shared), "shared camera/group identities", flush=True)
    teacher_rows = {r["file_path"]: r for r in prep["teacher_provenance"]["frames"]}
    totals, rows, replay = {}, [], []
    fixed_keys = {key(f) for f in fixed_dev}
    for i, f in enumerate(meta["frames"]):
        image_mask = cv2.imread(str(OUTPUT / f["mask_path"]), 0)
        support_raw = cv2.imread(str(OUTPUT / f["lidar_depth_mask_path"]), 0)
        support = support_raw > 0
        depth = np.load(OUTPUT / f["lidar_depth_path"], allow_pickle=False)
        evidence = teacher_rows[f["file_path"]]
        require(sha(OUTPUT / f["lidar_depth_path"]) == evidence["depth_sha256"], "Teacher bytes changed")
        require(sha(OUTPUT / f["lidar_depth_mask_path"]) == evidence["support_sha256"], "Teacher mask bytes changed")
        require(np.isin(support_raw, [0, 255]).all() and np.isfinite(depth).all(), "Invalid teacher/support")
        require(np.all(depth[~support] == 0) and np.all(depth[support] > 0), "Depth/support mismatch")
        require(not np.any(support & (image_mask == 0)), "Teacher escaped fixed environment mask")
        measured, unknown = nearest_unknown(ref, f)
        environment = image_mask > 0
        counts = dict(environment_pixels=int(environment.sum()), measured_environment_pixels=int((measured & environment).sum()),
                      unknown_nearest_environment_pixels=int((unknown & environment).sum()),
                      supported_environment_pixels=int(support.sum()),
                      forbidden_unknown_nearest_supported_pixels=int((support & unknown).sum()))
        require(counts["forbidden_unknown_nearest_supported_pixels"] == 0, "Unknown nearest foreground admitted teacher behind it")
        split = memberships[f["file_path"]]
        for group in ("all", split, f["camera_id"]):
            target = totals.setdefault(group, dict(frames=0, **{k: 0 for k in counts}))
            target["frames"] += 1
            for name, value in counts.items(): target[name] += value
        rows.append(dict(file_path=f["file_path"], split=split, **counts))
        if key(f) in fixed_keys:
            expected, valid = joint.regenerate_teacher(ref, f, image_mask, policy=policy)
            require(np.array_equal(depth, expected) and np.array_equal(support, valid), "Fixed DEV v3 replay differs")
            replay.append(f["file_path"])
        if (i+1) % 40 == 0:
            print("Independent unknown-foreground audit:", i+1, "/", len(meta["frames"]), flush=True)
    require(len(replay) == len(fixed_dev), "Incomplete fixed015 DEV teacher replay")
    for path, expected in files.items(): require(sha(path) == expected, "Joint source changed during audit")
    safety = dict(status="PASS", policy=approval["policy"], policy_sha256=approval["policy_sha256"],
        approval_sha256=approval["approval_sha256"], oracle="independent camera-z minima over ALL merged records; nearest trusted float32 tie OR; no project_depth/project_surfel_depth calls for oracle",
        totals=totals, frames=rows, fixed015_dev_exact_replay_frames=len(replay),
        reference_points=len(ref["xyz"]), unknown_reference_records=int((~ref["normal_valid"]).sum()),
        independent_geometry_ground_truth=False, high_fidelity_accepted=False,
        source_sha256=dict(files, **{str(Path(__file__)): sha(__file__)}),
        gpu_used=False, cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_seconds=time.monotonic()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    safety_path = save("_unknown_foreground", safety)
    print(json.dumps(dict(status="PASS", closure=str(closure_path), safety=str(safety_path),
                          totals=totals["all"], peak_rss_mb=safety["peak_rss_mb"], elapsed_seconds=safety["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
