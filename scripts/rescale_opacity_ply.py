#!/usr/bin/env python3
"""Explicit CPU opacity ablations. Never change colors or claim opacity recovery."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
from pathlib import Path
import shutil

from assemble_core_gaussians import (
    ROOT, MAX_CHUNK_BYTES, AssemblyError, InputGuard, _state, inspect_ply,
    local_path, positive_int, resource_preflight, write_json,
)
import numpy as np
from scipy.special import expit


def transform_opacity(logits, method, factor, epsilon=1e-6):
    """Transform center opacity, with explicit clipping, in float64.

    optical-depth: tau=-log(1-alpha), tau'=factor*tau.
    alpha-multiply: alpha'=factor*alpha; clipping breaks linearity.
    Identity preserves original float32 bytes even outside the clip interval.
    """
    if method not in ("optical-depth", "alpha-multiply"):
        raise AssemblyError("Unknown opacity method")
    if not np.isfinite(factor) or factor <= 0 or not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise AssemblyError("factor must be positive finite; epsilon must be in (0, 0.5)")
    values = np.asarray(logits)
    if not np.isfinite(values).all():
        raise AssemblyError("Nonfinite opacity logit")
    if factor == 1.:
        return values.copy(), 0
    x = values.astype(np.float64)
    with np.errstate(over="ignore", under="ignore"):
        if method == "optical-depth":
            alpha = -np.expm1(-factor * np.logaddexp(0., x))
        else:
            alpha = factor * expit(x)
    clips = int(np.count_nonzero((alpha < epsilon) | (alpha > 1. - epsilon)))
    alpha = np.clip(alpha, epsilon, 1. - epsilon)
    return (np.log(alpha) - np.log1p(-alpha)).astype(values.dtype), clips


def rescale(source, output, *, method, factor, epsilon=1e-6, chunk_bytes=8 * 1024**2,
            project=ROOT, preflight_only=False):
    project = Path(project).resolve()
    if project != ROOT:
        local_path(ROOT, project)
    source, destination = local_path(project, source), local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("Output must be a NEW directory with an existing parent")
    positive_int(chunk_bytes, "chunk_bytes")
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise AssemblyError("chunk_bytes exceeds bounded budget")
    transform_opacity(np.asarray([0.], dtype="<f4"), method, factor, epsilon)
    state, guard = _state(source), InputGuard(project)
    with source.open("rb") as handle:
        spec = inspect_ply(handle)
    if state[2] != len(spec["header"]) + spec["count"] * spec["stride"]:
        raise AssemblyError("Incomplete or trailing PLY payload")
    if "opacity" not in spec["names"] or chunk_bytes < spec["stride"]:
        raise AssemblyError("Need opacity field and a chunk large enough for one record")
    preflight = resource_preflight(destination, disk_bytes=state[2] + 1024**2,
                                   memory_bytes=12 * chunk_bytes + 256 * 1024**2)
    if preflight_only:
        return {"status": "preflight_passed", "preflight": preflight, "assembly_started": False}
    destination.mkdir()
    try:
        output_path = destination / "gaussians.ply"
        source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        before_sum, after_sum, clips, changed, before_gt, after_gt = 0., 0., 0, 0, 0, 0
        column = spec["names"].index("opacity")
        with source.open("rb") as reader, output_path.open("xb") as writer:
            if inspect_ply(reader) != spec:
                raise AssemblyError("Source PLY schema changed")
            writer.write(spec["header"])
            source_hash.update(spec["header"]); output_hash.update(spec["header"])
            step = chunk_bytes // spec["stride"]
            for start in range(0, spec["count"], step):
                size = min(step, spec["count"] - start)
                raw = reader.read(size * spec["stride"])
                if len(raw) != size * spec["stride"]:
                    raise AssemblyError("Incomplete PLY payload")
                source_hash.update(raw)
                rows = np.frombuffer(raw, dtype="<f4").reshape(size, -1).copy()
                if not np.isfinite(rows).all():
                    raise AssemblyError("Nonfinite PLY attribute")
                old = rows[:, column].copy()
                new, clipped = transform_opacity(old, method, factor, epsilon)
                rows[:, column] = new
                before, after = expit(old.astype(np.float64)), expit(new.astype(np.float64))
                before_sum += float(before.sum()); after_sum += float(after.sum())
                before_gt += int((before > .5).sum()); after_gt += int((after > .5).sum())
                clips += clipped; changed += int(np.count_nonzero(old != new))
                raw = rows.tobytes()
                writer.write(raw); output_hash.update(raw)
            if reader.read(1):
                raise AssemblyError("Trailing PLY payload")
        guard.remember(source, state, source_hash.hexdigest())
        guard.check()  # Rehash input; don't publish if another worker changed it.
        report = {"schema_version": 1, "status": "opacity_experiment_pending_paired_native_evaluation",
                  "cpu_only": True, "method": method, "factor": factor, "epsilon": epsilon,
                  "preflight": preflight, "source": source.relative_to(project).as_posix(),
                  "source_sha256": source_hash.hexdigest(), "source_stat": guard.report(),
                  "output": output_path.relative_to(project).as_posix(), "output_sha256": output_hash.hexdigest(),
                  "gaussians": spec["count"], "changed_records": changed, "clipped_records": clips,
                  "alpha_mean_before": before_sum / spec["count"], "alpha_mean_after": after_sum / spec["count"],
                  "gaussian_alpha_gt_0_5_before": before_gt / spec["count"],
                  "gaussian_alpha_gt_0_5_after": after_gt / spec["count"],
                  "pixel_alpha_coverage_measured": False, "color_modified": False,
                  "other_fields_byte_identical": True, "geometry_modified": False,
                  "trained_opacity_recovered": False, "high_fidelity_accepted": False,
                  "physical_scope": "Center-opacity sensitivity only. Optical-depth scales center transmittance; "
                    "it is not a uniform per-pixel density scaling of alpha*Gaussian kernels. "
                    "Alpha multiplication can saturate. Neither operation inverts training or preserves rendered radiance.",
                  "color_policy": "Unchanged SH/DC. No global color correction preserves front-to-back "
                    "sum(T_i*alpha_i*c_i)+T_background*c_background for arbitrary views and occlusions."}
        write_json(destination / "opacity_experiment.json", report)
        return report
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True, help="NEW output directory")
    parser.add_argument("--method", required=True, choices=("optical-depth", "alpha-multiply"))
    parser.add_argument("--factor", required=True, type=float)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = rescale(args.source, args.output, method=args.method, factor=args.factor,
                         epsilon=args.epsilon, chunk_bytes=args.chunk_mib * 1024**2,
                         preflight_only=args.preflight_only)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
