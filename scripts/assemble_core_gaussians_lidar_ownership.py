#!/usr/bin/env python3
"""CPU sparse-ray attribution and conservative measured-LiDAR ownership assembly.

Attribution follows the local INRIA surfel forward equations for selected rays.
The separate --build-lidar-ownership experiment requires a retained same-floor
Gaussian counterpart before deletion. Neither mode uses camera-nearest selection.
Float32 kernel parity and candidate quality require independent native checks.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import re
import numpy as np
from scipy.special import expit
from PIL import Image
from scipy.spatial import cKDTree
from assemble_core_gaussians import (ROOT, AssemblyError, InputGuard, inspect_ply, local_path,
    resource_preflight, write_json, read_json, parse_plan, default_source_plan, reference_pool,
    DEFAULT_REFERENCE_POINTS, assemble, sha256)


def axes_from_quaternion(quats):
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    a = np.column_stack([1 - 2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    b = np.column_stack([2*(x*y-w*z), 1 - 2*(x*x+z*z), 2*(y*z+w*x)])
    return a, b, np.cross(a, b)


def surfel_projection(xyz, scales, quats, camera):
    pose = np.asarray(camera["camera_to_worlds_ns"], dtype=np.float64)
    rotation = pose[:3, :3] @ np.diag([1., -1., -1.])
    center = (xyz - pose[:3, 3]) @ rotation
    a, b, normal = axes_from_quaternion(quats)
    aa, bb = (a * scales[:, :1]) @ rotation, (b * scales[:, 1:2]) @ rotation
    tw = np.column_stack([aa[:, 2], bb[:, 2], center[:, 2]])
    tu = camera["fl_x"] * np.column_stack([aa[:, 0], bb[:, 0], center[:, 0]]) + (camera["cx"] - .5)*tw
    tv = camera["fl_y"] * np.column_stack([aa[:, 1], bb[:, 1], center[:, 1]]) + (camera["cy"] - .5)*tw
    t = np.array([9., 9., -1.])
    denominator = np.sum(t * tw**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = t / denominator[:, None]
        image = np.column_stack([np.sum(f*tu*tw, axis=1), np.sum(f*tv*tw, axis=1)])
        extent = np.sqrt(np.maximum(1e-4, image**2 - np.column_stack([np.sum(f*tu*tu, axis=1), np.sum(f*tv*tv, axis=1)])))
        radius = np.ceil(np.maximum(extent.max(axis=1), 3*.707106))
    valid = (center[:, 2] > .2) & np.isfinite(radius) & np.isfinite(image).all(axis=1) & (denominator != 0)
    return tu, tv, tw, image, radius, valid, normal


def ray_alpha(projection, opacity, pixel):
    tu, tv, tw, image, radius, valid, normal = projection
    x, y = pixel
    # Native kernel bins to 16x16 tiles, with integer truncation towards zero.
    tile = np.array([x//16, y//16])
    lower = np.trunc((image - radius[:, None])/16.)
    upper = np.trunc((image + radius[:, None] + 15.)/16.)
    tiled = np.all((tile >= lower) & (tile < upper), axis=1)
    p = np.cross(x*tw - tu, y*tw - tv)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = p[:, :2] / p[:, 2:3]
        rho3d = np.sum(s*s, axis=1)
        rho2d = 2.*np.sum((image - [x, y])**2, axis=1)
        depth = np.sum(s*tw[:, :2], axis=1) + tw[:, 2]
        alpha = np.minimum(.99, opacity*np.exp(-.5*np.minimum(rho3d, rho2d)))
    active = valid & tiled & (p[:, 2] != 0) & (depth >= .2) & (alpha >= 1/255.) & np.isfinite(alpha)
    return np.flatnonzero(active), alpha[active], depth[active]


def sh_rgb(rows, names, camera_position):
    xyz = rows[:, [names.index(n) for n in ("x", "y", "z")]]
    d = xyz-camera_position
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    x,y,z = d.T; xx,yy,zz = x*x,y*y,z*z
    basis = np.column_stack([np.full(len(d), .28209479177387814), -.4886025119029199*y,
        .4886025119029199*z, -.4886025119029199*x, 1.0925484305920792*x*y,
        -1.0925484305920792*y*z, .31539156525252005*(2*zz-xx-yy), -1.0925484305920792*x*z,
        .5462742152960396*(xx-yy), -.5900435899266435*y*(3*xx-yy), 2.890611442640554*x*y*z,
        -.4570457994644658*y*(4*zz-xx-yy), .3731763325901154*z*(2*zz-3*xx-3*yy),
        -.4570457994644658*x*(4*zz-xx-yy), 1.445305721320277*z*(xx-yy), -.5900435899266435*x*(xx-3*yy)])
    dc = rows[:, [names.index(f"f_dc_{i}") for i in range(3)]]
    rest = rows[:, [names.index(f"f_rest_{i}") for i in range(45)]].reshape(-1, 3, 15).transpose(0, 2, 1)
    sh = np.concatenate([dc[:, None, :], rest], axis=1)
    return np.maximum(0., np.sum(sh * basis[:, :, None], axis=1) + .5)


def training_evidence(project, source, rows, names, query_position, guard):
    """Counts are center/depth proxies, not actual rendered transmittance."""
    if len(source["core_ids"]) != 1:
        raise AssemblyError("Diagnostic needs atomic original sources")
    base = local_path(project, f"data/garage_surface_v1/{source['core_ids'][0]}")
    meta = guard.json(base/"transforms.json")
    train = set(meta["train_filenames"])
    xyz = rows[:, [names.index(n) for n in ("x", "y", "z")]]
    _, _, normals = axes_from_quaternion(rows[:, [names.index(f"rot_{i}") for i in range(4)]])
    query_d = query_position-xyz
    query_d /= np.maximum(np.linalg.norm(query_d, axis=1, keepdims=True), 1e-12)
    counts = np.zeros((len(rows), 6), dtype=np.int64)
    best = np.full(len(rows), -2.)
    nearest_frames = [None for _ in rows]
    groups = {}
    for frame in meta["frames"]:
        if frame["file_path"] not in train:
            continue
        pose = np.asarray(frame["transform_matrix"])
        p = (xyz-pose[:3, 3]) @ (pose[:3, :3] @ np.diag([1., -1., -1.]))
        with np.errstate(divide="ignore", invalid="ignore"):
            u = frame["fl_x"]*p[:, 0]/p[:, 2]+frame["cx"]-.5
            v = frame["fl_y"]*p[:, 1]/p[:, 2]+frame["cy"]-.5
        inside = np.isfinite(u) & np.isfinite(v) & (p[:, 2]>.2) & (u>=-.5) & (u<frame["w"]-.5) & (v>=-.5) & (v<frame["h"]-.5)
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        ui, vi = np.floor(u[ids]+.5).astype(int), np.floor(v[ids]+.5).astype(int)
        paths = {key: local_path(project, base/frame[key]) for key in ("lidar_depth_path", "lidar_depth_mask_path", "mask_path")}
        for path in paths.values():
            if path not in guard.files:
                guard.snapshot(path)
        depth = np.load(paths["lidar_depth_path"], allow_pickle=False)
        with Image.open(paths["lidar_depth_mask_path"]) as image:
            support = np.asarray(image)>0
        with Image.open(paths["mask_path"]) as image:
            mask = np.asarray(image)>0
        if depth.shape != (frame["h"], frame["w"]) or support.shape != depth.shape or mask.shape != depth.shape:
            raise AssemblyError("Training evidence requires grayscale masks in exact prepared resolution")
        visible_mask = mask[vi, ui]
        ids, ui, vi = ids[visible_mask], ui[visible_mask], vi[visible_mask]
        counts[ids, 0] += 1
        target = depth[vi, ui]
        known = support[vi, ui] & np.isfinite(target) & (target>.2)
        ids, target = ids[known], target[known]
        counts[ids, 1] += 1
        residual = p[ids, 2]-target
        tolerance = np.minimum(.25, .1+.01*target)
        foreground = np.abs(residual)<=tolerance
        fg = ids[foreground]
        counts[fg, 2] += 1
        counts[ids[residual < -tolerance], 3] += 1  # free-space conflict
        counts[ids[residual > tolerance], 4] += 1   # behind measured foreground
        direction = pose[:3, 3]-xyz[fg]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        cosine = np.sum(direction*query_d[fg], axis=1)
        for index in fg[cosine > best[fg]]:
            nearest_frames[index] = {key: frame[key] for key in ('file_path', 'group_id', 'camera_id')}
        best[fg] = np.maximum(best[fg], cosine)
        opposite = np.sum(direction*normals[fg], axis=1)*np.sum(query_d[fg]*normals[fg], axis=1)<0
        counts[fg[opposite], 5] += 1
        group = groups.setdefault(str(frame["group_id"]), np.zeros(len(rows), dtype=bool))
        group[fg] = True
    group_counts = np.sum(list(groups.values()), axis=0) if groups else np.zeros(len(rows), dtype=int)
    result = []
    for i, values in enumerate(counts):
        result.append(dict(zip(("in_frame_unmasked", "known_lidar_depth", "foreground_supported",
            "free_space_conflict", "behind_measured_foreground", "opposite_hemisphere_foreground_views"), values.tolist()),
            foreground_distinct_train_groups=int(group_counts[i]),
            nearest_supported_training_frame=nearest_frames[i],
            nearest_supported_training_angle_deg=float(np.degrees(np.arccos(np.clip(best[i], -1., 1.)))) if best[i]>=-1 else None))
    return result


TRAIN_SUPPORT_CATEGORIES = ('zero_zero', 'weak_one_group', 'weak_two_groups',
                            'strong_unilateral', 'mixed_sides')


def train_support_categories(positive, negative):
    """Partition frozen group evidence; zero evidence does not prove invisibility."""
    p, n = np.asarray(positive), np.asarray(negative)
    if p.shape != n.shape or p.ndim != 1 or p.dtype.kind not in 'ui' or n.dtype.kind not in 'ui':
        raise AssemblyError('TRAIN group arrays must be aligned nonnegative integer vectors')
    if np.any(p < 0) or np.any(n < 0):
        raise AssemblyError('TRAIN group counts must be nonnegative')
    result = np.zeros(p.shape, np.uint8)
    maximum = np.maximum(p, n)
    unilateral = (p == 0) ^ (n == 0)
    result[unilateral & (maximum == 1)] = 1
    result[unilateral & (maximum == 2)] = 2
    result[unilateral & (maximum >= 3)] = 3
    result[(p > 0) & (n > 0)] = 4
    return result


def attribute(assembly, metrics, frame_index, pixels, output, project=ROOT, direction_support=None):
    project = Path(project).resolve()
    output = local_path(project, output)
    if output.exists() or not output.parent.is_dir():
        raise AssemblyError("Output must be a NEW report directory")
    preflight = resource_preflight(output, disk_bytes=16*1024**2, memory_bytes=1024**3)
    guard = InputGuard(project)
    a = guard.json(local_path(project, assembly))
    metric_path = local_path(project, metrics)
    m = guard.json(metric_path)
    frame = m["frames"][frame_index]
    camera = frame["camera"]
    if any(not 0<=p[0]<camera['w'] or not 0<=p[1]<camera['h'] for p in pixels):
        raise AssemblyError("Diagnostic pixels outside the image")
    if m.get("inference_sh_degree") != 3:
        raise AssemblyError("Diagnostic currently requires native degree-3 SH")
    full = local_path(project, a["full"]["path"])
    if m.get("source_ply_sha256") != a["full"]["sha256"]:
        raise AssemblyError("Native image and assembly identify different PLY files")
    guard.snapshot(full)
    trace_path = local_path(project, a["full"]["provenance"]["path"])
    guard.snapshot(trace_path)
    if guard.files[full]["sha256"] != a["full"]["sha256"] or guard.files[trace_path]["sha256"] != a["full"]["provenance"]["sha256"]:
        raise AssemblyError("Artifact SHA differs from assembly")
    trace = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    camera_position = np.asarray(camera["camera_to_worlds_ns"])[:, 3]
    support = None
    positive = negative = None
    if bool(m.get("direction_support")) != bool(direction_support):
        raise AssemblyError("Direction-aware native metrics require the same explicit --direction-support")
    if direction_support:
        from direction_support import load_direction_support
        support_path = local_path(project, direction_support)
        guard.snapshot(support_path)
        if guard.files[support_path]["sha256"] != m["direction_support"]["sha256"]:
            raise AssemblyError("Native metrics used a different direction sidecar")
        with full.open("rb") as handle:
            header = inspect_ply(handle)
        mm = np.memmap(full, mode="r", dtype="<f4", offset=len(header["header"]), shape=(header["count"], len(header["names"])))
        if list(header["names"][:3]) != ["x", "y", "z"]:
            raise AssemblyError("Direction diagnostic expects standard export XYZ columns")
        q0 = header["names"].index("rot_0")
        support = load_direction_support(support_path, {"path": str(full), "gaussians": header["count"], "sha256": a["full"]["sha256"]},
                                         {"means": mm[:, :3], "quats": mm[:, q0:q0+4]})
        for path in support["asset_states"]:
            guard.snapshot(Path(path))
        side_meta = json.loads(support_path.read_text())
        positive, negative = [np.load(local_path(project, side_meta['training_evidence'][key]['path']), mmap_mode='r', allow_pickle=False)
                              for key in ('positive_train_groups', 'negative_train_groups')]
    full_support_counts = None
    if support is not None:
        labels = train_support_categories(positive, negative)
        full_support_counts = {key: int(np.count_nonzero(labels == i))
                               for i, key in enumerate(TRAIN_SUPPORT_CATEGORIES)}
        del labels
    selected = [[] for _ in pixels]
    with full.open("rb") as handle:
        spec = inspect_ply(handle); names = spec["names"]
        for start in range(0, spec["count"], 65536):
            count = min(65536, spec["count"]-start)
            raw = handle.read(count*spec["stride"])
            rows = np.frombuffer(raw, dtype="<f4").reshape(count, -1).astype(np.float64)
            xyz = rows[:, [names.index(n) for n in ("x", "y", "z")]]
            scales = np.exp(rows[:, [names.index(f"scale_{i}") for i in range(3)]])
            q = rows[:, [names.index(f"rot_{i}") for i in range(4)]]
            projection = surfel_projection(xyz, scales, q, camera)
            if support is not None:
                flags = support['flags'][start:start+count]
                direction = camera_position - xyz
                cosine = np.sum(projection[6]*direction, axis=1)/np.maximum(np.linalg.norm(direction, axis=1), 1e-12)
                reject = ((flags==1)&(cosine < -support['grazing'])) | ((flags==2)&(cosine > support['grazing']))
                projection[5][reject] = False
            opacity = expit(rows[:, names.index("opacity")])
            for i, pixel in enumerate(pixels):
                ids, alpha, depth = ray_alpha(projection, opacity, pixel)
                if len(ids):
                    selected[i].append((ids+start, alpha, depth, projection[2][ids, 2], rows[ids]))
    native_path = local_path(project, metric_path.parent/frame["artifacts"]["candidate_png"])
    guard.snapshot(native_path)
    with Image.open(native_path) as image:
        native = np.asarray(image)
    reports, unique = [], {}
    for pixel, chunks in zip(pixels, selected):
        if not chunks:
            raise AssemblyError("No ray contributors")
        ids, alpha, depth, center_z, rows = [np.concatenate([c[i] for c in chunks]) for i in range(5)]
        order = np.lexsort((ids, center_z))
        ids, alpha, depth, center_z, rows = [v[order] for v in (ids, alpha, depth, center_z, rows)]
        after = np.cumprod(1.-alpha)
        stop = np.flatnonzero(after<1e-4)
        limit = int(stop[0]) if len(stop) else len(ids)
        weight = alpha[:limit]*np.r_[1., after[:limit-1]] if limit else np.empty(0)
        rgb = np.sum(sh_rgb(rows[:limit], names, camera_position)*weight[:, None], axis=0)
        source_weight = {}
        category_weights = {}
        train_support_summary = {}
        unknown_contributors = []
        if support is not None:
            labels = train_support_categories(positive[ids[:limit]], negative[ids[:limit]])
            for label, key in enumerate(TRAIN_SUPPORT_CATEGORIES):
                selected_label = labels == label
                train_support_summary[key] = {'contributors': int(selected_label.sum()),
                                               'weight': float(weight[selected_label].sum())}
        top = []
        for i in range(limit):
            sid = int(trace[ids[i]]["source_index"])
            source_weight[sid] = source_weight.get(sid, 0.)+float(weight[i])
            if support is not None:
                flag = int(support['flags'][ids[i]])
                category = {0:'unknown', 1:'positive_supported', 2:'negative_supported', 3:'both_observed'}[flag]
                category_weights[category] = category_weights.get(category, 0.)+float(weight[i])
                if flag == 0:
                    index = int(ids[i])
                    unknown_contributors.append({'full_record_index': index,
                        'source_id': a['sources'][sid]['id'],
                        'source_record_index': int(trace[index]['source_record_index']),
                        'positive_groups': int(positive[index]), 'negative_groups': int(negative[index]),
                        'category': TRAIN_SUPPORT_CATEGORIES[int(labels[i])],
                        'alpha': float(alpha[i]), 'weight': float(weight[i]),
                        'ray_depth_m': float(depth[i]), 'sort_center_depth_m': float(center_z[i])})
        for i in np.argsort(-weight)[:20]:
            index = int(ids[i]); origin = trace[index]
            sid = int(origin["source_index"])
            unique[index] = (sid, rows[i])
            top.append({"full_record_index": index, "source_id": a["sources"][sid]["id"],
                "source_record_index": int(origin["source_record_index"]), "weight": float(weight[i]),
                "alpha": float(alpha[i]), "ray_depth_m": float(depth[i]), "sort_center_depth_m": float(center_z[i]),
                "xyz": rows[i, [names.index(n) for n in ("x", "y", "z")]].tolist()})
            if support is not None:
                normal = axes_from_quaternion(rows[i:i+1, [names.index(f'rot_{j}') for j in range(4)]])[2][0]
                direction = camera_position - np.asarray(top[-1]['xyz'])
                top[-1]['direction_evidence'] = {'flag': int(support['flags'][index]), 'positive_groups': int(positive[index]),
                    'negative_groups': int(negative[index]), 'view_cosine': float(np.dot(normal, direction)/np.linalg.norm(direction))}
        native_rgb = native[pixel[1], pixel[0], :3].astype(float)/255.
        reports.append({"pixel_xy": pixel, "cpu_rgb": np.clip(rgb, 0., 1.).tolist(), "saved_native_rgb": native_rgb.tolist(),
                        "max_rgb_abs_error": float(np.max(np.abs(np.clip(rgb, 0., 1.)-native_rgb))),
                        "attributed_alpha": float(weight.sum()), "contributors": limit,
                        "direction_category_weights": category_weights,
                        "train_support_categories": train_support_summary,
                        "all_unknown_contributors": sorted(unknown_contributors, key=lambda c: -c['weight']),
                        "source_weights": {a["sources"][sid]["id"]: w for sid,w in sorted(source_weight.items(), key=lambda item:-item[1])},
                        "top_contributors": top})
    evidence = {}
    for sid in sorted({value[0] for value in unique.values()}):
        indices = [key for key,value in unique.items() if value[0]==sid]
        rows = np.stack([unique[key][1] for key in indices])
        records = training_evidence(project, a["sources"][sid], rows, names, camera_position, guard)
        evidence.update({str(key): value for key,value in zip(indices, records)})
    guard.check({full})
    report = {"status": "cpu_sparse_ray_attribution_diagnostic", "cpu_only": True, "preflight": preflight,
              "assembly": str(assembly), "metrics": str(metrics), "frame_index": frame_index,
              "frame": {k: frame[k] for k in ("file_path", "group_id", "camera_id")}, "camera": camera,
              "rays": reports, "contributor_training_evidence": evidence, "inputs": guard.report(),
              "direction_support": support['info'] if support is not None else None,
              "full_ply_train_support_counts": full_support_counts,
              "protocol": "CPU float64 analog of local INRIA TIGHTBBOX=0, dual-sided surfel kernel; "
                "16px tiles, center-depth stable sort, ray-plane intersection, min(rho3d,rho2d), alpha cutoff and early termination. "
                "Pixel parity checked against saved native degree-3 SH images; not a full-frame renderer certification.",
              "evidence_scope": "TRAIN center projections and prepared LiDAR depth/masks only. This is a proxy for "
                "foreground observation, not measured native per-Gaussian transmittance. No DEV/TEST image enters source confidence.",
              "model_modified": False, "static_owner_selected": False}
    output.mkdir()
    write_json(output/"attribution.json", report)
    print(json.dumps({"output": str(output), "max_rgb_abs_error": max(r["max_rgb_abs_error"] for r in reports),
                      "sources": sorted({v[0] for v in unique.values()})}))
    return report


def lidar_witness_plan(project, sources, ledger, guard, *, margin=.1, support_radius=.2, vertical=1.5, witness_radius=.1, workers=4):
    """Measured support preference, with immutable retained replacement witnesses.

    A point with no closer supported replacement is never rejected by this rule.
    Proposals are computed before protection, then every nominated witness is
    retained. This prevents cycles from deleting all representatives of a patch.
    """
    block_map = {b['id']: b for b in ledger['blocks']}
    pools = []
    for source in sources:
        if len(source['core_ids']) != 1:
            raise AssemblyError("LiDAR pilot requires atomic original sources")
        floor = block_map[source['core_ids'][0]].get('floor_id')
        floors = frozenset(re.findall(r'F\d+', str(floor)))
        # Unknown floor is isolated to its own source, never a global wildcard.
        if not floors:
            floors = frozenset([source['id']])
        reference, info = reference_pool(project, source, guard, DEFAULT_REFERENCE_POINTS)
        guard.snapshot(source['path'])
        with source['path'].open('rb') as handle:
            spec = inspect_ply(handle)
        rows = np.memmap(source['path'], mode='r', dtype='<f4', offset=len(spec['header']), shape=(spec['count'], len(spec['names'])))
        xyz = np.asarray(rows[:, [spec['names'].index(n) for n in ('x','y','z')]], dtype=np.float64)
        del rows
        own_distance, own_index = reference.query(xyz, workers=workers)
        measured = (own_distance<=support_radius) & (np.abs(xyz[:, 2]-reference.data[own_index, 2])<vertical)
        witness_indices = np.flatnonzero(measured)
        witness_tree = cKDTree(xyz[witness_indices]) if len(witness_indices) else None
        pools.append({'source': source, 'floors': floors, 'reference': reference, 'xyz': xyz,
                      'own_distance': own_distance, 'witness_indices': witness_indices, 'witness_tree': witness_tree,
                      'protected': np.zeros(len(xyz), bool), 'proposed': np.zeros(len(xyz), bool),
                      'witness_source': np.full(len(xyz), -1, np.int16), 'witness_record': np.full(len(xyz), -1, np.int64)})
    for i, pool in enumerate(pools):
        xyz = pool['xyz']
        best = pool['own_distance'].copy()
        for j, other in enumerate(pools):
            if i==j or not (pool['floors'] & other['floors']) or other['witness_tree'] is None:
                continue
            distance, ref_index = other['reference'].query(xyz, workers=workers)
            possible = (distance+margin < pool['own_distance']) & (distance < best) & (distance <= support_radius)
            possible &= np.abs(xyz[:, 2]-other['reference'].data[ref_index, 2]) < vertical
            ids = np.flatnonzero(possible)
            if not len(ids):
                continue
            near, record = other['witness_tree'].query(xyz[ids], workers=workers)
            accept = near<=witness_radius
            ids, record = ids[accept], other['witness_indices'][record[accept]]
            best[ids] = distance[ids]
            pool['proposed'][ids] = True
            pool['witness_source'][ids] = j
            pool['witness_record'][ids] = record
        print(json.dumps({'lidar_ownership_proposals': pool['source']['id'], 'proposed': int(pool['proposed'].sum())}), flush=True)
    # Protect only final nominations. Protection can retain extra duplicates but
    # can never recursively remove the last representative of a proposed point.
    for pool in pools:
        for j in np.unique(pool['witness_source'][pool['proposed']]):
            ids = pool['witness_record'][pool['witness_source']==j]
            pools[j]['protected'][ids] = True
    masks, traces, reports = {}, {}, []
    for i, pool in enumerate(pools):
        reject = pool['proposed'] & ~pool['protected']
        ids = np.flatnonzero(reject)
        sid = pool['source']['id']
        masks[sid] = ~reject
        witness_source = pool['witness_source'][ids]
        witness_record = pool['witness_record'][ids]
        # This final proof is independent of proposal order and includes the
        # optional subsequent support gate: witnesses were prechecked against it.
        distances = np.empty(len(ids), dtype=np.float64)
        for j in np.unique(witness_source):
            selected = witness_source==j
            records = witness_record[selected]
            other = pools[j]
            assert other['protected'][records].all()
            assert (other['own_distance'][records]<=support_radius).all()
            distances[selected] = np.linalg.norm(pool['xyz'][ids[selected]]-other['xyz'][records], axis=1)
        if len(ids) and not np.all(distances<=witness_radius+1e-12):
            raise AssemblyError("Retained LiDAR replacement witness proof failed")
        traces[sid] = {'rejected_source_record': ids, 'retained_witness_source_index': witness_source,
                       'retained_witness_record': witness_record, 'witness_distance_m': distances}
        reports.append({'id': sid, 'candidate': len(pool['xyz']), 'proposed': int(pool['proposed'].sum()),
                        'protected_witnesses': int(pool['protected'].sum()), 'rejected': len(ids),
                        'retained_by_ownership': int((~reject).sum()), 'floor_tokens': sorted(pool['floors'])})
    guard.check({s['path'] for s in sources})
    report = {'mode': 'lidar-support', 'rule': 'Prefer closer measured LiDAR support only with a protected same-floor trained Gaussian replacement',
              'camera_distance_used_for_decisions': False, 'cross_source_reassignment': False,
              'support_radius_m': support_radius, 'vertical_max_m': vertical, 'boundary_margin_m': margin,
              'replacement_max_distance_m': witness_radius, 'unknown_floor_policy': 'isolate source',
              'floor_policy': 'same F-token or shared ramp endpoint plus 3D replacement <=0.10m; floor labels unverified',
              'unwitnessed_deletion': False, 'witnesses_protected_from_ownership_and_support_deletion': True,
              'appearance_or_native_visibility_certified': False, 'sources': reports,
              'source_index_to_id': [s['id'] for s in sources], 'exclusion_enabled': True,
              'note': 'Geometric counterpart does not guarantee color/opacity equivalence. GPU counterfactual decides usefulness.'}
    return {'masks': masks, 'traces': traces, 'report': report, 'guard': guard}


def build_lidar_ownership(output, source_plan=None, workers=4, project=ROOT):
    project = Path(project).resolve()
    if type(workers) is not int or not 1 <= workers <= 8:
        raise AssemblyError("LiDAR ownership workers must be in 1..8")
    destination = local_path(project, output)
    if destination.exists() or not destination.parent.is_dir():
        raise AssemblyError("LiDAR ownership output must be a NEW directory")
    guard = InputGuard(project)
    plan = default_source_plan() if source_plan is None else guard.json(local_path(project, source_plan))
    sources = parse_plan(project, plan)
    ledger = guard.json(local_path(project, 'maps/garage_manifest.json'))
    preflight = resource_preflight(destination, disk_bytes=3*sum(s['state'][2] for s in sources)+512*1024**2,
                                   memory_bytes=5*1024**3)
    selection = lidar_witness_plan(project, sources, ledger, guard, workers=workers)
    record = assemble(output, source_plan=source_plan, ownership_mode='lidar-support',
                      ownership_selection=selection, support_radius_m=.2, workers=workers, project=project)
    trace_paths = []
    for sid, values in selection['traces'].items():
        path = destination/(sid+'_ownership_witnesses.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle, **values)
        trace_paths.append({'source': sid, 'path': path.relative_to(project).as_posix(), 'sha256': sha256(path)})
    write_json(destination/'lidar_ownership_audit.json', {'preflight': preflight, **selection['report'], 'witness_traces': trace_paths,
               'candidate_gaussians': record['candidate_gaussians'], 'retained_gaussians': record['retained_gaussians'],
               'support_filter_kept_identical_to_step1_step2': True, 'gpu_used': False})
    print(json.dumps({'output': output, 'candidate': record['candidate_gaussians'], 'retained': record['retained_gaussians'],
                      'ownership_rejected': sum(s['ownership_rejected_gaussians'] for s in record['sources'])}))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly")
    parser.add_argument("--metrics")
    parser.add_argument("--build-lidar-ownership", action="store_true")
    parser.add_argument("--source-plan")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-index", type=int, default=8)
    parser.add_argument("--pixels", default="320,180;480,180;560,180;320,280;480,280;560,280;320,380;480,380;560,380")
    parser.add_argument("--output", required=True)
    parser.add_argument("--direction-support", help="Must match direction-aware native metrics exactly")
    args = parser.parse_args(argv)
    if args.build_lidar_ownership:
        if args.assembly or args.metrics or args.direction_support:
            raise AssemblyError("LiDAR build cannot combine with attribution options")
        return build_lidar_ownership(args.output, source_plan=args.source_plan, workers=args.workers)
    if not args.assembly or not args.metrics:
        parser.error('attribution requires --assembly and --metrics')
    pixels = [[int(v) for v in pair.split(",")] for pair in args.pixels.split(";")]
    if not 1<=len(pixels)<=32 or any(len(p)!=2 or min(p)<0 for p in pixels):
        raise AssemblyError("Require 1..32 nonnegative pixel pairs")
    return attribute(args.assembly, args.metrics, args.frame_index, pixels, args.output, direction_support=args.direction_support)


if __name__ == "__main__":
    main()
