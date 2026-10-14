#!/usr/bin/env python3
"""Read-only geometry diagnosis of an approximate, sensor-unspecified height hint.

All new outputs are diagnostics, never generated floor/depth/training geometry.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT/"reports/block000_ground_height_hint_audit"
START = time.monotonic()
ASSETS = {}
POLICY = dict(user_height_hint_m=2.65, user_hint_measured=False, user_hint_sensor="unspecified: LiDAR or camera",
    relative_z_band_m=[-3.3, -2.0], max_trajectory_xy_distance_m=25., voxel_m=.06,
    horizontal_abs_nz_min=float(np.cos(np.deg2rad(30))), normal_knn=16, normal_neighbor_radius_max_m=.5,
    normal_context_xy_margin_m=1., normal_context_z_margin_m=1.,
    ransac_distance_m=.05, ransac_trials=256, ransac_seed=265000, min_fit_points=12,
    partition_xy_tile_m=5., max_planes_per_tile=2, support_xy_cell_m=.5,
    suggested_slope_evidence_min_degrees=3., local_fit_min_hull_area_m2=.25,
    no_fitted_height_lock=True)


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


def pin(path):
    p=Path(path).resolve()
    if str(p) not in ASSETS:
        st=p.stat();ASSETS[str(p)]=dict(sha256=sha(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p


def read(path):return json.loads(pin(path).read_text())


def check():
    assert time.monotonic()-START<600,"CPU time bound exceeded"
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,"3GiB RSS bound exceeded"


def q(v):
    v=np.asarray(v)
    if not len(v):return dict(count=0)
    assert np.isfinite(v).all()
    return dict(count=len(v),min=float(v.min()),p05=float(np.quantile(v,.05)),p50=float(np.quantile(v,.5)),
        p95=float(np.quantile(v,.95)),max=float(v.max()),mean=float(v.mean(dtype=np.float64)))


def nearest_path(xyz,path):
    """Exact closest XY point on camera2 TRAIN+DEV polyline, interpolating Z."""
    best=np.full(len(xyz),np.inf);z=np.zeros(len(xyz));segment=np.zeros(len(xyz),np.int32);arc=np.zeros(len(xyz))
    cumulative=0.
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        v=b[:2]-a[:2];length=float(np.linalg.norm(v))
        t=np.clip((xyz[:,:2]-a[:2])@v/max(float(v@v),1e-20),0.,1.)
        sq=np.sum((xyz[:,:2]-a[:2]-t[:,None]*v)**2,axis=1);use=sq<best
        best[use]=sq[use];z[use]=a[2]+t[use]*(b[2]-a[2]);segment[use]=i;arc[use]=cumulative+t[use]*length
        cumulative+=length
    return np.sqrt(best),z,segment,arc


def frustum_seen(points,frames):
    seen=np.zeros(len(points),bool)
    for f in frames:
        if seen.all():break
        ids=np.flatnonzero(~seen);T=np.asarray(f["transform_matrix"]);p=(points[ids]-T[:3,3])@T[:3,:3]
        z=-p[:,2];safe=z>.1
        u=f["fl_x"]*p[:,0]/np.where(safe,z,1)+f["cx"]-.5
        v=-f["fl_y"]*p[:,1]/np.where(safe,z,1)+f["cy"]-.5
        seen[ids[safe&(u>=-.5)&(u<f["w"]-.5)&(v>=-.5)&(v<f["h"]-.5)]]=True
    return seen


def candidate_stats(xyz,normal,valid,path,first,lower,upper):
    distance,cz,_,_=nearest_path(xyz,path)
    band=(xyz[:,2]-cz>=-3.3)&(xyz[:,2]-cz<=-2.0)
    nz=abs(normal[:,2])/np.maximum(np.linalg.norm(normal,axis=1),1e-8)
    horizontal=valid&(nz>=POLICY["horizontal_abs_nz_min"])
    use=band&(distance<25)&horizontal
    near=np.linalg.norm(xyz[:,:2]-first[:2],axis=1)
    in_crop=((xyz>=lower)&(xyz<=upper)).all(1)
    return dict(total_points=len(xyz),height_band_all_distances=int(band.sum()),height_band_within25m=int((band&(distance<25)).sum()),
        valid_horizontal_height_band_all_distances=int((band&horizontal).sum()),valid_horizontal_within25m=int(use.sum()),
        horizontal_candidates_inside_old_crop=int((use&in_crop).sum()),horizontal_candidates_outside_old_crop=int((use&~in_crop).sum()),
        horizontal_candidates_distance_to_first_DEV_xy_counts={str(r):int((use&(near<r)).sum()) for r in (3,6,8,12,25)},
        horizontal_candidates_distance_to_trajectory_xy_counts={str(r):int((use&(distance<r)).sum()) for r in (3,6,8,12,25)},
        candidate_relative_z_m=q(xyz[use,2]-cz[use]),candidate_absolute_z_m=q(xyz[use,2]),
        candidate_distance_to_trajectory_m=q(distance[use]),height_band_invalid_normal=int((band&(distance<25)&~valid).sum()),
        height_band_valid_nonhorizontal=int((band&(distance<25)&valid&~horizontal).sum())),use


def hull_info(points,path,first):
    if len(points)<3:return dict(status="insufficient_points"),None
    try:hull=ConvexHull(points[:,:2])
    except QhullError:return dict(status="collinear_xy"),None
    vertices=points[hull.vertices,:2]
    inside=lambda p:np.all(p@hull.equations[:,:2].T+hull.equations[:,2]<=1e-8,axis=1)
    path_inside=inside(path[:,:2]);first_inside=bool(inside(first[None,:2])[0])
    tree=cKDTree(points[:,:2]);dist,_=tree.query(path[:,:2],workers=3)
    areas=len(np.unique(np.floor(points[:,:2]/.5).astype(int),axis=0))*.25
    return dict(status="convex_envelope_only",vertices_xy=vertices.tolist(),area_m2=float(hull.volume),
        bbox_xy_min=points[:,:2].min(0).tolist(),bbox_xy_max=points[:,:2].max(0).tolist(),
        occupied_half_metre_xy_cells=len(np.unique(np.floor(points[:,:2]/.5).astype(int),axis=0)),
        occupied_grid_area_over_hull_area=areas/hull.volume if hull.volume else None,
        trajectory_pose_xy_inside_count=int(path_inside.sum()),trajectory_pose_count=len(path),
        first_DEV_camera_xy_inside=first_inside,trajectory_to_nearest_observed_xy_m=q(dist),
        first_DEV_camera_to_nearest_observed_xy_m=float(tree.query(first[:2])[0]),
        warning="Convex hull bridges gaps; it is an outer envelope of measured support, not evidence for an observed continuous floor inside it."),hull.equations


def plane_fit(points,source_ids,path,first,seed):
    if len(points)<12:return dict(status="insufficient_points",points=len(points)),np.empty(0,int)
    rng=np.random.default_rng(seed);best=None
    for _ in range(POLICY["ransac_trials"]):
        a,b,c=points[rng.choice(len(points),3,replace=False)]
        n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<.02:continue
        n/=length
        if abs(n[2])<POLICY["horizontal_abs_nz_min"]:continue
        d=-float(n@a);res=abs(points@n+d);inside=res<=.05
        if inside.sum()<12:continue
        cells=len(np.unique(np.floor(points[inside,:2]/.5).astype(int),axis=0))
        score=(cells,int(inside.sum()),-float(np.median(res[inside])))
        if best is None or score>best[0]:best=(score,inside)
    if best is None:return dict(status="no_supported_plane",points=len(points)),np.empty(0,int)
    inside=best[1]
    for _ in range(3):
        centroid=points[inside].mean(0);_,_,vt=np.linalg.svd(points[inside]-centroid,full_matrices=False)
        n=vt[-1]
        if n[2]<0:n=-n
        if n[2]<POLICY["horizontal_abs_nz_min"]:return dict(status="refit_not_near_horizontal",points=len(points)),np.empty(0,int)
        d=-float(n@centroid);new=abs(points@n+d)<=.05
        if new.sum()<12:return dict(status="insufficient_refit_inliers",points=len(points)),np.empty(0,int)
        if np.array_equal(new,inside):break
        inside=new
    ids=np.flatnonzero(inside);selected=points[ids]
    residual=selected@n+d
    hull,equations=hull_info(selected,path,first)
    zspan=float(np.ptp(selected[:,2]));tilt=float(np.rad2deg(np.arccos(np.clip(n[2],-1,1))))
    distance,cz,_,_=nearest_path(selected,path)
    height=cz-selected[:,2]
    # Predict only at already measured inlier XY positions, never in unobserved holes.
    fit_z=-(selected[:,:2]@n[:2]+d)/n[2]
    directional_span=float(np.ptp(fit_z))
    area=hull.get("area_m2",0.)
    good_area=area>=.25
    ramp_compatible=bool(good_area and tilt>=3 and directional_span>=.2 and len(ids)/len(points)>=.7)
    return dict(status="measured_local_plane" if good_area else "small_or_degenerate_spatial_support",
        input_points=len(points),inliers=len(ids),inlier_fraction=len(ids)/len(points),
        diagnostic_point_indices=source_ids[ids].tolist(),unit_normal_xyz=n.tolist(),plane_offset_m=d,
        equation="n dot world_xyz + offset =0; diagnostic normal points toward unchanged+Z",tilt_degrees=tilt,
        slope_xy_dz_dx_dy=(-n[:2]/n[2]).tolist(),orthogonal_abs_residual_m=q(abs(residual)),
        orthogonal_rmse_m=float(np.sqrt(np.mean(residual**2))),all_input_abs_residual_m=q(abs(points@n+d)),
        observed_z_span_m=zspan,fitted_z_span_at_observed_inlier_xy_m=directional_span,
        camera_minus_observed_point_height_m=q(height),distance_to_trajectory_xy_m=q(distance),support_hull=hull,
        inclined_patch_compatible_with_ramp_but_semantics_unknown=ramp_compatible,
        fitted_plane_not_filled_or_exported=True),ids


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-prefix",type=Path,default=PREFIX)
    args=parser.parse_args();pin(Path(__file__))
    prep=read(ROOT/"data/garage_surface_v1/block_000/preparation.json")
    transforms=read(ROOT/"data/garage_surface_v1/block_000/transforms.json")
    metrics=read(ROOT/"runs/fidelity_v2/block000_seed_aa_cap003_acceptance/metrics.json")
    names=set(transforms["train_filenames"])|set(transforms["val_filenames"])
    frames=[f for f in transforms["frames"] if f["file_path"] in names]
    trajectory_frames=sorted([f for f in frames if f["camera_id"]=="camera_2"],key=lambda f:f["timestamp_ns"])
    path=np.asarray([np.asarray(f["transform_matrix"])[:3,3] for f in trajectory_frames])
    first=np.asarray(metrics["frames"][0]["camera"]["camera_to_worlds_ns"])[:,3]
    assert len(path)>2
    lower,upper=np.asarray(prep["map"]["crop_min"]),np.asarray(prep["map"]["crop_max"])
    reference_path=pin(ROOT/"data/garage_surface_v1/block_000/lidar_reference.npz")
    with np.load(reference_path,allow_pickle=False) as f:ref={k:f[k] for k in ("xyz","normals","normal_valid")}
    reference_stats,ref_use=candidate_stats(ref["xyz"],ref["normals"],ref["normal_valid"],path,first,lower,upper)
    source=pin(prep["map"]["source"]);release=read(source.with_name("release_file_gate.json"))
    assert ASSETS[str(source)]["sha256"]==release["raw_sha256"]
    pool=[];rawids=[];raw_candidate_count=raw_candidate_crop=0;total=0
    context_zmin=float(path[:,2].min()-4.3);context_zmax=float(path[:,2].max()-1.)
    bbox_min=path[:,:2].min(0)-26;bbox_max=path[:,:2].max(0)+26
    with source.open("rb") as f:
        header={}
        while True:
            line=f.readline().decode("ascii").strip()
            if not line or line.startswith("#"):continue
            k,*v=line.split();header[k]=v
            if k=="DATA":break
        assert header["FIELDS"]==["x","y","z","rgb"] and header["DATA"]==["binary"] and header["SIZE"]==["4"]*4
        count=int(header["POINTS"][0]);dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<u4")])
        assert source.stat().st_size==f.tell()+count*16
        for start in range(0,count,262144):
            check();rec=np.fromfile(f,dtype=dtype,count=min(262144,count-start));total+=len(rec)
            xyz=np.column_stack([rec[n] for n in "xyz"])
            use=np.isfinite(xyz).all(1)&(xyz[:,2]>=context_zmin)&(xyz[:,2]<=context_zmax)&(xyz[:,:2]>=bbox_min).all(1)&(xyz[:,:2]<=bbox_max).all(1)
            indices=np.flatnonzero(use);xyz=xyz[indices]
            distance,cz,_,_=nearest_path(xyz,path)
            wanted=(distance<25)&(xyz[:,2]-cz>=-3.3)&(xyz[:,2]-cz<=-2.)
            raw_candidate_count+=int(wanted.sum());raw_candidate_crop+=int((wanted&((xyz>=lower)&(xyz<=upper)).all(1)).sum())
            context=distance<26
            pool.append(xyz[context]);rawids.append(start+indices[context])
    xyz=np.concatenate(pool);pcd_ids=np.concatenate(rawids);del pool,rawids
    context_raw_count=len(xyz)
    cells=np.floor(xyz.astype(np.float64)/.06).astype(np.int32)
    _,first_index=np.unique(cells,axis=0,return_index=True);del cells
    first_index.sort();xyz=xyz[first_index];pcd_ids=pcd_ids[first_index]
    print(f"raw={total} hint-band={raw_candidate_count} normal-context-voxels={len(xyz)}",flush=True)
    distance,cz,segment,arc=nearest_path(xyz,path)
    wanted=(distance<25)&(xyz[:,2]-cz>=-3.3)&(xyz[:,2]-cz<=-2.)
    candidate=np.flatnonzero(wanted);assert len(candidate)<=300000
    tree=cKDTree(xyz)
    normals=np.zeros((len(candidate),3),np.float32);valid=np.zeros(len(candidate),bool);radius=np.zeros(len(candidate),np.float32)
    for start in range(0,len(candidate),8192):
        check();ids=candidate[start:start+8192];dist,near=tree.query(xyz[ids],k=16,workers=3)
        nxyz=xyz[near].astype(float);nxyz-=nxyz.mean(1,keepdims=True)
        eigen,vec=np.linalg.eigh(np.einsum("nki,nkj->nij",nxyz,nxyz))
        normals[start:start+len(ids)]=vec[:,:,0]
        valid[start:start+len(ids)]=(eigen[:,0]<.25*np.maximum(eigen[:,1],1e-12))&(eigen[:,1]>.01*np.maximum(eigen[:,2],1e-12))&(dist[:,-1]<=.5)
        radius[start:start+len(ids)]=dist[:,-1]
    candidate_xyz=xyz[candidate]
    raw_stats,horizontal=candidate_stats(candidate_xyz,normals,valid,path,first,lower,upper)
    # Save only actual source points and measured-neighborhood diagnostics, no fitted samples.
    observations=candidate_xyz[horizontal].astype(np.float64)
    obs_raw_ids=pcd_ids[candidate[horizontal]]
    obs_normals=normals[horizontal];obs_radius=radius[horizontal]
    dpath,czpath,segments,arcs=nearest_path(observations,path)
    in_crop=((observations>=lower)&(observations<=upper)).all(1)
    seen=frustum_seen(observations,frames)
    raw_stats.update(horizontal_candidates_in_any_TRAIN_DEV_camera_frustum=int(seen.sum()),
        normal_radius_over_0_5m_rejected=int((radius>.5).sum()))
    print(f"horizontal candidates ref={reference_stats['valid_horizontal_within25m']} raw-uncapped={len(observations)} frustum={int(seen.sum())}",flush=True)
    global_fit,_=plane_fit(observations,np.arange(len(observations)),path,first,POLICY["ransac_seed"])
    all_hull,_=hull_info(observations,path,first)
    tiles=np.floor(observations[:,:2]/5).astype(np.int32)
    unique_tiles=np.unique(tiles,axis=0)
    patches=[];tile_stats=[]
    for ti,tile in enumerate(unique_tiles):
        check();ids=np.flatnonzero((tiles==tile).all(1));remaining=ids.copy()
        tile_stats.append(dict(tile_xy=tile.tolist(),candidate_count=len(ids),inside_old_crop=int(in_crop[ids].sum())))
        for layer in range(2):
            fit,local=plane_fit(observations[remaining],remaining,path,first,POLICY["ransac_seed"]+ti*2+layer+1)
            fit.update(tile_xy=tile.tolist(),layer=layer)
            patches.append(fit)
            if not len(local):break
            keep=np.ones(len(remaining),bool);keep[local]=False;remaining=remaining[keep]
            if len(remaining)<12:break
    fitted=[p for p in patches if p["status"]=="measured_local_plane"]
    union_inside=np.zeros(len(path),bool);first_local=False;adjacent=[]
    for patch in fitted:
        pts=observations[patch["diagnostic_point_indices"]]
        _,eq=hull_info(pts,path,first)
        union_inside|=np.all(path[:,:2]@eq[:,:2].T+eq[:,2]<=1e-8,axis=1)
        first_local|=bool(np.all(first[:2]@eq[:,:2].T+eq[:,2]<=1e-8))
    for i,left in enumerate(fitted):
        for j,right in enumerate(fitted[:i]):
            if np.max(abs(np.array(left["tile_xy"])-np.array(right["tile_xy"])))>1:continue
            a=observations[left["diagnostic_point_indices"]];b=observations[right["diagnostic_point_indices"]]
            d,idx=cKDTree(b[:,:2]).query(a[:,:2],workers=3);where=int(np.argmin(d));other=int(idx[where])
            if d[where]>1.:continue
            angle=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(left["unit_normal_xyz"],right["unit_normal_xyz"])),0,1))))
            adjacent.append(dict(left_tile=left["tile_xy"],right_tile=right["tile_xy"],left_layer=left["layer"],right_layer=right["layer"],
                closest_observed_xy_gap_m=float(d[where]),closest_observed_signed_z_difference_m=float(a[where,2]-b[other,2]),
                normal_angle_degrees=angle,left_observed_xyz=a[where].tolist(),right_observed_xyz=b[other].tolist(),
                note="Observed edge pair only; neither plane evaluated outside its own measured hull."))
    output_npz=Path(str(args.output_prefix)+".points.npz")
    with output_npz.open("xb") as f:
        np.savez_compressed(f,xyz=observations.astype(np.float32),raw_pcd_row_index=obs_raw_ids,
            estimated_unit_normals=obs_normals,normal_knn16_radius_m=obs_radius,
            distance_to_trajectory_xy_m=dpath.astype(np.float32),relative_z_to_nearest_camera_m=(observations[:,2]-czpath).astype(np.float32),
            closest_trajectory_segment=segments,trajectory_arc_m=arcs.astype(np.float32),inside_old_reference_crop=in_crop,
            in_any_TRAIN_DEV_frustum=seen)
    check()
    for file,old in ASSETS.items():
        st=Path(file).stat();assert st.st_size==old["bytes"] and st.st_mtime_ns==old["mtime_ns"],"Input changed"
    report=dict(schema_version=1,status="bounded_height_hint_geometry_diagnostic_complete",policy=POLICY,
        user_hint_statement="2.65m is an unmeasured user approximation with unspecified sensor (LiDAR or camera), never ground truth or a fitted plane-height constraint.",
        cpu_only=True,threads=3,gpu_used=False,training_sources_modified=False,existing_reference_depth_gate_worldup_modified=False,
        test_images_or_depth_read=False,independent_ground_truth=False,physical_floor_confirmed=False,
        source_pcd=dict(path=str(source),sha256=ASSETS[str(source)]["sha256"],total_points=total,release_hash_matches=True),
        trajectory=dict(definition="camera_2 TRAIN+DEV centers joined in timestamp order; exact closest XY segment gives interpolated cameraZ. A geometric camera trajectory proxy, not a measured sensor mounting height.",
            points=path.tolist(),timestamps_ns=[f["timestamp_ns"] for f in trajectory_frames],count=len(path),
            camera_z_m=q(path[:,2]),arc_length_xy_m=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()),
            first_DEV_camera_xyz=first.tolist(),FOV_definition="Rectified TRAIN+DEV camera geometric frusta only; no occlusion/visibility/semantic or RGB evidence. No TEST observations read."),
        selection=dict(raw_height_band_points_within25m=raw_candidate_count,raw_height_band_points_inside_old_crop=raw_candidate_crop,
            raw_height_band_points_outside_old_crop=raw_candidate_count-raw_candidate_crop,
            normal_context_raw_points=context_raw_count,normal_context_6cm_voxels=len(xyz),normal_context_absolute_z=[context_zmin,context_zmax],
            normal_context_xy_distance_limit_m=26.,uncapped_height_band_6cm_points=len(candidate),
            normal_estimation="k16 PCA on6cm actual source voxel representatives BEFORE selecting horizontal normals;1m XY/Z halo, maximum neighborhood radius.5m. Validity lambda0<.25lambda1 and lambda1>.01lambda2. No cap or point interpolation."),
        stored_400k_reference_candidates=reference_stats,uncapped_raw_candidates=raw_stats,
        observed_candidate_points=dict(path=str(output_npz),sha256=sha(output_npz),count=len(observations),
            content="Actual raw PCD row-indexed measured positions and estimated normals only; no synthetic plane, seed, depth, or acceptance output."),
        global_fit=global_fit,all_candidate_convex_envelope=all_hull,tiles=tile_stats,local_plane_patches=patches,
        observed_patch_consistency=adjacent,
        local_support_summary=dict(measured_local_plane_patches=len(fitted),trajectory_pose_xy_inside_any_local_hull=int(union_inside.sum()),
            trajectory_pose_count=len(path),first_DEV_camera_xy_inside_any_local_hull=first_local,
            inclined_patches_compatible_with_ramp_but_not_semantic_proof=sum(p["inclined_patch_compatible_with_ramp_but_semantics_unknown"] for p in fitted)),
        limitations=["Height-band selection itself conditions the candidate distribution; a small residual is not independent validation of2.65m or a semantic floor.",
            "PCA/RANSAC results describe observed local geometry in the existing map coordinate frame. No axis flip, world-up change, data correction or gate change.",
            "Convex hulls bridge gaps even inside a tile. Reported hull membership never licenses plane filling, new valid depth or observed coverage of every enclosed location.",
            "Global fit can join physically distinct surfaces; fixed5m local partitions and observed-neighbor comparisons are the stronger local evidence.",
            "Tilt alone cannot distinguish a ramp from map distortion, clutter or other sloped surfaces. Camera sensor identity and true height remain unverified.",
            "Missing low returns do not prove a physical floor is absent; extrapolation from distant patches into nearfield is outside this diagnostic."],
        inputs=ASSETS,elapsed_seconds=time.monotonic()-START,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    output=Path(str(args.output_prefix)+".json")
    with output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    summary=Path(str(args.output_prefix)+".md")
    lines=["2.65m is an unmeasured, sensor-unspecified user approximation. Diagnostic only; no physical floor or independent truth claim.",
        f"Raw points in broad height band/25m trajectory distance: {raw_candidate_count}; horizontal measured6cm representatives: {len(observations)}; stored400k reference: {reference_stats['valid_horizontal_within25m']}.",
        f"Supported local plane patches: {len(fitted)}; trajectory positions inside any local convex envelope: {int(union_inside.sum())}/{len(path)}; first DEV camera inside: {first_local}.",
        f"Global fit status: {global_fit['status']}; see JSON for residuals, local hull coordinates, partition consistency and slope caveats.",
        "All output points are original PCD observations. No fitted plane is filled or exported as reconstruction/teacher data."]
    with summary.open("x") as f:f.write("\n\n".join(lines)+"\n")
    print(json.dumps(dict(reference=reference_stats,raw=raw_stats,local=report['local_support_summary']),indent=2))
    print(f"saved={output} elapsed={report['elapsed_seconds']:.2f}s RSS={report['peak_rss_mib']:.1f}MiB")


if __name__=="__main__":
    with threadpool_limits(limits=3):main()
