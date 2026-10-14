"""Measured local surface support; no interpolation across unobserved holes."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfelDepthPolicy:
    """Explicit opt-in S1; default retains historical support thresholds.

    Relax only measured horizontal surfaces below the camera. Bound plane
    corrections by the pixel footprint AND a metric cap, never metres of
    z*sin(angle). Multiple separated returns cannot select a farther surface.
    """
    version: str = 'surfel_depth_v3'
    ground_support: bool = False
    min_cosine: float = .1
    ground_min_cosine: float = .05
    ground_normal_cosine: float = .85
    world_up: tuple = (0., 0., 1.)
    correction_base_m: float = .05
    correction_range_slope: float = .02
    correction_range_start_m: float = 5.
    max_correction_m: float = .25
    foreground_occlusion_tolerance_m: float = .10
    multilayer_mode: str = 'legacy'
    multilayer_trigger_m: float = 1.
    cluster_gap_m: float = .15
    front_cluster_max_span_m: float = .20
    front_cluster_min_points: int = 2

    @classmethod
    def trusted_ground(cls, **kwargs):
        return cls(**dict(dict(ground_support=True, multilayer_mode='front_cluster'), **kwargs))

    @classmethod
    def trusted_ground_v3(cls, **kwargs):
        """Singleton foreground may be trusted by its measured PCA normal.

        This does not let a trusted background borrow an unknown foreground's
        visibility, even when both returns lie below the multilayer trigger.
        """
        return cls.trusted_ground(**dict(dict(front_cluster_min_points=1), **kwargs))

    def __post_init__(self):
        for key in ('min_cosine', 'ground_min_cosine', 'ground_normal_cosine'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in (0,1]')
        for key in ('correction_base_m', 'correction_range_slope', 'correction_range_start_m',
                    'max_correction_m', 'foreground_occlusion_tolerance_m',
                    'multilayer_trigger_m', 'cluster_gap_m', 'front_cluster_max_span_m'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.cluster_gap_m <= 0 or self.max_correction_m <= 0:
            raise ValueError('Gap and correction cap must be positive')
        if self.multilayer_mode not in ('legacy', 'reject', 'front_cluster'):
            raise ValueError('Unknown multilayer mode')
        if type(self.front_cluster_min_points) is not int or self.front_cluster_min_points < 1:
            raise ValueError('front_cluster_min_points must be a positive integer')
        up = np.asarray(self.world_up, float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
            raise ValueError('world_up must be a finite nonzero 3-vector')


def estimate_normals(xyz,k=16):
    xyz=np.asarray(xyz,float)
    tree=cKDTree(xyz)
    _,neighbor=tree.query(xyz,k=min(k,len(xyz)),workers=4)
    normals=[];valid=[]
    for start in range(0,len(xyz),10000):
        pts=xyz[neighbor[start:start+10000]]
        centered=pts-pts.mean(axis=1,keepdims=True)
        values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered))
        normals.append(vectors[:,:,0])
        valid.append((values[:,0] < .25*np.maximum(values[:,1],1e-12)) &
                     (values[:,1]>.01*np.maximum(values[:,2],1e-12)))
    return np.vstack(normals).astype('float32'),np.concatenate(valid)


def _front_cluster_limits(pixel, z, size, policy):
    """Sort measured returns per pixel; never hop over an isolated foreground."""
    limits = np.full(size, np.inf)
    ambiguous = np.zeros(size, bool)
    if not len(z) or policy.multilayer_mode == 'legacy':
        return limits, ambiguous, 0
    order = np.lexsort((z, pixel))
    pixel, z = pixel[order], z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pixel)) + 1]
    ends = np.r_[starts[1:], len(z)]
    separated = z[ends - 1] - z[starts] > policy.multilayer_trigger_m
    for start, end in zip(starts[separated], ends[separated]):
        pid = pixel[start]
        gaps = np.flatnonzero(np.diff(z[start:end]) > policy.cluster_gap_m)
        stop = start + int(gaps[0]) + 1 if len(gaps) else end
        if (policy.multilayer_mode == 'reject' or stop - start < policy.front_cluster_min_points
                or z[stop - 1] - z[start] > policy.front_cluster_max_span_m):
            ambiguous[pid] = True
        else:
            limits[pid] = z[stop - 1]
    return limits, ambiguous, int(separated.sum())


def project_surfel_depth(xyz,normals,normal_valid,T_world_camera,K,width,height, *, policy=None, diagnostics=None):
    """Intersect each occupied pixel ray with its local measured tangent plane.

    Only a projected measured point can support a pixel. Grazing/uncertain
    intersections and large corrections are rejected, not filled. This is a
    derived reference conditioned on the map's normals and registration.
    """
    policy = SurfelDepthPolicy() if policy is None else policy
    if not isinstance(policy, SurfelDepthPolicy):
        raise TypeError('policy must be a SurfelDepthPolicy')
    xyz, normals = np.asarray(xyz, float), np.asarray(normals, float)
    normal_valid = np.asarray(normal_valid, bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape or normal_valid.shape != (len(xyz),):
        raise ValueError('xyz/normals/normal_valid shape mismatch')
    from .points import project_depth
    # Also validates pose, K and dimensions; same integer-centre foreground rule.
    foreground,foreground_valid=project_depth(xyz,T_world_camera,K,width,height)
    foreground_flat=foreground.ravel()
    foreground_trusted=np.zeros(width*height,bool)
    depth=np.full(width*height,np.inf,dtype=np.float32)
    measured_pixels, measured_z, candidates = [], [], []
    up=np.asarray(policy.world_up,float); up=up/np.linalg.norm(up)
    stats=dict(projected_points=0, normal_rejected=0, grazing_rejected=0,
               correction_rejected=0, trusted_ground_candidates=0)
    R=T_world_camera[:3,:3];t=T_world_camera[:3,3]
    for start in range(0,len(xyz),65536):
        p=(xyz[start:start+65536]-t)@R
        n=normals[start:start+65536]@R
        valid=np.isfinite(p).all(1) & (p[:,2]>.1)
        trusted=normal_valid[start:start+65536][valid]
        p,n=p[valid],n[valid]
        uv=(p@K.T);uv=uv[:,:2]/uv[:,2:3]
        ok=np.isfinite(uv).all(1)&(uv[:,0]>=-.5)&(uv[:,0]<width-.5)&(uv[:,1]>=-.5)&(uv[:,1]<height-.5)
        p,n,uv,trusted=p[ok],n[ok],uv[ok],trusted[ok]
        u,v=np.floor(uv+.5).astype(int).T
        pid=v*width+u
        measured_pixels.append(pid); measured_z.append(p[:,2])
        stats['projected_points'] += len(p)
        length=np.linalg.norm(n,axis=1)
        trusted &= np.isfinite(n).all(1) & (length>1e-8)
        # Preserve the nearest return's measured normal evidence separately
        # from candidate plane correction. Unknown foreground is an occluder,
        # including when a farther return is less than 10cm behind it.
        nearest=p[:,2].astype(np.float32)==foreground_flat[pid]
        np.logical_or.at(foreground_trusted,pid[nearest],trusted[nearest])
        stats['normal_rejected'] += int((~trusted).sum())
        n=np.divide(n,length[:,None],out=np.zeros_like(n),where=length[:,None]>1e-8)
        ray=np.column_stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(len(u))])
        denominator=(n*ray).sum(1)
        cosine=np.abs(denominator)/np.linalg.norm(ray,axis=1)
        z=np.divide((n*p).sum(1),denominator,out=np.full(len(u),np.nan),where=np.abs(denominator)>1e-8)
        ground=policy.ground_support & (np.abs((n@R.T)@up)>=policy.ground_normal_cosine) & ((p@R.T)@up<0)
        minimum=np.where(ground,policy.ground_min_cosine,policy.min_cosine)
        # One half-pixel footprint in both axes, converted to a plane depth
        # uncertainty. The metric cap prevents permissive grazing tolerances.
        footprint=p[:,2]*np.hypot(.5/K[0,0],.5/K[1,1])
        adaptive=policy.correction_base_m+policy.correction_range_slope*np.maximum(0,p[:,2]-policy.correction_range_start_m)
        adaptive+=footprint*np.sqrt(np.maximum(0,1-cosine**2))/np.maximum(cosine,minimum)
        tolerance=np.where(ground,np.minimum(adaptive,policy.max_correction_m),.1+.01*p[:,2])
        finite=np.isfinite(z)&(z>.1)
        facing=cosine>minimum
        close=np.abs(z-p[:,2])<tolerance
        stats['grazing_rejected'] += int((trusted & ~facing).sum())
        stats['correction_rejected'] += int((trusted & facing & (~finite|~close)).sum())
        ok=trusted & facing & finite & close
        stats['trusted_ground_candidates'] += int((ok&ground).sum())
        candidates.append((pid[ok],p[ok,2],z[ok],tolerance[ok]))
    pixels=np.concatenate(measured_pixels) if measured_pixels else np.empty(0,int)
    zs=np.concatenate(measured_z) if measured_z else np.empty(0)
    limits,ambiguous,multilayer_count=_front_cluster_limits(pixels,zs,width*height,policy)
    visibility_tolerance=np.maximum(.1,foreground_flat*.01)
    enhanced=policy.ground_support or policy.multilayer_mode!='legacy'
    if enhanced:
        # A distant uncertain foreground must not acquire a metre-scale
        # occlusion allowance merely because its Z coordinate is large.
        visibility_tolerance=np.minimum(visibility_tolerance,policy.foreground_occlusion_tolerance_m)
    stats['source_occlusion_rejected']=0
    stats['plane_occlusion_rejected']=0
    for pid,source_z,z,tolerance in candidates:
        ok=(~ambiguous[pid]) & (source_z<=limits[pid])
        if enhanced:
            # v3: the source return and its ray-plane correction have different
            # uncertainties. A generous grazing-plane correction must NEVER
            # turn a farther measured return into a visible foreground source.
            # Only a candidate measured at the actual nearest return may use
            # its own plane correction across the old visibility threshold.
            # The foreground z-buffer stores float32; compare in that exact
            # storage convention, with no widened equality tolerance.
            is_nearest=source_z.astype(np.float32)==foreground_flat[pid]
            source_visible=is_nearest | (foreground_trusted[pid] &
                (source_z<=foreground_flat[pid]+visibility_tolerance[pid]))
            plane_tolerance=np.where(is_nearest,
                np.maximum(visibility_tolerance[pid],tolerance),visibility_tolerance[pid])
            plane_visible=z<=foreground_flat[pid]+plane_tolerance
            stats['source_occlusion_rejected']+=int((ok&~source_visible).sum())
            stats['plane_occlusion_rejected']+=int((ok&source_visible&~plane_visible).sum())
            ok &= source_visible & plane_visible
        pid,z,tolerance=pid[ok],z[ok],tolerance[ok]
        np.minimum.at(depth,pid,z.astype('float32'))
    depth=depth.reshape(height,width);valid=np.isfinite(depth);depth[~valid]=0
    # An uncertain foreground normal is an unknown occluder, not empty space.
    # Never label a confident background surface through that foreground.
    valid &= foreground_valid
    if not enhanced:
        # Preserve the historical default support/correction semantics. The
        # opt-in v3 branch already checked occlusion per candidate before min.
        valid &= depth<=foreground+visibility_tolerance.reshape(height,width)
    depth[~valid]=0
    if diagnostics is not None:
        diagnostics.update(stats,policy=asdict(policy), valid_pixels=int(valid.sum()),
            measured_pixels=int(foreground_valid.sum()), multilayer_pixels=multilayer_count,
            unknown_foreground_pixels=int((foreground_valid.ravel()&~foreground_trusted).sum()),
            ambiguous_pixels=int(ambiguous.sum()), unknown_pixels=int((~valid).sum()))
    return depth,valid
