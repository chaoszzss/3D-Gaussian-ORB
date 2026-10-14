"""Training-only camera-z constraints from a prepared LIO map (NS 1.1.5).

The map is source-supported supervision, not independent survey ground truth.
Evaluation and RGB losses/metrics retain Nerfstudio's implementations.
"""

from __future__ import annotations

import json
import math
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from .bounded_strategy import BoundedDefaultStrategy


class AuditedRefinementStrategy(BoundedDefaultStrategy):
    """Record actual growth/pruning, including equal-count split/prune events."""

    def _grow_gs(self, *args, **kwargs):
        duplicate, split = super()._grow_gs(*args, **kwargs)
        self.last_growth = (int(duplicate), int(split))
        return duplicate, split

    def _prune_gs(self, *args, **kwargs):
        count = super()._prune_gs(*args, **kwargs)
        self.last_pruned = int(count)
        return count


def _plane(value, name):
    value = torch.as_tensor(value)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must have shape (H,W) or (H,W,1)")
    return value


def resize_lidar_targets(depth, valid, size):
    """Return HWC targets; both depth and support use nearest interpolation.

    Invalid/nonpositive/nonfinite source depths never become valid after resize.
    The target size is the actual renderer H/W, not a guessed schedule factor.
    """
    depth = _plane(depth, "lidar_depth").detach()
    if not depth.is_floating_point():
        depth = depth.float()
    valid = _plane(valid, "lidar_depth_mask").to(depth.device).detach()
    if valid.shape != depth.shape:
        raise ValueError("LiDAR depth and support mask dimensions differ")
    known = torch.isfinite(valid) & (valid > 0) & torch.isfinite(depth) & (depth > 0)
    depth = torch.where(known, depth, 0)
    if tuple(depth.shape) != tuple(size):
        # interpolate on CPU also supports float64, for precise gradient tests.
        depth = F.interpolate(depth[None, None], size=size, mode="nearest")[0, 0]
        known = F.interpolate(known.float()[None, None], size=size, mode="nearest")[0, 0].bool()
    return depth[..., None], known[..., None]


def compute_lidar_losses(outputs, depth, valid, wdepth=0.2, walpha=0.05, *,
                         wmetric=0.0, metric_delta_m=0.15):
    """Relative Huber, alpha coverage, and optional additive metre Huber.

    wmetric=0 preserves the original two loss entries and their exact values.
    The metric term is Huber(D-Z, delta metres), not SmoothL1 divided by delta.

    No penalty is applied to unknown pixels. Empty support yields graph-connected
    zeros. Missing/nonfinite renderer outputs raise, including with empty support,
    so a broken depth renderer cannot silently turn supervision off.
    """
    for name, weight in (("wdepth", wdepth), ("walpha", walpha), ("wmetric", wmetric)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(metric_delta_m) or metric_delta_m <= 0:
        raise ValueError('metric_delta_m must be finite and positive in metres')
    prediction = outputs.get("depth")
    if prediction is None:
        raise ValueError("outputs['depth'] is required; enable output_depth_during_training")
    prediction = _plane(prediction, "outputs['depth']")
    alpha = outputs.get("accumulation")
    if alpha is None:
        raise ValueError("outputs['accumulation'] is required for LiDAR alpha support")
    alpha = _plane(alpha, "outputs['accumulation']")
    if not prediction.is_floating_point() or not alpha.is_floating_point():
        raise ValueError("Renderer depth and accumulation must be floating point")
    if prediction.shape != alpha.shape or prediction.device != alpha.device:
        raise ValueError("Renderer depth and accumulation must have matching H/W and device")
    if not torch.isfinite(prediction).all() or not torch.isfinite(alpha).all():
        raise ValueError("Nonfinite renderer depth/accumulation; refusing silent unsupervised training")
    if torch.any((alpha < 0) | (alpha > 1)):
        raise ValueError("Renderer accumulation must be in [0,1]")
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    prediction = prediction.to(dtype=dtype)
    alpha = alpha.to(dtype=dtype)
    target = _plane(depth, "lidar_depth").to(device=prediction.device, dtype=dtype)
    mask = _plane(valid, "lidar_depth_mask").to(prediction.device)
    target, mask = resize_lidar_targets(target, mask, prediction.shape)
    mask = mask[..., 0]
    if not torch.any(mask):
        # Empty reductions preserve autograd without multiplying bad unknowns by 0.
        losses = {"lidar_depth_loss": prediction.reshape(-1)[:0].sum() * wdepth,
                  "lidar_alpha_loss": alpha.reshape(-1)[:0].sum() * walpha}
        if wmetric > 0:
            losses['lidar_metric_depth_loss'] = prediction.reshape(-1)[:0].sum() * wmetric
        return losses
    relative_error = (prediction[mask] - target[..., 0][mask]) / target[..., 0][mask]
    if not torch.isfinite(relative_error).all():
        raise ValueError("Nonfinite relative depth residual on known map pixels")
    losses = {
        "lidar_depth_loss": wdepth * F.huber_loss(
            relative_error, torch.zeros_like(relative_error), delta=0.1, reduction="mean"),
        "lidar_alpha_loss": walpha * (1 - alpha[mask]).mean(),
    }
    if wmetric > 0:
        residual_m = prediction[mask] - target[..., 0][mask]
        if not torch.isfinite(residual_m).all():
            raise ValueError('Nonfinite metric depth residual on known map pixels')
        metric_loss = wmetric * F.huber_loss(residual_m, torch.zeros_like(residual_m),
                                           delta=metric_delta_m, reduction='mean')
        if not torch.isfinite(metric_loss):
            raise ValueError('Nonfinite weighted metric depth loss')
        losses['lidar_metric_depth_loss'] = metric_loss
    return losses


def tangent_scale_penalties(log_scales, target_m=.03, camera_z=None, focal=None,
                            visible=None, max_diameter_px=2.5):
    """Soft metric hinge on the two actual 2DGS axes; optional screen hinge.

    Camera depth is detached: shrinking a footprint must not reward moving a
    surface away from the camera. Physical target is a prior, not a p95 claim.
    """
    if not math.isfinite(target_m) or target_m <= 0:
        raise ValueError('Tangent metric target must be finite and positive')
    tangent_log=log_scales[:, :2].amax(dim=-1)
    metric=F.relu(tangent_log-math.log(target_m)).mean() if len(tangent_log) else log_scales.sum()*0
    screen=log_scales.sum()*0
    if camera_z is not None:
        if not math.isfinite(max_diameter_px) or max_diameter_px <= 0:
            raise ValueError('Tangent screen diameter must be finite and positive')
        z=camera_z.detach()
        known=torch.isfinite(z)&(z>.2)
        if visible is not None:
            known &= visible
        if known.any():
            tangent=tangent_log.exp().expand_as(z)
            focal=torch.as_tensor(focal,device=z.device,dtype=z.dtype).detach().expand_as(z)
            diameter=2*tangent[known]*focal[known]/z[known]
            # Normalized hinge avoids multiplying the entire objective by 20.
            screen=F.softplus(diameter/max_diameter_px-1,beta=20).mean()
    return metric,screen


def warm_start_read_budget(max_gaussians):
    """Warm loading follows the training count cap, within reader hard bounds."""
    if type(max_gaussians) is not int or not 1 <= max_gaussians <= 6_000_000:
        raise ValueError('Warm Gaussian growth budget must be an integer in 1..6000000')
    return dict(max_gaussians=max_gaussians, max_ply_bytes=2 * 1024**3)


@dataclass
class LidarSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: LidarSplatfactoModel)
    output_depth_during_training: bool = True
    lidar_depth_loss_weight: float = 0.2
    lidar_alpha_loss_weight: float = 0.05
    lidar_metric_depth_loss_weight: float = 0.0
    lidar_metric_depth_huber_delta_m: float = 0.15
    allow_depth_missing: bool = False
    lidar_surface_anchor_weight: float = 0.02
    gaussian_max_scale_m: float = 0.5
    surface_splats: bool = False
    surface_thickness_m: float = 0.015
    surface_normal_weight: float = 0.03
    lidar_reference_path: Optional[Path] = None
    surface_max_normal_displacement_m: float = 0.03
    use_2dgs_renderer: bool = False
    surface_distortion_weight: float = 0.01
    max_gaussians: int = 500000
    use_inria_surfel_renderer: bool = False
    inria_backend: Literal['original', 'aa-depth-vjp'] = 'original'
    optimize_shared_rig: bool = False
    shared_rig_prior_weight: float = 0.0005
    gaussian_initialization_path: Optional[Path] = None
    warm_opacity_cap: float = 1.0
    enable_refinement: bool = True
    reset_opacity_during_refinement: bool = True
    training_audit_path: Optional[Path] = None
    training_iterations: Optional[int] = None
    tangent_scale_target_m: float = .03
    tangent_scale_loss_weight: float = 0.
    tangent_screen_loss_weight: float = 0.
    tangent_max_diameter_px: float = 2.5


class LidarSplatfactoModel(SplatfactoModel):
    config: LidarSplatfactoModelConfig

    def populate_modules(self):
        warm_budget = (warm_start_read_budget(self.config.max_gaussians)
                       if self.config.gaussian_initialization_path is not None else None)
        backend=getattr(self.config,'inria_backend','original')
        if backend not in ('original','aa-depth-vjp') or (backend!='original' and not self.config.use_inria_surfel_renderer):
            raise ValueError('INRIA backend variant requires the matching surface renderer')
        super().populate_modules()
        self.strategy.__class__=AuditedRefinementStrategy
        self.strategy.max_gaussians=self.config.max_gaussians
        if self.config.optimize_shared_rig:
            from .rig_optimizer import SharedRigOptimizer
            self.shared_rig=SharedRigOptimizer(device=self.means.device,dtype=self.means.dtype)
        # Clone before optimizers can move the Gaussian means: seed_points may
        # share storage with the parent's initial Parameter.
        self.register_buffer('_fixed_lidar_reference', self.means.detach().clone(), persistent=False)
        self._surface_tree = None
        self._surface_reference_cpu = None
        self._nearest_surface = None
        self._last_surface_update = -1
        self._surface_normals_cpu = None
        self._nearest_normal = None
        if self.config.use_2dgs_renderer or self.config.use_inria_surfel_renderer:
            self.strategy.absgrad=False
            self.strategy.key_for_gradient='means2d' if self.config.use_inria_surfel_renderer else 'gradient_2dgs'
        self._surface_normals_valid_cpu = None
        if self.config.lidar_reference_path is not None:
            import numpy as np
            reference=np.load(self.config.lidar_reference_path)
            ref=reference['xyz'].astype('float32')
            self._fixed_lidar_reference=torch.as_tensor(ref,device=self.means.device).clone()
            self._surface_reference_cpu=ref
            self._surface_normals_cpu=reference['normals'].astype('float32')
            self._surface_normals_valid_cpu=reference['normal_valid'].astype(bool)
        if self.config.surface_splats:
            import numpy as np
            from scipy.spatial import cKDTree
            from scipy.spatial.transform import Rotation
            ref = self._fixed_lidar_reference.detach().cpu().numpy().copy()
            self._surface_tree = cKDTree(ref)
            if self._surface_normals_cpu is None:
                _, neighbors = self._surface_tree.query(ref, k=min(16,len(ref)), workers=4)
                normals=[]
                for start in range(0,len(ref),10000):
                    pts=ref[neighbors[start:start+10000]]
                    centered=pts-pts.mean(axis=1,keepdims=True)
                    covariance=np.einsum('nki,nkj->nij',centered,centered)
                    _, vectors=np.linalg.eigh(covariance)
                    normals.append(vectors[:,:,0])
                normal=np.concatenate(normals)
            else:
                normal=self._surface_normals_cpu
            normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
            guide=np.tile(np.array([1.,0.,0.]),(len(normal),1))
            guide[np.abs(normal[:,0])>.9]=[0.,1.,0.]
            tangent=np.cross(guide,normal);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
            bitangent=np.cross(normal,tangent)
            rotation=np.stack([tangent,bitangent,normal],axis=-1)
            _,seed_indices=self._surface_tree.query(self.means.detach().cpu().numpy(),workers=4)
            quat=Rotation.from_matrix(rotation[seed_indices]).as_quat()[:,[3,0,1,2]].copy()
            with torch.no_grad():
                self.quats.copy_(torch.as_tensor(quat,device=self.quats.device,dtype=self.quats.dtype))
                self.scales[:,2].clamp_(max=math.log(self.config.surface_thickness_m))
            self._surface_reference_cpu=ref
            self._surface_normals_cpu=normal.astype('float32')
        if self.config.gaussian_initialization_path is not None:
            # Populate before Trainer creates optimizers, so Adam references the
            # actual warm parameters rather than discarded seed Parameters.
            import sys
            project = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(project / 'scripts'))
            from evaluate_gaussian_ply import read_candidate
            arrays, info = read_candidate(self.config.gaussian_initialization_path, **warm_budget)
            if info['gaussians'] > self.config.max_gaussians:
                raise ValueError('Warm Gaussian count exceeds configured growth budget')
            if info['sh_degree'] != self.config.sh_degree or set(arrays) != set(self.gauss_params):
                raise ValueError('Warm source Gaussian schema/SH differs from model')
            cap = self.config.warm_opacity_cap
            if not 0 < cap <= 1:
                raise ValueError('Warm opacity cap must be in (0,1]')
            device = self.means.device
            if cap < 1:
                import numpy as np
                cap_logit = math.log(cap / (1 - cap))
                changed = int((arrays['opacities'] > cap_logit).sum())
                arrays['opacities'] = np.minimum(arrays['opacities'], cap_logit)
            else:
                changed = 0
            for key, values in arrays.items():
                self.gauss_params[key] = torch.nn.Parameter(torch.as_tensor(values, device=device))
            self.warm_start_info = dict(info, opacity_cap=cap, initial_opacities_capped=changed,
                                        optimizer_protocol='new Adam after parameter initialization')

        self.refinement_audit = dict(schema_version=1, initial_gaussians=len(self.means), events=[],
            enabled=self.config.enable_refinement,
            opacity_resets_enabled=self.config.reset_opacity_during_refinement)
        self._audit_resumed = False
        if self.config.training_audit_path is not None and Path(self.config.training_audit_path).is_file():
            previous=json.loads(Path(self.config.training_audit_path).read_text())
            if (previous.get('initial_gaussians') != len(self.means)
                    or previous.get('enabled') != self.config.enable_refinement):
                raise ValueError('Existing refinement audit belongs to a different initialization/config')
            self.refinement_audit=previous
            self._audit_resumed=True
        if not self.config.reset_opacity_during_refinement:
            # Upstream resets at step zero too; the callback below skips it.
            self.strategy.reset_every = max(1, self.config.stop_split_at + 1)
            self.strategy.pause_refine_after_reset = 0

    def step_post_backward(self, step):
        if getattr(self,'_audit_resumed',False):
            # A crash may leave an audit newer than the last saved checkpoint.
            # Preserve only events before the resumed iteration.
            self.refinement_audit['events']=[e for e in self.refinement_audit['events'] if e['step']<step]
            self._audit_resumed=False
        strategy = self.strategy
        active = (self.config.enable_refinement and step < strategy.refine_stop_iter)
        event = (active and step > strategy.refine_start_iter
                 and step % strategy.refine_every == 0
                 and step % strategy.reset_every >= strategy.pause_refine_after_reset)
        before = len(self.means)
        if active and (step != 0 or self.config.reset_opacity_during_refinement):
            super().step_post_backward(step)
        # Apply the metric bound after optimizer/refine too, so final saved PLY
        # cannot exceed it by one optimizer update.
        if self.config.gaussian_max_scale_m > 0:
            with torch.no_grad():
                self.scales.clamp_(max=math.log(self.config.gaussian_max_scale_m))
                if self.config.surface_splats:
                    self.scales[:, 2].clamp_(max=math.log(self.config.surface_thickness_m))
        if event:
            duplicate, split = strategy.last_growth
            if duplicate or split or strategy.last_pruned:
                # Topology changes invalidate index-aligned references even when
                # growth and pruning cancel out to the same Gaussian count.
                self._nearest_surface = None
                self._nearest_normal = None
                self._nearest_normal_valid = None
                self._last_surface_update = -1
            self.refinement_audit['events'].append(dict(step=int(step), before=before,
                after=len(self.means), duplicated=duplicate, split=split, pruned=strategy.last_pruned))
        entries=self.refinement_audit.get('events',[])
        self.refinement_audit.update(last_callback_step=int(step), final_gaussians=len(self.means),
            refine_events=len(entries), duplicated_total=sum(e['duplicated'] for e in entries),
            split_total=sum(e['split'] for e in entries), pruned_total=sum(e['pruned'] for e in entries))
        last_step=self.config.training_iterations is not None and step==self.config.training_iterations-1
        if self.config.training_audit_path is not None and (event or step % 100 == 0 or last_step):
            path = Path(self.config.training_audit_path)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.refinement_audit, indent=2))
            temporary.replace(path)

    def get_outputs(self, camera):
        if self.config.optimize_shared_rig:
            corrected=self.shared_rig.correct(camera)
            camera=copy.copy(camera)
            if camera.metadata is not None:camera.metadata=dict(camera.metadata)
            camera.camera_to_worlds=corrected
        if self.training and self.config.gaussian_max_scale_m > 0:
            with torch.no_grad():
                self.scales.clamp_(max=math.log(self.config.gaussian_max_scale_m))
                if self.config.surface_splats:
                    self.scales[:,2].clamp_(max=math.log(self.config.surface_thickness_m))
                    if self._nearest_surface is not None and len(self._nearest_surface)==len(self.means):
                        delta=self.means-self._nearest_surface
                        trusted=self._nearest_normal_valid & (torch.linalg.vector_norm(delta,dim=-1)<.5)
                        normal=self._nearest_normal
                        signed=(delta*normal).sum(-1)
                        limit=self.config.surface_max_normal_displacement_m
                        correction=(signed-signed.clamp(-limit,limit))*trusted
                        self.means.sub_(correction[:,None]*normal)
                        q=self.quats/torch.linalg.vector_norm(self.quats,dim=-1,keepdim=True).clamp_min(1e-12)
                        w,x,y,z=q.unbind(-1)
                        current=torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],dim=-1)
                        cosine=(current*normal).sum(-1)
                        target=normal*torch.where(cosine<0,-1.,1.)[:,None]
                        cosine=cosine.abs().clamp(0,1)
                        aw=torch.sqrt((1+cosine)*.5)
                        av=torch.linalg.cross(current,target,dim=-1)/(2*aw[:,None]).clamp_min(1e-12)
                        qw=aw*w-(av*q[:,1:]).sum(-1)
                        qv=aw[:,None]*q[:,1:]+w[:,None]*av+torch.linalg.cross(av,q[:,1:],dim=-1)
                        aligned=torch.cat([qw[:,None],qv],dim=-1)
                        self.quats.copy_(torch.where(trusted[:,None],aligned,q))
        if not (self.config.use_2dgs_renderer or self.config.use_inria_surfel_renderer):
            return super().get_outputs(camera)
        import inspect
        import nerfstudio.models.splatfacto as ns_splat
        from gsplat.rendering import rasterization_2dgs
        original=ns_splat.rasterization
        accepted=set(inspect.signature(rasterization_2dgs).parameters)
        def surface_rasterization(**kwargs):
            if self.config.use_inria_surfel_renderer:
                from .inria_surfel_backend import render_surfel
                result=render_surfel(**{k:kwargs[k] for k in ('means','quats','scales','opacities','colors','viewmats','Ks','width','height','sh_degree')},
                    backend=getattr(self.config,'inria_backend','original'))
                rendered=torch.cat([result.rgb,result.depth_expected],dim=-1) if kwargs['render_mode']=='RGB+ED' else result.rgb
                result.info['render_distort']=result.distortion
                result.info['camera_z']=torch.einsum('cj,nj->cn',kwargs['viewmats'][:,2,:3],kwargs['means'])+kwargs['viewmats'][:,2,3:4]
                result.info['focal_max']=kwargs['Ks'][:,(0,1),(0,1)].amax(-1,keepdim=True)
                return rendered,result.alpha,result.info
            values=rasterization_2dgs(**{k:v for k,v in kwargs.items() if k in accepted},distloss=True)
            meta=values[-1]
            meta['camera_z']=torch.einsum('cj,nj->cn',kwargs['viewmats'][:,2,:3],kwargs['means'])+kwargs['viewmats'][:,2,3:4]
            meta['focal_max']=kwargs['Ks'][:,(0,1),(0,1)].amax(-1,keepdim=True)
            return values[0],values[1],meta
        ns_splat.rasterization=surface_rasterization
        try:
            return super().get_outputs(camera)
        finally:
            ns_splat.rasterization=original

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        losses = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self.training:
            return losses
        if self.config.tangent_scale_loss_weight > 0 or self.config.tangent_screen_loss_weight > 0:
            screen_enabled=self.config.tangent_screen_loss_weight>0
            if screen_enabled and 'camera_z' not in self.info:
                raise ValueError('Screen tangent constraint requires surface renderer camera metadata')
            radii=self.info['radii'] if screen_enabled else None
            visible=(radii.amax(-1)>0 if radii.ndim==3 else radii>0) if screen_enabled else None
            metric,screen=tangent_scale_penalties(self.scales,self.config.tangent_scale_target_m,
                self.info['camera_z'] if screen_enabled else None,
                self.info['focal_max'] if screen_enabled else None,visible,self.config.tangent_max_diameter_px)
            losses['tangent_scale_loss']=self.config.tangent_scale_loss_weight*metric
            if screen_enabled:
                losses['tangent_screen_loss']=self.config.tangent_screen_loss_weight*screen
        if self.config.optimize_shared_rig:
            for key,value in self.shared_rig.regularization().items():
                losses[key]=self.config.shared_rig_prior_weight*value
            # Fixed raster depths describe the old camera. They cannot supervise
            # an evolving extrinsic; measured surface anchors remain active below.
        if "lidar_depth" not in batch:
            if self.config.allow_depth_missing or batch.get("lidar_depth_missing_allowed") is True:
                return losses
            raise ValueError("Training batch has no lidar_depth; use LidarFullImageDatamanagerConfig")
        if "lidar_depth_mask" not in batch:
            raise ValueError("Training batch has no lidar_depth_mask")
        depth, valid = batch["lidar_depth"], batch["lidar_depth_mask"]
        if "mask" in batch:
            # Keep the original RGB mask path in the parent loss untouched.
            image_mask = _plane(batch["mask"], "mask").to(torch.as_tensor(depth).device)
            valid = _plane(valid, "lidar_depth_mask")
            if image_mask.shape != valid.shape:
                image_mask = F.interpolate(
                    image_mask.float()[None, None], size=valid.shape, mode="nearest")[0, 0]
            valid = (valid > 0) & torch.isfinite(image_mask) & (image_mask > 0)
        if not self.config.optimize_shared_rig:
            losses.update(compute_lidar_losses(
                outputs, depth, valid,
                self.config.lidar_depth_loss_weight, self.config.lidar_alpha_loss_weight,
                wmetric=self.config.lidar_metric_depth_loss_weight,
                metric_delta_m=self.config.lidar_metric_depth_huber_delta_m))
        if self.config.use_2dgs_renderer or self.config.use_inria_surfel_renderer:
            losses['surfel_distortion_loss']=self.config.surface_distortion_weight*self.info['render_distort'].mean()
        if self.config.lidar_surface_anchor_weight > 0:
            from scipy.spatial import cKDTree
            if self._surface_tree is None:
                self._surface_reference_cpu = self._fixed_lidar_reference.detach().cpu().numpy().copy()
                self._surface_tree = cKDTree(self._surface_reference_cpu)
            changed = self._nearest_surface is None or len(self._nearest_surface) != len(self.means)
            if changed or self.step - self._last_surface_update >= 250:
                _, ids = self._surface_tree.query(self.means.detach().cpu().numpy(), workers=4)
                self._nearest_surface = torch.from_numpy(self._surface_reference_cpu[ids]).to(self.means)
                if self.config.surface_splats:
                    self._nearest_normal = torch.from_numpy(self._surface_normals_cpu[ids]).to(self.means)
                    self._nearest_normal_valid = torch.ones(len(ids),dtype=torch.bool,device=self.means.device) if self._surface_normals_valid_cpu is None else torch.as_tensor(self._surface_normals_valid_cpu[ids],device=self.means.device)
                self._last_surface_update = self.step
            distance = torch.linalg.vector_norm(self.means - self._nearest_surface, dim=-1)
            # A measured-surface prior, not independent GT; permits local subvoxel
            # refinements while discouraging arbitrarily distant image billboards.
            losses['lidar_surface_anchor_loss'] = self.config.lidar_surface_anchor_weight * F.smooth_l1_loss(
                distance / .1, torch.zeros_like(distance), beta=1.)
            if self.config.surface_splats:
                quat=self.quats/torch.linalg.vector_norm(self.quats,dim=-1,keepdim=True).clamp_min(1e-12)
                w,x,y,z=quat.unbind(-1)
                normal=torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],dim=-1)
                cosine=(normal*self._nearest_normal).sum(-1)
                error=(1-cosine.square()).clamp_min(0)
                losses['lidar_surface_normal_loss']=self.config.surface_normal_weight*error[self._nearest_normal_valid].mean() if self._nearest_normal_valid.any() else error.sum()*0
        return losses

    def get_param_groups(self):
        groups=super().get_param_groups()
        if self.config.optimize_shared_rig:groups.update(self.shared_rig.get_param_groups())
        return groups


@dataclass
class LidarFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: LidarFullImageDatamanager)
    allow_depth_missing: bool = False
    lidar_depth_cache_max_bytes: int = 512 * 1024 * 1024
    """Bounded CPU LRU cache; zero disables retention. No on-disk cache."""


def _asset(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Prepared asset path must be a nonempty relative string")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Prepared asset must stay inside transforms directory: {name}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Prepared asset resolves outside transforms directory: {name}")
    return resolved


def _mask_file(path, shape):
    if path.suffix == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            mask = np.array(image)
    mask = _plane(mask, str(path))
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"Prepared mask dimensions differ from depth: {path}")
    return torch.isfinite(mask) & (mask > 0)


class LidarFullImageDatamanager(FullImageDatamanager[InputDataset]):
    config: LidarFullImageDatamanagerConfig

    def _initialize_lidar(self):
        if hasattr(self, "_lidar_frames"):
            return
        budget = self.config.lidar_depth_cache_max_bytes
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("lidar_depth_cache_max_bytes must be a nonnegative integer")
        data_path = Path(self.config.dataparser.data).resolve()
        transforms = data_path if data_path.suffix == ".json" else data_path / "transforms.json"
        root = transforms.parent
        with transforms.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        declared_missing = metadata.get("allow_depth_missing", False)
        if not isinstance(declared_missing, bool) or not isinstance(self.config.allow_depth_missing, bool):
            raise ValueError("allow_depth_missing must be an explicit boolean")
        allow_missing = self.config.allow_depth_missing or declared_missing
        if "train_filenames" not in metadata:
            raise ValueError("LiDAR training requires explicit train_filenames in prepared transforms")
        train = {_asset(root, name) for name in metadata["train_filenames"]}
        held_out = {_asset(root, name) for split in ("val", "test")
                    for name in metadata.get(f"{split}_filenames", [])}
        if train & held_out:
            raise ValueError("Prepared training and held-out filenames overlap")
        # NS sorts frames and can redirect images to images_2, images_4, etc.
        # Build by actual selected filename, never by transforms frame position.
        downscale = self.dataparser.downscale_factor
        if downscale is None:
            raise ValueError("Nerfstudio dataparser downscale factor is unset")
        by_image = {}
        for frame in metadata["frames"]:
            original = _asset(root, frame["file_path"])
            if original not in train:
                continue  # Do not open, check, or retain held-out depth assets.
            image = (root / f"images_{downscale}" / original.name).resolve() if downscale > 1 else original
            if image in by_image:
                raise ValueError(f"Ambiguous prepared training image: {image}")
            by_image[image] = frame
        frames = []
        physical_ids=[]
        for image in self.train_dataset.image_filenames:
            frame = by_image.get(Path(image).resolve())
            if frame is None:
                raise ValueError(f"Training image has no declared training frame: {image}")
            name=frame.get('camera_id','')
            physical_ids.append(int(name.split('_')[-1]) if name.startswith('camera_') else -1)
            paths = []
            for key in ("lidar_depth_path", "lidar_depth_mask_path"):
                path = _asset(root, frame[key]) if frame.get(key) else None
                paths.append(path)
            if any(path is None or not path.is_file() for path in paths):
                if not allow_missing:
                    raise ValueError(f"Missing LiDAR depth/mask for training image: {image}")
                frames.append(None)
                continue
            image_mask = _asset(root, frame["mask_path"]) if frame.get("mask_path") else None
            if image_mask is not None and not image_mask.is_file():
                raise ValueError(f"Missing image validity mask: {image_mask}")
            shape = (int(frame.get("h", metadata.get("h", 0))),
                     int(frame.get("w", metadata.get("w", 0))))
            if min(shape) <= 0:
                raise ValueError(f"Prepared frame needs positive h/w: {image}")
            frames.append((*paths, image_mask, shape))
        cameras = self.train_dataparser_outputs.cameras
        if torch.any(cameras.camera_type != CameraType.PERSPECTIVE.value) or (
            cameras.distortion_params is not None and torch.any(cameras.distortion_params != 0)
        ):
            raise ValueError("LiDAR supervision requires already rectified perspective images")
        scale = float(self.train_dataparser_outputs.dataparser_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("dataparser_scale must be finite and positive")
        self._lidar_scale = scale
        self._lidar_allow_missing = allow_missing
        self._lidar_depth_cache = OrderedDict()
        self._lidar_depth_cache_bytes = 0
        self._lidar_frames = frames
        self._physical_camera_ids=physical_ids

    def _load_lidar_frame(self, image_idx):
        cached = self._lidar_depth_cache.get(image_idx)
        if cached is not None:
            self._lidar_depth_cache.move_to_end(image_idx)
            return cached
        entry = self._lidar_frames[image_idx]
        if entry is None:
            return None
        depth_path, support_path, image_mask_path, shape = entry
        if depth_path.suffix != ".npy":
            raise ValueError(f"LiDAR camera-z depth must be a float .npy in metres: {depth_path}")
        array = np.load(depth_path, allow_pickle=False)
        if array.dtype.kind != "f" or array.shape != shape:
            raise ValueError(f"LiDAR depth must be floating point with prepared H/W: {depth_path}")
        depth = torch.from_numpy(array.astype(np.float32, copy=True)) * self._lidar_scale
        valid = _mask_file(support_path, shape)
        if image_mask_path is not None:
            valid &= _mask_file(image_mask_path, shape)
        depth, valid = resize_lidar_targets(depth, valid, shape)
        value = (depth, valid)
        nbytes = sum(t.numel() * t.element_size() for t in value)
        budget = self.config.lidar_depth_cache_max_bytes
        if nbytes <= budget:
            while self._lidar_depth_cache_bytes + nbytes > budget:
                _, old = self._lidar_depth_cache.popitem(last=False)
                self._lidar_depth_cache_bytes -= sum(t.numel() * t.element_size() for t in old)
            self._lidar_depth_cache[image_idx] = value
            self._lidar_depth_cache_bytes += nbytes
        return value

    def next_train(self, step):
        self._initialize_lidar()  # Fail on missing train metadata on the first call.
        camera, data = super().next_train(step)
        if camera.metadata is None:camera.metadata={}
        camera.metadata['physical_camera_id']=self._physical_camera_ids[int(data['image_idx'])]
        image_idx = int(data["image_idx"])
        if not 0 <= image_idx < len(self._lidar_frames):
            raise ValueError(f"Training image_idx outside selected dataset: {image_idx}")
        targets = self._load_lidar_frame(image_idx)
        if targets is None:
            data["lidar_depth_missing_allowed"] = self._lidar_allow_missing
            return camera, data
        # Keep cached targets on CPU; only the current image goes to the device.
        depth, valid = resize_lidar_targets(*targets, data["image"].shape[:2])
        if "mask" in data:
            mask = _plane(data["mask"], "mask").to("cpu")
            if mask.shape != valid.shape[:2]:
                raise ValueError("Training image/mask dimensions differ")
            valid = valid & (torch.isfinite(mask) & (mask > 0))[..., None]
        data["lidar_depth"] = depth.to(self.device)
        data["lidar_depth_mask"] = valid.to(self.device)
        return camera, data
