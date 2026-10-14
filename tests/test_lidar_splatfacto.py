"""CPU-only contract checks; these are not reconstruction/held-out scores."""

import json
import math
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import SSIM

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

from orbvi_gaussian.lidar_splatfacto import (
    LidarFullImageDatamanager,
    LidarFullImageDatamanagerConfig,
    LidarSplatfactoModel,
    LidarSplatfactoModelConfig,
    compute_lidar_losses,
    resize_lidar_targets,
)


def check_perfect_depth_and_full_alpha_are_zero():
    depth = torch.tensor([[2., 4.]], dtype=torch.float64)
    predicted = depth.clone().requires_grad_()
    alpha = torch.ones_like(depth, requires_grad=True)
    losses = compute_lidar_losses({"depth": predicted, "accumulation": alpha}, depth, depth > 0)
    assert all(loss.item() == 0 for loss in losses.values())
    sum(losses.values()).backward()
    assert torch.equal(predicted.grad, torch.zeros_like(depth))


def check_relative_huber_exact_value_gradients_and_unknown_pixels():
    target = torch.tensor([[2., 99.], [0., float("nan")]], dtype=torch.float64)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 1] = False
    predicted = torch.tensor([[3., 700.], [8., 9.]], dtype=torch.float64, requires_grad=True)
    alpha = torch.full_like(predicted, 0.4, requires_grad=True)

    def objective(p, a):
        return sum(compute_lidar_losses({"depth": p, "accumulation": a}, target, valid).values())

    assert torch.autograd.gradcheck(objective, (predicted, alpha))
    losses = compute_lidar_losses({"depth": predicted, "accumulation": alpha}, target, valid)
    assert math.isclose(losses["lidar_depth_loss"].item(), 0.2 * (0.1 * 0.5 - 0.005))
    assert math.isclose(losses["lidar_alpha_loss"].item(), 0.05 * 0.6)
    sum(losses.values()).backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([[0.01, 0.], [0., 0.]], dtype=torch.float64))
    torch.testing.assert_close(alpha.grad, torch.tensor([[-0.05, 0.], [0., 0.]], dtype=torch.float64))
    scaled = compute_lidar_losses({"depth": predicted * 10, "accumulation": alpha}, target * 10, valid)
    torch.testing.assert_close(scaled["lidar_depth_loss"], losses["lidar_depth_loss"])


def check_nearest_depth_and_mask_use_actual_odd_output_dimensions():
    depth = torch.arange(1, 36, dtype=torch.float64).reshape(5, 7)
    mask = torch.zeros(5, 7, dtype=torch.bool)
    mask[0, 0] = mask[0, 2] = mask[2, 4] = mask[1, 1] = True
    depth[0, 2] = float("nan")
    target, known = resize_lidar_targets(depth, mask, (2, 3))
    torch.testing.assert_close(target[..., 0], torch.tensor([[1., 0., 0.], [0., 0., 19.]], dtype=torch.float64))
    assert torch.equal(known[..., 0], torch.tensor([[True, False, False], [False, False, True]]))
    losses = compute_lidar_losses(
        {"depth": target.clone().requires_grad_(), "accumulation": torch.ones_like(target)}, depth, mask)
    assert losses["lidar_depth_loss"].item() == 0


def check_empty_support_has_differentiable_zero():
    predicted = torch.ones(2, 3, 1, requires_grad=True)
    alpha = torch.zeros(2, 3, 1, requires_grad=True)
    losses = compute_lidar_losses(
        {"depth": predicted, "accumulation": alpha}, torch.zeros(4, 6), torch.ones(4, 6))
    assert all(value.requires_grad and value.item() == 0 for value in losses.values())
    sum(losses.values()).backward()
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))
    assert torch.equal(alpha.grad, torch.zeros_like(alpha))


def check_broken_renderer_raises_even_without_valid_targets(broken):
    predicted = None if broken is None else torch.full((2, 3), broken)
    with unittest.TestCase().assertRaisesRegex(ValueError, "depth|Nonfinite"):
        compute_lidar_losses({"depth": predicted, "accumulation": torch.ones(2, 3)},
                             torch.zeros(2, 3), torch.zeros(2, 3))


class CpuLossHarness(LidarSplatfactoModel):
    """Use the real parent RGB loss, without renderer or perceptual downloads."""

    def populate_modules(self):
        self.step = 0
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.camera_optimizer = CameraOptimizerConfig(mode="off").setup(num_cameras=1, device="cpu")


def check_model_preserves_parent_rgb_loss_and_does_not_supervise_eval(metric_weight=0.):
    config = LidarSplatfactoModelConfig(num_downscales=1,lidar_surface_anchor_weight=0.,
                                      lidar_metric_depth_loss_weight=metric_weight)
    assert config._target is LidarSplatfactoModel and config.output_depth_during_training
    model = CpuLossHarness(config, SceneBox(aabb=torch.tensor([[-1.] * 3, [1.] * 3])), 1)
    image = torch.linspace(0, 1, 33 * 35 * 3).reshape(33, 35, 3)
    batch = {"image": image, "mask": torch.ones(33, 35, 1),
             "lidar_depth": torch.full((33, 35, 1), 2.),
             "lidar_depth_mask": torch.ones(33, 35, 1, dtype=torch.bool)}
    batch["mask"][:3] = 0
    outputs = {"rgb": torch.zeros(16, 17, 3, requires_grad=True), "background": torch.zeros(3),
               "depth": torch.full((16, 17, 1), 3., requires_grad=True),
               "accumulation": torch.full((16, 17, 1), 0.5, requires_grad=True)}
    parent = SplatfactoModel.get_loss_dict(model, outputs, batch)
    losses = model.get_loss_dict(outputs, batch)
    assert ('lidar_metric_depth_loss' in losses)==(metric_weight>0)
    for key, value in parent.items():
        torch.testing.assert_close(losses[key], value)
    sum(losses.values()).backward()
    assert torch.all(outputs["depth"].grad[:2] == 0)  # nearest source rows 0,2
    assert torch.all(outputs["accumulation"].grad[:2] == 0)
    with unittest.TestCase().assertRaisesRegex(ValueError, "no lidar_depth"):
        model.get_loss_dict(outputs, {"image": image})
    assert "lidar_depth_loss" not in model.get_loss_dict(
        outputs, {"image": image, "lidar_depth_missing_allowed": True})
    model.eval()
    # Even accidentally attached depth cannot affect evaluation loss/metrics.
    batch["lidar_depth"][:] = float("nan")
    outputs["rgb"] = torch.zeros_like(image)
    assert set(model.get_loss_dict(outputs, batch)) == set(parent)
    model.step = 500  # Pipeline's default periodic eval batch has no LiDAR GT.
    eval_batch = {"image": image, "mask": batch["mask"]}
    outputs["depth"] = None
    outputs["accumulation"] = None
    eval_losses = model.get_loss_dict(outputs, eval_batch)
    parent_eval = SplatfactoModel.get_loss_dict(model, outputs, eval_batch)
    assert set(eval_losses) == set(parent_eval)
    for key, value in parent_eval.items():
        torch.testing.assert_close(eval_losses[key], value)
    assert LidarSplatfactoModel.get_metrics_dict is SplatfactoModel.get_metrics_dict
    assert LidarSplatfactoModel.get_image_metrics_and_images is SplatfactoModel.get_image_metrics_and_images


def test_model_metric_term_uses_same_rgb_mask_resize_and_training_only():
    check_model_preserves_parent_rgb_loss_and_does_not_supervise_eval(metric_weight=.05)


def prepare_dataset(tmp_path):
    for folder in ("images", "masks", "depth", "depth_masks"):
        (tmp_path / folder).mkdir()
    frames = []
    for number, name in enumerate(("c", "d", "b", "a")):
        Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(tmp_path / "images" / f"{name}.png")
        image_mask = np.full((2, 3), 255, dtype=np.uint8)
        image_mask[0, 1] = 0
        Image.fromarray(image_mask).save(tmp_path / "masks" / f"{name}.png")
        value = {"a": 2., "b": 4., "c": 8., "d": 6.}[name]
        depth = np.array([[value] * 3, [float("nan"), 0., -1.]], dtype=np.float32)
        if name != "c":  # Held-out depth is deliberately absent and must never be read.
            np.save(tmp_path / "depth" / f"{name}.npy", depth)
            Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(
                tmp_path / "depth_masks" / f"{name}.png")
        pose = np.eye(4)
        pose[0, 3] = number
        frames.append(dict(file_path=f"images/{name}.png", mask_path=f"masks/{name}.png",
                           lidar_depth_path=f"depth/{name}.npy", lidar_depth_mask_path=f"depth_masks/{name}.png",
                           h=2, w=3, fl_x=2., fl_y=2., cx=1.5, cy=1., transform_matrix=pose.tolist()))
    metadata = dict(frames=frames, train_filenames=["images/d.png", "images/a.png", "images/b.png"],
                    val_filenames=["images/c.png"], test_filenames=[])
    (tmp_path / "transforms.json").write_text(json.dumps(metadata))
    return tmp_path, metadata


def make_manager(prepared, budget=30, downscale=1):
    root, _ = prepared
    config = LidarFullImageDatamanagerConfig(
        dataparser=NerfstudioDataParserConfig(data=root, downscale_factor=downscale,
                                            auto_scale_poses=False, scale_factor=2.,
                                            orientation_method="none", center_method="none", load_3D_points=False),
        cache_images="cpu", lidar_depth_cache_max_bytes=budget)
    assert config._target is LidarFullImageDatamanager
    manager = config.setup(device="cpu")
    # Avoid CUDA-backed pinned-memory allocation, while testing real NS sampling.
    manager.cached_train = [manager.train_dataset.get_data(i) for i in range(3)]
    manager.cached_eval = [manager.eval_dataset.get_data(0)]
    manager.train_cameras = manager.train_dataset.cameras
    manager.train_unseen_cameras = [1, 0, 2]
    return manager


def check_loader_maps_subset_index_applies_scale_masks_and_bounds_cpu_cache(prepared):
    manager = make_manager(prepared)
    _, batch = manager.next_train(999)
    assert batch["image_idx"] == 1
    assert batch["lidar_depth"][0, 0, 0].item() == 8  # 'b', not global frame 1 ('d')
    assert torch.equal(batch["lidar_depth_mask"][..., 0],
                       torch.tensor([[True, False, True], [False, False, False]]))
    assert manager._lidar_depth_cache_bytes == 30
    assert all(t.device.type == "cpu" for t in manager._lidar_depth_cache[1])
    assert "lidar_depth" not in manager.cached_train[1]
    manager._load_lidar_frame(1)  # Cache hit does not change accounting.
    assert manager._lidar_depth_cache_bytes == 30
    manager.next_train(1000)
    assert list(manager._lidar_depth_cache) == [0]  # Evicts previous frame.
    _, evaluation = manager.next_eval(0)
    assert "lidar_depth" not in evaluation and "lidar_depth_mask" not in evaluation


def check_missing_training_assets_fail_first_call_unless_metadata_explicit(prepared, allow_missing):
    root, metadata = prepared
    del metadata["frames"][1]["lidar_depth_path"]  # 'd', which is not first sampled.
    metadata["allow_depth_missing"] = allow_missing
    (root / "transforms.json").write_text(json.dumps(metadata))
    manager = make_manager(prepared)
    if not allow_missing:
        with unittest.TestCase().assertRaisesRegex(ValueError, "Missing LiDAR"):
            manager.next_train(0)
    else:
        manager.train_unseen_cameras = [2, 0, 1]
        _, batch = manager.next_train(0)
        assert "lidar_depth" not in batch and batch["lidar_depth_missing_allowed"] is True


def check_parser_image_downscale_uses_nearest_lidar_targets(prepared):
    root, _ = prepared
    for folder in ("images_2", "masks_2"):
        (root / folder).mkdir()
    for name in ("a", "b", "c", "d"):
        Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(root / "images_2" / f"{name}.png")
        Image.fromarray(np.full((1, 1), 255, dtype=np.uint8)).save(root / "masks_2" / f"{name}.png")
    _, batch = make_manager(prepared, downscale=2).next_train(0)
    assert batch["lidar_depth"].shape == batch["lidar_depth_mask"].shape == (1, 1, 1)
    assert batch["lidar_depth"].item() == 8 and batch["lidar_depth_mask"].item() is True


def check_config_can_be_mapped_without_preserving_parent_target():
    base = SplatfactoModelConfig(num_downscales=1, output_depth_during_training=False)
    mapped = {f.name: getattr(base, f.name) for f in fields(base)
              if f.name not in {"_target", "output_depth_during_training"}}
    candidate = LidarSplatfactoModelConfig(**mapped)
    assert candidate._target is LidarSplatfactoModel and candidate.output_depth_during_training


class LidarSplatfactoCpuTests(unittest.TestCase):
    # unittest is included with Python; the pinned training venv has no pytest.
    # This TestCase is also discoverable by pytest when used in a larger suite.
    test_perfect_depth = staticmethod(check_perfect_depth_and_full_alpha_are_zero)
    test_gradients_and_unknown_pixels = staticmethod(check_relative_huber_exact_value_gradients_and_unknown_pixels)
    test_odd_nearest_downscale = staticmethod(check_nearest_depth_and_mask_use_actual_odd_output_dimensions)
    test_empty_valid = staticmethod(check_empty_support_has_differentiable_zero)
    test_parent_loss_and_eval = staticmethod(check_model_preserves_parent_rgb_loss_and_does_not_supervise_eval)
    test_config_mapping = staticmethod(check_config_can_be_mapped_without_preserving_parent_target)

    def test_broken_renderer(self):
        for broken in (None, float("nan"), float("inf")):
            with self.subTest(broken=broken):
                check_broken_renderer_raises_even_without_valid_targets(broken)

    def test_loader(self):
        with tempfile.TemporaryDirectory(prefix="lidar-test-") as directory:
            check_loader_maps_subset_index_applies_scale_masks_and_bounds_cpu_cache(
                prepare_dataset(Path(directory)))

    def test_missing_assets_policy(self):
        for allow_missing in (False, True):
            with self.subTest(allow_missing=allow_missing), tempfile.TemporaryDirectory(prefix="lidar-test-") as directory:
                check_missing_training_assets_fail_first_call_unless_metadata_explicit(
                    prepare_dataset(Path(directory)), allow_missing)

    def test_parser_downscale(self):
        with tempfile.TemporaryDirectory(prefix="lidar-test-") as directory:
            check_parser_image_downscale_uses_nearest_lidar_targets(prepare_dataset(Path(directory)))


def test_training_cli_12k_and_inria_rejects_unsupported_camera_gradients():
    import importlib.util
    import pytest
    script = Path(__file__).resolve().parents[1] / 'scripts/train_lidar_block.py'
    spec = importlib.util.spec_from_file_location('train_lidar_cli_cpu', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    args = cli.parser_for_cli().parse_args(['--dataset', 'data/x', '--output', 'runs/x', '--renderer-inria'])
    cli.validate_args(args)
    assert (args.iterations, args.refine_every, args.stop_split_at) == (12000, 100, 11000)
    assert not args.no_refine and not args.reset_opacity and args.warm_opacity_cap == 1.
    assert args.vis == 'viewer'
    for mode in ('tensorboard', 'viewer+tensorboard'):
        with __import__('tempfile').TemporaryDirectory() as logdir:
            from torch.utils.tensorboard import SummaryWriter
            logger_args=cli.parser_for_cli().parse_args([
                '--dataset','data/x','--output','runs/x','--vis',mode])
            cli.validate_args(logger_args)
            assert logger_args.vis == mode
            # Verify the installed dependency can persist actual loss scalars.
            with SummaryWriter(logdir) as writer:
                writer.add_scalar('Train Loss Dict/lidar_depth_loss',.001,1)
            assert list(Path(logdir).glob('events.out.tfevents.*'))
    args.camera_optimizer_mode = 'SO3xR3'
    with pytest.raises(ValueError, match='no camera gradients'):
        cli.validate_args(args)
    args.camera_optimizer_mode = 'off'; args.stop_split_at = 500
    with pytest.raises(ValueError, match='zero events'):
        cli.validate_args(args)


def test_no_refine_callback_does_not_call_strategy_or_reset_opacity():
    from types import SimpleNamespace
    config = LidarSplatfactoModelConfig(enable_refinement=False, gaussian_max_scale_m=.12)
    means = torch.nn.Parameter(torch.ones(4, 3))
    scales = torch.nn.Parameter(torch.zeros(4, 3))
    harness = SimpleNamespace(config=config, means=means, scales=scales,
        strategy=SimpleNamespace(refine_stop_iter=11000), refinement_audit={})
    LidarSplatfactoModel.step_post_backward(harness, 0)
    assert scales.exp().max() <= .120001
    assert harness.refinement_audit['final_gaussians'] == 4


def test_actual_cpu_refinement_grows_and_records_optimizer_shapes(tmp_path):
    from orbvi_gaussian.lidar_splatfacto import AuditedRefinementStrategy
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.zeros(4, 3)),
        'scales': torch.nn.Parameter(torch.full((4, 3), math.log(.001))),
        'quats': torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * 4)),
        'opacities': torch.nn.Parameter(torch.full((4, 1), 3.)),
    })
    optimizers = {key: torch.optim.Adam([value], lr=.001) for key, value in params.items()}
    sum(value.sum() for value in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step(); optimizer.zero_grad()
    strategy = AuditedRefinementStrategy(refine_start_iter=500, refine_stop_iter=11000,
        reset_every=11001, refine_every=100, grow_grad2d=.0008, absgrad=False)
    strategy.max_gaussians = 8
    state = strategy.initialize_state(scene_scale=1.)
    proxy = torch.zeros(1, 4, 2, requires_grad=True)
    proxy.grad = torch.ones_like(proxy)
    info = dict(width=640, height=540, n_cameras=1, radii=torch.ones(1, 4), means2d=proxy, gaussian_ids=None)
    strategy.step_post_backward(params, optimizers, state, 600, info, packed=False)
    assert len(params['means']) == 8
    assert strategy.last_growth == (4, 0) and strategy.last_pruned == 0
    assert torch.sigmoid(params['opacities']).min() > .9
    for key, optimizer in optimizers.items():
        assert optimizer.param_groups[0]['params'][0] is params[key]
        assert optimizer.state[params[key]]['exp_avg'].shape == params[key].shape


def test_upstream_step_zero_reduces_warm_opacity_to_point_two_and_fixed_callback_preserves_it():
    from types import SimpleNamespace
    from gsplat.strategy import DefaultStrategy
    params = torch.nn.ParameterDict({'means': torch.nn.Parameter(torch.zeros(4, 3)),
        'opacities': torch.nn.Parameter(torch.full((4, 1), math.log(9.)))})
    optimizers = {key: torch.optim.Adam([value]) for key, value in params.items()}
    proxy = torch.zeros(1, 4, 2, requires_grad=True); proxy.grad = torch.ones_like(proxy)
    info = dict(width=640, height=540, n_cameras=1, radii=torch.ones(1, 4),
                means2d=proxy, gaussian_ids=None)
    # Exact old no-refine schedule: it still entered DefaultStrategy at step 0.
    strategy = DefaultStrategy(prune_opa=.1, refine_every=10000, refine_stop_iter=1000,
                               reset_every=300000, absgrad=False)
    strategy.step_post_backward(params, optimizers, strategy.initialize_state(), 0, info, packed=False)
    torch.testing.assert_close(params['opacities'].sigmoid(), torch.full((4, 1), .2))
    config = LidarSplatfactoModelConfig(enable_refinement=True, reset_opacity_during_refinement=False)
    preserved = torch.nn.Parameter(torch.full((4, 1), math.log(9.)))
    harness = SimpleNamespace(config=config, means=params['means'], opacities=preserved,
        scales=torch.nn.Parameter(torch.full((4, 3), -5.)), strategy=strategy,
        refinement_audit={})
    LidarSplatfactoModel.step_post_backward(harness, 0)
    torch.testing.assert_close(preserved.sigmoid(), torch.full((4, 1), .9))


def test_tangent_constraint_uses_only_two_axes_and_cannot_push_geometry_away():
    from orbvi_gaussian.lidar_splatfacto import tangent_scale_penalties
    scales=torch.tensor([[.01,.02,1.], [.08,.04,.001]],dtype=torch.float64).log().requires_grad_()
    camera_z=torch.tensor([[4.,4.]],dtype=torch.float64,requires_grad=True)
    metric,screen=tangent_scale_penalties(scales,camera_z=camera_z,focal=320.,
                                        visible=torch.tensor([[True,False]]))
    (metric+screen).backward()
    assert scales.grad[0,2]==0 and scales.grad[1,2]==0
    assert scales.grad[1,0]>0 and camera_z.grad is None
    # Invisible large splat contributes metric loss but no screen-space loss.
    _,isolated=tangent_scale_penalties(scales[:1],camera_z=camera_z[:,:1],focal=320.)
    torch.testing.assert_close(screen,isolated)


def test_screen_constraint_has_connected_zero_without_visible_front_splats():
    from orbvi_gaussian.lidar_splatfacto import tangent_scale_penalties
    scales=torch.full((2,3),-4.,requires_grad=True)
    metric,screen=tangent_scale_penalties(scales,camera_z=torch.tensor([[-1.,float('nan')]]),focal=320.)
    assert metric==0 and screen==0
    screen.backward()
    assert torch.equal(scales.grad,torch.zeros_like(scales))


def test_real_post_optimizer_callback_persists_refine_audit_and_hard_scale_bound(tmp_path):
    from orbvi_gaussian.lidar_splatfacto import AuditedRefinementStrategy
    # Construct only the CPU state needed by the actual model/NS callback;
    # parent populate_modules hard-codes CUDA and is deliberately not called.
    model=LidarSplatfactoModel.__new__(LidarSplatfactoModel)
    torch.nn.Module.__init__(model)
    model.config=LidarSplatfactoModelConfig(gaussian_max_scale_m=.12, training_iterations=601,
        reset_opacity_during_refinement=False, training_audit_path=tmp_path/'refinement.json')
    model.gauss_params=torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.zeros(4,3)),
        'scales':torch.nn.Parameter(torch.full((4,3),math.log(.5))),
        'quats':torch.nn.Parameter(torch.tensor([[1.,0,0,0]]*4)),
        'opacities':torch.nn.Parameter(torch.full((4,1),3.))})
    model.optimizers={key:torch.optim.Adam([value]) for key,value in model.gauss_params.items()}
    model.strategy=AuditedRefinementStrategy(refine_every=100,reset_every=11001,absgrad=False)
    model.strategy.max_gaussians=8
    model.strategy_state=model.strategy.initialize_state()
    proxy=torch.zeros(1,4,2,requires_grad=True);proxy.grad=torch.ones_like(proxy)
    model.info=dict(width=640,height=540,n_cameras=1,radii=torch.ones(1,4),means2d=proxy,gaussian_ids=None)
    model.step=600
    model.refinement_audit=dict(initial_gaussians=4,events=[])
    model.step_post_backward(600)
    audit=json.loads((tmp_path/'refinement.json').read_text())
    assert audit['refine_events']==1 and audit['split_total']==4
    assert audit['last_callback_step']==600 and audit['final_gaussians']==8
    assert model.scales.exp().max().item() <= .12
    assert model.opacities.sigmoid().min() > .9


def test_native_ns_strategy_inherits_extended_screen_split_parameters_on_cpu(monkeypatch):
    # NS 1.1.5 initializes SH with .cuda() even for a CPU model. Redirect only
    # that allocation in this fixture; run its actual populate_modules logic.
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    config=LidarSplatfactoModelConfig(stop_screen_size_at=11000,split_screen_size=.02,
        stop_split_at=11000,reset_opacity_during_refinement=False)
    xyz=torch.tensor([[0.,0,0],[1.,0,0],[0.,1.,0],[0.,0,1.]])
    model=config.setup(scene_box=SceneBox(aabb=torch.tensor([[-10.]*3,[10.]*3])),
        num_train_data=1,metadata={},device='cpu',seed_points=(xyz,torch.ones_like(xyz)*128))
    assert model.strategy.refine_scale2d_stop_iter==11000
    assert model.strategy.grow_scale2d==.02
    assert model.strategy_state['scene_scale']==1.
    assert model.strategy.refine_stop_iter==11000
