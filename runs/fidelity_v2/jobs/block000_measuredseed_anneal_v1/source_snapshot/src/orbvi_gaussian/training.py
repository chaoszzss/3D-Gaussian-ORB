"""Pinned Splatfacto launch planning with explicit metric-coordinate handling."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .io import PROJECT_ROOT, output_path


def confined_environment():
    env = os.environ.copy()
    cache = output_path(PROJECT_ROOT / "runtime/cache")
    temp = output_path(PROJECT_ROOT / "runtime/tmp")
    for path in [cache, temp]:
        path.mkdir(parents=True, exist_ok=True)
    for key, sub in {"XDG_CACHE_HOME": "xdg", "PIP_CACHE_DIR": "pip", "UV_CACHE_DIR": "uv",
                     "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
                     "TORCHINDUCTOR_CACHE_DIR": "torch_inductor", "TRITON_CACHE_DIR": "triton",
                     "HF_HOME": "huggingface", "MPLCONFIGDIR": "matplotlib",
                     "NUMBA_CACHE_DIR": "numba", "WANDB_DIR": "wandb",
                     "WANDB_CONFIG_DIR": "wandb_config"}.items():
        env[key] = str(cache / sub)
    env["TMPDIR"] = str(temp)
    env["TMP"] = str(temp)
    env["TEMP"] = str(temp)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = str(PROJECT_ROOT / ".venv-train/bin") + os.pathsep + env.get("PATH", "")
    env["CUDA_CACHE_PATH"] = str(cache / "cuda")
    env["__GL_SHADER_DISK_CACHE_PATH"] = str(cache / "nvidia_gl")
    env["__GL_SHADER_DISK_CACHE"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MAX_JOBS"] = "4"
    return env


def splatfacto_command(dataset, output, executable=None, iterations=7000, allow_random_init=False, allow_candidate_calibration=False):
    root = Path(dataset).expanduser().resolve()
    with (root / "preparation.json").open(encoding="utf-8") as f:
        prepared = json.load(f)
    with (root / "transforms.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    status_ok = prepared.get("status") == "prepared" or (allow_candidate_calibration and prepared.get("status") == "prepared_experimental")
    if not status_ok or (root / "INCOMPLETE.json").exists():
        raise ValueError("Dataset preparation is incomplete")
    if not meta.get("train_filenames") or not meta.get("val_filenames"):
        raise ValueError("Require independent camera groups in both training and validation")
    colored_init = bool(meta.get("ply_file_path") and meta.get("point_cloud_has_rgb"))
    if colored_init:
        cloud = (root / meta["ply_file_path"]).resolve()
        if root not in cloud.parents or not cloud.is_file():
            raise ValueError("Colored initialization PLY is missing or outside the dataset")
        from .points import read_point_cloud
        xyz, rgb = read_point_cloud(cloud)
        if not len(xyz) or rgb is None or not np.isfinite(xyz).all():
            raise ValueError("Colored initialization PLY must contain finite XYZ and measured RGB")
    if not colored_init and not allow_random_init:
        raise ValueError("No measured-color point initialization. Supply an aligned RGB map or colorize it "
                         "with calibrated training images; --allow-random-init is an explicit diagnostic baseline.")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    output = output_path(output)
    if output.exists():
        raise FileExistsError("Training output exists: %s" % output)
    centers = np.array([f["transform_matrix"] for f in meta["frames"]], dtype=float)[:, :3, 3]
    if not np.isfinite(centers).all():
        raise ValueError("Nonfinite camera centers")
    # AABB bound changes SceneBox only, not pose or point-cloud scale/origin.
    half_extent = max(1.0, float(np.abs(centers).max()) + 20.0)
    executable = Path(executable).expanduser().resolve() if executable else PROJECT_ROOT / ".venv-train/bin/ns-train"
    command = [str(executable), "splatfacto", "--data", str(root), "--output-dir", str(output),
               "--vis", "viewer", "--max-num-iterations", str(iterations),
               "--pipeline.datamanager.cache-images", "cpu", "--pipeline.datamanager.cache-images-type", "uint8",
               "--pipeline.model.random-init", "False" if colored_init else "True",
               "--pipeline.model.num-random", "100000", "--pipeline.model.use-scale-regularization", "True",
               "--pipeline.model.camera-optimizer.mode", "off", "nerfstudio-data",
               "--auto-scale-poses", "False", "--orientation-method", "none", "--center-method", "none",
               "--downscale-factor", "1", "--scale-factor", "1.0", "--scene-scale", str(half_extent),
               "--load-3D-points", "True" if colored_init else "False"]
    return command


GPU_PROBE = r'''
import json
import torch
import gsplat
from importlib.metadata import version
from gsplat.rendering import rasterization
assert version("nerfstudio") == "1.1.5", "Expected pinned Nerfstudio 1.1.5"
assert version("gsplat") == "1.4.0", "Expected pinned gsplat 1.4.0"
assert torch.__version__.split("+")[0] == "2.4.0", "Expected pinned PyTorch 2.4.0"
assert torch.cuda.is_available(), "CUDA runtime is unavailable"
means = torch.tensor([[0., 0., 3.]], device="cuda", requires_grad=True)
quats = torch.tensor([[1., 0., 0., 0.]], device="cuda", requires_grad=True)
scales = torch.tensor([[.2, .2, .2]], device="cuda", requires_grad=True)
opacities = torch.tensor([.5], device="cuda", requires_grad=True)
colors = torch.tensor([[.7, .3, .1]], device="cuda", requires_grad=True)
view = torch.eye(4, device="cuda")[None]
K = torch.tensor([[[25., 0., 16.], [0., 25., 16.], [0., 0., 1.]]], device="cuda")
rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, view, K, 32, 32)
loss = rgb.mean() + alpha.mean()
loss.backward()
assert torch.isfinite(rgb).all() and torch.isfinite(means.grad).all()
print(json.dumps({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
 "gsplat": version("gsplat"), "nerfstudio": version("nerfstudio"),
 "gpu": torch.cuda.get_device_name(), "forward_backward": "passed"}))
'''


def launch_splatfacto(command):
    executable = Path(command[0])
    if not executable.is_file():
        raise FileNotFoundError("Training environment missing; run scripts/bootstrap_training.sh --install")
    python = executable.parent / "python"
    if not python.is_file():
        raise ValueError("ns-train must belong to a dedicated Python environment")
    env = confined_environment()
    # Revalidate before GPU compilation/training as datasets can change after planning.
    dataset = Path(command[command.index("--data") + 1])
    with (dataset / "transforms.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if command[command.index("--pipeline.model.random-init") + 1] == "False":
        from .points import read_point_cloud
        cloud = (dataset / meta["ply_file_path"]).resolve()
        if dataset not in cloud.parents:
            raise ValueError("Initialization PLY escapes dataset")
        xyz, rgb = read_point_cloud(cloud)
        if not len(xyz) or rgb is None:
            raise ValueError("Rejecting implicit random initialization fallback")
    subprocess.run([str(python), "-c", GPU_PROBE], env=env, cwd=str(PROJECT_ROOT), check=True)
    return subprocess.run(command, env=env, cwd=str(PROJECT_ROOT), check=False).returncode
