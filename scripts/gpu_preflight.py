#!/usr/bin/env python3
"""Tiny gsplat CUDA forward/backward probe; no dataset or training output."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal", action="store_true", help="Do not require Nerfstudio")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    # Configure caches before importing torch, matplotlib, or CUDA extensions.
    for key, suffix in {
        "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg", "HF_HOME": "huggingface",
        "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
    }.items():
        os.environ[key] = str(project / "runtime/cache" / suffix)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(project / "runtime/tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("MAX_JOBS", "4")
    sys.dont_write_bytecode = True
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if args.output:
        target = args.output.resolve()
        if not target.is_relative_to(project / "runtime"):
            parser.error("--output must be inside project/runtime")

    import torch
    from gsplat.rendering import rasterization

    assert version("gsplat") == "1.4.0", "Expected pinned gsplat 1.4.0"
    assert torch.__version__ == "2.4.0+cu124", "Expected Torch 2.4.0+cu124"
    assert version("torchvision") == "0.19.0+cu124"
    assert torch.version.cuda == "12.4", "Expected cu124 runtime"
    if not args.minimal:
        assert version("nerfstudio") == "1.1.5", "Expected Nerfstudio 1.1.5"
    assert torch.cuda.is_available(), "CUDA runtime is unavailable"
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    means = torch.tensor([[0.12, -0.08, 3.0], [-0.3, 0.18, 4.0]], device="cuda", requires_grad=True)
    quats = torch.tensor([[1.0, 0.05, 0.03, 0.01], [1.0, -0.03, 0.02, 0.04]], device="cuda", requires_grad=True)
    scales = torch.tensor([[0.2, 0.15, 0.1], [0.15, 0.2, 0.12]], device="cuda", requires_grad=True)
    opacities = torch.tensor([0.5, 0.6], device="cuda", requires_grad=True)
    colors = torch.tensor([[0.7, 0.3, 0.1], [0.1, 0.4, 0.8]], device="cuda", requires_grad=True)
    view = torch.eye(4, device="cuda")[None]
    intrinsics = torch.tensor([[[25.0, 0.0, 16.0], [0.0, 25.0, 16.0], [0.0, 0.0, 1.0]]], device="cuda")
    rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, view, intrinsics, 32, 32)
    torch.cuda.synchronize()
    assert tuple(rgb.shape) == (1, 32, 32, 3)
    assert tuple(alpha.shape) == (1, 32, 32, 1)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert alpha.max() > 0 and rgb.max() > 0, "Render contains no splats"
    # A spatially weighted loss avoids symmetry masking position gradients.
    ramp = torch.linspace(0.5, 1.5, 32, device="cuda").reshape(1, 1, 32, 1)
    loss = (rgb * ramp).mean() + (alpha * ramp).mean()
    loss.backward()
    torch.cuda.synchronize()
    gradients = {}
    for name, parameter in {"means": means, "quats": quats, "scales": scales,
                            "opacities": opacities, "colors": colors}.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        norm = parameter.grad.norm().item()
        assert norm > 0, "Zero gradient: " + name
        gradients[name] = norm
    from gsplat.cuda import _backend
    mapped_cuda_libraries = sorted({
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("libcudart", "libcudnn", "libcusparse", "libnvJitLink"))
    })
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "torchvision": version("torchvision"), "gsplat": version("gsplat"),
        "nerfstudio": None if args.minimal else version("nerfstudio"),
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "rgb_shape": list(rgb.shape), "alpha_max": alpha.max().item(), "loss": loss.item(),
        "gradient_norms": gradients, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - started, "forward_backward": "passed",
        "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
        "gsplat_extension": _backend._C.__file__,
        "mapped_cuda_libraries": mapped_cuda_libraries,
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
