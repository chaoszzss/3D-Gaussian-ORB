# 本机训练环境实际验证

2026-10-06，环境状态：**ready_for_pilot_training**。已实际执行 `bash scripts/bootstrap_training.sh --install`，完整 Nerfstudio 依赖安装成功，gsplat CUDA 扩展在本机编译成功，tiny 渲染 forward/backward 通过。2026-10-06 17:57:17（Asia/Shanghai）完成 GPU 验证，随后立即通知主代理接管 GPU。本代理没有启动数据训练。

## 实际版本与硬件

| 项目 | 实际值 |
| --- | --- |
| Python | 3.10.21，uv 管理，位于 `runtime/python/` |
| 虚拟环境 | `.venv-train/`，与系统 Python 3.8、base Python 3.13 分离 |
| PyTorch | `2.4.0+cu124` |
| torchvision | `0.19.0+cu124` |
| Nerfstudio / gsplat | `1.1.5` / `1.4.0` |
| NumPy / OpenCV | `1.26.4` / `opencv-python-headless==4.10.0.84` |
| Ninja / setuptools | `1.11.1.4` / `75.8.2` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB，compute capability 8.9 |
| 驱动 | 570.133.07 |
| 本机 nvcc / C++ | CUDA 12.8.61 / GCC 9.4.0 |
| Torch 构建的 CUDA 版本 | 12.4 |
| CUDA 编译目标 | `compute_89` PTX 与 `sm_89`，C++17，`MAX_JOBS=4` |
| 已安装包 | 252 项，实际版本见 `runtime/training.freeze.txt` |

联网核查仍确认 Nerfstudio 1.1.5 的官方依赖固定 `gsplat==1.4.0`；gsplat 当前发行版是 1.5.3，本环境保持项目原合同 1.4.0。Torch 的 `+cu124` 后缀被写入约束，保证后续解析保持所需 CUDA wheel。版本合同没有升级。[Nerfstudio 固定版本元数据](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/pyproject.toml)、[PyTorch 官方 2.4.0 安装矩阵](https://pytorch.org/get-started/previous-versions/)、[gsplat 官方 PyPI 元数据](https://pypi.org/pypi/gsplat/json)。

## CUDA forward/backward 证据

实际执行 `scripts/gpu_preflight.py`，使用固定版本的 `gsplat.rendering.rasterization`：两个非对称、各向异性高斯，一个针孔相机，32×32 RGB 和 alpha，默认 packed 路径。用空间加权损失执行 `backward()`，核验输出有限、非空，以及五类参数梯度有限且非零。此调用遵循 [gsplat v1.4.0 官方 rasterization 实现](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/gsplat/rendering.py)。

| 验证项 | 实际结果 |
| --- | --- |
| forward/backward | passed |
| RGB shape | `[1, 32, 32, 3]` |
| alpha 最大值 | 0.5728291273 |
| loss | 0.0170126613 |
| means 梯度范数 | 0.0090241609 |
| quats 梯度范数 | 0.0009221351 |
| scales 梯度范数 | 0.0779946670 |
| opacities 梯度范数 | 0.0224341620 |
| colors 梯度范数 | 0.0051186671 |
| Torch peak allocated / reserved | 60,416 / 2,097,152 bytes |
| 首次 kernel 编译与 probe 耗时 | 299.870 s |

实际加载的扩展为 `runtime/cache/torch_extensions/gsplat_cuda/gsplat_cuda.so`。上述 tiny Torch 显存统计不含完整 CUDA context，也不是 160 组图像或车库训练的显存预算；真实 pilot 的资源和质量由主代理训练验证。

`/proc/self/maps` 同时记录了实际动态链接路径：`libcudart`、`libcudnn`、`libcusparse` 来自 `.venv-train` 中的 CUDA wheel；`libnvJitLink.so.12.8.61` 实际来自已有系统 CUDA 12.8。这个 12.4 wheel / 12.8 Toolkit 组合已通过本机 tiny 测试，未安装额外 Toolkit 或修改驱动。此实测范围不扩大为所有第三方 CUDA 扩展的兼容认证。[NVIDIA CUDA 同主版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[PyTorch 2.4 扩展源码](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/cpp_extension.py)。

完整机器可读证据：`runtime/gpu_preflight.json`。按主代理最新要求，GPU 就绪后停止后续 GPU probe、依赖安装和 `.so` 编译，以便其立即运行真实 pilot；没有另外启动第二个进程执行 `training.py` 的单高斯 `GPU_PROBE`。本 probe 已覆盖同一版本、同一 rasterization 调用，并增加了所有高斯参数的梯度核验。

## 完整 Nerfstudio 与启动合同

完整环境已安装，没有退回最小环境。`uv pip check` 检查 252 包并输出 `All installed packages are compatible`。`ns-train splatfacto --help` 实际返回 0。

还通过 Tyro 实际解析了主代理计划使用的完整参数组合，并检查生成的配置对象：CPU / uint8 图像缓存、`random-init False`、100,000 初始高斯配置、尺度正则启用、相机优化 off，以及 `nerfstudio-data` 的自动缩放 False、方向/中心 none、downscale 1、scale factor 1.0、scene scale 20、load 3D points True。此验证只解析配置，未读入数据或开始训练。证据为 `runtime/ns-contract-cli.json`。

当前未额外固定的依赖解析结果包括 Tyro 1.0.16、jaxtyping 0.3.7、torchmetrics 1.9.0、viser 0.2.7、websockets 16.1.1。CLI 已用这些实际版本测试；完整版本快照可供后续复现。安装时避免同时安装 GUI 与 headless 两套 OpenCV provider。

## 现实问题与修复

首次安装：Python 下载 6.86 s，Torch 与 CUDA 核心 wheel 准备 5 min 06 s，Nerfstudio 后续 216 包准备 1 min 34 s；官方源没有发生下载失败，`fpsample`、`pyliblzfse` 源码包在隔离目录成功构建。uv 缓存实际约 7.4 GiB，托管 Python 约 83 MiB，扩展目录约 26 MiB；缓存包含展开后的包，并与虚拟环境使用硬链接。

第一轮 GPU preflight 报 `RuntimeError: Ninja is required to load C++ extensions`。Ninja 已在虚拟环境中，但直接调用 `.venv-train/bin/python` 不会自动把该环境的 `bin` 放入 PATH。bootstrap 已显式 prepend `.venv-train/bin`；独立 probe 也 prepend 当前 Python 的 bin。修复后实际重跑 `bootstrap --install`，安装缓存全部复用，然后 CUDA 编译、forward/backward 和 CLI 检查成功，整个命令退出 0。

编译时出现未设置 `TORCH_CUDA_ARCH_LIST` 的提示，实际生成的目标已核对为本机 8.9。Nerfstudio 导入产生旧 AMP decorator 的 FutureWarning；这些提示未阻止验证。没有修改第三方源代码来绕过问题。

## 使用与交接

在项目根目录启动主代理的训练流程前，继承项目内环境路径：

```bash
eval "$(bash scripts/bootstrap_training.sh --print-env)"
# 然后运行主代理准备好的 .venv-train/bin/ns-train 或训练封装。
```

关键启动环境为 `.venv-train/bin` 的 PATH、`TORCH_EXTENSIONS_DIR=runtime/cache/torch_extensions`、`CUDA_CACHE_PATH=runtime/cache/cuda`、`TMPDIR/TMP/TEMP=runtime/tmp`、`PYTHONDONTWRITEBYTECODE=1` 和 `PYTHONNOUSERSITE=1`，以及脚本设置的 Torch、HF、Matplotlib、Numba、wandb、XDG、Jupyter 等项目内路径。主代理已确认在其独占的 `src/orbvi_gaussian/training.py` 中补齐 PATH 与 CUDA cache 设置。

bootstrap 的 `--install` 可恢复已有环境，不再拒绝已有 `.venv-train`；`--check` 可重做 tiny 验证；`--minimal` 仅在明确选择时安装/验证 Torch+gsplat；默认仍安装完整 Nerfstudio。安装或 preflight 命令应在训练进程退出后使用，避免同时触发 JIT 构建。

本代理代码写入范围为 `scripts/bootstrap_training.sh`、`configs/training.constraints.txt`、新增 `scripts/gpu_preflight.py` 和本报告；依赖、下载记录、官方源码快照、日志及临时文件只放在 `runtime/` 和 `.venv-train/`。没有修改 HOME 配置、系统 Python、全局驱动、主代理源码、标定或数据文件。

## 可复核产物

| 文件 | 内容 |
| --- | --- |
| `runtime/bootstrap_training.log` | 首次完整安装日志及 Ninja PATH 失败 |
| `runtime/bootstrap_training_retry1.log` | 修复后完整重跑、CUDA 成功与退出路径 |
| `runtime/gpu_preflight.json` | GPU、版本、实际动态链接路径、输出及所有梯度 |
| `runtime/training.freeze.txt` | 252 项实际安装版本 |
| `runtime/nerfstudio.resolved.txt` | 安装前官方索引依赖解析 |
| `runtime/ns-train-cli-smoke.log` | 实际 Splatfacto 帮助输出 |
| `runtime/ns-train-splatfacto-help.txt` | bootstrap 最终 CLI 检查输出 |
| `runtime/ns-contract-cli.json` | 主代理拟用参数组合实际解析结果 |
| `runtime/training-sources/provenance.json` | 六份官方固定版本源码/元数据 URL、时间与 SHA256 |
| `runtime/cache/torch_extensions/gsplat_cuda/build.ninja` | 实际 nvcc、include、ABI 与 GPU 架构编译参数 |

最终产物 SHA256：

```text
48acc534ab291321df02a3b9380eb1481af890e35903e62ad291ec1af46709e4  scripts/bootstrap_training.sh
d78c01c6ababa29b6823cc85b299d232a2648a15b738b7d578b19f1d345d7813  scripts/gpu_preflight.py
76b222d7244ea0988f0ff658935d481cf3bcd3aaa6a2df9516aefb793996b3f0  configs/training.constraints.txt
0d5ac07de539ae24dc03249075b2537ef36c87de8005578aa93e9de61069d21c  runtime/gpu_preflight.json
b6433feeb66f7271b3dc7a20d23d04cda65ec5a7ba2025f071e51c3deff40461  runtime/cache/torch_extensions/gsplat_cuda/gsplat_cuda.so
```

## 最新 pilot 参数与 LPIPS 预训练权重缓存

主代理正在准备 `data/pilot_block000_board`：首 40 s、约 78 个相机组 × 4 视角、740 crop、LiDAR 初始化上限 80,000 点，独立 dev/test。新增的 `--pipeline.model.num-downscales 1`、`--pipeline.model.resolution-schedule 500`、`--steps-per-save 1000` 已在 CPU 上用已安装的 Tyro 完整解析，并核验配置值为 1 / 500 / 1000；证据为 `runtime/ns-pilot-cli.json`。本代理没有修改主代理的训练命令或数据文件。

固定 Nerfstudio Splatfacto 实际构造 `LearnedPerceptualImagePatchSimilarity(normalize=True)`，当前 TorchMetrics 默认使用 AlexNet。所需 ImageNet 骨干权重已经从官方 URL 预缓存；TorchVision 0.19.0 的官方实现明确提供该 URL 并启用下载哈希核对。[TorchVision 0.19.0 AlexNet 权重定义](https://github.com/pytorch/vision/blob/v0.19.0/torchvision/models/alexnet.py)。

| 项目 | 实际结果 |
| --- | --- |
| `TORCH_HOME` | `/home/chaoszzss/dataset/rosbag/3D-Gaussian/runtime/cache/torch` |
| 骨干 checkpoint | `runtime/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` |
| 官方 URL | `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth` |
| checkpoint 大小 | 244,408,911 bytes |
| checkpoint SHA256 | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| LPIPS 标定层权重 | 已随 TorchMetrics 安装，位于 `.venv-train/lib/python3.10/site-packages/torchmetrics/functional/image/lpips_models/alex.pth` |
| CPU 初始化与同图得分 | passed，score = 0.0 |
| CUDA initialized | false，该检查进程设置了 `CUDA_VISIBLE_DEVICES=''` |
| 预缓存与 CPU 检查耗时 | 23.906 s |
| 实际证据 | `runtime/lpips_cpu_preflight.json`、`runtime/lpips_cpu_preflight.log` |

训练进程应继续继承上述项目内 `TORCH_HOME`，即可复用骨干权重；TorchMetrics 的 LPIPS 标定层从虚拟环境包内读取。此次补充只下载项目内缓存并使用 CPU，没有再运行 GPU probe、安装包或修改 CUDA `.so`；扩展 SHA256 保持前述值。LPIPS 缓存完成与新增 CLI 验证结果均已通知主代理。
