# ORBVI 六层车库：LiDAR 几何与 Gaussian 外观重建

本轮全部 **28/28 个区域**已完成实际训练、模型导出、统一米制世界坐标装配，以及完整模型的 DEV/TEST 评估。此处的完成指六层采集数据的重建流程闭合；地面和侧墙仍有空缺，纹理、接缝和亮区质量未通过高保真验收。按本轮要求，针对性优化留到下一轮。

本轮完整模型是 **`maps/garage_core_all_v1/gaussians_full.ply`，6,780,816 个高斯，1.68 GB**。全程四路鱼眼通过 TripleSphere 虚拟针孔、同次 LIO 全姿态和 LiDAR 表面约束训练；没有把点云预览当作高斯模型。原生后端为 INRIA 2DGS，输出为含薄第三尺度、SH3 和旋转/透明度的标准 Gaussian PLY。

- [直接查看真实结果与文件位置](reports/CURRENT_ROUND_DELIVERY_20261007.html)
- [完整交付记录、指标和不足](reports/CURRENT_ROUND_DELIVERY_20261007.md)
- [文件清单、六层区块对应、交互与重启命令](docs/VIEW_CURRENT_MAP.md)
- [全量模型与可视化交付审计](reports/CURRENT_ROUND_DELIVERY_20261007.json)

整图总览：<http://127.0.0.1:8769/>，点击“六层车库 · 全部28区域总览”的“加载”。内部查看：<http://127.0.0.1:8770/>，加载目标和相邻区块，点目标“定位”，再按“进入区域”；视角1–4对应四路原始相机。鼠标左键旋转、右键平移、滚轮缩放。总览显示全部28区域的120万真实高斯样本，完整678万记录保存在全量PLY；浏览器按3块/150万高斯预算显示内部细节。

本轮每个核心区域从已有邻域模型启动，额外训练1000步，真实检查点位于 `runs/core_joint_v1/`，所有检查点与导出文件一致性已核对。完整模型DEV为408张去重图像，PSNR **18.10 dB**、SSIM **0.673**；最终TEST为404张去重图像，PSNR **18.28 dB**、SSIM **0.675**。原生完整模型渲染确认的地面空缺和污染已保存；几何参考使用同次LIO/map，不构成独立测绘验收。楼层标签仍为历史时间段估计，`floor_verified:false`。

旧首版 `maps/garage_full_v1/gaussians_full.ply`（10,029,806个高斯、2.487GB）和 `runs/garage_surface_v1/` 的28块12000步模型保留。旧版8765/8766查看器用于历史比较，不能与本轮8769/8770结果混同。旧整图的全量配对对照评估因同时可见记录超预算中断，保留原始失败记录，不宣称对照完整；本轮模型已独立完成全28区域的DEV和TEST校验。

这是独立工程。bag、标定和外部历史建图文件只读；历史NVIDIA/Triton/Torch缓存写入例外已如实记录在 [写入范围记录](reports/WRITE_SCOPE_EXCEPTION_20261007.md)，后续进程缓存限定到本目录。

- [详细实施方案](docs/RECONSTRUCTION_PLAN.md)
- [数据审计](docs/DATA_AUDIT.md)
- [标定来源与坐标链](reports/RIG_SOURCE_EVIDENCE.md)
- [全量模型原生评估方式](docs/FULL_SCENE_EVALUATION.md)
- [核心区域训练与恢复](docs/CORE_JOINT_TRAINING.md)

## 当前发现与已生成数据

原始 VIO 在 **458.294 秒附近严重发散**，尾坐标达到百亿米量级。`prepare` 和 `plan-blocks` 会拒绝这份全程 VIO。相机使用 **TripleSphere**，五项参数为 `xi,lambda,alpha,b,c`，不能作为 OpenCV 的 `k1..k4` 使用。`header.stamp` 与 bag 时间相差约 65,737,402 秒，全部同步以整数 `header_ns` 为依据。

| 输出 | 内容 |
| --- | --- |
| `data/keyframes_2hz/` | 全程四路原始 JPEG 各 1,627 张，共 6,508 张；164 张 IPM；完整 252,946 条消息索引、VIO/IMU CSV。约 711 MB。 |
| `data/pilot_20s/` | 20 秒试验，164 张 raw、11 张 IPM、10 份 Livox 与 10 份设备 depth 点云 NPZ。 |
| `data/all_lio_candidate/` | 与本 bag 时间匹配的 8,245 个原生 LIO 姿态 CSV、来源和运动 QA。约 681 米路径，最大单步 0.175 米。 |
| `reports/lio_candidate_blocks.json` | 候选 LIO 的 28 个分块，30 m 长度、5 m 重叠；保留统一世界坐标。 |
| `data/garage_surface_v1/` | 已准备全部 28 块：640×540 RGB、mask、固定 train/dev/test、真实图像着色初始化与 LiDAR 表面深度。 |
| `runs/garage_surface_v1/` | 逐块真实训练检查点、dev/test 图像比较、指标与标准 Gaussian PLY。未完成区块不伪造模型。 |
| `maps/garage_manifest.json` | 查看器的实时全区块索引；包含训练、评估、模型与估计楼层状态。 |
| `reports/device_vio_trajectory_quality.json` | 数值有限但物理发散的 VIO 诊断。 |
| `reports/rectified_v6_preview/` | 四路中央 640×540 针孔诊断图与有效 mask。候选内参不构成完整标定。 |
| `reports/samples/contact_sheet_sparse.jpg` | 跨全程稀疏相机样例，观察光照、结构和画质。 |

LIO 候选来自 V2 `height_continuity_release/registered_and_semantics.bag` 的 `/aft_mapped_to_init`，采用 `camera_init/aft_mapped` 消息参考系。源 topic 有 8,246 条，本次按原 bag 时间窗口保留 8,245 条。它与自身注册云可以追溯；不能直接接到 V6 多会话合并图。运动合理性通过也不证明全局闭环精度或相机外参正确。

## 运行环境

ROS 提取使用现有 **系统 Python 3.8**，依赖 ROS Noetic、NumPy、SciPy、OpenCV、PyYAML。不要用当前 conda base Python 3.13 运行 ROS 阶段。不需要启动 roscore。

```bash
cd /home/chaoszzss/dataset/rosbag/3D-Gaussian
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py --help
```

所有输出必须位于本目录；已经存在的输出目录会被拒绝覆盖，复跑请选新名字。ROS 临时消息类和测试临时文件也放在 `runtime/`。

## 数据、轨迹与预览

下例使用新输出名字，避免覆盖本轮结果。完整提取保存抽样图片及完整轻量索引；默认不保存所有点云载荷。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py inspect
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py extract \
  --output data/new_keyframes --camera-hz 2 --ipm-hz 0.2
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py extract \
  --output data/new_pilot --duration-s 20 --include-clouds --cloud-hz 0.5
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py trajectory-qa \
  --pose-csv data/all_lio_candidate/odometry.csv
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py rectify-preview \
  --extracted data/pilot_20s --calibration configs/v6_board_candidate.unverified.yaml \
  --output reports/new_rectified_preview
```

早期预览配置保留缺失外参。当前训练采用 `configs/direct_lio_refined.experimental.yaml`，使用同次 LIO 的 body→LiDAR、FAST 的 LiDAR→raw camera 与 board TS 内参，显式闭合 `T_world_body @ T_body_lidar @ inverse(T_camera_lidar)`。FAST 原始编号映射、来源哈希、旋转舍入修复及受限训练特征微调均保留在 `reports/RIG_SOURCE_EVIDENCE.md`。该配置仍为 `verified:false`：来源与数学链得到核对，不等同独立测量验收。

如果找到其他同次采集的 LIO 原生证据包，可导出为统一 CSV：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py export-poses \
  --pose-bag /absolute/path/to/registered.bag --pose-topic /aft_mapped_to_init \
  --extracted data/keyframes_2hz --output data/new_lio_poses
```

外部 CSV 使用 `header_ns,bag_ns,frame_id,child_frame_id,tx,ty,tz,qx,qy,qz,qw`。姿态必须为所声明的 `T_world_body`，单位 m、四元数 xyzw，不能只提供 XYZ。单份 CSV 不能变换 body/world 参考系。≤1 ms 的局部时间乱序只排序观测，不修改原始 ns；更大回退、冲突同戳、姿态断档及物理发散会被检查或拒绝。

## 标定闭合后准备与训练

先在本目录生成 `configs/calibration.verified.yaml`，明确实测版本、topic 对应、body/world、时差和 `T_body_camera`。使用外部 LIO 时设置 `pose.source: external_csv`、对应的 `world_frame`，并确认 `aft_mapped` 的物理参考点。不要把虚拟 PINHOLE 的 `body_T_camN` 直接给 raw 鱼眼使用。`verified:true` 是使用者对实测来源的声明，程序只校验完整性与数学合法性。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py check-calibration \
  --calibration configs/calibration.verified.yaml
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/orbvi_pipeline.py prepare \
  --extracted data/keyframes_2hz --calibration configs/calibration.verified.yaml \
  --pose-csv data/all_lio_candidate/odometry.csv \
  --block-plan reports/lio_candidate_blocks.json --block-id block_000 \
  --output data/block_000_prepared
```

输出含 RGB、mask、COLMAP 文本模型、Nerfstudio `transforms.json` 和来源记录。当前全图准备使用 `--separate-test --global-split`：全程源时间组 index%8!=0 为 train，index%16==8 为 dev，index%16==0 为 test。四路同源相机共同划分，相邻重叠区块也使用同一划分，颜色初始化仅读取训练组。它评估采集轨迹附近的插值外观，不能代替独立再访、未知空间视角或测绘验证。COLMAP 没有伪造 feature tracks，因此不能直接把它当完成的 SfM/BA 结果。

可追加同一世界系、同次初始化的地图：

```text
--initialization-map /absolute/path/to/aligned_map.pcd
--map-world-frame camera_init --map-alignment-verified
--map-voxel-m 0.1 --max-points 100000
--colorize-from-images --save-depth
```

这些参数必须合并到上面的 `prepare` 命令。只有配准来源确实成立才使用 `--map-alignment-verified`。PCD 中的 RGB 默认被当作未核验；如已有实测图像 RGB，可显式 `--map-rgb-verified`。否则 `--colorize-from-images` 仅用训练组、有效 mask 和稀疏 z-buffer 着色，未观察点不进彩色初始化。它不替代动态物体剔除、曝光补偿或稠密遮挡验证。`--save-depth` 是地图投影的相机 z-depth/m，标准 Splatfacto 不自动消费它作为深度损失。

本机 8 GB 显存，先用一个 640×540 视图/物理相机、100k 初始化点、受控区块，实测显存后再加视角/密度。训练环境与 ROS 环境分开：

```bash
bash scripts/bootstrap_training.sh --install
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/run_splatfacto.py \
  --dataset data/block_000_prepared --output runs/block_000 --iterations 7000
# 上一行打印命令；加入 --execute 才测试 CUDA/gsplat 并启动训练。
```

训练封装固定 NS 1.1.5、gsplat 1.4.0、Torch 2.4.0/cu124，关闭自动重定向、居中、缩放和独立逐帧相机优化。图片缓存在 CPU；实际彩色点初始化文件须存在并可读。训练环境已安装在 `.venv-train/`，本机 nvcc 12.8 + Torch cu124 的 CUDA forward/backward 已实测。当前批次通过 `train_lidar_block.py --surface-splats --renderer-inria` 使用原始 2DGS 光线与表面交点渲染，输出标准 Gaussian PLY；这仍是高斯外观模型，不是普通彩色点云。此后端实际来源及预检见 `reports/INRIA_SURFACE_BACKEND.md`。

楼层标签由历史时间区间估计，仍为 `floor_verified:false`。模型当前存在地面细纹、车位文字、反光与遮挡区域的模糊，首块实测审查见 `reports/VISUAL_QUALITY_REVIEW.md`。完整首版目标优先；未采到的角落、车辆背面等区域不会凭空补成已观测结构。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest tests -q \
  --basetemp="$PWD/runtime/pytest_manual" -p no:cacheprovider
```

测试包含真实 ROS 序列化端到端、跨时钟同步、TS 投影、位姿约定、点云压缩/字节序、遮挡、训练组着色、深度文件、分组隔离、发散拒绝和输出路径约束。测试夹具明确标识为合成数据，不能作为真实车库标定或重建结果。
