# 深度管线独立诊断：交接版

结论：没有发现 camera-z/range、米/毫米、矩阵方向、重复 α 归一或训练帧索引错误。发现早期 depth resize 射线偏移和 reference 可见性过滤不吻合；二者不足以解释全部最终米级偏差。**本机 gsplat 1.4 的 2DGS `RGB+ED` 也返回中心 z 的期望，单换 renderer/tuple bridge 不能得到 ray-plane depth。**

范围只包括本报告与 [diagnose_depth_pipeline.py](/home/chaoszzss/dataset/rosbag/3D-Gaussian/scripts/diagnose_depth_pipeline.py)。实验、日志、temp/cache 均在 runtime，禁止 bytecode；未改源模块、训练脚本或标定，未安装依赖，未运行既有 sensor/loss 回归、训练或优化搜索。

完整实测、输入 SHA256 和版本：[depth_pipeline_diagnostic.json](/home/chaoszzss/dataset/rosbag/3D-Gaussian/runtime/depth_pipeline_diagnostic.json)。实测 Torch 2.4.0+cu124 / gsplat 1.4.0 / NS 1.1.5。

## 1. 已排除的核心接口错误

- CV c2w 右乘 diag(1,-1,-1,1) 导出 NS；NS get_viewmat 再翻 Y/Z 并求逆，恢复 CV w2c。非零 translation/rotation 合成相机与独立 CV inverse 最大差 1.19e-7。真实 frame 与保存 COLMAP w2c 差 4.44e-16。
- gsplat `RGB+ED`：投影中心 camera-z 作 depth 通道，之后除累计 α；NS 直接返回该通道。实际 ED 与 D/α 的最大差 **0**。不是 range，不应再除 α。
- `project_surfel_depth` 的行向量 `(xyz-t)@R` 正确；交点 z=(n·p)/(n·ray) 是 camera-z 米。reference 使用 CV 整数中心。
- 实际 parser scale=1、transform identity、distortion 全零。自定义 loader 只乘 dataparser_scale；stock `depth_unit_scale_factor=.001` 不参与自定义 lidar_depth_path，未出现千倍单位差。
- 实际 manager：276 train / 20 dev，声明 16 test。全部 276 train 文件→LiDAR depth/support 路径无错配，camera pose 与对应 transforms frame 全通过。
- 抽查实际 `next_train` 首/中/尾共 10 个 idx：RGB、image mask、depth、support 与对应文件逐元素一致；camera.metadata.cam_idx=image_idx。抽查 NS cx/cy 比原 COLMAP CV 值大 .5。
- 312 帧 depth 非零但 support=false：**0** 像素；support=true 但 depth 非有限/非正：**0**。
- 调用真实 parent `get_loss_dict`，用 CPU shell model 避免创建训练模型：uint8 RGB / bool mask / 完美预测在 2×、1× 的 main_loss 均 **0**，batch image 与 outputs RGB 未被修改。未发现 parent 读取错 GT。

为了有界，manager 的 cache 只装入抽查图像。零 distortion 的真实 `_load_images` 原本直接返回 dataset.get_data，所以省略全量 cache 不改变被查路径。parent 源码见 [splatfacto.py:696](/home/chaoszzss/dataset/rosbag/3D-Gaussian/.venv-train/lib/python3.10/site-packages/nerfstudio/models/splatfacto.py:696)。

[evaluate_reconstruction.py:108](/home/chaoszzss/dataset/rosbag/3D-Gaussian/scripts/evaluate_reconstruction.py:108) 确实遗漏读取 lidar_depth_mask_path，但这批 depth/support 完全等价，故对本批评分无影响，不能当作当前米级偏差原因。

## 2. 明确的早期 depth resize bug

[resize_lidar_targets:55](/home/chaoszzss/dataset/rosbag/3D-Gaussian/src/orbvi_gaussian/lidar_splatfacto.py:55) 使用 nearest；2× depth/support 读取原 raster `[2v,2u]`，已实际断言。修复 PixelHalf 后，低分辨率 K_NS/2 和半整数像素对应原 CV `[2u+.5,2v+.5]`；RGB area 平均中心也在这里。

例如 fx=320、cx_CV=319.5、cx_NS=320：低像素 i 的横向 ray 为 `(i+.5-160)/160`，对应原 CV u=2i+.5，nearest 却取 2i。两轴各错半个原像素，独立于已修复的全分辨率 PixelHalf。合成斜平面纯 GT/ray 差 median .0131m / p95 .0272m / max .0389m。

12000-step run 的 num_downscales=1/resolution_schedule=1500，所以只影响前 1500 step。**不能据此解释最终全部 >1m 偏差。**应按实际低分辨率射线重投影 reference，或明确使深度监督在相同分辨率下比较；对 sparse depth 跨遮挡做 area 平均不是正确修复。本次未改实现。

## 3. 实际 3DGS / 2DGS oracle

GPU 每次 raster 64×64，平面 4096 splats，两层 2 splats，单面 1 splat；合成相机只是算法 fixture，不是实际标定估计。使用现有 backend，未修改 kernel 或安装依赖。

| 实验 | median abs | p95 abs | max abs |
|---|---:|---:|---:|
| 正面 5m：surfel reference vs analytic | 0 | 0 | 0 |
| 正面 5m：3DGS ED vs reference | 4.77e-7m | 1.43e-6m | 2.38e-6m |
| 斜面：surfel reference vs analytic | 1.06e-7m | 2.28e-7m | 4.57e-7m |
| 密集小斜面：3DGS ED vs reference | .0216m | .0289m | .0319m |
| 密集小斜面：2DGS ED vs reference | .0142m | .0212m | .0267m |
| 单个大倾斜 2DGS：ED vs ray-plane hit | **1.364m** | **2.333m** | **3.000m** |

两个平面 ED=D/α 的实际最大差均为 0；meta.depths 与独立中心 CV z 差至多 1.15e-6m。3DGS 斜面内部 α 为 .647–.974，已正确归一。

两层反例：中心前层 z=5m、后层 z=15m，opacity 各 .4，中心累计 α=.64。实际 ED=**8.750001m**，解析式 `(.4*5+.6*.4*15)/(.4+.6*.4)=8.75`；两层 RGB 和 background 都为 .6 时最大 RGB error **5.96e-8**。因此 RGB 可以完全匹配，而 expected-z 与第一层相差 3.75m。

### 2DGS 不自动提供 ray-plane depth

[rendering.py:1234](/home/chaoszzss/dataset/rosbag/3D-Gaussian/.venv-train/lib/python3.10/site-packages/gsplat/rendering.py:1234) 仍拼接中心 depths；CUDA [rasterize_to_pixels_2dgs_fwd.cu:210](/home/chaoszzss/dataset/rosbag/3D-Gaussian/.venv-train/lib/python3.10/site-packages/gsplat/cuda/csrc/rasterize_to_pixels_2dgs_fwd.cu:210) 按 c_ptr[k]*vis 累计。ray-plane 局部坐标 s 用于权重，depth 通道没有替换成交点 z；median 分支也读取中心 depth。

实际 primitive：mean=[0,0,5]、normal=[.8,0,.6]、tangent scales=[2,2]、第三 scale=.01、opacity=.8。2780 个 α>.1 像素 ED 范围 **[4.999999,5.000001]m**，analytic plane hit 是 **[2.666667,8]m**。所以 dense small surfels 的较小误差只能说明数值逼近，不说明返回量已经是 ray-plane hit。单换 renderer 可能改变外观和权重，不能作为深度语义已修复的证据。

2DGS 7-tuple / packed=False 已实测：render=[1,64,64,4]、alpha=[1,64,64,1]、meta.means2d=[1,N,2]。一次真实 backward：means2d 的 grad/absgrad 均存在且有限；gradient_2dgs 有 grad、没有 absgrad。真实 DefaultStrategy(absgrad=True, 默认 means2d)._update_state 也通过。主代理后来声明 v2 使用 absgrad=False/key_for_gradient=gradient_2dgs，与本机该 tensor 的 grad 相符；本 sidecar 未检查或改变 v2 训练进程。

### 补充：render_median 与独立 backward（最终实测）

按主代理追加请求，只再运行 1 个 Gaussian 的 64×64 median oracle，未重跑 loader/真实 frame。设置 depth_mode=median，仍用同一 mean=[0,0,5] / normal=[.8,0,.6] / scales=[2,2,.01] / opacity=.8。

2780 个受支持像素的 **render_median 精确为 [5,5]m**；analytic plane hit 为 [2.666667,8]m。对交点深度的 median abs error=**1.363636m**、p95=**2.333333m**、max=**3m**，对中心 z 的误差则完全为 0。没有额外除 α。

因此本机 render_median 也是选中的 primitive 中心 z，不能作为 ray-plane depth 替代。CUDA [forward:236](/home/chaoszzss/dataset/rosbag/3D-Gaussian/.venv-train/lib/python3.10/site-packages/gsplat/cuda/csrc/rasterize_to_pixels_2dgs_fwd.cu:236) 用 T>.5 选择 c_ptr[COLOR_DIM-1]；[backward:276](/home/chaoszzss/dataset/rosbag/3D-Gaussian/.venv-train/lib/python3.10/site-packages/gsplat/cuda/csrc/rasterize_to_pixels_2dgs_bwd.cu:276) 将 median gradient 加到该 depth color channel。

以 SmoothL1(median,analytic plane hit) **单独**反传，loss=.893648，结果如下，未混入 RGB 或 ED 的 gradient：

| 参数/代理 | gradient 有限 | gradient norm |
|---|---|---:|
| mean | 是 | .450841 |
| quaternion | 是 | 0 |
| scales | 是 | 0 |
| opacity | 是 | 0 |
| RGB colors | 是 | 0 |
| means2d / gradient_2dgs | 是 | 0 / 0 |

所以 backward 可运行且 finite，但这个单面 depth loss 不提供法向、尺度或 opacity 的交点几何梯度。它不能以“梯度正常”作为 ray-plane 语义的证据。JSON 键为 synthetic.2DGS_single_tilted_surfel.render_median_probe；GPU peak allocated 9.28MB。

快速复现补充实验：先运行主命令生成 JSON，再执行 `.venv-train/bin/python -B scripts/diagnose_depth_pipeline.py --median-only`，仅追加 median probe，保留已有机读结果。该 flag 明确运行一次 GPU oracle。

## 4. 真实 cam5 reference：量化不是普遍米级差，过滤会改变可见层

帧：images/1720267884750228828_camera_5_center.png，400000 reference points，640×540。

- 重新投影保存 XYZ/normals vs 保存 depth：28664 个 support 像素逐值完全相同；image mask 内 support 差 **0**。
- 已接受点的像素 ray error median .001247rad / p95 .001872rad，乘 f=320 约 .399/.599px。
- 单个已接受点的 tangent z correction：median abs **.00657m** / p95 **.11689m** / max **.50484m**；5803 个 grazing candidate 被拒绝。不能把普遍米级误差归因于这些已接受点的量化修正。
- raw 全点最近 z-buffer vs filtered reference：28651 共同像素，median abs **.00488m** / p95 **2.748m** / max **33.955m**。

### 可复现的可见性过滤不吻合

[surfels.py:33](/home/chaoszzss/dataset/rosbag/3D-Gaussian/src/orbvi_gaussian/surfels.py:33) 在建立近层可见性前删除 normal-invalid，随后拒绝 grazing/大修正，仅在剩余点中 z-buffer；不记录更近但被拒绝的点。

本帧 **1830** 个 valid-reference 像素的 raw winner 与 reference 差 >1m；**全部 1830 是 raw 更近**。回溯 raw winner：1732 个 normal-invalid，97 个 normal-valid 但 grazing，1 个 intersection 不合格；0 个仍通过 surfel 筛选。这是前点被过滤后另一远点成为 valid GT，不是同一点的亚像素修正。

raw 点本身也只在这个像素附近，不能由此断言它是精确连续表面交点。保守修复是把存在不可信近层的像素标为未知，避免自动替换成较远层 confident GT；可先建立全点近深度/可见性候选，再在带容差的同一近层中检查法向置信度。本次未改变生成器。

## 5. 旧模型实际局部渲染：剩余误差仍明显

CPU mmap 读取 step-000011999.ckpt。根据用户给定的旧版本历史，显式 legacy-cv，取保存 COLMAP 原 K，并验证 source pose 未变。**没有用新 K/保存的新 depth raster 与旧模型比较**：同源 map 被重新投影到旧 renderer 的 `[u+.5,v+.5]` 物理射线，等价整数-center reference K=K_old-.5。

crop=[544,169,64,64]，固定 4×4 位置中按 support 数量选取；不是全帧无偏统计。全模型 523704 点留在 CPU，只送保守 screen bbox 选出的 **3874** 点到 GPU。projection 使用原 full-camera K/FOV，然后平移 means2d，仅调用 64×64 实际 gsplat tile/raster kernel；没有因裁切改变 covariance clamp，也没有随机抽样。

| 局部结果 | 数值 |
|---|---:|
| 比较像素 α>.5 | 1711 |
| ED−reference signed median | −6.961m |
| median / p95 absolute error | 7.226 / 18.228m |
| α median | .999750 |
| α>.95 的 median absolute error | 7.232m |
| 各层 z 的权重标准差 median / p95 | 3.234 / 8.414m |
| 权重中距 map>.25m 中心比例 median / p95 | .00776 / .4164 |
| 权重中 normal-invalid 或 distance≥.5m 中心比例 median / p95 | .1113 / .6853 |
| raw 更近 >1m 像素 | 138 |
| 去掉这 138 后 median abs error | **6.606m**，1573 像素 |
| ED vs raw 最近 z 的 median abs error | **6.443m** |

所以 reference 的前景过滤问题不解释全部残差；α 高也没有消除它。中心贴近 map 不保证中心属于该像素的首个可见表面，也不保证 footprint 内的中心-z 等于 plane hit。该 crop 有明显跨深度层混合，不能由 overall mean NN median 推出 depth 正确；也不能把这个局部结果外推成整 dev 指标。

当前 [lidar_splatfacto.py:183](/home/chaoszzss/dataset/rosbag/3D-Gaussian/src/orbvi_gaussian/lidar_splatfacto.py:183) 硬投影条件是 normal_valid & distance<.5，只约束这些点的法向位移；不约束门槛外中心，也不是全局切向距离界。因此启用硬投影不保证全体 mean NN p95 小。这里描述检查时源码，不声称旧 checkpoint 已执行后来添加的硬投影。

## 6. 解释边界与可复现交接

map 同时参与初始化、监督，以上是同源实现/一致性诊断，**不是独立几何 GT**。绝对 body 标定偏差、SDK producer 绑定未知、intrBoard/oldQR 光学坐标不同及三角测深不确定性仍保留。高 cheirality / 较小 held-out epipolar 不能被本报告升级为绝对深度证据。没有构造或拟合新的 calibration matrix。

当前没有证据支持继续将结构失败简单归因于 depth loss 权重。应先明确 visibility reference 和 renderer depth 的物理协议；本机 3DGS、2DGS 的 ED 都不是实际 ray-plane hit。修复参考过滤后仍需评估剩余真实误差，而不是宣称单点问题已解释全部失败。

脚本 CPU threads=2，reference≤400000，checkpoint<512MiB；GPU≤50000候选点，raster≤64×64；超过点数预算明确跳过。synthetic peak allocated 约10MiB，实际 crop 约9MiB。没有再做阈值/超参数搜索。机读结果含完整统计和被检查输入 SHA256。

```bash
cd /home/chaoszzss/dataset/rosbag/3D-Gaussian
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-train/bin/python -B scripts/diagnose_depth_pipeline.py --gpu \
  --checkpoint runs/surfel_reference_12000/block000_surfel_reference/splatfacto/2026-10-06_192526/nerfstudio_models/step-000011999.ckpt \
  --checkpoint-pixel-center legacy-cv \
  > runtime/depth_pipeline_stdout.log 2> runtime/depth_pipeline_stderr.log
```

省略 checkpoint：只跑 loader、合成 oracle/小 backward 和真实 reference；再省略 gpu：只查 CPU loader/reference。旧 checkpoint convention 必须显式指定，不自动套用新 K。实验已成功退出 0，日志只含 Torch 提示，无 traceback。
