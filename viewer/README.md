# 车库本地 Gaussian 地图查看器

这是现有 Python 工程的本地查看器。`scripts/serve_map.py` 使用 Python 3.8 标准库，前端使用维护者官方 Spark 的 Gaussian 解码和 GPU 渲染，支持 position / log-scale / wxyz rotation / opacity logit / SH0–3。所有资源从本机读取；运行期间不使用 CDN、不上传模型、不发布网站。

当前尚无训练完成的实景 PLY。默认读取工程根下的 `maps/garage_manifest.json`；文件缺失或模型待导出时显示待训练状态。不存在自动演示回退。`viewer/fixtures/` 的文件与截图均为 **合成自检，非实景**，不能证明车库重建质量、标定精度或六层地图已完成。

## 启动

本次已将依赖安装在 `viewer/node_modules/`，保留了固定版本与 package-lock。没有构建步骤，不需要全局 npm 包。

```bash
cd /home/chaoszzss/dataset/rosbag/3D-Gaussian
mkdir -p viewer/runtime/tmp viewer/runtime/npm-cache
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/viewer/runtime/tmp"
export npm_config_cache="$PWD/viewer/runtime/npm-cache"
# 首次安装或依赖目录未交付时：
npm --prefix viewer ci --ignore-scripts --no-audit --no-fund --update-notifier=false
# 实景模式（清单缺失时也可启动）：
/usr/bin/python3 scripts/serve_map.py
```

浏览器打开 `http://127.0.0.1:8765/`。默认绑定 `127.0.0.1`；服务器只读。`--port` 可修改端口，`--manifest viewer/garage_manifest.example.json` 可检查待训练的接口示例。清单路径必须相对工程根，不能访问工程外文件。

合成自检需要显式使用 `--demo`：

```bash
# 以上三个本地环境变量仍需保留
/usr/bin/python3 viewer/make_fixture.py
/usr/bin/python3 scripts/serve_map.py --demo --port 8766
```

打开 `http://127.0.0.1:8766/`，页面顶部持续显示“合成自检模型 · 非实景”。自检生成四个区块，每块 882 个各向异性高斯，覆盖 SH0–3、非单位内部旋转、不同透明度和不同米制位置；另有 81 个显式合成 LiDAR 辅层点。没有复制或转换实景地图冒充训练模型。

## 清单协议 v1

规范文件：[manifest.schema.json](manifest.schema.json)。待训练示例：[garage_manifest.example.json](garage_manifest.example.json)。生产者负责写 `maps/garage_manifest.json`，查看器不会创建或修改此文件。默认文件允许 `blocks: []`。清单可保留没有 PLY 的待训练区块，页面可定位其包围盒，模型加载按钮保持禁用。

| 字段 | 约定 |
| --- | --- |
| `schema_version` | 整数 `1` |
| `coordinates`、`units` | 固定 `camera_init`、`m` |
| `dataset_kind` | `real` 或 `synthetic`，缺省为 `real`；合成模式必须明确声明 |
| `camera_up` | 三个有限数，非零，默认 `[0,0,1]`；仅决定相机和参考网格的上方向 |
| `blocks[].id` | 唯一 `[A-Za-z0-9_-]{1,96}`；API 使用此编号，不暴露任意文件下载接口 |
| `floor_id` | 字符串或 `null`；不依据高度自动命名楼层，跨层坡道可用独立人工索引 |
| `floor_verified` | 可选；默认未核验，界面持续说明 |
| `path` | 工程根相对 POSIX 路径，如 `runs/block_000/gaussians.ply`；不相对清单目录 |
| `coordinates` | 每块必须再次声明 `camera_init` |
| `bbox_m` | `{"min":[x,y,z],"max":[x,y,z]}`，为统一世界系下的保守 AABB；用于定位、近邻选择与适配视角 |
| `training` | 必需对象；建议 `status: pending / trained / failed`、算法、迭代数、训练集来源与标定核验状态；未知信息可用空对象 |
| `eval` | 必需对象；建议 `status: not_evaluated / evaluated`、留出划分、`psnr_db`、SSIM、LPIPS、几何 RMSE/m 和报告来源；不编造缺失指标 |
| `lidar` | 可选独立辅层：`{path,coordinates:'camera_init',alignment_verified:true,source}`；当前接受 PLY XYZ 点云 |

`training` / `eval` 支持附加字段；界面只显示简洁状态和已声明的 PSNR，不输出训练日志、命令或内部调用信息。指标是生产者声明，不构成查看器对精度的签收。JSON Schema 是编辑器契约；Python 服务器还执行有限数值、唯一 ID、路径、模型文件及非零观察上方向等检查。

**统一空间要求**：PLY 高斯中心本身就应为 `camera_init` 的米制世界坐标。每块 Object3D 保持位置 `[0,0,0]`、旋转 xyzw `[0,0,0,1]`、缩放 `[1,1,1]`，没有逐块居中、归一化、flatten-z 或官方示例中用于特定模型的 OpenCV/OpenGL 翻转。使用 `ExtSplats` 保存 float32 中心，避免标准 packed 中心的 half 精度降低车库大范围位置精度。生产者须撤销训练中的 auto-scale / centering / orientation，并把局部模型变换烘焙回标准 PLY：中心、尺度、旋转及方向相关 SH 必须同时正确处理。查看器不读取 `transform` 等额外元数据实施坐标补偿。

包围盒应覆盖高斯实际显示范围。轨迹块计划中的 `bbox_min_m / bbox_max_m` 可作为训练前临时索引，但不一定覆盖训练后高斯。导出后应更新 `bbox_m`；查看器没有使用它裁切模型。楼层筛选改变可见性和列表，保留所有高斯的原始 Z。

**发布到本地清单**：训练结束后先写入完整 PLY 临时文件，再用同目录原子替换到最终路径，最后原子替换清单。避免浏览器读取正在追加的数据。刷新清单会卸载当前所有模型与辅层，重新获取索引；需重新选择加载。此操作只改变浏览器工作集。

## PLY 兼容边界

接受标准 **`binary_little_endian 1.0`、单一 `vertex` 元素、float32 Gaussian 属性**：`x y z`、`scale_0..2`（自然对数尺度，单位 m）、`rot_0..3`（wxyz）、`opacity`（logit）、`f_dc_0..2`、可选连续 `f_rest_0..8/23/44`（对应 1/2/3 阶，按标准 INRIA 的通道排列）。可有 float 法线和其他普通标量属性。SH0 不需要 `f_rest`。服务器检查字节长度与属性数量匹配，检查首个 vertex 的有限性与非零旋转，然后交由维护者解码器处理完整模型。

普通 XYZ / RGB PLY 会被拒绝作为 Gaussian，即使 Spark 本身提供点云导入能力，此查看器也不会借此替代训练模型。ASCII Gaussian PLY 在 Spark 2.3.1 浏览器实测报 `Unsupported PLY format: ascii`，入口会明确拒绝；请导出小端二进制标准 PLY。ASCII **LiDAR 辅层**由 Three.js 的 PLYLoader 加载，可以使用。压缩 PLY、SPZ、SPLAT、KSPLAT、网格面、big-endian 和 float64 Gaussian 属性尚未纳入本清单协议。

这是结构兼容前检，不遍历验证每个顶点、不签收数值稳定性，也不保证模型确实来自声明的训练数据。位置维持 float32；尺度、旋转、颜色和 SH 仍经过 Spark 自身的内部编码，不能当作所有属性无损往返工具。

## 操作与资源

进入页面 **不自动加载任何模型**。手动按块加载，或用“加载观察中心附近区块”选择当前楼层索引下离观察中心 AABB 最近的 1–3 块；近邻按钮会替换当前 Gaussian 工作集。选中楼层会隐藏不匹配的已加载区块，不会卸载它们。

默认预算为最多 3 个 Gaussian 块、总计 150 万高斯、输入总计 512 MiB；LiDAR 独立为最多 3 层、30 万点、32 MiB。超预算会在下载/分配前阻止加载。隐藏仍计入预算，卸载会移除并 dispose 模型数据；Spark 排序与渲染缓冲可能保留已分配容量供复用，预算不是实际 VRAM 测量。大单块须由训练/导出流程进一步分块，本次没有完整多层 LOD、自动漫游流式替换或 GPU 总容量探测，不能保证任意 8 GB GPU 上的帧率或显存峰值。

定位依据米制包围盒，仅移动相机；斜视、俯视、正视、侧视和“适配已显示区块”均可用。参考网格为 1 m 间距。LiDAR 辅层独立蓝色点材质与标签，不使用 Gaussian 渲染，也不声明恢复纹理。

## 本地验证

```bash
cd /home/chaoszzss/dataset/rosbag/3D-Gaussian
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/viewer/runtime/tmp"
export npm_config_cache="$PWD/viewer/runtime/npm-cache"
/usr/bin/python3 viewer/tests/test_server.py
# 浏览器仅用于自检，可选；安装和profile全部位于viewer/runtime
export PLAYWRIGHT_BROWSERS_PATH="$PWD/viewer/runtime/browsers"
node viewer/node_modules/playwright/cli.js install chromium
node viewer/tests/browser.mjs
```

服务器测试包含 PLY 与点云区分、SH 阶数、缺失清单空态、ASCII Gaussian 拒绝、截断/首顶点非有限值拒绝、同一空间协议、符号链接、重复 JSON key、Host、防跨站子资源访问，以及 HTTP Range/HEAD。浏览器测试通过真正的 WebGL2 / Spark 创建高斯纹理、排序和绘制，用 framebuffer 像素变化检查可见/隐藏，验证 SH3 数据保留、多块世界系、楼层、四种视角、预算、卸载、近邻替换、LiDAR、移动布局和离线请求。

截图与机器报告存放在 `viewer/runtime/verification/`；所有截图保持醒目的合成自检标记。本地浏览器验证用 Chromium SwiftShader，尚不能代表本机物理 GPU 的性能。固定低数量模型自检也不等同于大型实景 PLY 压测。

## 访问范围与许可来源

服务只提供固定前端资源、`GET/HEAD /api/manifest` 及清单登记的 `/api/blocks/{id}/model|lidar`。没有写接口、目录列表或通用工程文件下载。路径使用 `dir_fd + O_NOFOLLOW` 逐层打开，拒绝 `..`、绝对路径、符号链接、隐藏目录与非普通文件；拒绝非本地主机名及跨站子资源请求。正常 `--host` 参数仍可显式改绑定地址，本次不开放网络监听。生产清单和模型文件只读。

核验日期 2026-10-06。实际安装的 npm 包锁定版本和完整性见 `package-lock.json`；未修改包内容。

| 依赖 / 来源 | 核验 |
| --- | --- |
| [Spark 官方仓库](https://github.com/sparkjsdev/spark) / [官方安装示例](https://sparkjs.dev/docs/) | `@sparkjsdev/spark 2.3.1`，官方示例使用 Three.js 0.180.0；本地 peerDependency `three >=0.180.0` |
| [官方 SplatMesh 文档](https://sparkjs.dev/docs/splat-mesh/) | fileBytes、initialized、SH3、extSplats float32、dispose、多对象场景集成 |
| [官方加载说明](https://sparkjs.dev/docs/loading-splats/) | Gaussian PLY；说明 Spark 也接受点云，查看器前检故意拒绝此 Gaussian 替代入口 |
| [官方性能说明](https://sparkjs.dev/docs/performance/) | Gaussian 栅格化与从后向前透明度混合，关闭 MSAA 与限制像素比 |
| [Spark LICENSE](https://github.com/sparkjsdev/spark/blob/main/LICENSE) | MIT，版权 WORLD LABS TECHNOLOGIES, INC.，本地 `node_modules/@sparkjsdev/spark/LICENSE` 保留原文 |
| [Three.js 官方 0.180.0 LICENSE](https://github.com/mrdoob/three.js/blob/r180/LICENSE) | MIT，本地 `node_modules/three/LICENSE` 保留原文 |
| [Playwright 官方](https://github.com/microsoft/playwright) | 仅开发自检依赖 `1.56.1`，Apache-2.0；不被页面导入 |

依赖许可证随原始包保留。训练算法、模型与采集数据的许可及使用范围由相应原始产物决定；本查看器的 MIT 渲染器许可不改变它们。
