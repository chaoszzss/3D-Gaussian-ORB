# 本地交付与验证记录

验证日期：2026-10-06。所有写入均为 `3D-Gaussian/viewer/` 与 `3D-Gaussian/scripts/serve_map.py`；没有修改训练、标定、原工程或 git 状态。本查看器没有上传、发布或使用 Sites。安装、npm 缓存、浏览器下载、浏览器 profile、Python 临时目录与验证截图全部位于 `viewer/`。Python 运行设置 `PYTHONDONTWRITEBYTECODE=1` 与工程内 `TMPDIR`。

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `../scripts/serve_map.py` | Python 3.8 标准库服务器，默认 127.0.0.1:8765，清单与模型前检、只读白名单、无符号链接访问、Range/HEAD |
| `index.html`、`style.css`、`app.js` | 中文 UI、官方 Spark GPU Gaussian 渲染、统一米制空间、多块可见性、楼层索引、观察角度、近邻工作集、独立 LiDAR 辅层 |
| `package.json`、`package-lock.json` | Spark 2.3.1、Three.js 0.180.0、开发自检 Playwright 1.56.1 的本地依赖与固定完整性 |
| `manifest.schema.json` | 模型清单协议 v1 |
| `garage_manifest.example.json` | 待训练接口示例；包含占位路径与包围盒，不能当实景地图 |
| `README.md` | 启动、协议、导出属性、兼容性、内存预算、验证复跑与官方许可来源 |
| `make_fixture.py` | 明确生成合成 Gaussian 与辅层自检数据 |
| `fixtures/synthetic_manifest.json` | 明确标记 `dataset_kind: synthetic` 的四区块清单 |
| `fixtures/synthetic_a.ply` 至 `synthetic_d.ply` | 每块 882 个高斯，分别 SH3、SH1、SH2、SH0，全部小端二进制；坐标分别偏移 [0,0,0]、[8,0,0]、[0,0,5]、[8,0,5] |
| `fixtures/synthetic_lidar.ply` | 81 个合成 XYZ 几何辅层点，独立标签及 Three.js 点材质 |
| `fixtures/synthetic_ascii_unsupported.ply` | ASCII Gaussian 的明确负例，不登记为可加载区块 |
| `tests/test_server.py` | 9 项协议、安全及 HTTP 测试 |
| `tests/browser.mjs` | 实际 Chromium/WebGL2/Spark 浏览器交互与绘制验证，自行启动并停止临时 loopback 服务器 |
| `runtime/verification/browser-report.json` | 本次浏览器测试的机器报告 |
| `runtime/verification/synthetic-gaussian.png`、`synthetic-mobile.png` | 本次合成自检的桌面与移动布局截图 |
| `node_modules/` | 已安装的本地依赖及原始 LICENSE，未改包内容 |
| `.gitignore` | 仅查看器内的依赖与运行缓存忽略规则 |

`runtime/` 和 `node_modules/` 不需要进版本控制。依赖可用 `npm --prefix viewer ci` 复原，合成数据与验证产物可由以上脚本复原。生产者的 `maps/garage_manifest.json` 未被创建或修改，接口与兼容边界已通知主代理。

## 结果

`/usr/bin/python3 viewer/tests/test_server.py`：**9 项通过**。确认缺失生产清单时空态为 real 而非 synthetic；XYZ/RGB 无法冒充 Gaussian；SH 阶数、损坏长度、首条非有限数值、非法坐标与尺度单位、符号链接、非法路径、重复 JSON key、Range/HEAD、本地主机名及跨站请求限制均经过测试。

`node viewer/tests/browser.mjs`：**15 项检查通过，错误列表为空，31 个请求全部为 loopback**。真实使用 Spark 解码 PLY、加载 Gaussian 属性纹理、排序、GPU 绘制。以渲染画布截图的 SHA256 差异核验：加载后的画面与空画面不同，隐藏后画面恢复为空；测试时去掉网格和空态文字的影响。

| 绘制状态 | 本次画布 SHA256 |
| --- | --- |
| 空画面 / 隐藏所有高斯 | `9ba7542c0f92f00f44ca9547e7d0df96a18ad3b5140cc359ab508f9ba102c83b` |
| 显示首个合成区块 | `4673444828b29d7395628fd73fe3e9e2ef9cff891bfb77340332a88ef231879e` |

额外确认：模型 Object3D 位置 [0,0,0]、四元数 xyzw [0,0,0,1]、scale [1,1,1]，中心使用 ExtSplats float32。读取保留 SH3，其他三个区块分别保留 SH1、SH2、SH0；第二块解码后中心 X=8 m，不存在逐块自动重置原点。两个下层区块适配后的观察中心约 X=4 m。第四块超出默认三块预算而被阻止；隐藏仍计入内存预算；卸载从工作集中删除并处理模型资源，卸载 Gaussian 同时卸载其 LiDAR 辅层。

浏览器验证还覆盖：两层与未知楼层索引、四种观察角度、近邻按钮替换工作集、独立 LiDAR 标记、刷新清单卸载、窄屏布局、初始无自动模型请求、实景模式与合成模式隔离。`node --check` 对前端与浏览器验证脚本通过。全部 Python 实际使用系统 Python **3.8.10**；服务仅使用标准库。

## 限制

没有 trained 实景 PLY，当前不构成真实车库重建、楼层准确性或高保真质量验收。截图全部是合成自检。浏览器检查使用 Chromium SwiftShader；本机物理 GPU 帧率、训练同时运行时的显存余量、百万级 PLY 加载峰值尚未测量。默认用户按需选择/近邻块，预算 3 块 / 150 万高斯 / 512 MiB 输入，尚未提供完整分层 LOD。Spark 2.3.1 拒绝 ASCII Gaussian PLY，因此 Gaussian 接口只接小端二进制标准 float32 PLY；LiDAR 辅层支持 ASCII。

生产清单的 `training` 与 `eval` 为生产者声明，模型可读不代表训练或几何精度已验收。前检是文件头与首个 vertex 的结构检查，不是整模型数值或物理正确性证明。卸载 dispose 模型资源后，渲染器仍可保留全局排序缓冲容量复用，预算不等于显存实测。
