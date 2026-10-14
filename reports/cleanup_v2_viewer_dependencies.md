# 最终在线查看器清理依赖审计

结论：26份PLY迁至 `maps/garage_fidelity_v2_full_preview/blocks/{id}.ply` 并同步manifest后，四个指定入口无需依赖其余训练目录。8768与8770共用最新000资产，无须复制第二份。本文只作依赖审计；实际迁移、删除由主代理执行。

审计时间：2026-10-09T09:09:15.743304+00:00。仅新增本报告及同名JSON；没有修改现有文件，没有GPU、浏览器或服务启停操作。没有全盘du，只统计viewer运行/测试依赖目录。

| 保留入口 | 必需文件与模型 |
| --- | --- |
| 8767 `/reports/full_preview_v2/` | landing HTML、3张实际截图、`full_preview_v2_viewer.json/.md`、其链接的assembly、full PLY、source audit JSON |
| 8769总览 | `overview_manifest.json`、`gaussians_overview.ply`（1,200,000点）＋共享server/viewer依赖 |
| 8770详图 | `manifest.json`、26份原始PLY（13,681,252点）＋共享server/viewer依赖 |
| 8768最新000诊断 | `maps/garage_fidelity_v2_block000_anneal_diagnostic/manifest.json`，共用26资产中的000（2,495,734点） |

**最小不可删文件**（JSON的 `runtime.minimum_nonmodel_keep` 与 `migration.mapping` 可直接用于主代理保留清单）：

- `scripts/serve_map.py`：系统Python 3.8+标准库运行，三个查看器服务均无需Torch、CUDA、训练venv或Node开发服务。
- `viewer/index.html`、`viewer/style.css`、`viewer/app.js`、`viewer/gaussian_sh_restore.js`、`viewer/gaussian_direction_support.js`。
- `maps/garage_reference_views.json`：94,412字节，自带各区块世界位姿及FOV。其 `source_metadata` 是历史来源，不读取对应transforms或图像；删除本JSON会丢失“进入区域”功能。
- 上述三个当前manifest、26份源PLY、full／overview PLY。full虽不参与总览渲染，但landing提供最终完整模型下载，属于当前交付强依赖。
- `reports/full_preview_v2/index.html`、`overview.png`、`overview_side.png`、`f6_block_027.png`；`reports/full_preview_v2_viewer.json/.md`；`maps/garage_fidelity_v2_full_preview/assembly.json`；链接的 `reports/full_preview_v2_source_audit.json`。

26份PLY合计 3392991812 字节；full 3392952092 字节；overview 297601595 字节。共28份不同用途的PLY，总计 7083545499 字节（不含主代理另选的重建点云）。000诊断不新增模型存储。当前三份manifest均无LiDAR或direction sidecar，毋须因此保留历史sidecar训练包。

**JS最小模块闭包**：

```text
viewer/app.js
├─ gaussian_sh_restore.js
├─ gaussian_direction_support.js（v3动态导入）
├─ three/build/three.module.js → three.core.js
├─ three/examples/jsm/controls/OrbitControls.js → three
├─ three/examples/jsm/loaders/PLYLoader.js → three
└─ @sparkjsdev/spark/dist/spark.module.js
   ├─ three
   └─ three/examples/jsm/postprocessing/Pass.js → three
```

这些路径均位于现有 `viewer/node_modules`；server的STATIC白名单直接使用它们。Spark 2.3.1的WASM在bundle第2922行内嵌，worker在第2942行内嵌并通过Blob创建，无另行下载的worker/wasm强依赖。已对全部STATIC路由逐一比对HTTP字节和本地SHA。

最稳妥建议保留整个 `viewer/node_modules`：60098645 字节（约57.31 MiB），同时保留package.json、package-lock.json及依赖许可证。精确模块闭包仅是静态核实，未物理裁剪后复测；没有必要为几十MB承担裁剪遗漏风险。`viewer/runtime`约0.98 GiB，是测试用Chromium等缓存，用户浏览器和当前服务不使用；删除仅影响未来旧自动化测试。`viewer/tests`亦非在线运行依赖。

**26份PLY迁移约定**：

每份目标为 `maps/garage_fidelity_v2_full_preview/blocks/{id}.ply`；完整from/to、真实count、bytes和内容SHA在JSON `migration.mapping`。000为最新切向尺度上限退火；015为joint014–016单份联合资产；其余24份保留历史模型。不得按source_blocks再复制joint三遍。

1. 主代理先落地真实目标文件、验证原SHA/count，再原子更新8770的 `blocks[].path`，以及8768的 `blocks[0].path` 与 `source_export.path`。保留id、source_blocks、bbox、楼层未核验标记、renderer、内容SHA及历史分数。
2. API `/api/blocks/{id}/model` 不变。MapStore每次请求重读manifest，因此路径修改不要求重启server；浏览器“刷新清单”会卸载并重新索引。旧路径待迁移路由验证后再清理，以兼顾正在进行的请求。
3. 不可用软链接代替真实文件：`open_local`对每层目录/文件使用 `O_NOFOLLOW`。临时硬链接是普通文件、可供同文件系统暂存，最终目录中必须有真实可读数据。
4. 当前预算保持：8768为250万／640MiB／1块；8769为150万／512MiB／3块；8770为400万／640MiB／4块。仅改manifest不能提高服务预算。

本审计已在内存验证修改路径后的manifest符合schema；没有创建目标PLY，故不将未来迁移路由宣称已通过。主代理迁移后应检查三个API身份、26+1模型HEAD/Range及实际count/SHA；当前HTTP强依赖检查均通过。

**hash与历史证据**：

纯移动不改PLY内容，因此26源模型、full、overview、参考相机、renderer和既有截图的内容hash保持不变；8770／8768 manifest的hash必变。现有viewer/容量/native报告中的manifest hash、旧runs路径，是当时的证据，不应悄悄替换为新hash并沿用“已测”结论。另存迁移映射、前后manifest hash和迁移后CPU检查即可。

建议保留原assembly.json作构建来源记录，另添迁移映射。若主代理选择修改 `assembly.sources[].path`，先保存原bytes及SHA，再保留original_path并记录新SHA；已有viewer报告中的assembly hash应继续解释为历史版本。旧 `maps/garage_manifest.json`、旧28查看器、joint015单独诊断及历史parity作业不会自动修复，非本轮选定入口，应归档标历史，不能因它们仍有路径就递归保留训练树。

**运行依赖以外的小证据**：

当前manifest直接引用的小JSON共 34 份、965742 字节；详见JSON `historical_provenance.direct_small_json_files`。训练结果、640／1280指标、几何报告、checkpoint导出证明及单帧容量报告值得独立归档。runs中的小JSON可移至 `reports/cleanup_v2_history/runs/...`，记录原路径与SHA，毋须为其保留PNG、float npz、checkpoint、配置目录或数据集。
建议另保留full source audit JSON/MD、原015 v3全20结果及capture JSON、joint015及最新000单帧结果及capture JSON、原manifest快照、两新模型的几何/导出proof JSON。具体存在文件及大小、SHA见 `additional_evidence_to_archive`；不递归追踪它们内部的图像、模型、dataset、checkpoint路径。

文档可明确标注：“历史训练/评估中间文件已清理；本文件为当时记录。现存模型位置见迁移清单，历史分数和PLY内容哈希不变。” `reference_views.source_metadata`、`assembly.input_sha256` 与指标中的source images都属于此类来源说明。

旧 `reports/fidelity_v2_visual_review.html` 的逐帧比较依赖大量runs图像，它不属于此次保留的full-preview landing。若清掉图像，应停用/标明旧页已归档，不能留作仍可用入口；不因旧页而保留全部历史测试输出。训练环境与data目录不被本次四入口使用，但重建点云/源数据的保留由主代理另行裁决，本报告不代替其全局清理策略。

完整机器可读清单：[`cleanup_v2_viewer_dependencies.json`](cleanup_v2_viewer_dependencies.json)。
