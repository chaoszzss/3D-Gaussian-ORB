# 最终成果整理与永久清理

时间：2026-10-09T17:22:48.921309+08:00。按用户明确要求，已永久删除旧实验与中间结果，没有移入回收站。

工程实际占用：**271.46 GB → 19.78 GB**（252.82 GiB → 18.42 GiB），减少 **251.68 GB / 234.40 GiB**。删除314,308个文件/符号链接和9,941个空目录，无跳过项。以du对整个工程的实际分配块统计为准，已考虑训练环境与缓存之间的硬链接；各删除目录逻辑大小不能直接相加作为释放空间。

## 保留成果

| 内容 | 当前位置 |
| --- | --- |
| 完整三维高斯PLY，13,681,252条记录 | [maps/garage_fidelity_v2_full_preview/gaussians_full.ply](../maps/garage_fidelity_v2_full_preview/gaussians_full.ply) |
| 网页总览，120万条真实抽样记录 | [maps/garage_fidelity_v2_full_preview/gaussians_overview.ply](../maps/garage_fidelity_v2_full_preview/gaussians_overview.ply) |
| 全部26份分区模型，覆盖28个采集区域 | [maps/garage_fidelity_v2_full_preview/blocks/](../maps/garage_fidelity_v2_full_preview/blocks/) |
| 最新000、014–016联合训练最终权重及配置 | [maps/garage_fidelity_v2_full_preview/checkpoints/](../maps/garage_fidelity_v2_full_preview/checkpoints/) |
| 最终原生渲染对照图、指标 | [evaluation/index.html](../maps/garage_fidelity_v2_full_preview/evaluation/index.html) |
| 在线查看入口与实际截图 | [full_preview_v2/](full_preview_v2/) |
| 固定LiDAR参考、初始化点云 | data/garage_surface_v1/block_*/lidar_reference.npz、points3D.ply |
| 轨迹、标定、输入索引与准备配方 | data/all_lio_candidate/、configs/、保留的transforms/preparation与CSV |

保留工程源码、已有未提交修改、Git与标定工作树、训练Python环境、已验证的自定义CUDA后端及必要网页依赖。`runs/`只剩小型训练/评估来源记录；最终模型与权重已集中到上述成果目录。工程外原始bag、原始PCD和标定来源均未动。

## 已清理内容及影响

删除旧轮次及失败候选的整图/分区导出、旧训练checkpoint、重复RGB/深度/掩码、float渲染数组、TensorBoard事件、测试截图与临时结果、浏览器/安装/编译缓存。源码和手写验证脚本保留。

最终PLY可继续直接浏览，训练环境仍保留。继续训练需重新准备已删除的RGB/深度等输入，依赖原始bag、工程外LIO地图/标定和留存配方；不能承诺仅靠用户bag逐字节复原全部历史来源。以前的审计和status是历史证据，其中一些中间路径已经删除。旧source路径到现存路径见[迁移及SHA清单](cleanup_v2_migration.json)。`assembly.json`与旧审计保持原始字节，不把历史hash冒充迁移后manifest的hash。

保留的是已交付的混合全场预览：000及014–016为本轮新模型，其余24区使用历史模型。覆盖全部28采集区，不代表六层接近1:1高保真验收通过。

## 记录与检查

- [精确删除清单（gzip JSONL）](cleanup_v2_delete_plan.jsonl.gz)、[执行记录](cleanup_v2_execution.json)
- [独立保留审计](cleanup_v2_retention_audit.md)、[删除清单复核](cleanup_v2_plan_review.md)
- [删除后模型/权重/源码/后端完整性核验](cleanup_v2_integrity.json)
- [删除后网页、模型路由与依赖核验](cleanup_v2_viewer_postcheck.json)

删除后检查已通过：477份留存文件（含26份分区、完整/总览PLY及两份最终权重）SHA一致；57份固定点云参考SHA一致；77份未计划修改的已跟踪文件字节不变；原始与AA-depth-VJP两套后端导入及固定证据核验通过。8767入口、8768诊断、8769总览、8770全部26分区、四路参考视角、108张对照图和11份网页静态依赖均通过HTTP核验。检查没有使用GPU或重新做质量验收。
