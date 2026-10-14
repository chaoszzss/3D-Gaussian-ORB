# 清理前独立保留审计

本轮只读：未删除、移动或修改现有文件。主代理必须先形成具体删除清单，再对照本报告检查父目录、实际路径及符号链接；本报告不是删除执行清单。

## 最终交付不可丢失

26个源PLY覆盖28区块，共13,681,252条记录。分区viewer仍直接读runs中的路径；只保留full/overview却删掉这些源PLY，会破坏26区分区查看。源SHA来自刚完成的逐字节独立审计，本轮重新核对header/count/size而不重复读取全部大PLY。

|资产|源区块|记录数|路径|SHA256|
|---|---|---:|---|---|
|block_000|block_000|2495734|`runs/fidelity_v2/block000_measuredseed_anneal_v1/gaussians.ply`|`450121da2b23919aa3831d7ee41e893e28b93b2d78d35bae6b07110a4e227bfc`|
|block_001|block_001|342875|`runs/garage_surface_v1/block_001/eval_dev/gaussians.ply`|`3785efef2d38b5d42e78c5459a2b591aa35aafa98d018fcec7b8ff443207187f`|
|block_002|block_002|364881|`runs/garage_surface_v1/block_002/eval_dev/gaussians.ply`|`e0bdd8a676fb5f9ce5f6183e57e69b8628fd29167b4f71bba1951cf6e42fc100`|
|block_003|block_003|368896|`runs/garage_surface_v1/block_003/eval_dev/gaussians.ply`|`a8994bbfcf73dde83ce417788a86e48d17beac785f6731a44dcd8a964f3cd6ec`|
|block_004|block_004|384304|`runs/garage_surface_v1/block_004/eval_dev/gaussians.ply`|`0e3b74ffda27ba810cfe2df96f529857a20425767644b2a090876d090f13e8f7`|
|block_005|block_005|409298|`runs/garage_surface_v1/block_005/eval_dev/gaussians.ply`|`52c5fbcf93d96569899b5011a2d067abfdb0eb01d3fd9d51f09ccac9b7416f24`|
|block_006|block_006|400192|`runs/garage_surface_v1/block_006/eval_dev/gaussians.ply`|`e66cd2217726eb5740b3a3565f3061417cf7b19be63d07d647e6b18837532e1c`|
|block_007|block_007|389736|`runs/garage_surface_v1/block_007/eval_dev/gaussians.ply`|`c52388c31b3483517305ff8258fbc3da9118ff01a486435bc4cbfa9cd7c746a0`|
|block_008|block_008|378061|`runs/garage_surface_v1/block_008/eval_dev/gaussians.ply`|`e6babdf6effef3f561d7029eff8cc122058de3fcf15f1b30b478ac24ad9ddf69`|
|block_009|block_009|378567|`runs/garage_surface_v1/block_009/eval_dev/gaussians.ply`|`b1bf5b73294f71968fed0828a0f93a4f511304caf64d6a8de26ab5d13adebc51`|
|block_010|block_010|361041|`runs/garage_surface_v1/block_010/eval_dev/gaussians.ply`|`b2ad7dded0ead1b9e5dd9a347bd38c01e25ab458223b6ab4bdbb1fd500842696`|
|block_011|block_011|369946|`runs/garage_surface_v1/block_011/eval_dev/gaussians.ply`|`cf4df8161f9c58762925bf70a23a3a7217d481084c1aeb5fe157acb9ef62ec06`|
|block_012|block_012|370745|`runs/garage_surface_v1/block_012/eval_dev/gaussians.ply`|`98b51e2de919d6bc9727167bc265982736bdde0be58a3f37e885257d2a5e8242`|
|block_013|block_013|368601|`runs/garage_surface_v1/block_013/eval_dev/gaussians.ply`|`136c01fca355a1a862477651201af2c6f92a94e71b4896d33bfd501f182f41ba`|
|block_015|block_014,block_015,block_016|2498837|`runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/gaussians.ply`|`870258a2a53790fe9571d46a423b0e9b6d5b37afd9f0ad55b3a85369e9700eba`|
|block_017|block_017|358029|`runs/garage_surface_v1/block_017/eval_dev/gaussians.ply`|`05c16010823c3896a3c010cedf50b16e2863b1e7f1ac37b7fa0ec650600c6f30`|
|block_018|block_018|343041|`runs/garage_surface_v1/block_018/eval_dev/gaussians.ply`|`af2a59435a0fd684ff9a5bc062dd80966456ec72210fb3d08808ae95dd46e796`|
|block_019|block_019|303023|`runs/garage_surface_v1/block_019/eval_dev/gaussians.ply`|`5302cc005bf085ba0df91728f01dfa511009b9001543c49467ac86dd0bb7ff0a`|
|block_020|block_020|364368|`runs/garage_surface_v1/block_020/eval_dev/gaussians.ply`|`1da513fafc87d7a1ab7d196ffa9d57eece720bcb18f9f0936818af130a0bcd3b`|
|block_021|block_021|343186|`runs/garage_surface_v1/block_021/eval_dev/gaussians.ply`|`7bdae5fe622432ce4f5e24387d0c296d3f41a0ef2e38a377eb7bdf1e01bc12b7`|
|block_022|block_022|360323|`runs/garage_surface_v1/block_022/eval_dev/gaussians.ply`|`dd2a5a50fbf240b2fb723b2a05a98f7ffdf5d78c174dd98a388eb688a63919b5`|
|block_023|block_023|362611|`runs/garage_surface_v1/block_023/eval_dev/gaussians.ply`|`3388513577b698d0203d42b4b4025f67fa460dd9b72261cd6d3fe2dbd886a71a`|
|block_024|block_024|331951|`runs/garage_surface_v1/block_024/eval_dev/gaussians.ply`|`b6eeb699a224c4369a3908e6c0aa0ad2fa9bb5b6b8d0ff5822013c19c5a19cb6`|
|block_025|block_025|405967|`runs/garage_surface_v1/block_025/eval_dev/gaussians.ply`|`6cb2162815604b07cfa21c812d89da8056d746915be8cddeb3e32da3ebdb694e`|
|block_026|block_026|391296|`runs/garage_surface_v1/block_026/eval_dev/gaussians.ply`|`e5c7fee6cd401eda38d7ff2249f89cc4d8a9c4a0cd59bd55a4cff2bac8004d1e`|
|block_027|block_027|235743|`runs/garage_surface_v1/block_027/eval_dev/gaussians.ply`|`119085e52d23426528da10ca6abb2c195a0c6ef565948806ed9044518f0bf11e`|

|合并交付|记录数|文件字节|SHA256|
|---|---:|---:|---|
|`maps/garage_fidelity_v2_full_preview/gaussians_full.ply`|13681252|3392952092|`d1f458d96d4b33865c817e5eb1b5e1db80720ec1eaba8fdcdeeb184fa66c0c83`|
|`maps/garage_fidelity_v2_full_preview/gaussians_overview.ply`|1200000|297601595|`d3df427141a27568e6812733467d48d465891754bb93613aabc58be4f2d5ec18`|

full/overview、assembly/manifest/overview_manifest、maps/garage_reference_views.json、reports/full_preview_v2入口和截图必须保留。源路径若迁移，需要保留旧→新路径与SHA映射、由主代理更新查看器引用并核验，不能删除仍被使用的源路径。

## 最终训练状态与可渲染PLY分开

建议仅保留以下两个最新14999 checkpoint，加上对应training_result.json、launch.json、refinement.json、config.yml和恢复证明。网页查看无需checkpoint；但PLY不能恢复Adam状态及被导出过滤的00032条/joint5条记录。删除历史/中间checkpoint不影响当前PLY预览，但不能再声称可恢复那次训练。

- `runs/fidelity_v2/train_block000_measuredseed_anneal_v1/block_000/splatfacto/2026-10-09_150023/nerfstudio_models/step-000014999.ckpt`：1,776,903,634B，SHA `cfb20371011a5d4c7832183fef5e9a0cdd9403566ecd576f374418379d6241cf`。
- `runs/fidelity_v2/train_joint015_depthv3_aa_metric_cap003/core_015/splatfacto/2026-10-09_121414/nerfstudio_models/step-000014999.ckpt`：1,779,081,106B，SHA `8c969f712d8d3b3588a9ef3fbfe4a8cd8612e79daf6a9ceed0da9979436dc3fa`。

两者合计3.312GiB；不要求保留所有旧checkpoint或训练环境。checkpoint本身仍需数据/配置/依赖才能继续训练，且不保证RNG逐位连续。

## 不能按“可再生成data”一刀切

保留data/all_lio_candidate/odometry.csv及provenance.json：8245条LIO轨迹来源是另一份工程外registered_and_semantics.bag，不是仅凭用户bag即可保证复原的原始VIO。保留configs全部标定与修订、worktrees/rig-board和rig-refine中的candidate/prior/refined_test_gate及源码。

data中未发现原始PCD；真正原始重建地图在工程外：
`/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/czy_joint_raw_map.pcd`（307,186,588B）。外部原始标定、LIO bag及用户bag均只读、不得触碰，具体路径见JSON。

建议保留28份`data/garage_surface_v1/block_NNN/lidar_reference.npz`、28份`points3D.ply`及`data/reference_map_voxel01.npz`，共363.1MiB。它们是有用的采集点云/固定参考衍生品；不是完整原始PCD或独立真值。无需保留各重复准备版本的同一份点云。

保留28块和新000/joint的transforms.json、preparation.json、keyframes source/manifest和CSV索引、固定小掩码。RGB/深度/深度掩码、重复prepared datasets可列候选，但再生成依赖工程外PCD、标定、LIO轨迹和冻结recipe；不能承诺仅靠原bag逐字节重现。

## 在线查看与源码保护

scripts/serve_map.py是Python标准库服务，无需CUDA或.venv-train。必须保留viewer源码，以及实际静态路由依赖的viewer/node_modules/@sparkjsdev/spark和three（约45MiB，包含被import的兄弟文件）。Playwright/浏览器测试缓存与训练环境属于不同类别，可独立评估清理。

.git、已跟踪文件、现有修改、未跟踪源码/配置一律不能按缓存删。runtime/inria2dgs_src及inria2dgs_aa_depth_vjp_src包含自定义CUDA源码/许可证/构建来源，不是整个目录都能删；build缓存、.venv-train需与源文件分开。runtime/*.py、reports/*.py也是现有工作。工作树的独有未跟踪校准结果已列入JSON。

## 最小证据与删除后限制

保留源/拼接/浏览器验收JSON和说明、最终000/joint配对结果和checkpoint/geometry证明、固定plan、bag full-SHA证据、006历史恢复更正、标定来源与backend证明，以及所选26资产的小config/training/eval元数据。JSON中给出逐文件路径/大小/SHA清单。

原006保留更正：loop11999/声明12000，但Adam更新11001。楼层F1–F6和坡道未独立验证；不平均各来源PSNR为整场分数，不把源覆盖和无损拼接当高保真。

可清理候选包括旧训练/中间checkpoint、已替换旧模型、调试float渲染与PNG、重复图像/深度、pytest/build缓存。先排除本报告具体保留文件及其父目录、活跃网页引用和用户源码。删除float预测/教师深度后，旧配对报告只是历史证据，不能再宣称完整可重跑。

## 定向容量估算（非全局du）

- sourcePLY_logical_bytes：3,392,991,812B（3.160GiB）
- full_overview_logical_bytes：3,690,553,687B（3.437GiB）
- minimum_metadata_and_live_software_logical_bytes：127,994,058B（0.119GiB）
- canonical_reference_and_initializer_pointcloud_bytes：380,772,253B（0.355GiB）
- latest_two_checkpoint_logical_bytes：3,555,984,740B（3.312GiB）

本审计只新增reports/cleanup_v2_retention_audit.json和.md；尚未收到/审核主代理实际删除清单，下一步是独立核对该清单没有覆盖保留项。
