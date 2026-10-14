# 全场预览独立源覆盖核查

PASS仅指源覆盖和资产身份，不是高保真质量验收。

26个资产覆盖28个源区块，无重复归属：24个历史资产＋000新资产＋014/015/016联合资产。共13,681,252条导出PLY记录。
坐标camera_init，单位m；全部parser变换为单位矩阵、scale=1。实际源并集bbox：{'min': [-38.333412170410156, -71.57218170166016, -12.985081672668457], 'max': [104.46508026123047, 80.49833679199219, 21.970640182495117]}。

|资产|覆盖源区块|PLY记录数|源路径|
|---|---|---:|---|
|block_000|block_000|2495734|`runs/fidelity_v2/block000_measuredseed_anneal_v1/gaussians.ply`|
|block_001|block_001|342875|`runs/garage_surface_v1/block_001/eval_dev/gaussians.ply`|
|block_002|block_002|364881|`runs/garage_surface_v1/block_002/eval_dev/gaussians.ply`|
|block_003|block_003|368896|`runs/garage_surface_v1/block_003/eval_dev/gaussians.ply`|
|block_004|block_004|384304|`runs/garage_surface_v1/block_004/eval_dev/gaussians.ply`|
|block_005|block_005|409298|`runs/garage_surface_v1/block_005/eval_dev/gaussians.ply`|
|block_006|block_006|400192|`runs/garage_surface_v1/block_006/eval_dev/gaussians.ply`|
|block_007|block_007|389736|`runs/garage_surface_v1/block_007/eval_dev/gaussians.ply`|
|block_008|block_008|378061|`runs/garage_surface_v1/block_008/eval_dev/gaussians.ply`|
|block_009|block_009|378567|`runs/garage_surface_v1/block_009/eval_dev/gaussians.ply`|
|block_010|block_010|361041|`runs/garage_surface_v1/block_010/eval_dev/gaussians.ply`|
|block_011|block_011|369946|`runs/garage_surface_v1/block_011/eval_dev/gaussians.ply`|
|block_012|block_012|370745|`runs/garage_surface_v1/block_012/eval_dev/gaussians.ply`|
|block_013|block_013|368601|`runs/garage_surface_v1/block_013/eval_dev/gaussians.ply`|
|block_015|block_014, block_015, block_016|2498837|`runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/gaussians.ply`|
|block_017|block_017|358029|`runs/garage_surface_v1/block_017/eval_dev/gaussians.ply`|
|block_018|block_018|343041|`runs/garage_surface_v1/block_018/eval_dev/gaussians.ply`|
|block_019|block_019|303023|`runs/garage_surface_v1/block_019/eval_dev/gaussians.ply`|
|block_020|block_020|364368|`runs/garage_surface_v1/block_020/eval_dev/gaussians.ply`|
|block_021|block_021|343186|`runs/garage_surface_v1/block_021/eval_dev/gaussians.ply`|
|block_022|block_022|360323|`runs/garage_surface_v1/block_022/eval_dev/gaussians.ply`|
|block_023|block_023|362611|`runs/garage_surface_v1/block_023/eval_dev/gaussians.ply`|
|block_024|block_024|331951|`runs/garage_surface_v1/block_024/eval_dev/gaussians.ply`|
|block_025|block_025|405967|`runs/garage_surface_v1/block_025/eval_dev/gaussians.ply`|
|block_026|block_026|391296|`runs/garage_surface_v1/block_026/eval_dev/gaussians.ply`|
|block_027|block_027|235743|`runs/garage_surface_v1/block_027/eval_dev/gaussians.ply`|

每个PLY完整流式SHA、全部float有限性、实际bbox/count/stride与清单及export元数据一致；全部训练config及split归属已核对。

已有相机keyframes共6508张（另IPM164张）；28块去重后6496张，TRAIN/DEV/TEST分别5684/408/404，全局互斥。替换模型的训练/DEV/TEST集合与被替换源区块的并集相同。
实际keyframe header范围：{'min_ns': 1720267880900293988, 'max_ns': 1720268705743587726, 'span_s': 824.843293738}；已准备图像范围：{'min_ns': 1720267881750246842, 'max_ns': 1720268705243734726, 'span_s': 823.493487884}。未进入prepared blocks的已保存相机帧12张，逐条见JSON。

用户bag完整SHA历史证据：`c0e602ec6f8eb866e62c60aab7c0ff68759470f818262e3a2e00e4d07faebfc1`，24545241324 bytes。本次仅核对既有证据与当前stat，未打开或重新hash bag。images/messages CSV记录独立核对；header采集时间与bag写入时间分开报告。

006更正保留：loop11999/声明12000，六组Adam计数11001；不写成12000次优化，也不写成未经训练。

F1-F6＋坡道沿用未核实标签；完整仅指28个现有采集区块覆盖，不等于每条bag消息均训练或未观测空间完整。新000/joint现有导出分别省略32/5条checkpoint记录，未在本审计改写。
各来源分数未作混合平均；全场分数为空。后续无损拼接核查同样不代表高保真。

命令：`CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -B reports/full_preview_v2_source_audit.py`
CPU≤3线程、无GPU；耗时4.40s，RSS峰值431.2MiB。
