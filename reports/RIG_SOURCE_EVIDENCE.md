# Rig source evidence：raw topic、FAST/board 编号与 LIO 坐标链

只读证据核查已完成。明确的 FAST-Calib 声明为 `/rk_cam/3→cam0、/rk_cam/2→cam1、/rk_cam/4→cam2、/rk_cam/5→cam3`。原 bag 的 raw 编号来自 SDK metadata 的 `camera_id`，当前及历史 bridge 代码均保留这个规则。**raw2/3/4/5→FAST1/0/2/3 现在已有源码编号契约、bag source frame 和主代理 QR 开发集几何旁证；完整采集版本与 board cam0..3 的物理绑定仍保留实验验证边界。LIO body 参考点和内部 T_WB/T_BL 数学链已建立，不是阻塞项。**

本次仅创建本报告和 `reports/rig_source_evidence.json`。未安装包、未改配置/源码、未做 LiDAR projection 或外观猜测。Python 禁止 bytecode，TMPDIR 在项目 reports 内。搜索限于所给包和选定目录的有限深度，排除了图片、标签及构建数据。

**1. 编号结论及其证据边界**

FAST 声明见 [qr_params.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/qr_params.yaml:4) 的第 4–8 行及各 camera.topic 字段（33、51、69、87 行）。bag raw topic 和 frame_id 已实际读取，四路 connection callerid 都是 `/orbvi_ros_bridge`，首帧均为 JPEG，采集 header_ns 都是 `1720267880900293988`，尺寸由既有 bag 核查记录为 1280×1080。

| 原 bag topic（均以 /image/compressed 结尾） | SDK ID token | 同物理 ID 契约下的 FAST 模型 | board 同序假设 | 当前旧候选的 board 假设 |
| --- | ---: | --- | --- | --- |
| `/orbvi/raw/camera_2` | 2 | `cam1` | `cam1` | `cam0` |
| `/orbvi/raw/camera_3` | 3 | `cam0` | `cam0` | `cam1` |
| `/orbvi/raw/camera_4` | 4 | `cam2` | `cam2` | `cam2` |
| `/orbvi/raw/camera_5` | 5 | `cam3` | `cam3` | `cam3` |

FAST 一列是配置中已证明的 `/rk_cam/N` 声明应用到 bag 的优先候选；最后两列都是待试验项。不能因 raw camera_2 是排序后第一路，就宣告其为 board cam0。board YAML 只有 camN 和 TwcN，没有 source ID、topic 或采集版本标识。

具体 raw 消息桥接源码当前可读于 [frame_conversions.cpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/orbvi_ros_bridge/src/frame_conversions.cpp:1031)。bundle 分支第 1061–1064 行用 `camera.i.camera_id` 构造 topic，只有缺失/空 token 才回退到 `i`；第 1053–1056 行保留 source_frame_id 和采集 timestamp。单图分支见 613–616 行，decoded raw 分支见 1139–1141 行。[bridge_node.cpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/orbvi_ros_bridge/src/bridge_node.cpp:1331) 将 suffix 接到 `/orbvi` 并发布。此桥接代码没有 FAST/board 模型编号重排表。bag 的 source frame_id 与同一 ID 相符；没有保存原始 SDK metadata，不能仅凭 ROS message 恢复缺失 ID 的上游生成历史。

另一个 [raw_image_fusion_bridge_node.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_raw_image_fusion_bridge/src/raw_image_fusion_bridge_node.cpp:63) 是 Image→CompressedImage 重编码桥：输入/输出按同一 index 配对，162 行保留 header，182 行发布。**本 bag 的 callerid 和 JPEG 信息直接指向 orbvi_ros_bridge，不能用该重编码桥的默认 PNG/排序参数替代本 bag 的采集配置。**

当前 vendor bridge README 在 [README.zh-CN.md](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/README.zh-CN.md:184) 记载 front=3/right=2/rear=4/left=5；[orbvi_4vl_bag_four.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/config/orbvi_4vl_bag_four.yaml:18) 也以 raw3/raw2/raw4/raw5 绑定 FAST 参数。这些属于参考契约和候选配置，不能提升为本 bag 的实测标定值。当前挂载中旧 workspace 路径确实可读，报告记录的是本次实际看到的源码与 Git 对象。

**已实际读取的 source frame 身份与 QR 开发集旁证**

| bag raw topic | 首帧 source frame_id | FAST 声明与开发集对应 |
| --- | --- | --- |
| `/orbvi/raw/camera_2/image/compressed` | `orbvi/raw_fisheye_stream/camera_2` | FAST cam1，`/rk_cam/2/mjpeg` |
| `/orbvi/raw/camera_3/image/compressed` | `orbvi/raw_fisheye_stream/camera_3` | FAST cam0，`/rk_cam/3/mjpeg` |
| `/orbvi/raw/camera_4/image/compressed` | `orbvi/raw_fisheye_stream/camera_4` | FAST cam2，`/rk_cam/4/mjpeg` |
| `/orbvi/raw/camera_5/image/compressed` | `orbvi/raw_fisheye_stream/camera_5` | FAST cam3，`/rk_cam/5/mjpeg` |

主代理已完成 [QR dev scores](/home/chaoszzss/dataset/rosbag/3D-Gaussian/experiments/rig_candidates/qr_dev/scores.json) 与 [candidate.yaml](/home/chaoszzss/dataset/rosbag/3D-Gaussian/experiments/rig_candidates/qr_dev/candidate.yaml)：20 个 camera/frame 样本、5 个 mosaic，mean normalized MI=`0.115496071837`，mean reflectance/luma Spearman=`0.251504355107`；candidate 采用上述 raw2/3/4/5→FAST1/0/2/3 映射。其 7 个来源 SHA 引用均与本核查一致。本次只读这些既有工件，没有重复 projection。**这些分数支持开发集几何一致性，不是绝对像素误差；candidate 仍保留 verified=false 和 topic_mapping_verified=false。**编号声明、metadata rule、bag source frame 和开发集旁证形成可直接交接的 source-supported mapping，采集 firmware hash 缺失不应被改写成无关的 body 未知。board 实验正在 worktrees/rig-board，尚不宣告 board permutation。

**2. raw bridge 的历史稳定性与 release SHA 证据**

本地 vendor Git HEAD 为 `9bdca7ea700127dee02f3cc601ba8d4c5cd560fa`。工作树 frame_conversions.cpp 与 HEAD blob 的 SHA256 一致，下面列出跨版本对象；完整历史和逐段规则 hash 在 JSON。

| Git commit | author date | converter 文件 SHA256 |
| --- | --- | --- |
| `270d823e5023ba5b2bc97aa7696ba83ab690f632` | 2026-08-16T14:22:24+08:00 | `684fd875b08aa21d88f6b24af4d36d0c67f1d6056c7f94b7217d2b8a6929e441` |
| `9bdca7ea700127dee02f3cc601ba8d4c5cd560fa` | 2026-07-30T09:14:56+00:00 | `eb4135118d2a5373fed2739e67152ffab137d058003a0a0ea2c414566fd92f84` |
| `f39dc7101273fc374720c10bdfe8d5d9cea237fe` | 2026-07-28T02:02:46+00:00 | `ec1f98bdacf20c653ae67d412d28eed3cc83fa65b998578394da11a4db127772` |
| `979ef3a9b9bbd257dba0bcd0ec788a61ee6b3884` | 2026-07-21T11:42:29+08:00 | `a1b71678a5ee2209abab29813399f0275c8f5ce154c30c2052ac1405fb0fb163` |
| `ac16fde516daeca6ef260ba14c421bbcc4f73ca3` | 2026-06-17T15:50:33+08:00 | `92abe2fac00241d53edc0a73d851051a4c1ffe32bf8bd401adb2cf8b8756b325` |
| `1f19949bbfe19a79e3bbba10501eab831901e64e` | 2026-06-15T14:27:59+08:00 | `fbf5e840134a92f4deae37b6cb485e21daedec745afbfc27267465ecfc756a14` |

全部检查到的 11 个 converter 历史版本均以 metadata camera_id 构造 raw 编号；2026-06-17 的 ac16fde 后缀改成 `/image/compressed`，之后至 2026-08-16 保留此规则。bag recording clock 在 2026-08-06，采集 header clock 在 2024-07-06；两种时钟不能用于直接断言采集 firmware/SDK revision。当前 checkout 或 Git 日期也不构成 bag 所用二进制的证明。

V2 release 的 [runtime_source_sha256.json](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/runtime_source_sha256.json:53) 有 57 个文件条目，本次逐项重算有 57 个与 recorded SHA 一致；逐项差异保存在 JSON。关键的 `launch/czy_lio_only.launch`、`launch/joint_mapping2.launch` 与历史清单一致。该清单**不含 AVP 的 LIVMapper.cpp、IMU_Processing.cpp 或 fastlivo_mapping_avp 二进制 SHA**，所搜 release 及限定深度文档中未找到这些 mapper 的历史源码 snapshot；workspace/AVP 没有可用的 Git 历史（所见 workspace .git 为空）。因此报告给出当前 C++ 全文件 SHA，明确不宣称其与 release 二进制字节级一致。

作为运行语义的独立旁证，runtime seq_name 指向 `/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/Log/result/czy_height_continuity_release3_20260925.txt`；该 TUM 的前 8245 行与 bag 导出 CSV 逐行比较，最大时间差 `7.15e-07` s（float64/六位小数），最大平移分量差 `5e-07` m，考虑 q/−q 后最大 quaternion L2 差 `9.63e-07`。TUM 共 8246 行，多最后一条 `1720268705.800401` s；不能说 TUM 是 bag 的完整字节级副本。实际 registered bag 的 odometry 和 cloud connection callerid 均为 `/laserMapping`，首帧 header_ns=`1720267881300294161`，world 为 camera_init，odom child 为 aft_mapped。

**3. 已闭合的 LIO body/雷达坐标数学链**

统一定义 `T_A_B` 把 B 中的列向量变到 A。`W=camera_init`，`B=aft_mapped` 是由 `/orbvi/imu` 驱动的 IMU 状态参考点，`L` 是输入 `/orbvi/lidar/custom` 的数值坐标（bag frame_id 为 livox_frame），`C_i` 是对应 raw optical camera。

[IMU_Processing.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/IMU_Processing.cpp:675) 将 R_imu/pos_imu 回写 state；[LIVMapper.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/LIVMapper.cpp:2573) 的 set_posestamp 和 publish_odometry 原样输出 `_state.pos_end` 与该旋转，child 是 aft_mapped。[LIVMapper.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/LIVMapper.cpp:1363) 对雷达点使用 `_state.rot_end * (extR * p_lidar + extT) + _state.pos_end`。set_extrinsic 在 [IMU_Processing.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/IMU_Processing.cpp:96) 明确把 extR/extT 保存为 Lid_rot_to_IMU/Lid_offset_to_IMU。因此已发布 odometry 是 `T_W_B`，记录的 extrinsic 是 `T_B_L`。

本次 release 的实际参数取 [runtime_parameters.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/runtime_parameters.yaml:67)，不是从当前 reference 配置替换：

```text
T_B_L = [ 1  0  0  -0.01100
          0 -1  0  -0.02329
          0  0 -1   0.04412
          0  0  0   1       ]

T_L_B = [ 1  0  0   0.01100
          0 -1  0  -0.02329
          0  0 -1   0.04412
          0  0  0   1       ]
```

[czy_lio_only.launch](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/launch/czy_lio_only.launch:21) 说明 Rx(pi) 来自该数据的轴向/场景检查，平移保留 upstream Mid360 参数。**这是该 run 的 estimator 内部 transform，足够把同 run 的 map/pose/camera 候选闭合；不能把此平移描述成另行实测的整套物理标定。**当前源和上游副本都使用同一 body 链，JSON 保存各自 SHA 与代码行。

首条 LIO quaternion（xyzw）为 `[0.6027908003054441, -0.7976944837440637, -3.762994212998189e-05, 0.01807651115124351]`，已从 registered bag 原消息读取。继续保留这个姿态，不手动翻转重力、不归零成 identity。该数学链和上述 source-supported mapping 可以交给主代理开始一块 pilot；无需等待与内部链无关的绝对物理 body 实测。主代理继续控制其训练与验证声明，本报告不启动训练或改 verified 标志。

runtime img_en=0、imu_en=true。虽然 dmabuf_camera_order=[3,2,4,5]，runtime 只有 logical cam0/source3 和 logical cam1/source4；其 camera model 数值对应 FAST cam0 与 FAST cam2。这里的 logical cam1 并非 FAST cam1。相机列表只是 LIO-only 启动时仍载入的配置，不能据此认为 release 用过四路 raw 图像或 board 内参。

雷达 bridge 在 [frame_conversions.cpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/orbvi_ros_bridge/src/frame_conversions.cpp:1664) 仅解包 point.x/y/z；[preprocess.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/preprocess.cpp:159) 复制输入坐标。[IMU_Processing.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/IMU_Processing.cpp:783) deskew 的结果仍是 end LiDAR 坐标，不能把 deskew 输出误认成 IMU 点。odom 时间是 last_lio_update_time；runtime 明写 lidar header 不是 scan end。重投影原始逐点 scan 需要按采集时间/deskew 处理，由主代理已有实验负责。gravityAlignment 仅左乘世界状态旋转，见 [LIVMapper.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/LIVMapper.cpp:746)，不会新增一个待定 body 外参。

FAST 的 `Rcl/Pcl` 方向为 camera←LiDAR。[vio.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/vio.cpp:94) 先对 `T_B_L` 取逆，再在 151–152 行以 `Rcl*Rli` 和 `Rcl*Pli+Pcl` 形成 camera←IMU，印证方向。对任一已绑定物理相机的 FAST 外参：

```text
T_W_L(t)   = T_W_B(t) T_B_L
T_B_Ci     = T_B_L inverse(T_Ci_L)
T_W_Ci(t)  = T_W_B(t) T_B_L inverse(T_Ci_L)
p_Ci       = T_Ci_L inverse(T_B_L) inverse(T_W_B(t)) p_W

R_B_Ci = R_B_L R_Ci_L^T
t_B_Ci = t_B_L - R_B_L R_Ci_L^T t_Ci_L
```

**4. board 备选链、模型差异及待试验 permutation**

[camera_config.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v6/maps/board-calib-backup/camera_config.yaml:3) 包含四组 raw TripleSphere 九参数和 Twc0..3。局部历史 audit 的 118 行将 Twc 平移乘 0.001，rig-anchor profile 将它解释成 cam0 reference 中的 camera pose；备份 YAML 自身未标注单位。可用备选链按该局部文档解释 `Twc_j=T_boardcam0_boardcamj`，将毫米转换为米。

| model | QR fx/fy/cx/cy | board fx/fy/cx/cy |
| --- | --- | --- |
| cam0 | 617.389773, 622.048384, 644.022091, 552.244888 | 619.972724, 624.499892, 645.778270, 495.918375 |
| cam1 | 616.277046, 620.726726, 671.836453, 537.261021 | 619.451637, 623.831879, 625.438413, 528.606660 |
| cam2 | 619.214661, 623.935440, 681.144474, 529.167400 | 611.847442, 616.058548, 663.210701, 527.468458 |
| cam3 | 612.375680, 616.879215, 650.856118, 506.715096 | 617.002888, 621.267381, 673.990925, 538.356548 |

两套内参不同，不能用相同的 camN 字符串证明物理相机相同。保留以下备选路径供主代理评估：

1. QR model + 同编号 FAST camera←LiDAR，整组 model/extrinsic 按 raw→FAST permutation 绑定。
2. board model + 已确认对应同一物理 raw 相机的 FAST 外参；raw→board permutation 单独记录，不能在替换内参时默默重排物理相机。
3. board rigid rig + 单个 FAST anchor：先确定 board a 与 FAST f 是同一 raw 相机，再按下式传播到其余 board 相机。

```text
T_boardj_L   = inverse(Twc_j) Twc_a T_FASTf_L
T_W_boardj  = T_W_B T_B_L inverse(T_boardj_L)

如果 anchor a=0：Twc_0=I，
T_boardj_L = inverse(Twc_j) T_FASTf_L
```

Twc 本身已经选 boardcam0 为 reference，不能把它当成地图 camera_init 或 IMU body 的 pose。fisheye_8s 下的 left/right PINHOLE 是 320×256 虚拟视图；没有 raw→virtual rectification 时，把 VINS body_T_camN 接到 raw Rcl 不成立。上述 FAST 或 board-anchor 数学链无需 VINS 虚拟模型，也无需先求那个虚拟 body。

permutation 固定以 raw topic 顺序 `[camera_2,camera_3,camera_4,camera_5]` 表示，向量第 k 位是该 topic 使用的模型编号：FAST 优先 `[1,0,2,3]`；board 同序假设 `[1,0,2,3]`；现有未经确认的候选是 `[0,1,2,3]`。**JSON 给出 24 个完整 permutation，FAST p_F 与 board p_B 保持独立，除非像素证据明确绑定。**不能将优先候选声明成已通过实验的事实。

TripleSphere 实现见 [triple_sphere_camera.cpp](/home/chaoszzss/orbvi_proj_ws/src/rpg_vikit/vikit_common/src/triple_sphere_camera.cpp:35)，其 denominator 为 `z3 + alpha/(1-alpha)*d3`。九参数顺序为 fx/fy/cx/cy/xi/lambda/alpha/b/c；JSON 原样存入 QR 和 board 值。FAST 六位小数 Rcl 的 orthogonality Frobenius 误差为 cam0=1.82e-06, cam1=1.77e-06, cam2=8.31e-07, cam3=1.16e-06。主代理的 bounded SVD 应留存原值、文件 SHA、修正量与舍入界；本次未写入任何修正参数。

**5. 主代理可用原 bag 场景解决的剩余项**

- raw SDK ID 与 FAST source ID 的物理对应：上述 `[1,0,2,3]` 已有 QR dev 旁证；后续留出帧可继续检查稳定性和量化标定误差，不把 MI 当成绝对像素误差。
- board camN 与 raw topic 的绑定及采集内参版本：比较 QR/board 各候选和独立 p_B；board rig 传播时记录选定 anchor。
- 图像/scan 的采集时间和地图关联：在运动时间窗检查 header 时差、逐点 deskew 和同 run 点云/pose 的注册一致性。
- 历史 mapper C++/二进制 SHA 缺失是来源归档限制；同 run 点云、trajectory、runtime transform 和原 scan 场景可以验证所需的可观测数学关系，不需要把 body 标成未知而中断候选评估。

上述实验由主代理进行，本次未重复。只读报告的任务已经收尾；剩余 permutation 和实测模型选择有明确候选及可验证链。

**6. 当前文件 SHA256（全文件，不是代码片段）**

| 文件 | SHA256 |
| --- | --- |
| [qr_params.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/qr_params.yaml) | `b5b04c2cb9b7fcdabc02f14f92c25edb59428683e74eea01483a0651e8ff804b` |
| [camera_config.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v6/maps/board-calib-backup/camera_config.yaml) | `1651d01dfd2d2d949888ecf99334d551b0a5675ef5249d5bea83b80e0eb7f55f` |
| [runtime_parameters.yaml](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/runtime_parameters.yaml) | `c777315bc25b4ebd5bb236fa029fdac55d278cbc73b433bcc06dc8411aeb2786` |
| [runtime_source_sha256.json](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/maps/CZY_DATA/height_continuity_release/runtime_source_sha256.json) | `4aabf1f9dbb6fe4b52ce5313dc9e76df9867574736535d56c1c2db5141ecf3cd` |
| [czy_lio_only.launch](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/launch/czy_lio_only.launch) | `6ff0ba74f2d934271ed76f586de534701677fd69f2cb0db04e1e2112fc631fb8` |
| [joint_mapping2.launch](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/launch/joint_mapping2.launch) | `86644eba9aaa0965175173778a0d1a98eff7004e5dd6eaad237fcb8c189359be` |
| [LIVMapper.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/LIVMapper.cpp) | `6215c89cbc2f2aec0ca1c5ddeece89a6199f032ec1430e114beefc05ece0782f` |
| [IMU_Processing.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/IMU_Processing.cpp) | `7af9f9c371437691fea6c6fa4be58691d14448c9badca81c60ad7413be8d469d` |
| [vio.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/vio.cpp) | `fd33b09bb31afc469a1f49cbc244620c034ca79bd2d0d11423eb1f254bc44d62` |
| [preprocess.cpp](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/preprocess.cpp) | `b9c83e979b678a5ba0ffa4438b0ba5ed31e097e9ec773e2ba9e9909d8ef00a38` |
| [triple_sphere_camera.cpp](/home/chaoszzss/orbvi_proj_ws/src/rpg_vikit/vikit_common/src/triple_sphere_camera.cpp) | `149a4b0d3cf9fc075f8570c5d90366f54843fc03375454343adb46db68228b64` |
| [frame_conversions.cpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/orbvi_ros_bridge/src/frame_conversions.cpp) | `eb4135118d2a5373fed2739e67152ffab137d058003a0a0ea2c414566fd92f84` |
| [bridge_node.cpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/orbvi_ros_bridge/src/bridge_node.cpp) | `1c0624b139e5101c3f117b0a779a462aeac72cfdaf11dc6454641211472b5427` |
| [cam0_multi_calib_result.txt](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/cam0_multi_calib_result.txt) | `c11fa353c4cf0db5d830b5c5899a17cab02c4c98f075352518488735aa923931` |
| [cam1_multi_calib_result.txt](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/cam1_multi_calib_result.txt) | `1c81e9541c49aad145f11f44c109256369a74df831b30791f775b881ff0d5dd2` |
| [cam2_multi_calib_result.txt](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/cam2_multi_calib_result.txt) | `697c4f03dc98272d715b0e0def27a1b99ac13e52607e4e946bd6422c8b246ce8` |
| [cam3_multi_calib_result.txt](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final/cam3_multi_calib_result.txt) | `08744cc4addef692e4181144a5bfc3c35b0e8a1df7e2f3f008dc33d0cc67c60f` |
| [audit_board_calibration.py](/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v2/scripts/audit_board_calibration.py) | `6548858916dd886fc18e9cab6822999e18f077e4b592668be7c5dad93f1ef242` |
| [odometry.csv](/home/chaoszzss/dataset/rosbag/3D-Gaussian/data/all_lio_candidate/odometry.csv) | `5a0f49cd5fce90bea63a9cd0f57bd8c8944d727695d0ccf5c9b852fe5dcb9b42` |

其余来源的完整路径、SHA、代码行范围、原文、Git 历史对象、实际 bag connection、24 个 permutation、runtime transform 和候选模型均在相邻 JSON。未计算 25 GB 原 bag 或 8.8 GB registered bag 的全文件 SHA；bag 证据只读其 connection index 与首条相关消息。

QR dev 只读来源 SHA256：`scores.json` = `4c1cd5d39fc2576872178b40bffe14f2975b5449f89a702875303f7dd00aa0f3`；`candidate.yaml` = `64126577a73fc2d1a0ae7628960049ed75e72cbcbe8ce324098ee8d698fbd5f7`。

**7. 进一步 source 查证的结果与实验声明**

当前 SDK build cache [CMakeCache.txt](/home/chaoszzss/orbvi_ros1_ws/build/CMakeCache.txt:354) 的 include/library 都指向 third_party bundle，`ORBVI_HOST_SDK_SOURCE_DIR` 为空；Git 同步提交 ac3edb04 只给出 SDK revision hint `0daaf911`。本次没有恢复这个 SDK 的 metadata 生产端或 board camN→camera_id 加载实现。三个历史 README 对象（ac3edb04、9bdca7ea、270d823e）都保留 front=3/right=2/rear=4/left=5 声明，完整对象 SHA 与行号已入 JSON；这进一步支持参考编号契约，仍不是本 bag capture firmware 的逐通道证明。官方 vendor 页面未成功打开，namespace API 返回 404；检索中的无关 Orbbec SDK 结果没有采用。

SDK [image.hpp](/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge/third_party/orbvi_sdk/include/orbvi_sdk/image.hpp:140) 的 decoded image 结构有 `camera_id`、`calibration_version`、`calibration_hash`，但此 bag 的 `sensor_msgs/CompressedImage` 只有 ROS header、format、data。raw 的 camera ID token 和 source frame 身份可核查；没有采集时 metadata/标定 hash，不能从该 ROS 消息重新构造原物理通道与 board camN 的绑定。交付结论为：**topic↔SDK ID 的 source 链明确，FAST 3/2/4/5 编号声明明确，raw→FAST `[1,0,2,3]` 有开发集旁证；跨采集真实 physical channel↔FAST/board 的强证明仍未取得。**这属于实验验证边界，不新增 body 未知、不阻滞 warmup。

主代理最新报告（本段数值来自用户更新，未独立读取其对应新工件）：board+FAST dev MI≈0.133，QR≈0.1155，交换 raw2/raw3 后 MI≈0.1076；错误映射 rho≈0.276 反而高于正确映射≈0.262。因此 reflectance/luma rho 不能单独证明映射，MI 也不等于绝对像素误差。dataset/model 应显式标为 experimental，verified 保持 false；由主代理安排真实 held-out 训练检验。首 LIO quaternion 和已经闭合的 `T_W_B T_B_L inverse(T_C_L)` 保持原样。
