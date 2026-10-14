#!/usr/bin/python3
"""只读 ROS1 bag 审计；输出严格限制在本项目的三个审计路径。

用系统 Python 3.8，不导入自定义 ROS 消息包，不逐点构造 Python 对象。
所有消息的 header/轻量元数据均扫描；图像解码、稠密点云坐标稀疏抽样。
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
# genpy的动态类加载会调用tempfile；在导入ROS及读取bag之前限制隐式写入。
TEMP = ROOT / 'runtime/tmp'
TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(TEMP)
tempfile.tempdir = str(TEMP)
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import cv2
import numpy as np
import rosbag

REPORT = ROOT / 'reports/dataset_audit.json'
DOC = ROOT / 'docs/DATA_AUDIT.md'
SAMPLES = ROOT / 'reports/samples'
PKG = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_lioslam_fusion_CZY_v6')
OLD_PKG = PKG.with_name('orbvi_omni_livo2_lioslam_fusion_CZY')
V5 = PKG.with_name('orbvi_omni_livo2_lioslam_fusion_CZY_v5')
BRIDGE = Path('/home/chaoszzss/orbvi_ros1_ws/src/orbvi_ros_bridge')
CAMERAS = ['/orbvi/raw/camera_{}/image/compressed'.format(i) for i in (2, 3, 4, 5)]


def string_at(data, offset):
    length = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    return data[offset:offset + length].decode('utf-8', errors='replace'), offset + length


def header_at(data, offset=0):
    seq, sec, nsec = struct.unpack_from('<III', data, offset)
    frame, end = string_at(data, offset + 12)
    return seq, sec * 1000000000 + nsec, frame, end


def stats(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'count': 0}
    q = np.percentile(a, [0, 1, 5, 50, 95, 99, 100])
    return dict(count=int(a.size), min=float(q[0]), p01=float(q[1]), p05=float(q[2]),
                median=float(q[3]), p95=float(q[4]), p99=float(q[5]), max=float(q[6]),
                mean=float(np.mean(a)), std=float(np.std(a)))


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    if isinstance(v, Path):
        return str(v)
    return v


def save(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # 独占报告文件，原位覆盖，不在其他目录创建临时文件。
    REPORT.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def parse_path(data, offset):
    n = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    if n == 0:
        return np.empty(0, np.int64), np.empty((0, 7)), []
    frame_len = struct.unpack_from('<I', data, offset + 12)[0]
    stride = 72 + frame_len
    if offset + n * stride == len(data):
        dt = np.dtype({'names': ['sec', 'nsec', 'frame_len', 'pose'],
                       'formats': ['<u4', '<u4', '<u4', ('<f8', 7)],
                       'offsets': [4, 8, 12, 16 + frame_len], 'itemsize': stride})
        a = np.frombuffer(data, dt, count=n, offset=offset)
        if np.all(a['frame_len'] == frame_len):
            return (a['sec'].astype(np.int64) * 1000000000 + a['nsec'],
                    a['pose'].copy(), [data[offset + 16:offset + 16 + frame_len].decode('utf-8')])
    ts, poses, frames = [], [], set()
    for _ in range(n):
        _, stamp, frame, offset = header_at(data, offset)
        ts.append(stamp)
        frames.add(frame)
        poses.append(struct.unpack_from('<7d', data, offset))
        offset += 56
    if offset != len(data):
        raise ValueError('Path serialized length mismatch')
    return np.asarray(ts, np.int64), np.asarray(poses), sorted(frames)


def trajectory_summary(records):
    if not records:
        return {}
    t = np.asarray([r[0] for r in records], np.int64)
    poses = np.asarray([r[2:9] for r in records], float)
    xyz, q = poses[:, :3], poses[:, 3:]
    dt = np.diff(t) * 1e-9
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    norms = np.linalg.norm(q, axis=1)
    qn = q / np.maximum(norms[:, None], 1e-30)
    angles = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qn[:-1] * qn[1:], axis=1)), 0, 1)))
    speeds = ds / np.maximum(dt, 1e-9)
    idx = np.where((ds > .5) | (speeds > 20) | (angles > 30) | (dt <= 0))[0]
    first_over = {}
    for threshold in (1., 2., 5., 100., 1000.):
        where = np.where(ds > threshold)[0]
        if len(where):
            i = int(where[0] + 1)
            first_over[str(threshold)] = {'index': i, 'header_ns': int(t[i]),
                'relative_s': (int(t[i]) - int(t[0])) * 1e-9, 'step_m': float(ds[i - 1]), 'xyz': xyz[i]}
    return {'count': len(records), 'first_header_ns': int(t[0]), 'last_header_ns': int(t[-1]),
            'duration_s': (int(t[-1]) - int(t[0])) * 1e-9,
            'xyz_min': xyz.min(axis=0), 'xyz_max': xyz.max(axis=0),
            'first_pose_xyz_xyzw': poses[0], 'last_pose_xyz_xyzw': poses[-1],
            'path_length_m': float(np.sum(ds)), 'net_displacement_m': float(np.linalg.norm(xyz[-1] - xyz[0])),
            'z_m': stats(xyz[:, 2]), 'z_net_m': float(xyz[-1, 2] - xyz[0, 2]),
            'step_m': stats(ds), 'speed_m_s_from_position': stats(speeds[dt > 0]),
            'quaternion_norm': stats(norms), 'step_rotation_deg': stats(angles),
            'jump_screen_definition': '相邻位移>0.5m 或位移速度>20m/s 或旋转>30deg 或 dt<=0；只是筛查，不是物理真值',
            'jump_screen_count': len(idx), 'jump_examples': [
                {'index': int(i + 1), 'header_ns': int(t[i + 1]), 'relative_s': (int(t[i + 1]) - int(t[0])) * 1e-9,
                 'dt_s': float(dt[i]), 'step_m': float(ds[i]), 'rotation_deg': float(angles[i]),
                 'before_xyz': xyz[i], 'after_xyz': xyz[i + 1]} for i in idx[:100]],
            'first_step_over_m': first_over,
            'largest_steps': [{'index': int(i + 1), 'relative_s': (int(t[i + 1]) - int(t[0])) * 1e-9,
                               'step_m': float(ds[i]), 'dz_m': float(xyz[i + 1, 2] - xyz[i, 2]),
                               'rotation_deg': float(angles[i])} for i in np.argsort(ds)[-10:][::-1]]}


def image_sample(data, offset, datatype, topic, index, stamp):
    if datatype == 'sensor_msgs/CompressedImage':
        encoding, offset = string_at(data, offset)
        size = struct.unpack_from('<I', data, offset)[0]
        a = cv2.imdecode(np.frombuffer(data, np.uint8, count=size, offset=offset + 4), cv2.IMREAD_UNCHANGED)
    else:
        height, width = struct.unpack_from('<II', data, offset)
        encoding, offset = string_at(data, offset + 8)
        endian = data[offset]
        step, size = struct.unpack_from('<II', data, offset + 1)
        if encoding not in ('bgr8', 'rgb8', 'mono8'):
            return {'index': index, 'error': 'unsupported encoding ' + encoding}
        channels = 1 if encoding == 'mono8' else 3
        a = np.frombuffer(data, np.uint8, count=size, offset=offset + 9).reshape(height, step)
        a = a[:, :width * channels].reshape(height, width, channels)
        if encoding == 'rgb8':
            a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        if channels == 1:
            a = a[:, :, 0]
    if a is None:
        return {'index': index, 'header_ns': stamp, 'decode_ok': False}
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    same = bool(np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 1], a[:, :, 2])) if a.ndim == 3 else True
    label = topic.split('/')[3] if '/raw/' in topic else 'ipm'
    name = '{}_{:05d}_{}.jpg'.format(label, index, stamp)
    if not cv2.imwrite(str(SAMPLES / name), a):
        raise RuntimeError('cannot write sample ' + name)
    result = {'index': index, 'header_ns': stamp, 'encoding_or_format': encoding,
              'decode_ok': True, 'shape_hwc': list(a.shape), 'dtype': str(a.dtype),
              'all_channels_identical': same, 'mean_bgr': np.mean(a, axis=(0, 1)),
              'gray_mean': float(np.mean(gray)), 'gray_std': float(np.std(gray)),
              'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
              'black_pixel_fraction': float(np.mean(gray <= 5)),
              'white_pixel_fraction': float(np.mean(gray >= 250)),
              'sample_path': 'reports/samples/' + name}
    return result


def pc_metadata(data, offset):
    height, width, nf = struct.unpack_from('<III', data, offset)
    offset += 12
    fields = []
    for _ in range(nf):
        name, offset = string_at(data, offset)
        pos, datatype, count = struct.unpack_from('<IBI', data, offset)
        offset += 9
        fields.append({'name': name, 'offset': pos, 'datatype': datatype, 'count': count})
    endian = data[offset]
    point_step, row_step, size = struct.unpack_from('<III', data, offset + 1)
    payload = offset + 13
    dense = bool(data[payload + size])
    return {'height': height, 'width': width, 'fields': fields, 'is_bigendian': bool(endian),
            'point_step': point_step, 'row_step': row_step, 'data_bytes': size, 'is_dense': dense}, payload


def pc_sample(data, meta, offset, index, stamp):
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    endian = '>' if meta['is_bigendian'] else '<'
    dt = np.dtype({'names': [f['name'] for f in meta['fields']],
                   'formats': [endian + types[f['datatype']] for f in meta['fields']],
                   'offsets': [f['offset'] for f in meta['fields']], 'itemsize': meta['point_step']})
    a = np.ndarray((meta['height'], meta['width']), dtype=dt, buffer=data, offset=offset,
                   strides=(meta['row_step'], meta['point_step'])).reshape(-1)
    xyz = np.column_stack([a[k] for k in ('x', 'y', 'z')])
    finite = np.all(np.isfinite(xyz), axis=1)
    p = xyz[finite]
    result = {'index': index, 'header_ns': stamp, 'points': len(a), 'finite_points': int(finite.sum()),
              'zero_xyz_points': int(np.sum(np.all(p == 0, axis=1))),
              'xyz_min': p.min(axis=0) if len(p) else [], 'xyz_max': p.max(axis=0) if len(p) else [],
              'range_m_assuming_SI': stats(np.linalg.norm(p, axis=1)), 'xyz_examples': p[:5]}
    if 'rgb' in a.dtype.names:
        bits = a['rgb'].astype(np.float32).view(np.uint32)
        rgb = np.column_stack([(bits >> 16) & 255, (bits >> 8) & 255, bits & 255])
        result['packed_rgb_examples'] = rgb[:10]
        result['packed_rgb_equal_channels_fraction'] = float(np.mean((rgb[:, 0] == rgb[:, 1]) & (rgb[:, 1] == rgb[:, 2])))
    return result


def sync_summary(topic_data):
    ref = np.asarray(topic_data[CAMERAS[0]]['header_ns'], np.int64)
    result = {'reference_topic': CAMERAS[0], 'four_way': {}, 'nearest_to_reference': {}}
    all_equal = all(topic_data[c]['header_ns'] == topic_data[CAMERAS[0]]['header_ns'] for c in CAMERAS)
    result['four_way']['identical_ordered_header_sequences'] = all_equal
    result['four_way']['message_counts'] = {c: len(topic_data[c]['header_ns']) for c in CAMERAS}
    if len(set(len(topic_data[c]['header_ns']) for c in CAMERAS)) == 1:
        h = np.asarray([topic_data[c]['header_ns'] for c in CAMERAS], np.int64)
        b = np.asarray([topic_data[c]['bag_ns'] for c in CAMERAS], np.int64)
        spread = h.max(axis=0) - h.min(axis=0)
        result['four_way']['same_ordinal_header_spread_ms'] = stats(spread * 1e-6)
        vals, counts = np.unique(spread, return_counts=True)
        result['four_way']['same_ordinal_header_spread_ns_histogram'] = {int(v): int(c) for v, c in zip(vals, counts)}
        result['four_way']['same_ordinal_exact_header_group_count'] = int(np.sum(spread == 0))
        result['four_way']['same_ordinal_groups_within_6ms'] = int(np.sum(spread <= 6000000))
        result['four_way']['same_ordinal_grouping_note'] = '本bag四路数量相同且所有序号对应组最大spread受限；仍需基于header容差配对，不能据此保证曝光硬件同步'
        result['four_way']['bag_receive_spread_ms'] = stats((b.max(axis=0) - b.min(axis=0)) * 1e-6)
    for c in CAMERAS[1:] + ['/orbvi/ipm/image', '/orbvi/vio/odometry', '/orbvi/depth/points', '/orbvi/lidar/custom', '/orbvi/imu']:
        a = np.sort(np.asarray(topic_data[c]['header_ns'], np.int64))
        idx = np.searchsorted(a, ref)
        left = np.clip(idx - 1, 0, len(a) - 1)
        right = np.clip(idx, 0, len(a) - 1)
        nearest = np.where(np.abs(a[left] - ref) <= np.abs(a[right] - ref), left, right)
        delta = a[nearest] - ref
        result['nearest_to_reference'][c] = {'signed_delta_ms': stats(delta * 1e-6),
            'exact_stamp_count': int(np.sum(delta == 0)), 'within_5ms_count': int(np.sum(np.abs(delta) <= 5000000)),
            'within_6ms_count': int(np.sum(np.abs(delta) <= 6000000)),
            'note': '最近邻可能多对一；边缘覆盖差异会放大最大偏差，不是严格同步对'}
    ipm = np.asarray(topic_data['/orbvi/ipm/image']['header_ns'], np.int64)
    result['ipm_stamps_found_in_each_camera'] = {c: int(np.sum(np.isin(ipm, topic_data[c]['header_ns']))) for c in CAMERAS}
    return result


def bounded_inventory():
    roots = [(PKG / 'maps/CZY_DATA', 4), (PKG / 'maps/board-calib-backup', 2),
             (PKG / 'config', 1), (Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final'), 3),
             (Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/Log/result'), 0),
             (Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/Log/PCD'), 1),
             (OLD_PKG / 'maps/CZY_DATA/lio_diagnostics', 1), (OLD_PKG / 'maps/CZY_DATA/mapping2', 0),
             (V5 / 'maps/CZY_DATA/v4_all', 0),
             (V5 / 'maps/CZY_DATA/lifelong_run_02/lifelong/completion', 1),
             (Path('/home/chaoszzss/lioslam_ws/data/maps'), 2), (Path('/home/chaoszzss/lioslam_ws/data/trajectories'), 2)]
    selected, skipped = [], []
    skip = {'.git', 'images', 'image', 'frames', 'labels', 'annotations', 'registered_clouds', 'scans', 'clouds', 'keyframes'}
    dirs = 0
    def visit(root, depth):
        nonlocal dirs
        if not root.is_dir() or dirs >= 250:
            return
        dirs += 1
        entries = []
        with os.scandir(str(root)) as it:
            for e in it:
                entries.append(e)
                if len(entries) > 300:
                    skipped.append({'path': str(root), 'reason': '目录条目超过300，停止，不遍历大型帧目录'})
                    return
        for e in sorted(entries, key=lambda x: x.name):
            if e.is_dir(follow_symlinks=False) and depth > 0:
                if e.name.lower() in skip or any(s in e.name.lower() for s in ('image', 'frame', 'label')):
                    skipped.append({'path': e.path, 'reason': '帧/标签目录排除'})
                else:
                    visit(Path(e.path), depth - 1)
            elif e.is_file(follow_symlinks=False):
                p = Path(e.path)
                if p.suffix.lower() in ('.pcd', '.ply', '.tum', '.yaml', '.yml') or any(w in p.name.lower() for w in ('trajectory', 'traj', 'calib', 'extrinsic', 'camera_config', 'mid360', 'audit', 'fusion_report')) or (p.suffix == '.txt' and p.name.startswith(('czy_', 'fusion_full', 'validated_'))):
                    selected.append({'path': str(p), 'size_bytes': e.stat().st_size})
    for root, depth in roots:
        visit(root, depth)
    for item in selected:
        p = Path(item['path'])
        if p.suffix.lower() == '.pcd':
            lines = []
            with p.open('rb') as f:
                for _ in range(30):
                    line = f.readline(2048).decode('ascii', errors='replace').strip()
                    lines.append(line)
                    if line.startswith('DATA '):
                        break
            item['pcd_header'] = lines
        if p.suffix.lower() == '.tum' or (p.suffix == '.txt' and ('lio' in p.name or 'trajectory' in p.name or 'height' in p.name or p.name.startswith('czy_dense_'))):
            try:
                a = np.loadtxt(str(p), ndmin=2)
                item['tum_summary'] = {'rows': len(a), 'columns': int(a.shape[1]), 'first': a[0].tolist(), 'last': a[-1].tolist(),
                                       'duration_s': float(a[-1, 0] - a[0, 0]), 'xyz_min': a[:, 1:4].min(axis=0).tolist(),
                                       'xyz_max': a[:, 1:4].max(axis=0).tolist()}
                if a.shape[1] == 8:
                    item['tum_summary']['path_length_m'] = float(np.sum(np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1)))
                    item['tum_summary']['z_max_drawdown_m'] = float(np.max(np.maximum.accumulate(a[:, 3]) - a[:, 3]))
                    item['tum_summary']['header_time_matches_czy_all'] = bool(abs(a[0, 0] - 1720267880.9) < 5 and a[-1, 0] <= 1720268707)
            except (ValueError, OSError) as exc:
                item['inspection_error'] = str(exc)
        if p.name == 'czy_joint_fusion_report.json' and p.stat().st_size < 10000000:
            j = json.loads(p.read_text(encoding='utf-8'))
            keys = ['frame_id', 'finalized', 'finalize_mode', 'final_output_stamp', 'latest_input_stamp',
                    'fusion_time_axis', 'sensor_clock_offset_normalized', 'snapshot_source',
                    'cloud_callback_count', 'raw_points_written', 'semantic_callback_count',
                    'ground_surface_supported_tracks', 'ground_surface_unverified_tracks', 'sensor_height',
                    'bev_pixel_scale_m_per_px', 'pose_to_bev_yaw_deg', 'projection_mode']
            item['fusion_report_evidence'] = {k: j[k] for k in keys if k in j}
    return {'roots_and_max_depth': [{'path': str(r), 'max_depth': d} for r, d in roots],
            'visited_directories': dirs, 'limits': '250目录、每目录至多300条，不跟随符号链接；PCD仅读header',
            'skipped': skipped, 'files': selected}


def calibration_evidence():
    board = PKG / 'maps/board-calib-backup/camera_config.yaml'
    result = {'raw_camera_yaml': str(board), 'raw_camera_matrices': {}, 'source_files': [],
              'raw_topic_to_cam_index': {'status': '未查到明确SDK导出索引表，camera_2..5不能直接等于cam0..3',
                                        'candidate_only': '可能为2→0、3→1、4→2、5→3；需索引或视觉独立验证'},
              'verified_complete_camera_to_vio_body_chain': False}
    if board.is_file():
        fs = cv2.FileStorage(str(board), cv2.FILE_STORAGE_READ)
        for i in range(4):
            result['raw_camera_matrices']['cam' + str(i)] = fs.getNode('cam' + str(i)).mat().tolist()
            result['raw_camera_matrices']['Twc' + str(i)] = fs.getNode('Twc' + str(i)).mat().tolist()
        fs.release()
    for rel in ['maps/board-calib-backup/camera_config.yaml', 'maps/board-calib-backup/MID360s_config.json',
                'maps/board-calib-backup/fisheye_8s/fisheye_stereo_imu_config.yaml',
                'docs/CZY_CALIBRATION_OPEN_ITEMS.md', 'docs/CZY_BOARD_CALIBRATION_AUDIT.md', 'config/joint_fusion.yaml',
                'docs/version6/BEV_SOURCE_AUDIT.zh-CN.md', 'docs/version6/DELIVERY.zh-CN.md',
                'docs/version6/LIDAR_GEOMETRY_AUDIT.zh-CN.md', 'docs/version6/evidence/final_export_check.json']:
        p = PKG / rel
        if p.is_file():
            result['source_files'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    result['evidence_notes'] = [
        '板端camera_config包含cam0..3的1×9参数及Twc0..3；已有工程标定审计定义为TripleSphere，Twc平移mm，参考cam0。参数不是OpenCV标准fisheye四系数。',
        'fisheye_8s是320×256 PINHOLE虚拟视图，body_T_cam0..7及td属于该链；不能直接使用同名raw矩阵。',
        'MID360驱动零外参不是LiDAR到视觉IMU单位变换证据。',
        '工程已有标定审计报告raw/virtual缺rectification，标定版本差异；本次重新读取原文件，但未将历史报告中候选外参认证为真值。']
    result['vio_coordinate_evidence'] = {
        'bridge_source': str(BRIDGE / 'orbvi_ros_bridge/src/frame_conversions.cpp'),
        'line_reference': 'MakeVioOdometry约1720行；AppendVioOdometryToPath约1754行',
        'behavior': 'header/child_frame_id来自SDK metadata source_frame_id/source_child_frame_id；姿态、位置、速度和协方差逐字段复制，不施加坐标变换。缺child metadata才默认body。',
        'path_behavior': '当前桥接AppendVioOdometryToPath追加pose；切frame或时间倒退清空；超过max_poses裁前缀；不回写历史pose。当前源码未证明采集时版本完全相同。',
        'physical_pose_reference': '未证实优化VIO pose的物理参考点/光轴与轴向；bag world/world标签不构成T_world_body定义。'}
    vikit = Path('/home/chaoszzss/orbvi_proj_ws/src/rpg_vikit')
    result['triple_sphere_model'] = {
        'parameter_order': ['fx', 'fy', 'cx', 'cy', 'xi', 'lambda', 'alpha', 'b', 'c'],
        'projection_source': str(vikit / 'vikit_common/src/triple_sphere_camera.cpp'),
        'header': str(vikit / 'vikit_common/include/vikit/triple_sphere_camera.h'),
        'loader': str(vikit / 'vikit_ros/include/vikit/camera_loader.h'),
        'sdk_header': str(BRIDGE / 'third_party/orbvi_sdk/include/orbvi_sdk/fisheye_rig.hpp'),
        'forward_equations': ['d1=sqrt(x*x+y*y+z*z)', 'z2=z+xi*d1', 'd2=sqrt(x*x+y*y+z2*z2)',
            'z3=z2+lambda*d2', 'd3=sqrt(x*x+y*y+z3*z3)', 'den=z3+alpha/(1-alpha)*d3',
            'mx=x/den; my=y/den', 'u=fx*mx+b*my+cx; v=c*mx+fy*my+cy'],
        'loader_parameter_spelling': 'cam_lamda（工程ROS参数拼写；模型成员lambda）',
        'scope': '这是当前本机模型实现与SDK接口证据，不证明采集时设备固件/标定版本完全相同；反投影与有效域见源码'}
    for p in [vikit / 'vikit_common/src/triple_sphere_camera.cpp', vikit / 'vikit_common/include/vikit/triple_sphere_camera.h',
              vikit / 'vikit_ros/include/vikit/camera_loader.h', BRIDGE / 'third_party/orbvi_sdk/include/orbvi_sdk/fisheye_rig.hpp',
              BRIDGE / 'orbvi_ros_bridge/src/frame_conversions.cpp']:
        if p.is_file():
            result['source_files'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    fast = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/fast-calib_final')
    result['fast_calib_raw_files'] = []
    for p in sorted(fast.glob('cam*_multi_calib_result.txt')):
        result['fast_calib_raw_files'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'raw_text': p.read_text(encoding='utf-8'), 'direction_and_units_evidence': 'Rcl/Pcl；现有工程标定审计解释camera←LiDAR，平移m；未当作已验证相机→VIO链'})
    result['referenced_historical_paths'] = [{'path': str(PKG / 'maps/CZY_DATA' / rel), 'exists': (PKG / 'maps/CZY_DATA' / rel).exists()}
        for rel in ['dense_surface_v4/full_grounded_v2', 'dense_surface_v4/input', 'board_calibration_audit']]
    result['current_v6_vs_historical_board_files'] = []
    for rel in ['maps/board-calib-backup/camera_config.yaml', 'maps/board-calib-backup/MID360s_config.json',
                'maps/board-calib-backup/fisheye_8s/fisheye_stereo_imu_config.yaml']:
        current, historical = PKG / rel, OLD_PKG / rel
        result['current_v6_vs_historical_board_files'].append({'current_path': str(current), 'historical_path': str(historical),
            'identical_bytes': current.read_bytes() == historical.read_bytes() if current.is_file() and historical.is_file() else None})
    result['v6_calibration_gap_assessment'] = {
        'physical_camera_to_vio_chain_closed': False,
        'evidence': 'v6板端文件与旧CZY相同；version6/BEV_SOURCE_AUDIT明确固定杆臂/yaw/时间偏移拟合未改善保留集，本轮未改变标定。',
        'v6_scope': '新版本主要修复语义地面支持、跨会话身份与geometry_refine；不是raw鱼眼/IMU/LiDAR重新标定。',
        'full_se3_contract': 'online.py要求显式pose_to_bev_full_se3和bev_ground_height_m；默认planar/经验45deg不能用作raw相机物理6DoF外参。'}
    return result


def current_v6_artifacts():
    final_dir = PKG / 'maps/CZY_DATA/run02_geometry_final_v6'
    export = PKG / 'docs/version6/evidence/final_export_check.json'
    final_report = final_dir / 'czy_joint_fusion_report.json'
    result = {'primary_package': str(PKG), 'latest_delivered_map_dir': str(final_dir),
              'final_map_report': str(final_report), 'final_export_check_path': str(export),
              'same_run_all_trajectory_available': False,
              'note': '最终图是V5 prior+run02 corrected_session的多会话合并；不能套用另一轮all轨迹。'}
    if export.is_file():
        result['declared_export_evidence'] = json.loads(export.read_text(encoding='utf-8'))
    if final_report.is_file():
        r = json.loads(final_report.read_text(encoding='utf-8'))
        result['final_report_evidence'] = {k: r[k] for k in ['frame_id', 'finalized', 'raw_points_written', 'total_raw_points',
                                                           'reused_raw_points', 'session_raw_points', 'sensor_height'] if k in r}
        l = r.get('lifelong', {})
        result['final_report_evidence']['inputs'] = l.get('inputs')
        result['final_report_evidence']['alignment'] = l.get('alignment')
    source_bag = V5 / 'maps/CZY_DATA/lifelong_run_02/session/registered_and_semantics.bag'
    correction = V5 / 'maps/CZY_DATA/lifelong_run_02/lifelong/completion/session_correction.json'
    result['matching_session_pose_source'] = {'registered_bag': str(source_bag), 'registered_bag_exists': source_bag.exists(),
                                             'session_correction_json': str(correction), 'correction_exists': correction.exists(),
                                             'status': '可追溯同次run02的原始pose与校正；最终大图含prior，非单一all轨迹'}
    if correction.is_file():
        cr = json.loads(correction.read_text(encoding='utf-8'))
        result['matching_session_pose_source']['correction_keys'] = list(cr)
        result['matching_session_pose_source']['correction_metadata'] = {k: cr[k] for k in
            ['transform_convention', 'transform_model', 'accepted', 'source_trajectory_sha256', 'max_stamp_extrapolation_sec'] if k in cr}
        result['matching_session_pose_source']['correction_metadata'].update(reference_pose_count=len(cr.get('reference_poses', [])),
                                                                            correction_knot_count=len(cr.get('knots', [])))
    if source_bag.is_file():
        with rosbag.Bag(str(source_bag), 'r') as b:
            infos = b.get_type_and_topic_info().topics
            result['matching_session_pose_source']['topics'] = {k: {'type': v.msg_type, 'count': v.message_count} for k, v in infos.items()}
            for topic, info in infos.items():
                if info.msg_type != 'nav_msgs/Odometry':
                    continue
                records, frames, children = [], set(), set()
                for _, raw, t in b.read_messages(topics=[topic], raw=True):
                    data = raw[1]
                    _, stamp, frame, offset = header_at(data)
                    child, offset = string_at(data, offset)
                    values = struct.unpack_from('<7d', data, offset)
                    records.append([stamp, t.to_nsec()] + list(values))
                    frames.add(frame)
                    children.add(child)
                result['matching_session_pose_source']['odom_topic'] = topic
                result['matching_session_pose_source']['frame_ids'] = sorted(frames)
                result['matching_session_pose_source']['child_frame_ids'] = sorted(children)
                result['matching_session_pose_source']['raw_session_trajectory_summary'] = trajectory_summary(records)
    return result


def delivery_handoff(r):
    """仅汇总已有审计，不再读取bag、PCD或标定源文件。"""
    v6 = r['current_v6']
    pose_source = v6['matching_session_pose_source']
    summary = pose_source['raw_session_trajectory_summary']
    return {
        'status': '审计已结稿；剩余项为证据缺口，不是待执行扫描',
        'map': {
            'directory': v6['latest_delivered_map_dir'],
            'raw_filename': 'czy_joint_raw_map.pcd', 'raw_points': 30456613,
            'prior_points': 19196862, 'corrected_session_points': 11259751,
            'frame_id': 'camera_init', 'new_online_run': False,
            'source_run': 'V5 lifelong_run_02', 'inputs': v6['final_report_evidence']['inputs'],
            'input_hash_scope': '这些SHA256来自v6来源报告，本轮未重新整文件哈希大型PCD',
            'same_run_single_all_map_and_camera_trajectory_verified': False},
        'matching_session_pose': {
            'registered_bag': pose_source['registered_bag'], 'topic': pose_source['odom_topic'],
            'header_frame_ids': pose_source['frame_ids'], 'child_frame_ids': pose_source['child_frame_ids'],
            'count': summary['count'], 'first_header_ns': summary['first_header_ns'],
            'last_header_ns': summary['last_header_ns'], 'duration_s': summary['duration_s'],
            'correction_path': pose_source['session_correction_json'],
            'correction_metadata': pose_source['correction_metadata'],
            'transform_relation': 'T_prior_sensor(t)=T_prior_session(t) @ T_session_sensor(t)；必须按session_correction定义应用157个时变knots，不能只取一矩阵',
            'scope': '仅对应最终图的新run02会话；prior的all同次逐帧pose链未认证'},
        'original_all_synchronization': {
            'time_axis': 'header整数ns；bag整数ns只保留来源/接收顺序',
            'four_way_group_count': 16234, 'exact_header_groups': 14159,
            'groups_with_5003000ns_spread': 2075, 'suggested_group_tolerance_ns': 6000000,
            'reference_for_ipm': '/orbvi/raw/camera_3/image/compressed',
            'ipm_count': 3556, 'ipm_exact_camera3_count': r['synchronization']['ipm_stamps_found_in_each_camera'][CAMERAS[1]],
            'pose_warning': '原始VIO的world/world物理坐标定义未证实；458.294363s开始大跳变；不得跨约12.5s断档插值',
            'local_reorder_policy_reported_by_main_agent': '只对<=1ms局部header回退排序，raw CSV保持原始；真正clock reset拒绝。odom row4129回退23127ns，pose差约1.221mm，不据此将原header判为时钟重置。',
            'imu_nonmonotonic_scope': 'IMU原流另有95057036ns回退，超出1ms局部重排例外；未认证原因，不能自动修正或宣称发生硬件clock reset',
            'one_timestamp_is_not_one_pose_definition': 'Log/czy_lio_lifelong_v5.txt与v6.txt时间范围相同但XYZ不同；同stamp不证明相同初始化或重建来源'},
        'rgb_provenance': {
            'final_map_has_rgb_storage_field': True,
            'final_map_photometric_rgb_source_verified': False,
            'scope': 'PCD中的rgb字段/彩色显示不证明来自相机；语义类别RGB不能作为真实辐射颜色。当前报告未核验最终raw RGB的逐点来源或train/val隔离。',
            'delivery_contract': '--map-rgb-verified需有独立来源声明；intensitymap从train图像着色时保留图像/pose/标定与分割来源，不能用此标志补齐未确认的几何链'},
        'remaining_execution': [],
        'evidence_blockers': [
            'raw camera_2..5到板端cam0..3显式映射',
            'raw→virtual rectification与采集实际标定版本',
            'camera↔VIO物理参考点的完整外参及世界轴向',
            '最终图prior/v4_all同次注册证据或相机轨迹初始化对应',
            '最终raw地图RGB的相机/强度来源与split声明']}


def finalize_timing_rates(r):
    """从已审计的首末header和实际count计算记录平均速率，不用rosbag.info.frequency。"""
    for topic, d in r['topics'].items():
        duration = (d['last_header_ns'] - d['first_header_ns']) * 1e-9
        count = d['count']
        median_ms = d['header_interval_ms']['median']
        d['recorded_rate_hz'] = {
            'header_duration_s': duration,
            'count_over_header_duration_hz': count / duration if duration > 0 else None,
            'mean_interval_rate_hz': (count - 1) / duration if duration > 0 and count > 1 else None,
            'median_interval_implied_rate_hz': 1000. / median_ms if median_ms > 0 else None,
            'definition': '前者N/(last_header-first_header)，后者(N-1)/duration；median间隔反推速率另列。只表示实际记录数据，不用bag frequency推断设备标称采样率。'}
    r['timing_rate_note'] = 'rosbag info.frequency是其索引/接收间隔估计，不用于本报告平均采样率；实际记录平均以全部消息count和sensor header duration计算。'
    return r


def mark_critical_vio_failure(r):
    v = r['vio']['summary']
    r['critical_vio_failure'] = {
        'full_vio_initialization_allowed': False,
        'independent_confirmation': '原主bag ROS正常反序列化与本审计wire解析均核对过4388/4391/4392/4395/5025/8054等异常pose；不是仅CSV或解析格式推测。',
        'last_xyz': v['last_pose_xyz_xyzw'][:3], 'first_step_over_2m': v['first_step_over_m']['2.0'],
        'first_step_over_100m': v['first_step_over_m']['100.0'],
        'full_path_length_m_diagnostic_only': v['path_length_m'],
        'block_plan_3525': '主代理报告产生3525块；这是发散轨迹的无效诊断产物，不是可交付全程分块/初始化证据。',
        'required_action': '全程重建使用另行核验的LIO/registered SE3；相机rig外参和地图初始化缺证据时保持阻塞，禁止fake identity。'}
    r['delivery_handoff']['original_all_synchronization']['full_vio_initialization_allowed'] = False
    return r


def probe_all_lio_evidence():
    """定向只读索引、5个odom/cloud header和一个8246行TUM；不扫描bag正文。"""
    base = PKG.with_name('orbvi_omni_livo2_lioslam_fusion_CZY_v2') / 'maps/CZY_DATA/height_continuity_release'
    path = base / 'registered_and_semantics.bag'
    tum = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/Log/result/czy_height_continuity_release_20260925.txt')
    result = {'registered_bag': str(path), 'exists': path.is_file(), 'same_time_interval_as_original_all': False,
              'same_initialization_as_v6_final_map_verified': False, 'camera_rig_extrinsic_verified': False,
              'method': 'bag connection/index metadata + 5定点odom/cloud header样本 + 小型完整TUM；没有全量扫描registered bag',
              'evidence_scope': '已记录完整LIO的xyz+xyzw SE3；不认证全局图优化、测绘精度或最终v6 prior坐标同一',
              'rig_policy': '缺rig外参时禁止把identity当作标定', 'samples': []}
    if not path.is_file():
        return result
    result['bag_size_bytes'] = path.stat().st_size
    with rosbag.Bag(str(path), 'r') as b:
        infos = b.get_type_and_topic_info().topics
        result['topics'] = {k: {'type': v.msg_type, 'count': v.message_count} for k, v in infos.items()}
        od_connections = [c for c in b._get_connections() if c.topic == '/aft_mapped_to_init']
        cloud_connections = [c for c in b._get_connections() if c.topic == '/cloud_registered']
        entries = list(b._get_entries(od_connections))
        cloud_entries = list(b._get_entries(cloud_connections))
        for index in [0, len(entries) // 4, len(entries) // 2, 3 * len(entries) // 4, len(entries) - 1]:
            e = entries[index]
            for _, m, t in b.read_messages(topics=['/aft_mapped_to_init'], start_time=e.time, end_time=e.time):
                p, q = m.pose.pose.position, m.pose.pose.orientation
                sample = {'index': index, 'header_ns': m.header.stamp.to_nsec(), 'bag_ns': t.to_nsec(),
                          'frame_id': m.header.frame_id, 'child_frame_id': m.child_frame_id,
                          'xyz_xyzw': [p.x, p.y, p.z, q.x, q.y, q.z, q.w]}
                result['samples'].append(sample)
                break
            if index < len(cloud_entries):
                ce = cloud_entries[index]
                for _, raw, t in b.read_messages(topics=['/cloud_registered'], start_time=ce.time, end_time=ce.time, raw=True):
                    _, stamp, frame, offset = header_at(raw[1])
                    meta, _ = pc_metadata(raw[1], offset)
                    sample['same_ordinal_cloud_header_ns'] = stamp
                    sample['cloud_frame_id'] = frame
                    sample['odom_cloud_exact_header_match'] = stamp == sample['header_ns']
                    sample['cloud_metadata'] = meta
                    break
    result['first_header_ns'] = result['samples'][0]['header_ns']
    result['last_header_ns'] = result['samples'][-1]['header_ns']
    result['same_time_interval_as_original_all'] = 1720267880000000000 <= result['first_header_ns'] <= 1720267885000000000 and 1720268704000000000 <= result['last_header_ns'] <= 1720268707000000000
    result['sampled_pose_finite'] = bool(np.all(np.isfinite([s['xyz_xyzw'] for s in result['samples']])))
    result['sampled_odom_cloud_exact_header_match_count'] = sum(s.get('odom_cloud_exact_header_match', False) for s in result['samples'])
    if tum.is_file():
        a = np.loadtxt(str(tum), ndmin=2)
        comparisons = []
        for s in result['samples']:
            k = int(np.argmin(np.abs(a[:, 0] - s['header_ns'] * 1e-9)))
            op = np.asarray(s['xyz_xyzw'])
            xyz_err = float(np.linalg.norm(a[k, 1:4] - op[:3]))
            qa, qb = a[k, 4:8], op[3:]
            cosine = abs(float(np.dot(qa, qb))) / max(float(np.linalg.norm(qa) * np.linalg.norm(qb)), 1e-30)
            comparisons.append({'sample_index': s['index'], 'tum_row': k, 'position_error_m': xyz_err,
                                'sign_invariant_rotation_error_deg': float(np.degrees(2 * np.arccos(np.clip(cosine, 0, 1))))})
        records = [[int(round(row[0] * 1e9)), 0] + list(row[1:8]) for row in a]
        result['tum'] = {'path': str(tum), 'sha256': hashlib.sha256(tum.read_bytes()).hexdigest(), 'rows': len(a),
            'columns': int(a.shape[1]), 'all_finite': bool(np.all(np.isfinite(a))),
            'stamp_precision': '文本秒的6位小数；此处生成ns为舍入值，精确首末ns以bag样本为准',
            'quaternion_norm': stats(np.linalg.norm(a[:, 4:8], axis=1)),
            'full_trajectory_summary': trajectory_summary(records), 'sample_bag_comparisons': comparisons,
            'full_tum_is_not_full_bag_value_comparison': True}
    declaration = PKG / 'docs/version6/evidence/lidar/summary.json'
    if declaration.is_file():
        d = json.loads(declaration.read_text(encoding='utf-8'))
        result['v6_provenance_declaration'] = {k: d[k] for k in ['all_floor5_source_bag', 'all_floor5_source_same_initialization_as_v4_all', 'all_floor5_source_provenance_note'] if k in d}
    return result


def audit(bag_path, samples_per_topic):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema_version': 1, 'status': 'scanning', 'generated_utc': datetime.datetime.utcnow().isoformat() + 'Z',
              'bag_path': str(bag_path), 'bag_size_bytes': bag_path.stat().st_size,
              'method': {'complete_scan': '原始序列化消息；全部header、所有IMU/odom、所有Path历史重叠比较、所有Custom点offset/tag/line；不依赖自定义消息安装',
                         'sampled': '各图像topic等序号稀疏解码；PointCloud2和Custom坐标等序号抽样；未逐帧视觉质检',
                         'timestamp_precision': '原始整数ns；统计单位换算后浮点显示'}, 'connections': []}
    td = {}
    odometry, imu = [], []
    path_info = {'message_count': 0, 'lengths': [], 'history_position_change_messages': 0, 'history_rotation_change_messages': 0,
                 'max_history_position_change_m': 0., 'max_history_rotation_change_deg': 0., 'overlap_pose_comparisons': 0,
                 'examples': [], 'nested_frame_ids': set(), 'nonmonotonic_nested_stamp_messages': 0, 'odom_comparisons': 0,
                 'max_path_odom_position_error_m': 0., 'count_decreases': 0, 'first_stamp_advances': 0,
                 'max_path_odom_rotation_error_deg': 0., 'path_odom_mismatch_over_1e_6m_count': 0}
    prev_ts, prev_pose, final_ts, final_pose = None, None, None, None
    odom_seen = {}
    custom = {'messages': 0, 'total_points': 0, 'point_num_mismatch': 0, 'timebase_minus_header_ns': [],
              'offset_first_ns': [], 'offset_last_ns': [], 'offset_max_ns': [], 'offset_decrease_messages': 0,
              'tag_histogram': collections.Counter(), 'line_histogram': collections.Counter(), 'lidar_ids': collections.Counter(),
              'coordinate_samples': [], 'zero_xyz_points': 0, 'nonfinite_xyz_points': 0,
              'zero_xyz_tag_histogram': collections.Counter()}
    custom_dt = np.dtype([('offset', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                          ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1')])
    with rosbag.Bag(str(bag_path), 'r') as bag:
        report['bag_time'] = {'start_s': bag.get_start_time(), 'end_s': bag.get_end_time(),
                              'duration_s': bag.get_end_time() - bag.get_start_time(), 'total_messages': bag.get_message_count()}
        infos = bag.get_type_and_topic_info().topics
        for c in bag._get_connections():
            report['connections'].append({'id': c.id, 'topic': c.topic, 'datatype': c.datatype, 'md5sum': c.md5sum,
                                          'connection_header': plain(c.header), 'message_definition': c.msg_def})
        for topic, info in infos.items():
            td[topic] = {'type': info.msg_type, 'expected_count': info.message_count, 'header_ns': [], 'bag_ns': [],
                         'seq': [], 'frame_ids': collections.Counter(), 'samples': [], 'metadata_variants': collections.Counter()}
        wanted = {topic: set(int(x) for x in np.linspace(0, d['expected_count'] - 1, samples_per_topic)) for topic, d in td.items()}
        report['topic_index'] = {topic: {'type': d['type'], 'expected_count': d['expected_count']} for topic, d in td.items()}
        save(report)
        for total, (topic, raw, record_stamp) in enumerate(bag.read_messages(raw=True), 1):
            datatype, data = raw[0], raw[1]
            seq, stamp, frame, offset = header_at(data)
            d = td[topic]
            index = len(d['header_ns'])
            d['header_ns'].append(stamp)
            d['bag_ns'].append(record_stamp.to_nsec())
            d['seq'].append(seq)
            d['frame_ids'][frame] += 1
            if datatype in ('sensor_msgs/CompressedImage', 'sensor_msgs/Image'):
                if datatype == 'sensor_msgs/CompressedImage':
                    fmt, end = string_at(data, offset)
                    size = struct.unpack_from('<I', data, end)[0]
                    d['metadata_variants'][fmt] += 1
                else:
                    h, w = struct.unpack_from('<II', data, offset)
                    enc, end = string_at(data, offset + 8)
                    step, size = struct.unpack_from('<II', data, end + 1)
                    d['metadata_variants'][json.dumps({'height': h, 'width': w, 'encoding': enc, 'step': step, 'data_bytes': size})] += 1
                if index in wanted[topic]:
                    s = image_sample(data, offset, datatype, topic, index, stamp)
                    d['samples'].append(s)
                    print('IMAGE {} index={} header={} shape={}'.format(topic, index, stamp, s.get('shape_hwc')), flush=True)
            elif datatype == 'sensor_msgs/Imu':
                v = struct.unpack_from('<37d', data, offset)
                imu.append([stamp] + list(v))
            elif datatype == 'nav_msgs/Odometry':
                child, offset = string_at(data, offset)
                v = struct.unpack_from('<85d', data, offset)
                d['metadata_variants'][child] += 1
                r = [stamp, record_stamp.to_nsec()] + list(v[:7]) + list(v[43:49])
                odometry.append(r)
                odom_seen[stamp] = np.asarray(v[:7])
                d.setdefault('pose_covariance_all_zero_count', 0)
                d['pose_covariance_all_zero_count'] += int(not any(v[7:43]))
                d.setdefault('twist_covariance_all_zero_count', 0)
                d['twist_covariance_all_zero_count'] += int(not any(v[49:85]))
                d.setdefault('pose_covariance_diagonal', []).append([v[7 + k * 7] for k in range(6)])
            elif datatype == 'nav_msgs/Path':
                ts, poses, frames = parse_path(data, offset)
                path_info['message_count'] += 1
                path_info['lengths'].append(len(ts))
                path_info['nested_frame_ids'].update(frames)
                path_info['nonmonotonic_nested_stamp_messages'] += int(np.any(np.diff(ts) <= 0))
                if prev_ts is not None and len(ts) and len(prev_ts):
                    common, ia, ib = np.intersect1d(prev_ts, ts, assume_unique=False, return_indices=True)
                    path_info['overlap_pose_comparisons'] += len(common)
                    err = np.linalg.norm(prev_pose[ia, :3] - poses[ib, :3], axis=1)
                    qa, qb = prev_pose[ia, 3:], poses[ib, 3:]
                    cos = np.abs(np.sum(qa * qb, axis=1)) / np.maximum(np.linalg.norm(qa, axis=1) * np.linalg.norm(qb, axis=1), 1e-30)
                    ang = np.degrees(2 * np.arccos(np.clip(cos, 0, 1)))
                    maximum = float(np.max(err)) if len(err) else 0.
                    ma = float(np.max(ang)) if len(ang) else 0.
                    path_info['max_history_position_change_m'] = max(path_info['max_history_position_change_m'], maximum)
                    path_info['max_history_rotation_change_deg'] = max(path_info['max_history_rotation_change_deg'], ma)
                    path_info['history_position_change_messages'] += int(maximum > 1e-6)
                    path_info['history_rotation_change_messages'] += int(ma > 1e-4)
                    path_info['count_decreases'] += int(len(ts) < len(prev_ts))
                    path_info['first_stamp_advances'] += int(ts[0] > prev_ts[0])
                    if maximum > 1e-6 and len(path_info['examples']) < 20:
                        k = int(np.argmax(err))
                        path_info['examples'].append({'path_header_ns': stamp, 'changed_pose_header_ns': int(common[k]), 'change_m': maximum,
                                                      'previous_xyz': prev_pose[ia[k], :3], 'current_xyz': poses[ib[k], :3]})
                # 当前Path尾部与相同stamp的odom比较；所有历史重叠已在上方全量比较。
                if len(ts) and int(ts[-1]) in odom_seen:
                    op = odom_seen[int(ts[-1])]
                    e = float(np.linalg.norm(op[:3] - poses[-1, :3]))
                    oq, pq = op[3:], poses[-1, 3:]
                    cos = abs(float(np.dot(oq, pq))) / max(float(np.linalg.norm(oq) * np.linalg.norm(pq)), 1e-30)
                    ae = float(np.degrees(2 * np.arccos(np.clip(cos, 0, 1))))
                    path_info['odom_comparisons'] += 1
                    path_info['max_path_odom_position_error_m'] = max(path_info['max_path_odom_position_error_m'], e)
                    path_info['max_path_odom_rotation_error_deg'] = max(path_info['max_path_odom_rotation_error_deg'], ae)
                    path_info['path_odom_mismatch_over_1e_6m_count'] += int(e > 1e-6)
                if prev_ts is None:
                    path_info['first_path_first_header_ns'] = int(ts[0]) if len(ts) else None
                    path_info['first_path_last_header_ns'] = int(ts[-1]) if len(ts) else None
                prev_ts, prev_pose, final_ts, final_pose = ts, poses, ts, poses
            elif datatype == 'sensor_msgs/PointCloud2':
                meta, payload = pc_metadata(data, offset)
                d['metadata_variants'][json.dumps({k: v for k, v in meta.items() if k not in ('width', 'row_step', 'data_bytes')}, sort_keys=True)] += 1
                d.setdefault('point_counts', []).append(meta['width'] * meta['height'])
                d.setdefault('invalid_data_length_count', 0)
                d['invalid_data_length_count'] += int(meta['data_bytes'] != meta['row_step'] * meta['height'])
                if index in wanted[topic]:
                    d['samples'].append(dict(pc_sample(data, meta, payload, index, stamp), metadata=meta))
            elif datatype == 'livox_ros_driver2/CustomMsg':
                timebase, pn, lidar_id = struct.unpack_from('<QIB', data, offset)
                n = struct.unpack_from('<I', data, offset + 16)[0]
                a = np.frombuffer(data, custom_dt, count=n, offset=offset + 20)
                custom['messages'] += 1
                custom['total_points'] += n
                custom['point_num_mismatch'] += int(pn != n)
                custom['lidar_ids'][lidar_id] += 1
                custom['timebase_minus_header_ns'].append(timebase - stamp)
                if n:
                    custom['offset_first_ns'].append(int(a['offset'][0]))
                    custom['offset_last_ns'].append(int(a['offset'][-1]))
                    custom['offset_max_ns'].append(int(a['offset'].max()))
                    custom['offset_decrease_messages'] += int(np.any(np.diff(a['offset'].astype(np.int64)) < 0))
                    for key, field in [('tag_histogram', 'tag'), ('line_histogram', 'line')]:
                        vals, counts = np.unique(a[field], return_counts=True)
                        custom[key].update({int(v): int(c) for v, c in zip(vals, counts)})
                    zeros = (a['x'] == 0) & (a['y'] == 0) & (a['z'] == 0)
                    custom['zero_xyz_points'] += int(np.sum(zeros))
                    custom['nonfinite_xyz_points'] += int(np.sum(~(np.isfinite(a['x']) & np.isfinite(a['y']) & np.isfinite(a['z']))))
                    vals, counts = np.unique(a['tag'][zeros], return_counts=True)
                    custom['zero_xyz_tag_histogram'].update({int(v): int(c) for v, c in zip(vals, counts)})
                    if index in wanted[topic]:
                        xyz = np.column_stack([a[k] for k in ('x', 'y', 'z')])
                        p = xyz[np.all(np.isfinite(xyz), axis=1)]
                        custom['coordinate_samples'].append({'index': index, 'header_ns': stamp, 'timebase_ns': timebase,
                           'points': n, 'finite_points': len(p), 'xyz_min': p.min(axis=0), 'xyz_max': p.max(axis=0),
                           'range_m_assuming_SI': stats(np.linalg.norm(p, axis=1)), 'xyz_examples': p[:5],
                           'offset_first_ns': int(a['offset'][0]), 'offset_last_ns': int(a['offset'][-1]),
                           'reflectivity': stats(a['reflectivity'])})
            if total % 10000 == 0:
                print('SCAN {}/{} elapsed={:.1f}s topic={}'.format(total, report['bag_time']['total_messages'], time.monotonic() - started, topic), flush=True)
    report['topics'] = {}
    for topic, d in td.items():
        h, b, s = np.asarray(d['header_ns'], np.int64), np.asarray(d['bag_ns'], np.int64), np.asarray(d['seq'], np.int64)
        dh = np.diff(h)
        r = {k: v for k, v in d.items() if k not in ('header_ns', 'bag_ns', 'seq', 'point_counts', 'pose_covariance_diagonal')}
        r.update(count=len(h), count_matches_index=len(h) == d['expected_count'], first_header_ns=int(h[0]), last_header_ns=int(h[-1]),
                 first_bag_ns=int(b[0]), last_bag_ns=int(b[-1]), header_duration_s=(int(h[-1]) - int(h[0])) * 1e-9,
                 header_interval_ms=stats(dh * 1e-6), duplicate_adjacent_header_count=int(np.sum(dh == 0)),
                 backwards_header_count=int(np.sum(dh < 0)), seq_delta=stats(np.diff(s)),
                 header_seq_first=int(s[0]), header_seq_last=int(s[-1]), bag_minus_header_s=stats((b - h) * 1e-9),
                 bag_minus_header_first_s=(int(b[0]) - int(h[0])) * 1e-9,
                 bag_minus_header_last_s=(int(b[-1]) - int(h[-1])) * 1e-9)
        gap_threshold = max(float(np.median(dh[dh > 0])) * 5, 250000000.)
        r['gap_screen_threshold_s'] = gap_threshold * 1e-9
        gi = np.where(dh > gap_threshold)[0]
        bi = np.where(dh <= 0)[0]
        r['gap_screen_count'] = len(gi)
        r['gap_examples'] = [{'index': int(i + 1), 'before_header_ns': int(h[i]), 'after_header_ns': int(h[i + 1]),
                             'relative_before_s': (int(h[i]) - int(h[0])) * 1e-9, 'gap_s': float(dh[i]) * 1e-9} for i in gi[:30]]
        r['nonincreasing_header_examples'] = [{'index': int(i + 1), 'before_header_ns': int(h[i]), 'after_header_ns': int(h[i + 1]),
                                              'delta_ns': int(dh[i])} for i in bi[:30]]
        x = (h - h[0]) * 1e-9
        y = (b - h) * 1e-9
        slope = float(np.dot(x - x.mean(), y - y.mean()) / max(np.sum((x - x.mean()) ** 2), 1e-30))
        residual = y - (y.mean() + slope * (x - x.mean()))
        r['record_offset_linear_fit'] = {'slope_s_per_sensor_s': slope, 'equivalent_drift_ppm': slope * 1e6,
                                          'residual_ms': stats(residual * 1000), 'interpretation': '接收/回放时基拟合，不是传感器间时钟标定'}
        if 'point_counts' in d:
            r['point_counts'] = stats(d['point_counts'])
        if 'pose_covariance_diagonal' in d:
            ca = np.asarray(d['pose_covariance_diagonal'])
            r['pose_covariance_diagonal_stats'] = [stats(ca[:, k]) for k in range(6)]
        report['topics'][topic] = r
    report['synchronization'] = sync_summary(td)
    report['vio'] = {'summary': trajectory_summary(odometry), 'trajectory_columns': ['header_ns', 'bag_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
                     'trajectory': odometry, 'frame_semantics': '保留发布值；未假定body、ENU、NED、车辆或相机坐标'}
    split = report['vio']['summary']['first_step_over_m'].get('2.0')
    if split:
        report['vio']['before_first_step_over_2m'] = trajectory_summary(odometry[:split['index']])
        report['vio']['before_first_step_over_2m']['interpretation'] = '诊断截段，不等于所有前段pose通过几何精度验收；仍有时间断档和一次header回退'
    path_info['length_stats'] = stats(path_info.pop('lengths'))
    path_info['nested_frame_ids'] = sorted(path_info['nested_frame_ids'])
    path_info['final_path_columns'] = ['header_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    path_info['final_path'] = [[int(t)] + list(p) for t, p in zip(final_ts, final_pose)] if final_ts is not None else []
    path_info['final_path_oldest_precedes_first_recorded_odom'] = bool(len(final_ts) and final_ts[0] < odometry[0][0]) if final_ts is not None else False
    report['vio_path'] = path_info
    ia = np.asarray(imu, float)
    # 列0为stamp，列1..37对应ROS IMU wire layout。
    acc, gyro, q = ia[:, 26:29], ia[:, 14:17], ia[:, 1:5]
    report['imu'] = {'count': len(imu), 'acceleration_xyz': [stats(acc[:, k]) for k in range(3)],
                      'acceleration_norm': stats(np.linalg.norm(acc, axis=1)), 'angular_velocity_xyz': [stats(gyro[:, k]) for k in range(3)],
                      'angular_velocity_norm': stats(np.linalg.norm(gyro, axis=1)),
                      'orientation_quaternion_norm': stats(np.linalg.norm(q, axis=1)),
                      'orientation_all_zero_count': int(np.sum(np.all(q == 0, axis=1))),
                      'orientation_covariance_all_zero_count': int(np.sum(np.all(ia[:, 5:14] == 0, axis=1))),
                      'angular_velocity_covariance_all_zero_count': int(np.sum(np.all(ia[:, 17:26] == 0, axis=1))),
                      'linear_acceleration_covariance_all_zero_count': int(np.sum(np.all(ia[:, 29:38] == 0, axis=1))),
                      'unit_evidence': 'ROS message_definition要求m/s²和rad/s，桥接逐字段复制；加速度模长约重力可排除以g为数值的常见情况，但不能单凭统计证明陀螺轴向/量纲正确。',
                      'first_sample_wire_37d': imu[0][1:]}
    for key in ('timebase_minus_header_ns', 'offset_first_ns', 'offset_last_ns', 'offset_max_ns'):
        custom[key] = stats(custom[key])
    custom['units_and_semantics'] = 'bag定义无单位注释；timebase与header差仅ns级，offset每帧近1e8，与约0.1s scan间隔一致；按ns解释有数据证据。XYZ米及tag位含义须结合本地驱动/协议；不把tag=0当有效性保证。'
    custom['local_preprocess_evidence'] = {'path': '/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2_avp/src/preprocess.cpp',
        'lines': '154–175', 'offset_unit': 'offset_time / 1e6写入curvature，源码注释unit ms，支持offset_time ns',
        'xyz_handling': 'XYZ直接复制并与以m设置的blind比较，没有乘1000或0.001；当前工程按m使用',
        'tag_filter': '(tag & 0x30)为0x00或0x10，line<N_SCANS；这是下游现有过滤，不等同完整tag有效性或MID360协议认证'}
    report['livox_custom'] = custom
    report['local_artifacts'] = bounded_inventory()
    report['calibration'] = calibration_evidence()
    report['current_v6'] = current_v6_artifacts()
    finalize_timing_rates(report)
    report['delivery_handoff'] = delivery_handoff(report)
    mark_critical_vio_failure(report)
    report['limitations'] = [
        '没有/tf、/tf_static、CameraInfo连接；frame_id不证明外参。',
        'VIO位置/姿态物理参考点及轴向未证实，world/world不能自动解释为T_world_body。',
        'raw topic camera_2..5到camera_config cam0..3映射未获显式证据；TripleSphere不能直接用cv2.fisheye。',
        'raw→virtual rectification矩阵、采集实际标定版本、相机→VIO物理参考系的完整闭合链尚缺。',
        'VIO在458.3s附近发散；全程不能直接作为优化初始轨迹。',
        'v6最新最终图是多会话几何重建；匹配raw all图与同次初始化的完整相机轨迹未认证。',
        '稀疏图像和点云坐标抽样不保证全量每帧解码质量、几何准确性或地图最终质量。',
        '已有地图/轨迹仅定位与header检查，未认证为地面真值或与本bag姿态同一世界系。']
    report['status'] = 'complete'
    report['elapsed_s'] = time.monotonic() - started
    save(report)
    render(report)
    contact_sheet(report)
    print('COMPLETE report={} doc={} elapsed={:.1f}s'.format(REPORT, DOC, report['elapsed_s']), flush=True)
    return report


def contact_sheet(report):
    rows = []
    for c in CAMERAS:
        tiles = []
        for s in report['topics'][c]['samples']:
            if not s.get('decode_ok'):
                continue
            a = cv2.imread(str(ROOT / s['sample_path']))
            a = cv2.resize(a, (320, 270))
            tile = np.zeros((302, 320, 3), np.uint8)
            tile[32:] = a
            cv2.putText(tile, '{} idx{}'.format(c.split('/')[3], s['index']), (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            tiles.append(tile)
        if tiles:
            rows.append(np.hstack(tiles))
    if rows and len(set(a.shape[1] for a in rows)) == 1:
        cv2.imwrite(str(SAMPLES / 'contact_sheet_sparse.jpg'), np.vstack(rows))


def render(r):
    r = plain(r)
    if r.get('status') != 'complete':
        return
    v, p, sy, cu = r['vio']['summary'], r['vio_path'], r['synchronization'], r['livox_custom']
    lines = ['# 主bag只读数据审计', '', '输入：`{}`；生成于 {}。'.format(r['bag_path'], r['generated_utc']), '',
             '## 首要数据问题：禁止原VIO全程初始化', '',
             '**原VIO在458.294363s起严重发散，最后xyz为[998569458.5517, -7607714043.4815, 27781226527.8881]。全程VIO初始化不允许。** 原bag正常ROS反序列化和wire解析已独立对照；单位四元数或轻微header局部排序不能挽救位置发散。主代理报告的3525块属于无效轨迹诊断产物，不能作为有效全程block计划。', '',
             '需使用有来源证据的完整LIO SE3，并另行认证camera rig及地图初始化；未获外参时不得用fake identity补链。', '',
             '## 可直接用于提取的结论', '',
             '- 数据关联统一使用**传感器header整数ns**，同时保留bag ns；两个epoch相差约65,737,402秒，不能混用，也不能仅凭首帧常数偏移校正全程。',
             '- 四路实际topic为camera_2/3/4/5，均CompressedImage JPEG。camera编号与标定cam编号的映射仍需明确证据。',
             '- 四路完整有序header序列相同：`{}`；全程存在5.003ms的偏差，精确stamp交集不能覆盖全部四路组。建议显式6ms容差匹配并保留逐路真实ns；容差不是新标定结论。'.format(sy['four_way']['identical_ordered_header_sequences']),
             '- VIO只保留原始发布姿态；world/world和初始近绕X轴180°姿态不能自动构成车辆body坐标定义。',
             '- **VIO在458.3s附近严重发散，后续位置达到百亿量级；不能将全程作为重建位姿。**',
             '- **当前主参考为CZY_v6；最新最终地图来自V5 run02多会话重建，不是同次all单会话地图。**',
             '', '![首帧四路](../reports/samples/contact_sheet_first.jpg)', '',
             '[全程稀疏contact sheet](../reports/samples/contact_sheet_sparse.jpg)', '',
             '## 覆盖和真实connection', '',
             '扫描{}条，bag时长{:.6f}秒。全部header/IMU/odom/Path重叠历史以及Custom点时间/tag/line均已检查；图像解码与稠密坐标按序号等距抽样，不能称全量视觉质检。'.format(r['bag_time']['total_messages'], r['bag_time']['duration_s']), '',
             '完整connection header、消息定义、MD5和callerid保存在JSON `connections`，不以topic名字猜测消息类型。', '',
             '| topic | type | 数量 | frame_id | header时长s | header间隔中位ms | bag−header首/末s |',
             '|---|---|---:|---|---:|---:|---:|']
    for topic, d in r['topics'].items():
        lines.append('| {} | {} | {} | {} | {:.6f} | {:.3f} | {:.6f} / {:.6f} |'.format(
            topic, d['type'], d['count'], ', '.join(d['frame_ids']), d['header_duration_s'], d['header_interval_ms']['median'], d['bag_minus_header_first_s'], d['bag_minus_header_last_s']))
    if 'recorded_rate_hz' in r['topics']['/orbvi/imu']:
        ir = r['topics']['/orbvi/imu']['recorded_rate_hz']
        lines.extend(['', '### 实际记录速率与非单调时间', '',
            'IMU共154,657条，sensor header覆盖{:.9f}s；N/duration平均记录速率为**{:.6f}Hz**，按(N−1)/duration为{:.6f}Hz。header间隔中位4.949ms对应约{:.6f}Hz，它和含间歇丢帧/不均匀间隔的平均速率是不同统计量。'.format(
                ir['header_duration_s'], ir['count_over_header_duration_hz'], ir['mean_interval_rate_hz'], ir['median_interval_implied_rate_hz']),
            '', '**不采用rosbag info.frequency（例如7157Hz）作为IMU平均采样率。** 全topic速率保存在JSON `topics.*.recorded_rate_hz`，均以记录count及设备header首末时刻计算。', '',
            'odom row4129的header从1720268300747032166ns回退至1720268300747009039ns，差−23,127ns，pose位置差约1.221mm。主代理采用≤1ms局部排序、raw CSV保持原顺序，真正clock reset拒绝；此例不自动判为硬件clock reset。', '',
            'IMU另有一次−95,057,036ns回退，超过1ms例外，不得套用微小回退规则。图像/深度各一次约23µs回退，IPM/Path的header审计未发现回退；每topic的计数和具体事件分别保留。'])
    lines.extend(['', '## 图像与四路同步', ''])
    for c in CAMERAS + ['/orbvi/ipm/image']:
        samples = r['topics'][c]['samples']
        good = [s for s in samples if s.get('decode_ok')]
        lines.append('- `{}`：抽样{}/{}张成功；尺寸{}；Laplacian方差范围{}，仅作为相对清晰度指标，无未经验证阈值。彩色通道相同帧{}。'.format(
            c, len(good), len(samples), sorted(set(str(s['shape_hwc']) for s in good)),
            ('{:.2f}–{:.2f}'.format(min(s['laplacian_variance'] for s in good), max(s['laplacian_variance'] for s in good)) if good else '无'), sum(s['all_channels_identical'] for s in good)))
    if 'bag_receive_spread_ms' in sy['four_way']:
        spread = sy['four_way']['bag_receive_spread_ms']
        hs = sy['four_way']['same_ordinal_header_spread_ms']
        lines.append('\n按相同序号的四路组，header完全相同{}组/{}组，spread分布为`{}`ns；最大header分散{:.3f}ms。bag接收分散中位{:.3f}ms，p99={:.3f}ms，max={:.3f}ms。'.format(
            sy['four_way']['same_ordinal_exact_header_group_count'], sum(sy['four_way']['same_ordinal_header_spread_ns_histogram'].values()),
            sy['four_way']['same_ordinal_header_spread_ns_histogram'], hs['max'], spread['median'], spread['p99'], spread['max']))
    lines.extend(['', 'header回退/相邻重复及sequence变化、逐topic接收偏移分布/线性拟合、各传感器最近邻偏差详见JSON。最近邻可多对一，不能将覆盖边缘偏差认定为传感器固定时延。', '',
                  '## VIO全部轨迹、跳变及Path历史', '',
                  '- {}个odom，header时长{:.3f}s，累计位移{:.3f}m，净位移{:.3f}m；z范围{:.3f}–{:.3f}m，净z变化{:.3f}m。'.format(v['count'], v['duration_s'], v['path_length_m'], v['net_displacement_m'], v['z_m']['min'], v['z_m']['max'], v['z_net_m']),
                  '- 初始xyz/xyzw：`{}`；全部8055条原始轨迹数值保存在JSON `vio.trajectory`，列名与整数ns明确。'.format(plain(v['first_pose_xyz_xyzw'])),
                  '- 跳变筛查命中{}个；规则为{}。最大单步位移{:.6f}m、旋转{:.6f}°；详见jump_examples/largest_steps。'.format(v['jump_screen_count'], v['jump_screen_definition'], v['step_m']['max'], v['step_rotation_deg']['max']),
                  '- {}个Path，长度范围{}–{}；逐相邻Path按相同pose header进行了{}次历史比较。位置被改写消息{}个，最大{:.9g}m；旋转被改写消息{}个，最大{:.9g}°。'.format(p['message_count'], int(p['length_stats']['min']), int(p['length_stats']['max']), p['overlap_pose_comparisons'], p['history_position_change_messages'], p['max_history_position_change_m'], p['history_rotation_change_messages'], p['max_history_rotation_change_deg']),
                  '- Path末姿态与同stamp的已记录odom比较{}次，最大位置误差{:.9g}m。Path最早已有pose时间{}，主bag最早odom时间{}；bag录制前的历史不能据此视为已完整记录。'.format(p['odom_comparisons'], p['max_path_odom_position_error_m'], p['first_path_first_header_ns'], r['topics']['/orbvi/vio/odometry']['first_header_ns']),
                  '- 当前桥接源码是累加odom、超长裁前缀，不回写旧pose；数据全量比较才是本bag是否发生Path历史修正的直接证据。未发现回写不证明SDK没有优化/全局修正，也不能将Path当独立真值。', '',
                  '### 异常边界与时间断档', ''])
    for threshold, event in v['first_step_over_m'].items():
        lines.append('- 首次单步>{}m：index={}，header={}ns，相对最早odom {:.6f}s，单步{:.6f}m。'.format(
            threshold, event['index'], event['header_ns'], event['relative_s'], event['step_m']))
    if 'before_first_step_over_2m' in r['vio']:
        pv = r['vio']['before_first_step_over_2m']
        lines.append('- 首次>2m单步之前的诊断前段：{}个pose，路径{:.3f}m，z范围{:.3f}–{:.3f}m；这是截段统计，仍未认证前段所有pose精度。'.format(
            pv['count'], pv['path_length_m'], pv['z_m']['min'], pv['z_m']['max']))
    for topic in CAMERAS[:1] + ['/orbvi/depth/points', '/orbvi/vio/odometry', '/orbvi/ipm/image', '/orbvi/imu']:
        d = r['topics'][topic]
        for gap in d['gap_examples'][:3]:
            lines.append('- `{}`断档：{} → {}ns，间隔{:.6f}s。'.format(topic, gap['before_header_ns'], gap['after_header_ns'], gap['gap_s']))
        if d['nonincreasing_header_examples']:
            lines.append('- `{}`header回退：`{}`。'.format(topic, d['nonincreasing_header_examples']))
    lines.extend(['',
                  '## 点云、Livox与IMU', ''])
    dp = r['topics']['/orbvi/depth/points']
    lines.append('- Depth PointCloud2 frame=`base_link`，字段布局详见metadata_variants；首帧XYZ float32位于0/4/8，rgb float32位于16、point_step32，rgb按PCL packed RGB位解释，不作为普通浮点强度。样例点数与坐标范围在JSON。frame名不认证外参。')
    lines.append('- Custom共{}帧/{}点，point_num长度不一致{}帧；timebase−header范围{:.0f}–{:.0f}ns，首点offset中位{:.0f}，最大offset中位{:.0f}ns，offset倒退{}帧；tag频数`{}`，line频数`{}`。'.format(
        cu['messages'], cu['total_points'], cu['point_num_mismatch'], cu['timebase_minus_header_ns']['min'], cu['timebase_minus_header_ns']['max'],
        cu['offset_first_ns']['median'], cu['offset_max_ns']['median'], cu['offset_decrease_messages'], cu['tag_histogram'], cu['line_histogram']))
    lines.append('- timebase与header接近且offset跨度近0.1s，与scan周期一致，支持ns和scan起始时间解释。不能把header误当scan末尾；tag位语义需本地驱动/协议核查，不能擅用某种Livox过滤掩码。')
    lines.append('- Custom有{}个全零XYZ点（约{:.3f}%），按tag分布为`{}`，非有限XYZ点{}个。需结合零距离/近距离检查，不能只按tag=0判断有效。现有preprocess.cpp将offset_time/1e6转为ms，XYZ直接以m使用；这是实际工程的单位使用证据。'.format(
        cu['zero_xyz_points'], cu['zero_xyz_points'] / cu['total_points'] * 100, cu['zero_xyz_tag_histogram'], cu['nonfinite_xyz_points']))
    im = r['imu']
    lines.append('- IMU加速度模长中位{:.6f}，p05–p95={:.6f}–{:.6f}；角速度模长中位{:.6f}。消息定义要求m/s²/rad/s，桥接不做变换。{}个全零姿态，零四元数不能用于姿态；协方差为0表示未知，不能解读为零噪声。'.format(
        im['acceleration_norm']['median'], im['acceleration_norm']['p05'], im['acceleration_norm']['p95'], im['angular_velocity_norm']['median'], im['orientation_all_zero_count']))
    lines.extend(['', '## 标定和地图定位', '',
                  '只在已明确工程的标定/地图子目录作有界搜索（深度、条目上限、跳过目录清单保存在JSON）。未递归扫描图像/标注目录。', '',
                  '- 板端标定：`{}`，原始cam0..3矩阵已保存JSON。工程既有审计定义其为TripleSphere，Twc以cam0为参考、平移mm；不能当成OpenCV标准fisheye参数。'.format(r['calibration']['raw_camera_yaml']),
                  '- `fisheye_8s`含八路虚拟PINHOLE的内外参及图像→IMU td；缺raw→virtual rectification显式矩阵和索引映射，不能将同名cam直接接链。',
                  '- FAST-Calib已有raw camera←LiDAR候选文件；已有标定报告显示版本与闭环差异，尚未认证为一条一致的camera→VIO链。',
                  '- 优化VIO桥接明确保留SDK source_frame_id/source_child_frame_id，姿态逐字段复制；当前bag实际world/world。SDK发布参考点与轴向尚需证据。',
                  '', '### 当前v6与标定缺口', '',
                  '- 当前工程：`{}`；最新交付目录：`{}`。'.format(r['current_v6']['primary_package'], r['current_v6']['latest_delivered_map_dir']),
                  '- v6板端三份标定与历史CZY逐字节相同；`version6/BEV_SOURCE_AUDIT.zh-CN.md`明示固定杆臂/yaw/时偏新拟合未通过保留集，没有改变标定。v6修复语义几何，不证明相机→IMU→LiDAR→世界坐标链重新测量或闭合。',
                  '- v6 `final_export_check.json`：new_online_run=false，source_run=V5 lifelong_run_02，raw 30,456,613点；最终图含旧prior和新corrected_session，应追溯每个会话的T(t)。全图不存在可直接配本all原始图像的一条已认证相机轨迹。',
                  '- 同次run02的轨迹在`{}`（odom）与`{}`（时变校正），不是把旧all的TUM直接套最终大图。'.format(
                      r['current_v6']['matching_session_pose_source']['registered_bag'], r['current_v6']['matching_session_pose_source']['session_correction_json']),
                  '- 最终图prior来自V5 `maps/CZY_DATA/v4_all`；该目录本次未找到同次registered证据bag或独立轨迹文件，不能证明历史Log中的同名all轨迹与此prior初始化相同。',
                  '', '### TripleSphere实际模型', '',
                  '9参数顺序为`fx, fy, cx, cy, xi, lambda, alpha, b, c`。本机`rpg_vikit/vikit_common/src/triple_sphere_camera.cpp:35`含完整投影，`:55`含反投影；SDK `fisheye_rig.hpp`接口字段顺序相同。ROS loader参数拼写是`cam_lamda`。', '',
                  '```text', 'd1 = ||p||; z2 = z + xi*d1; d2 = sqrt(x*x+y*y+z2*z2)',
                  'z3 = z2 + lambda*d2; d3 = sqrt(x*x+y*y+z3*z3)', 'den = z3 + alpha/(1-alpha)*d3',
                  'mx = x/den; my = y/den', 'u = fx*mx + b*my + cx; v = c*mx + fy*my + cy', '```', '',
                  '有效域、零分母、反投影根号需按源码处理；该参数不是标准OpenCV fisheye的畸变四系数。完整路径及源码哈希在JSON `calibration.triple_sphere_model/source_files`。', '',
                  '| 已有地图/轨迹（当前v6优先，旧版仅对比） | 体积bytes | 文件证据 |', '|---|---:|---|'])
    for f in r['local_artifacts']['files']:
        if Path(f['path']).suffix.lower() in ('.pcd', '.tum') or 'tum_summary' in f:
            show = any(x in f['path'] for x in ['run02_geometry_final_v6/', 'all_replay_candidate/',
                'lifelong_run_02/lifelong/completion/corrected_session/', 'CZY_v5/maps/CZY_DATA/v4_all/'])
            show = show or Path(f['path']).name in ['czy_lio_v6_final_all.txt', 'czy_lio_lifelong_v5.txt',
                                                   'czy_lio_lifelong_v6.txt', 'czy_height_continuity_release3_20260925.txt']
            if not show:
                continue
            detail = '; '.join(x for x in f.get('pcd_header', []) if x.startswith(('POINTS ', 'FIELDS ', 'DATA ')))
            if 'tum_summary' in f:
                a = f['tum_summary']
                detail = '{}行，{:.3f}s，首/末时间{} / {}'.format(a['rows'], a['duration_s'], a['first'][0], a['last'][0])
            lines.append('| `{}` | {} | {} |'.format(f['path'], f['size_bytes'], detail))
    lines.extend(['', '定位地图只证明文件存在；v6 final是旧run02的几何重建，all_replay_candidate明确标为candidate。旧CZY mapping2仅保留历史对比，raw/fused/semantic及会话初始化必须区分；未确认逐帧来源的PCD不能拿另一轮TUM重新配准。', '',
                  '## 明确阻塞项', ''])
    lines.extend('- ' + s for s in r['limitations'])
    if 'candidate_all_lio_evidence' in r:
        e = r['candidate_all_lio_evidence']
        lines.extend(['', '## 匹配all时间的完整LIO SE3替代证据', '',
            '- 实际存在：`{}`，大小{}bytes；/aft_mapped_to_init与/cloud_registered各8246条，另有3556条semantic_primitives。'.format(e['registered_bag'], e.get('bag_size_bytes')),
            '- 精确odom首末header：{}–{}ns，与主all传感器时间覆盖匹配；frame=camera_init，child=aft_mapped。'.format(e.get('first_header_ns'), e.get('last_header_ns')),
            '- 本次仅读索引与五个定点odom/cloud样本，{}个样本header精确配对；未全量扫描8.76GB证据包，也未逐条认证全部bag pose数值。'.format(e.get('sampled_odom_cloud_exact_header_match_count')),
            '- 对应TUM：`{}`，8246行xyz+xyzw，完整文本有限，四元数接近单位长度；五个样本按q/−q同一旋转比较。精确位姿差与全TUM轨迹统计在JSON `candidate_all_lio_evidence.tum`。'.format(e.get('tum', {}).get('path')),
            '- v6自己的lidar/summary.json明确未建立V2轨迹与V4 prior相同初始化。该证据可作为与自身registered云一致的all LIO候选，**不能直接套到最终v6多会话map，不能当已验证camera pose**。',
            '- registered云字段保存在样本metadata；是否具备相机辐射颜色须单独认证。rig外参缺失仍为阻塞，禁止identity代填。'])
    if 'delivery_handoff' in r:
        h = r['delivery_handoff']
        sp = h['matching_session_pose']
        lines.extend(['', '## 最终交付：v6来源、同步pose与RGB边界', '',
            '审计已结稿，无剩余执行任务。以下来自已有审计结果的汇总，不追加bag或点云扫描。', '',
            '- v6最终raw由19,196,862个prior点和11,259,751个corrected_session点组成，共30,456,613点；new_online_run=false。来源报告的输入路径和声明SHA256在JSON `delivery_handoff.map.inputs`，本轮未重新整文件哈希大型PCD。',
            '- 同次run02 `/aft_mapped_to_init`共{}个pose，header范围{}–{}ns，持续{:.9f}s；header.frame_id=camera_init，child_frame_id=aft_mapped。'.format(
                sp['count'], sp['first_header_ns'], sp['last_header_ns'], sp['duration_s']),
            '- `session_correction.json`含5041个reference_poses及157个时变knots，约定T_prior_session、gravity_aligned_yaw_xyz。校正链为`T_prior_sensor(t)=T_prior_session(t) @ T_session_sensor(t)`，按文件定义插值，不能只取一个初始矩阵。',
            '- 原all的3556个IPM stamp全部精确出现在camera_3 header序列；四路使用6ms容差并保留真实stamp，避免精确交集损失2075组。header回退需显式排序/分段，不能按接收顺序默认严格递增。',
            '- Log中的`czy_lio_lifelong_v5.txt`与`czy_lio_lifelong_v6.txt`时间范围相同但XYZ不同；同stamp或文件名v6不能证明它们对应最终图的同次源pose。最终图的新会话以原registered bag和session_correction为可追溯输入。',
            '- 最终PCD确有RGB字段，但本轮未认证其相机颜色来源，也未核验其train/val隔离。语义类别RGB不是辐射颜色。`--map-rgb-verified`应保留独立来源声明；`--colorize-from-images`仅用train帧着色与当前审计边界一致，仍须有已确认的相机pose链。',
            '', '交付仍受以下证据缺口限制：'])
        lines.extend('- ' + s for s in h['evidence_blockers'])
    lines.extend(['', '## 实际执行与修改范围', '', '```bash',
                  'TMPDIR=/home/chaoszzss/dataset/rosbag/3D-Gaussian/runtime/tmp PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 /home/chaoszzss/dataset/rosbag/3D-Gaussian/scripts/audit_bag.py --bag /home/chaoszzss/dataset/rosbag/orbvi_4vl_data_czy_all.bag --samples-per-topic 6',
                  '```', '', '写入`scripts/audit_bag.py`、`reports/dataset_audit.json`、`docs/DATA_AUDIT.md`与`reports/samples/`；genpy隐式临时类固定于`runtime/tmp/`。未修改源bag、历史标定/地图、其他代理代码，未安装依赖。', ''])
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=ROOT.parent / 'orbvi_4vl_data_czy_all.bag')
    parser.add_argument('--samples-per-topic', type=int, default=6)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--refresh-evidence', action='store_true', help='保留已扫描bag结果，仅重查本地版本证据并更新报告')
    parser.add_argument('--handoff-only', action='store_true', help='只汇总已有JSON到最终交付，不读取bag/PCD/标定源文件')
    parser.add_argument('--probe-all-lio', action='store_true', help='定向读取V2 all registered索引、5个odom/cloud样本和小型完整TUM')
    args = parser.parse_args()
    if args.probe_all_lio:
        r = json.loads(REPORT.read_text(encoding='utf-8'))
        r['candidate_all_lio_evidence'] = probe_all_lio_evidence()
        r['delivery_handoff'] = delivery_handoff(r)
        finalize_timing_rates(r)
        mark_critical_vio_failure(r)
        r['delivery_handoff']['alternative_all_lio_evidence'] = {'registered_bag': r['candidate_all_lio_evidence']['registered_bag'],
            'same_time_interval_as_original_all': r['candidate_all_lio_evidence']['same_time_interval_as_original_all'],
            'same_initialization_as_v6_final_map_verified': False, 'camera_rig_extrinsic_verified': False}
        save(r)
        render(r)
        print('ALL_LIO_INDEX_PROBE_READY without full bag scan', REPORT, flush=True)
        return
    if args.handoff_only:
        r = json.loads(REPORT.read_text(encoding='utf-8'))
        if r.get('status') != 'complete':
            parser.error('现有bag审计未完成，不能直接结稿')
        finalize_timing_rates(r)
        r['delivery_handoff'] = delivery_handoff(r)
        mark_critical_vio_failure(r)
        r['handoff_generated_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
        save(r)
        render(r)
        print('HANDOFF_READY without bag/PCD scan', REPORT, flush=True)
        return
    if args.refresh_evidence:
        r = json.loads(REPORT.read_text(encoding='utf-8'))
        if r.get('status') != 'complete':
            parser.error('现有bag审计未完成，不能仅刷新证据')
        r['local_artifacts'] = bounded_inventory()
        r['calibration'] = calibration_evidence()
        r['current_v6'] = current_v6_artifacts()
        finalize_timing_rates(r)
        r['delivery_handoff'] = delivery_handoff(r)
        mark_critical_vio_failure(r)
        r['vio']['summary'] = trajectory_summary(r['vio']['trajectory'])
        split = r['vio']['summary']['first_step_over_m'].get('2.0')
        if split:
            r['vio']['before_first_step_over_2m'] = trajectory_summary(r['vio']['trajectory'][:split['index']])
            r['vio']['before_first_step_over_2m']['interpretation'] = '诊断截段，不等于所有前段pose通过几何精度验收；仍有时间断档和一次header回退'
        r['local_evidence_refreshed_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
        save(r)
        render(r)
        print('EVIDENCE_REFRESHED', REPORT, flush=True)
        return
    if args.render_only:
        r = json.loads(REPORT.read_text(encoding='utf-8'))
        render(r)
        return
    if args.samples_per_topic < 1 or args.samples_per_topic > 20:
        parser.error('samples-per-topic须为1..20')
    audit(args.bag.resolve(), args.samples_per_topic)


if __name__ == '__main__':
    main()
