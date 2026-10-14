#!/usr/bin/env python3
"""Publish a local delivery record only after actual all28 DEV/TEST completion."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['block_%03d' % i for i in range(28)]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    p = ROOT / path
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def main():
    assembly = read('maps/garage_core_all_v1/assembly.json')
    batch = read('runs/core_joint_v1/status.json')
    frozen = read('reports/CURRENT_ROUND_FROZEN_FOR_TEST_20261007.json')
    paths = {split: 'runs/full_scene_core_all_v1_final_%s/summary.json' % split
             for split in ('dev', 'test')}
    evaluations = {split: read(p) for split, p in paths.items()}
    assert batch['complete'] is True and batch['trained_exported_core_count'] == 28
    assert assembly['coverage']['nonzero_core_count'] == 28
    assert [c['id'] for c in assembly['cores']] == EXPECTED
    assert all(c['retained_gaussians'] > 0 and c['overview_gaussians'] > 0 for c in assembly['cores'])
    for split, d in evaluations.items():
        assert d['complete'] is True and d['native_all28_assessed'] is True
        assert d['split'] == split and d['validated_core_ids'] == EXPECTED
        assert d['models']['candidate'] == frozen['candidate']
        assert d['high_fidelity_accepted'] is False
        assert d['unique_images']['candidate']['unique_image_count'] > 0
    actual = {}
    for role in ('full', 'overview'):
        a = assembly[role]
        p = ROOT / a['path']
        assert p.stat().st_size == a['file_bytes']
        sha = digest(p)
        assert sha == a['sha256']
        actual[role] = {'path': a['path'], 'gaussians': a['gaussians'],
                        'file_bytes': p.stat().st_size, 'sha256': sha}
    overview = read('viewer/runtime/core_all_overview_zoom/verification.json')
    inside = read('viewer/runtime/core_all_inside/verification.json')
    assert overview['cores'] == 28 and not overview['js_errors']
    assert overview['overview_gaussians'] == actual['overview']['gaussians']
    assert {r['id'] for r in inside['regions']} == set(['F%d' % i for i in range(1, 7)] +
        ['ramp_F%d_F%d' % (i, i + 1) for i in range(1, 6)])
    assert all(not r['js_errors'] and 'NVIDIA' in r['renderer'] for r in inside['regions'])
    timestamp = datetime.now().astimezone().isoformat()
    report = {'status': 'current_round_dataset_reconstruction_loop_complete_with_limitations',
              'completed_at_local': timestamp, 'current_round_closed_loop_complete': True,
              'training_verified_cores': 28, 'assembly_nonzero_cores': 28,
              'artifacts': actual, 'coordinate_frame': 'camera_init', 'units': 'm',
              'floors_flattened': False, 'models_recentered': False,
              'evaluation_paths': paths,
              'evaluation_unique_images': {s: d['unique_images']['candidate'] for s, d in evaluations.items()},
              'evaluation_frame_observations': {s: d['frame_observations']['candidate'] for s, d in evaluations.items()},
              'visual_reports': ['viewer/runtime/core_all_overview_zoom/verification.json',
                                 'viewer/runtime/core_all_inside/verification.json'],
              'high_fidelity_accepted': False, 'surface_coverage_verified': False,
              'unobserved_building_complete': False, 'floor_verified': False,
              'calibration_verified': False, 'independent_geometry_acceptance': False,
              'whole_old_baseline_comparison_complete': False,
              'baseline_failure': 'Old10M model exact active records exceed the 4M/6M guards; original partial reports preserved, no sampled full-baseline claim.',
              'limitations': ['Native and browser views show missing floor and side-wall regions.',
                             'Markings, reflections, lights and vehicle edges remain blurred or contaminated.',
                             'Browser overview is a 1.2M subset; local views obey the 3-block/1.5M memory budget.',
                             'Depth reference is conditioned on the same input LIO/map and support pixels, not independent ground truth.',
                             'Six-floor labels are historical time-window estimates, not surveyed controls.'],
              'next_round': 'Targeted optimization deferred by explicit user request; no parameter tuning after frozen final TEST.',
              'viewers': {'overview': 'http://127.0.0.1:8769/', 'inside': 'http://127.0.0.1:8770/'},
              'instructions': 'docs/VIEW_CURRENT_MAP.md'}
    save('reports/CURRENT_ROUND_DELIVERY_20261007.json', report)

    # Whole-model renders: select the middle TEST group of camera_3 per floor,
    # rather than selecting the highest-scoring image.
    from PIL import Image, ImageDraw
    out = ROOT / 'reports/current_round'
    out.mkdir(exist_ok=True)
    sheet = Image.new('RGB', (1280, 1740), (16, 26, 34))
    draw = ImageDraw.Draw(sheet)
    image_records = []
    for i, core in enumerate((2, 8, 13, 17, 21, 26)):
        folder = ROOT / 'runs/full_scene_core_all_v1_final_test' / ('block_%03d_candidate' % core)
        rows = [r for r in json.loads((folder / 'metrics.json').read_text())['frames']
                if r['camera_id'] == 'camera_3']
        rows.sort(key=lambda r: int(r['group_id']))
        row = rows[len(rows) // 2]
        source = folder / ('%03d_candidate.png' % row['dataset_index'])
        image = Image.open(source).convert('RGB')
        assert image.size == (640, 540)
        x, y = (i % 2) * 640, (i // 2) * 580
        draw.text((x + 12, y + 12), 'F%d | full 6.78M model | held-out TEST' % (i + 1), fill=(117, 218, 193))
        sheet.paste(image, (x, y + 40))
        image_records.append({'estimated_floor': 'F%d' % (i + 1), 'core_id': 'block_%03d' % core,
                              'source': str(source.relative_to(ROOT)), 'source_sha256': digest(source),
                              'group_id': row['group_id'], 'camera_id': row['camera_id'],
                              'selection': 'middle TEST group of camera_3', 'psnr_db': row['psnr_db']})
    sheet.save(out / 'full_model_test_six_floors.png')
    save('reports/current_round/full_model_test_six_floors_provenance.json',
         {'status': 'layout_of_actual_native_whole_model_TEST_renders', 'records': image_records,
          'model_sha256': actual['full']['sha256'], 'repainted_or_filled': False})

    # Derived viewer metadata describes evaluation of the full model, not an
    # independent score for a displayed subset. Canonical assembly stays immutable.
    p = 'maps/garage_core_all_v1/viewer_manifest.json'
    viewer = read(p)
    viewer['native_fullscope_evaluated'] = True
    viewer['current_round_closed_loop_complete'] = True
    viewer['full_model_evaluation'] = paths
    for b in viewer['blocks']:
        b['eval'] = {'status': 'whole_map_evaluated', 'scope': 'complete_source_ply_not_displayed_subset',
                     'dev_report': 'runs/full_scene_core_all_v1_final_dev/%s_candidate/metrics.json' % b['id'],
                     'test_report': 'runs/full_scene_core_all_v1_final_test/%s_candidate/metrics.json' % b['id'],
                     'high_fidelity_accepted': False}
    save(p, viewer)
    p = 'maps/garage_core_all_v1/overview_manifest.json'
    navigation = read(p)
    navigation['native_fullscope_evaluated'] = True
    navigation['current_round_closed_loop_complete'] = True
    navigation['full_model_evaluation'] = paths
    for b in navigation['blocks']:
        b['label'] = '六层车库 · 全部28区域总览'
        b['eval'] = {'status': 'whole_map_evaluated', 'display_kind': 'overview',
                     'scope': 'complete_source_ply_not_overview_subset',
                     'reports': paths, 'high_fidelity_accepted': False}
    save(p, navigation)
    scores = evaluations['test']['unique_images']['candidate']
    markdown = '''# 本轮六层车库重建交付

完成时间：{time}。本轮全部 28 个区域的训练、导出、整图装配、完整模型 DEV/TEST 校验和真实可视化已完成。完整性指采集数据的区域处理，表面仍有空缺，高保真未验收。针对性优化按要求留到下一轮。

完整模型：`/home/chaoszzss/dataset/rosbag/3D-Gaussian/maps/garage_core_all_v1/gaussians_full.ply`，**6,780,816 个高斯，1.68 GB**。坐标为 `camera_init`、单位米；保留楼层高度，不做逐块居中。

整图查看：<http://127.0.0.1:8769/>，点击 `garage_overview` 的“加载”；内部查看：<http://127.0.0.1:8770/>，加载目标及相邻区块，点目标“定位”后“进入区域”。查看器为本地 Three.js/Spark Gaussian 工程。总览使用全部区域的 120 万真实高斯样本，分区用于查看细节；完整文件不受浏览预算限制。

完整文件列表、估计楼层与区块对应、交互和关闭电脑后的重启命令见 [查看说明](../docs/VIEW_CURRENT_MAP.md)。详细数量、SHA256、检查点和校验见 [交付审计](CURRENT_ROUND_DELIVERY_20261007.json)、[训练/装配证据](CURRENT_ROUND_TRAINING_ASSEMBLY_AUDIT_20261007.json)。

| 完整模型评估 | 区域数 | 去重图像数 | 平均 PSNR / dB | 像素加权 SSIM |
| --- | --- | --- | --- | --- |
{table}

原始固定划分、相机位姿、RGB/mask 和几何参考保持不变；评估仅读取模型，不进行拟合。重叠区同一图像去重，不把重复观测当独立图像。深度仅在固定 LiDAR 支持像素上评估，不能证明地面或全图没有空洞。旧版整图对照因同时可见记录超过预算中断，保留部分报告，不宣称有完整配对对照结果。

![真实整图总览](../viewer/runtime/core_all_overview_zoom/all28_oblique.png)

![真实完整模型的六层 TEST 渲染](current_round/full_model_test_six_floors.png)

另有 [六层浏览器内部视角](../viewer/runtime/core_all_inside/six_floors_contact_sheet.png) 和五段坡道截图，均为实际模型显示。截图排版没有补绘空缺或纹理。

当前问题包括原生渲染确认的地面/侧墙空缺、文字箭头模糊、亮区溢出、车辆边缘和接缝污染。楼层标签、标定精度与几何均未完成独立验收；未观察区域不能声明重建完成。历史工程外缓存写入例外见 [记录](WRITE_SCOPE_EXCEPTION_20261007.md)，后续缓存已限于本工程。
'''.format(time=timestamp, table='\n'.join('| %s | 28 | %d | %.2f | %.3f |' %
         (s.upper(), d['unique_images']['candidate']['unique_image_count'],
          d['unique_images']['candidate']['psnr_db_frame_mean'],
          d['unique_images']['candidate']['ssim_pixel_mean']) for s, d in evaluations.items()))
    (ROOT / 'reports/CURRENT_ROUND_DELIVERY_20261007.md').write_text(markdown)
    # Standalone local HTML report for direct browser viewing, with real images.
    body = ('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>六层车库 · 本轮高斯重建结果</title>
<style>body{margin:40px auto;max-width:1280px;padding:0 24px;background:#101a22;color:#e5edf2;font:17px/1.7 system-ui}a{color:#75dac1}img{width:100%%;height:auto}code{overflow-wrap:anywhere}small{color:#a6bcc6}</style>
<h1>六层车库：本轮完整数据集重建结果</h1><p>全部28区域训练、装配和完整模型DEV/TEST校验完成。678万真实高斯，1.68GB；局部表面仍有空缺，画质尚未验收。下一轮针对性优化。</p>
<p><a href="http://127.0.0.1:8769/">打开整图总览</a> · <a href="http://127.0.0.1:8770/">进入分区内部</a> · <a href="../docs/VIEW_CURRENT_MAP.md">完整文件与重启说明</a></p>
<p>模型：<code>/home/chaoszzss/dataset/rosbag/3D-Gaussian/maps/garage_core_all_v1/gaussians_full.ply</code></p>
<p>最终TEST：%d张去重图像，PSNR %.2f dB，SSIM %.3f。</p>
<h2>实际全区域总览</h2><small>120万高斯总览样本，保留全部28区域贡献；完整PLY保留678万。</small>
<img src="../viewer/runtime/core_all_overview_zoom/all28_oblique.png" alt="实际高斯整图总览">
<h2>完整模型的六层留出视角</h2><small>原生2DGS逐视角考虑全量源记录；每层选择camera_3的中间TEST时间组，没有挑选最高分样例或补绘纹理。</small>
<img src="current_round/full_model_test_six_floors.png" alt="实际完整模型六层TEST渲染">
<h2>浏览器内部分区视角</h2><small>局部加载与原生全量评估不同；黑暗空缺是需要下一轮处理的问题。</small>
<img src="../viewer/runtime/core_all_inside/six_floors_contact_sheet.png" alt="六层实际浏览器视角">
<p><a href="CURRENT_ROUND_DELIVERY_20261007.md">完整交付记录</a> · <a href="CURRENT_ROUND_DELIVERY_20261007.json">审计JSON</a></p></html>''' %
        (scores['unique_image_count'], scores['psnr_db_frame_mean'], scores['ssim_pixel_mean']))
    (ROOT / 'reports/CURRENT_ROUND_DELIVERY_20261007.html').write_text(body)
    print(json.dumps({'status': report['status'], 'files_verified': actual,
                      'test_unique_images': scores}, ensure_ascii=False))


if __name__ == '__main__':
    main()
