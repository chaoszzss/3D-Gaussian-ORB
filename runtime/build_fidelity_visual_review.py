"""CPU-only review of actual fixed DEV renders. Never generate or modify imagery."""
import hashlib
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'reports/fidelity_v2_visual_review.html'
links = set()

def read(path):
    return json.loads((ROOT / path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def link(path):
    path = (ROOT / path).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), path
    links.add(path)
    return os.path.relpath(path, out.parent)

def image(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', header[16:24]) == (640, 540), path
    return link(path)

# label, metrics, explicit status, optional paired conditioned-geometry report
specs = [
 ('block_015', [
  ('旧整图', 'runs/fidelity_v2/baseline_old_all28/block_015/metrics.json', '旧基线 · 未验收', None),
  ('原始单块', 'runs/fidelity_v2/original_single_blocks/block_015/metrics.json', '原始基线 · 未验收', None),
  ('严格v3 · 1280单块训练', 'runs/fidelity_v2/train_1280_depthv3_block015_acceptance/metrics.json', '未通过', 'reports/surface_quality_v2_depthv3_1280_015_rev2.json'),
  ('整图静态3cm裁剪', 'runs/fidelity_v2/tangentcap003_015/block_015/metrics.json', '仅诊断 · 不采用', 'reports/surface_quality_v2_tangentcap003_015.json'),
  ('严格v3 · 三块联合训练 · 3cm', 'runs/fidelity_v2/joint015_aa_cap003_acceptance/block_015/metrics.json', '质量未通过 · 不推广', 'reports/surface_quality_v2_joint015_aa_cap003.json'),
 ]),
 ('block_000', [
  ('旧整图 · 刷新评估', 'runs/fidelity_v2/old_all28_block000_refreshed_v2/metrics.json', '旧基线 · 未验收', None),
  ('原始单块 · 刷新评估', 'runs/fidelity_v2/original000_refreshed_v2/metrics.json', '原始基线 · 未验收', None),
  ('block000 · seed AA 3cm · 15000步', 'runs/fidelity_v2/block000_seed_aa_cap003_acceptance/metrics.json', '质量未通过 · 不推广', 'reports/surface_quality_v2_block000_seed_aa_cap003.json'),
  ('measuredseed_cap003_diagnostic · 静态诊断', 'runs/fidelity_v2/block000_measuredseed_cap003_diagnostic/metrics.json', '仅诊断 · 未验收', 'reports/surface_quality_v2_block000_measuredseed_cap003_diagnostic.json'),
  ('original_cap003_diagnostic · 静态诊断', 'runs/fidelity_v2/block000_original_cap003_diagnostic/metrics.json', '仅诊断 · 未验收', 'reports/surface_quality_v2_block000_original_cap003_diagnostic.json'),
  ('opacity600_control · 600步诊断', 'runs/fidelity_v2/block000_opacity600_control/metrics.json', '仅诊断 · 未验收', 'reports/surface_quality_v2_block000_opacity600_control_diagnostic.json'),
  ('opacity600_soft09 · 600步诊断', 'runs/fidelity_v2/block000_opacity600_soft09/metrics.json', '仅诊断 · 未验收', 'reports/surface_quality_v2_block000_opacity600_soft09_diagnostic.json'),
  ('measuredseed_anneal_v1 · 15000步实际训练', 'runs/fidelity_v2/block000_measuredseed_anneal_v1/metrics.json', '诊断候选 · 未通过验收',
   'reports/surface_quality_v2_block000_measuredseed_anneal_v1.json'
   if (ROOT / 'reports/surface_quality_v2_block000_measuredseed_anneal_v1.json').is_file() else None),
 ])]

# This candidate now has actual completed evaluation; missing geometry stays unmeasured.
pending = []

def build_block(block_id, sources):
    models, reference, expected = [], None, None
    for label, name, status, _ in sources:
        path = ROOT / name
        report = read(path)
        assert report['split'] == 'dev' and report['images'] == 20
        assert report['metric_domain'] == 'float_renderer' and report['primary_protocol'] == 'ego_v2'
        identities = []
        for f in report['frames']:
            assert f['block_id'] == block_id
            assert (f['camera']['w'], f['camera']['h']) == (640, 540)
            identities.append(dict(index=f['dataset_index'], group=str(f['group_id']), camera=f['camera_id'],
                pose_intrinsics=f['camera'], gt_sha256=sha(path.parent / f['artifacts']['gt_png']),
                mask_sha256=sha(f['acceptance_mask_path']), plan=report['acceptance_plan_id']))
        assert len(identities) == 20 and len({(f['group'], f['camera']) for f in identities}) == 20
        if expected is None:
            expected = identities
            reference = [dict(image=image(path.parent / f['artifacts']['gt_png']),
                camera=f['camera_id'], group=str(f['group_id'])) for f in report['frames']]
        else:
            assert identities == expected, (block_id, label, 'fixed frame/camera/GT/mask mismatch')
        values = [dict(image=image(path.parent / f['artifacts']['candidate_png']),
            error=image(path.parent / f['artifacts']['error_png']),
            psnr=f['protocols']['ego_v2']['psnr_db'], ssim=f['protocols']['ego_v2']['ssim_valid_pixels'])
            for f in report['frames']]
        models.append(dict(label=label, status=status, report=link(path), frames=values,
            mean=report['protocols']['ego_v2']['psnr_db_mean']))
    geometry, denominator, baseline = [], None, None
    for label, name, status, geo in sources:
        if geo is None:
            geometry.append(dict(label=label, status=status, measured=False))
            if 'block000_measuredseed_anneal_v1/' in name:
                geometry[-1]['evidence_note'] = '图像评估已完成；几何报告尚未生成，待评。'
            continue
        report = read(geo)
        assert report['block_id'] == block_id and report['independent_ground_truth'] is False
        provenance = report['render_measurement']['sources']
        for kind, source in [('candidate', name), ('baseline', sources[0][1])]:
            assert Path(provenance[kind]['path']).resolve() == (ROOT / source).resolve()
            assert provenance[kind]['sha256'] == sha(ROOT / source), (block_id, kind)
        g5, g4 = report['gate_evidence']['G5'], report['gate_evidence']['G4']
        assert g5['fixed_denominator']
        # Diagnostic measurements can exist while their acceptance provenance is unknown.
        # Preserve that distinction; never promote an unknown gate or substitute another report.
        if geo.endswith('_diagnostic.json'):
            assert report['high_fidelity_pass'] is False
            assert report['render_measurement']['status'] == 'measured'
            assert g5['status'] in ('known', 'unknown') and g4['status'] in ('known', 'unknown')
        else:
            assert g5['status'] == g4['status'] == 'known'
        support = dict(floor=g5['floor']['eligible_pixels'], wall=g5['wall']['eligible_pixels'], fixed_id=g5['fixed_denominator_id'])
        assert denominator is None or support == denominator
        denominator = support
        current_baseline = dict(floor=g5['floor']['baseline_hole_rate'], wall=g5['wall']['baseline_hole_rate'], p50_m=g4['baseline_p50_m'])
        assert baseline is None or baseline == current_baseline
        baseline = current_baseline
        geometry.append(dict(label=label, status=status, measured=True, report=link(geo),
            floor=g5['floor']['candidate_hole_rate'], wall=g5['wall']['candidate_hole_rate'], p50_m=g4['candidate_p50_m'],
            floor_reduction=g5['floor_relative_reduction'], wall_reduction=g5['wall_relative_reduction']))
        if geo.endswith('_diagnostic.json'):
            reasons = list(dict.fromkeys([g['reason'] for g in (g4, g5) if g.get('reason')]))
            geometry[-1].update(diagnostic_only=True, high_fidelity_pass=False,
                gate_status={'G4': g4['status'], 'G5': g5['status']},
                evidence_note=('诊断实测；G4/G5 unknown：报告未确认checkpoint/export绑定。'
                               if 'unknown' in (g4['status'], g5['status']) else '诊断实测；未通过新验收。'),
                evidence_reasons=reasons, geometry_report_sha256=sha(ROOT / geo))
    geometry[0].update(baseline, measured=True, report=next(g['report'] for g in geometry if g['measured']))
    return dict(id=block_id, models=models, reference=reference, geometry=geometry, denominator=denominator, identities=expected)

blocks = [build_block(block_id, sources) for block_id, sources in specs]
link('docs/FIDELITY_ROUND_V2.md')
resolution_path = 'runs/fidelity_v2/block000_measuredseed_anneal_v1_resolution/metrics.json'
resolution = read(resolution_path)
assert resolution['images'] == 20 and resolution['split'] == 'dev'
assert resolution['inference_model_step'] == 14999 and resolution['inference_sh_degree'] == 3
resolution_link = link(resolution_path)
resolution_psnr = resolution['protocols']['ego_v2']['psnr_db_mean']
payload = json.dumps(dict(blocks=blocks, pending=pending), ensure_ascii=False, allow_nan=False).replace('</', '<\\/')

body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>车库重建：固定验证视角</title><style>\nbody{font:16px/1.6 system-ui,sans-serif;margin:24px;background:#f3f5f6;color:#17212b}main{max-width:1600px;margin:auto}\nh1{font-size:25px}p{max-width:1100px}.status{padding:12px 18px;background:#fff1d6;border-left:4px solid #bd7a00}\n.controls{position:sticky;top:0;background:#f3f5f6;padding:14px 0;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}\nbutton,select{font:inherit;padding:5px 12px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}\nfigure{margin:0;padding:12px;background:white;border:1px solid #d4dce2;border-radius:6px}img{width:100%;display:block;background:#111}figcaption{min-height:50px;margin-bottom:8px}small{color:#4e5964}\ntable{border-collapse:collapse;margin:20px 0}td,th{padding:7px 14px;border:1px solid #c7d0d8;text-align:left}a{color:#07539e}\n</style><main><h1>车库重建：固定验证视角</h1>\n<p class="status">本轮仍未通过高保真验收。block_015与block_000各有20个固定DEV视角；区块内同帧比较，切换区块后独立选择帧，不能代表六层整图通过验收。</p>\n<p>所有图像均来自真实观测或实际模型渲染，没有生成或修补图像。block000保留原比较项，并加入measuredseed/original cap003静态诊断和opacity600 control/soft09各20帧实际评估；这些诊断不代表新验收通过。评分使用相同的 ego_v2 有效区和640×540相机；1280训练模型在此重新渲染到固定640验证相机。图片用于查看，数值来自浮点渲染。</p>\n<table><thead><tr><th>模型 / 状态</th><th>地面错误/缺失率</th><th>墙面错误/缺失率</th><th>条件双向p50/m</th><th>相对旧整图改善（地面/墙面）</th><th>几何证据状态</th></tr></thead><tbody id="geometry"></tbody></table><p id="support"></p><p class="status" id="pending"></p>\n<p>几何表直接读取链接中的实际固定分母报告；未测不代表零。地面/墙面属于几何分类，楼层未独立核验。高alpha不等于正确表面：错误深度也计入上述缺失率。参考LiDAR来自同一建图输入，不能作为独立测绘精度证明。<a href="../docs/FIDELITY_ROUND_V2.md">工程记录</a></p>\n<div class="controls"><label>区块 <select id="block"></select></label><label>模型 <select id="model"></select></label><button id="prev">上一帧</button><select id="frame"></select><button id="next">下一帧</button><label><input id="error" type="checkbox">查看误差图</label><span id="identity"></span></div>\n<section id="panels"></section></main><script>const data=PAYLOAD;\nconst blockSelect=document.querySelector(\'#block\'),select=document.querySelector(\'#frame\'),panels=document.querySelector(\'#panels\');let blockIndex=0,index=0;\nconst modelSelect=document.querySelector(\'#model\');\nconst current=()=>data.blocks[blockIndex];\ndata.blocks.forEach((b,i)=>{const o=document.createElement(\'option\');o.value=i;o.textContent=b.id+\' · \'+b.reference.length+\'帧\';blockSelect.append(o)});\nconst percent=v=>v===undefined?\'—\':(100*v).toFixed(2)+\'%\';\nfunction geometry(){const b=current(),body=document.querySelector(\'#geometry\');body.replaceChildren();b.geometry.forEach(g=>{const tr=document.createElement(\'tr\');const values=[g.label+\' · \'+g.status,g.measured?percent(g.floor):\'未测\',g.measured?percent(g.wall):\'未测\',g.measured?g.p50_m.toFixed(6):\'未测\',g.measured?percent(g.floor_reduction)+\' / \'+percent(g.wall_reduction):\'未测\',g.evidence_note||(g.measured?\'已有实测 · 未验收\':\'未测\')];values.forEach((v,i)=>{const td=document.createElement(\'td\');td.textContent=v;if(i===5&&g.evidence_reasons)td.title=g.evidence_reasons.join(\'\\n\');if(i===0&&g.report){const a=document.createElement(\'a\');a.textContent=\' 几何报告\';a.href=g.report;td.append(a)}tr.append(td)});body.append(tr)});document.querySelector(\'#support\').textContent=b.id+\'固定几何分母：地面 \'+b.denominator.floor+\' 像素，墙面 \'+b.denominator.wall+\' 像素。不同区块分母独立，不作跨区块总体外推。\';document.querySelector(\'#pending\').textContent=data.pending.filter(p=>p.block_id===b.id).map(p=>p.id+\' · \'+p.status+\' · \'+p.message).join(\'；\')}\nfunction card(label,path,detail,link){const f=document.createElement(\'figure\'),c=document.createElement(\'figcaption\'),img=document.createElement(\'img\');c.textContent=label+\' — \'+detail;if(link){const a=document.createElement(\'a\');a.textContent=\' 评分报告\';a.href=link;c.append(a)}img.src=path;img.alt=label;img.loading=\'lazy\';f.append(c,img);panels.append(f)}\nfunction draw(){panels.replaceChildren();const b=current(),r=b.reference[index],error=document.querySelector(\'#error\').checked;document.querySelector(\'#identity\').textContent=b.id+\' · 观测组 \'+r.group;card(\'真实留出图像\',r.image,r.camera);b.models.forEach((m,i)=>{if(modelSelect.value!==\'all\'&&Number(modelSelect.value)!==i)return;const v=m.frames[index];card(m.label,error?v.error:v.image,m.status+\' · 本帧 \'+v.psnr.toFixed(2)+\' dB · \'+b.reference.length+\'帧均值 \'+m.mean.toFixed(2)+\' dB\',m.report)})}\nfunction changeBlock(){blockIndex=Number(blockSelect.value)||0;index=0;select.replaceChildren();current().reference.forEach((f,i)=>{const o=document.createElement(\'option\');o.value=i;o.textContent=String(i).padStart(3,\'0\')+\' · \'+f.camera;select.append(o)});select.value=0;modelSelect.replaceChildren();const all=document.createElement(\'option\');all.value=\'all\';all.textContent=\'全部已有结果\';modelSelect.append(all);current().models.forEach((m,i)=>{const o=document.createElement(\'option\');o.value=String(i);o.textContent=m.label+\' · \'+m.status;modelSelect.append(o)});modelSelect.value=\'all\';geometry();draw()}\nblockSelect.onchange=changeBlock;modelSelect.onchange=draw;select.onchange=()=>{index=Number(select.value);draw()};document.querySelector(\'#error\').onchange=draw;\ndocument.querySelector(\'#prev\').onclick=()=>{index=(index+current().reference.length-1)%current().reference.length;select.value=index;draw()};document.querySelector(\'#next\').onclick=()=>{index=(index+1)%current().reference.length;select.value=index;draw()};changeBlock();</script></html>'

resolution_note = ('<p>新候选measuredseed_anneal_v1已完成15000步切向尺度上限0.12m→0.03m退火训练（warm_opacity_cap恒为1.0）；下方以640×540固定DEV20逐帧比较。'
                   '另有独立1280×1080评估：%.6f dB（<a href="%s">1280评分报告</a>），不混入640表格。'
                   '当前仍为未通过验收的诊断候选。</p>') % (resolution_psnr, resolution_link)
body = body.replace('<table>', resolution_note + '<table>', 1)
out.write_text(body.replace('PAYLOAD', payload))
validation = dict(status='PASS', cpu_only=True, generated_or_modified_images=0, html_sha256=sha(out),
    all_linked_files_exist=True, linked_files=len(links),
    pending=pending,
    blocks=[dict(id=b['id'], models=len(b['models']), paired_frames=len(b['reference']),
        exact_camera_GT_mask_identity=True, geometry_fixed_denominator=b['denominator'],
        model_reports=[m['report'] for m in b['models']]) for b in blocks],
    assets=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(links)])
# Print CPU evidence; do not overwrite an earlier task's validation artefacts.
print(json.dumps({key: value for key, value in validation.items() if key != 'assets'}, ensure_ascii=False))
print(out, [(b['id'], len(b['models']), len(b['reference'])) for b in blocks], 'links', len(links))
