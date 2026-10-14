import json,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'runs/fidelity_v2/work_state.json'
d=json.loads(p.read_text());d['updated_at']=datetime.datetime.now().astimezone().isoformat()
for item in d.get('experiments',[]):
 base=root/item['path']; files=list(base.glob('block_*/metrics.json'))
 if files:
  item['completed_blocks']=[f.parent.name for f in sorted(files)]
  item['status']='complete' if len(files)==28 else 'completed_local_diagnostic'
  item.pop('pid',None);item.pop('block',None)
  reports=[json.loads(f.read_text()) for f in files]
  n=sum(r['protocols']['ego_v2']['images'] for r in reports)
  item['images']=n;item['ego_v2']={'images':n,'psnr_db_mean':sum(r['protocols']['ego_v2']['psnr_db_mean']*r['protocols']['ego_v2']['images'] for r in reports)/n,'summary_protocol':'image-weighted; full per-pixel statistics remain in source metrics'}
d['status']='in_progress';d['high_fidelity_accepted']=False
d['latest_completed_pilot']={'id':'block000_depthv3_seed_aa_cap003','training':'runs/fidelity_v2/train_block000_depthv3_seed_aa_cap003/training_result.json','iterations':15000,'gaussians':2499616,'psnr_1280_db':17.542795694823777,'original_psnr_1280_db':24.30860442901614,'conditioned_distance_p50_m':.31731943413615227,'promotion_allowed':False,'geometry_report':'reports/surface_quality_v2_block000_seed_aa_cap003.json'}
d['viewer_large_model']={'status':'CAPACITY_CAPTURE_PASS','loaded_gaussians':2498837,'paired_psnr_delta_db':-.011003552342867096,'sampled_board_memory_max_mib':1519,'scope':'one fixed 640x540 frame; capacity and parity, not reconstruction quality'}
d['active_work']=['independent block000 paired acceptance artifact','compare fixed old/v3 depth teacher domains on block000','training gradient/objective and warm initialization budget investigation','combined block000/block015 visual review']
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
entry='''

2026-10-09 14:29 继续执行：000全分辨率试点与查看器容量检查完成
000直接激光种子试点实际完成15000步、134次refine，79406→2499616高斯；所有自动评估已完成。640固定20帧：新17.699661dB、原单块24.404413dB、旧整图15.259180dB。1280固定20帧：新17.542796dB、原单块24.308604dB，下降6.765809dB。alpha有效区覆盖约78.77%。固定旧参考的条件距离p50=0.317319m，旧整图0.303605m；地面错误/缺失率84.83%（相对仅改善4.35%），墙面84.35%（恶化7.87%）。地面分母仅725像素，不能推广为整块所有地面。没有通过高保真验收，不启动28块放量。
训练峰值torch allocated=5394204160B、reserved=6685720576B；本次未记录独立整卡训练峰值，不能与浏览器采样峰值混用。checkpoint与全参数导出真实性通过，证据reports/block000_seed_aa_cap003_checkpoint.json；几何reports/surface_quality_v2_block000_seed_aa_cap003.json。
查看器单帧大模型检查实际加载2498837高斯，browser-native配对差-0.011004dB，整卡显存采样峰值1519MiB，浏览器及服务进程已退出并释放GPU锁。这证明本机可加载该候选和显示一致性，不证明候选几何通过。
接下来分别核查000新旧深度参考的共同支持域、训练中不透明度饱和与致密化容量分配，并修复大模型warm初始化仍误用旧读取预算的问题。旧bag、模型、参考与失败实验均保留。
'''
with (root/'status.txt').open('a') as f:f.write(entry)
with (root/'docs/FIDELITY_ROUND_V2.md').open('a') as f:f.write(entry)
print('Updated status, engineering log, and work_state; no running GPU job recorded.')
