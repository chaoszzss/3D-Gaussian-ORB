"""Read-only coverage overlays and PCA estimability of the frozen300 points.

No feature detection, matching, tracking, triangulation, or integration occurs.
"""
import os,sys,json,hashlib,csv,resource,time,itertools,html
from pathlib import Path
from collections import Counter
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];OUT=BASE/'coverage_diagnostics'
if OUT.exists():raise FileExistsError('Do not overwrite a diagnostic report')
OUT.mkdir();(OUT/'overlays').mkdir();os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np,cv2
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
cv2.setNumThreads(3);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]))
resource.setrlimit(resource.RLIMIT_AS,(2800*1024**2,resource.RLIM_INFINITY))
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.titlesize':13,'savefig.facecolor':'white'})
START=time.monotonic();RADII=(.1,.25,.5,1.)
THRESHOLDS=dict(radii_m=list(RADII),minimum_external_neighbors=6,minimum_distinct_spacing_m=.01,
    maximum_smallest_to_middle_eigenvalue_ratio=.1,minimum_middle_to_largest_eigenvalue_ratio=.1,
    minimum_middle_eigenvalue_m2=.0001,plane_rms_limit='min(0.03m,0.05*radius)',
    stable_normal_angle_deg=10.,stable_requires_two_adjacent_radii=True,
    corroborated_requires_two_source_triplets_in_each_neighborhood=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def quantiles(x):
    return dict(zip(('min','median','p95','max'),map(float,np.percentile(x,[0,50,95,100])))) if len(x) else None
HANDOFF=BASE/'validated_handoff';manifest=read(HANDOFF/'handoff.json');source=read(HANDOFF/'tracks.json')
tracks=source['tracks'];frames=source['train_frame_table'];parent=read(BASE/'triplet_plan.json');source_report=read(HANDOFF/'report.json')
dataset=Path(source_report['source_dataset']);protected={}
for entry in manifest['canonical_files'].values():
    assert sha(entry['path'])==entry['sha256'];protected[entry['path']]=entry['sha256']
protected[str(HANDOFF/'handoff.json')]=sha(HANDOFF/'handoff.json')
for p,h in read(BASE/'protected_inputs_before.json').items():assert sha(ROOT/p)==h;protected[str(ROOT/p)]=h
for p,h in source_report['source_sha256'].items():assert sha(p)==h;protected[p]=h
for p,h in source_report['train_asset_sha256'].items():assert sha(p)==h;protected[p]=h
xyz=np.array([t['xyz'] for t in tracks]);band=np.array([t['geometric_floor_band'] for t in tracks],bool)
assert len(xyz)==300 and band.sum()==23
with np.load(HANDOFF/'tracks.npz',allow_pickle=False) as a:
    assert np.array_equal(a['xyz'],xyz) and not a['normal_valid'].any()
def pca_metrics(points,radius):
    centered=points-points.mean(0);cov=centered.T@centered/len(points)
    values,vectors=np.linalg.eigh(cov);values=np.maximum(values,0);normal=vectors[:,0]
    if normal[np.argmax(np.abs(normal))]<0:normal=-normal
    small_middle=float(values[0]/max(values[1],1e-30));middle_large=float(values[1]/max(values[2],1e-30))
    rms=float(np.sqrt(values[0]))
    passed=bool(values[1]>=.0001 and small_middle<=.1 and middle_large>=.1 and rms<=min(.03,.05*radius))
    return dict(eigenvalues_m2=values.tolist(),smallest_middle_ratio=small_middle,middle_largest_ratio=middle_large,
        plane_rms_m=rms,diagnostic_unoriented_normal=normal.tolist(),shape_thresholds_pass=passed)
# Analytic checks verify the report logic without changing any data.
grid=np.array([[x,y,0.] for x,y in itertools.product((-.1,0.,.1),repeat=2)])
assert pca_metrics(grid,.5)['shape_thresholds_pass']
assert not pca_metrics(np.array([[x,0.,0.] for x in np.linspace(-.2,.2,9)]),.5)['shape_thresholds_pass']
tree=cKDTree(xyz);all_dist,all_ids=tree.query(xyz,k=len(xyz),workers=1)
per_point=[];candidate_masks=[];radius_summaries=[]
for radius in RADII:
    entries=[];qualified=[]
    for i in range(len(xyz)):
        within=[int(j) for dist,j in zip(all_dist[i],all_ids[i]) if dist<=radius and j!=i]
        selected=[]
        for j in within:
            if np.linalg.norm(xyz[j]-xyz[i])<.01:continue
            if all(np.linalg.norm(xyz[j]-xyz[k])>=.01 for k in selected):selected.append(j)
        enough=len(selected)>=6;diversity=len({tracks[j]['triplet_id'] for j in [i]+selected})
        metrics=pca_metrics(xyz[[i]+selected],radius) if enough else None
        valid=bool(enough and metrics['shape_thresholds_pass'])
        row=dict(track_id=i,radius_m=radius,raw_external_neighbor_count=len(within),distinct_external_neighbor_count=len(selected),
            selected_neighbor_track_ids=selected,source_triplet_diversity=diversity,enough_density=enough,
            local_shape_candidate=valid,pca=metrics)
        entries.append(row);qualified.append(valid)
    candidate_masks.append(np.array(qualified,bool));per_point.append(entries)
    radius_summaries.append(dict(radius_m=radius,
        all_neighbor_counts=quantiles([e['distinct_external_neighbor_count'] for e in entries]),
        band_neighbor_counts=quantiles([entries[i]['distinct_external_neighbor_count'] for i in np.flatnonzero(band)]),
        all_enough_density=sum(e['enough_density'] for e in entries),band_enough_density=sum(entries[i]['enough_density'] for i in np.flatnonzero(band)),
        all_local_shape_candidates=sum(qualified),band_local_shape_candidates=int(np.array(qualified)[band].sum())))
stability=[];stable_any=np.zeros(len(xyz),bool);corroborated_any=np.zeros(len(xyz),bool)
for i in range(len(xyz)):
    comparisons=[]
    for ri in range(len(RADII)-1):
        a,b=per_point[ri][i],per_point[ri+1][i]
        if not a['local_shape_candidate'] or not b['local_shape_candidate']:continue
        na=np.array(a['pca']['diagnostic_unoriented_normal']);nb=np.array(b['pca']['diagnostic_unoriented_normal'])
        angle=float(np.degrees(np.arccos(np.clip(abs(na@nb),0,1))))
        stable=angle<=10;corroborated=bool(stable and a['source_triplet_diversity']>=2 and b['source_triplet_diversity']>=2)
        stable_any[i]|=stable;corroborated_any[i]|=corroborated
        comparisons.append(dict(radii_m=[RADII[ri],RADII[ri+1]],unoriented_normal_angle_deg=angle,stable=stable,multiple_triplets=corroborated))
    stability.append(dict(track_id=i,geometric_floor_band=bool(band[i]),stable_across_adjacent_scales=bool(stable_any[i]),
        stable_and_multiple_triplets=bool(corroborated_any[i]),comparisons=comparisons))
# Six-neighbor distance uses the same1cm separation rule, with no fixed cutoff.
six_dist=[]
for i in range(len(xyz)):
    selected=[]
    for j in all_ids[i]:
        if j==i or np.linalg.norm(xyz[j]-xyz[i])<.01:continue
        if all(np.linalg.norm(xyz[j]-xyz[k])>=.01 for k in selected):selected.append(int(j))
        if len(selected)==6:break
    six_dist.append(float(np.linalg.norm(xyz[selected[-1]]-xyz[i])) if len(selected)==6 else None)
all_six=np.array(six_dist,float)
emit(OUT/'neighborhoods.json',dict(thresholds=THRESHOLDS,per_radius=per_point,stability=stability,
    sixth_distinct_neighbor_distance_m=six_dist,normal_valid_written_back=False))
with (OUT/'neighborhoods.csv').open('x',newline='') as f:
    keys=['track_id','geometric_floor_band','radius_m','raw_external_neighbor_count','distinct_external_neighbor_count','source_triplet_diversity','enough_density','local_shape_candidate','plane_rms_m','smallest_middle_ratio','middle_largest_ratio']
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
    for entries in per_point:
        for e in entries:
            row={k:e[k] for k in keys if k in e};row['geometric_floor_band']=bool(band[e['track_id']])
            row.update({k:e['pca'][k] if e['pca'] else '' for k in keys[-3:]});writer.writerow(row)
centers=np.array([np.array(f['transform_matrix'])[:3,3] for f in frames])
fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
ax=axes[0,0]
for cid in sorted({f['camera_id'] for f in frames}):
    ids=[i for i,f in enumerate(frames) if f['camera_id']==cid];ax.plot(centers[ids,0],centers[ids,1],color='.65',lw=.7)
ax.scatter(xyz[~band,0],xyz[~band,1],s=12,c='#168ac4',label='Other visual points')
ax.scatter(xyz[band,0],xyz[band,1],s=30,c='#e68613',marker='^',label='Geometric height band (23)')
ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Measured visual points and TRAIN camera trajectory');ax.set_aspect('equal');ax.legend(fontsize=8)
ax=axes[0,1];ax.scatter(xyz[:,0],xyz[:,2],s=12,c=np.where(band,'#e68613','#168ac4'))
ax.plot(centers[:,0],centers[:,2],'.',color='.6',ms=2);ax.set(xlabel='World X (m)',ylabel='World Z (m)',title='Unmodified metric XYZ; no fitted floor')
ax=axes[1,0];bins=np.geomspace(max(.02,all_six.min()*.8),all_six.max()*1.1,22)
ax.hist(all_six,bins=bins,histtype='step',lw=2,color='#168ac4',label='All300')
ax.hist(all_six[band],bins=bins,histtype='step',lw=2,color='#e68613',label='Height-band23')
for radius in RADII:ax.axvline(radius,color='.6',lw=.6,ls='--')
ax.set_xscale('log');ax.set(xlabel='Distance to sixth distinct neighbor (m)',ylabel='Point count',title='Neighborhood density (1cm separation)');ax.legend()
ax=axes[1,1];x=np.arange(4)
ax.bar(x-.18,[r['all_local_shape_candidates'] for r in radius_summaries],.36,label='All shape candidates',color='#168ac4')
ax.bar(x+.18,[r['band_local_shape_candidates'] for r in radius_summaries],.36,label='Height-band candidates',color='#e68613',hatch='//')
ax.set_xticks(x,[str(r) for r in RADII]);ax.set(xlabel='PCA neighborhood radius (m)',ylabel='Point count',title='Local shape tests only; no normal_valid promotion');ax.legend(fontsize=8)
for ax in axes.ravel():ax.grid(alpha=.2)
fig.suptitle('Frozen300-point visual seed: coverage and normal estimability')
fig.savefig(OUT/'coverage_density.png',dpi=180);fig.savefig(OUT/'coverage_density.pdf');plt.close(fig)
overlay_records=[];occupied_cells=[]
for ti,triplet in enumerate(parent['triplets']):
    ids=triplet['train_indices'];subset=[t for t in tracks if t['triplet_id']==ti]
    fig,axes=plt.subplots(1,3,figsize=(19.2,6.4),constrained_layout=True)
    for j,(index,ax) in enumerate(zip(ids,axes)):
        f=frames[index];rgb=cv2.imread(str(dataset/f['file_path']))[:,:,::-1];mask=cv2.imread(str(dataset/f['mask_path']),0)
        assert rgb.shape==(1080,1280,3)
        ax.imshow(rgb,interpolation='nearest');shade=np.zeros((1080,1280,4));shade[mask==0]=[.25,.25,.25,.5];ax.imshow(shade,interpolation='nearest')
        for t in subset:
            u,v=t['uv'][j];color='#ffaf28' if t['geometric_floor_band'] else '#00e5ff'
            ax.scatter([u],[v],s=24,facecolors='none',edgecolors=color,linewidths=1.1)
            T=np.array(f['transform_matrix'])@np.diag([1.,-1.,-1.,1.]);p=np.linalg.inv(T)[:3]@np.r_[t['xyz'],1.]
            projected=np.array([f['fl_x']*p[0]/p[2]+f['cx']-.5,f['fl_y']*p[1]/p[2]+f['cy']-.5])
            ax.plot(projected[0],projected[1],'+',color=color,ms=3,mew=.7)
            if t['geometric_floor_band']:ax.annotate(str(t['track_id']),(u,v),xytext=(3,3),textcoords='offset points',color=color,fontsize=7)
        cells={(int(t['uv'][j][0]//80),int(t['uv'][j][1]//80)) for t in subset}
        occupied_cells.append(dict(triplet_id=ti,view=j,train_frame_index=index,file_id=f['file_path'],tracks=len(subset),occupied_80px_cells=len(cells)))
        ax.set_title('View%d | frame%d | %dns\n%s' % (j+1,index,f['timestamp_ns'],f['camera_id']))
        ax.set(xlim=(-.5,1279.5),ylim=(1079.5,-.5));ax.set_axis_off()
    nband=sum(t['geometric_floor_band'] for t in subset)
    fig.suptitle('Triplet%02d %s | %d retained tracks, %d geometric-band points\nCircle: observed TRAIN pixel; +: projected XYZ. Orange band is NOT a semantic floor label.' % (ti+1,triplet['camera_id'],len(subset),nband))
    filename='overlays/triplet_%02d_%s.png'%(ti+1,triplet['camera_id'])
    fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    overlay_records.append(dict(triplet_id=ti,plot=filename,train_frame_indices=ids,train_file_ids=[frames[i]['file_path'] for i in ids],
        retained_tracks=len(subset),geometric_floor_band=nband,track_ids=[t['track_id'] for t in subset]))
    print('OVERLAY',ti+1,'/32',len(subset),'points',nband,'band',flush=True)
def occupied(points,cell):return len({tuple(v) for v in np.floor(points[:,:2]/cell).astype(int)})
rows=dict(status='READONLY_COVERAGE_AND_NORMAL_ESTIMABILITY_COMPLETE_QUALITY_UNASSESSED',tracks=300,geometric_floor_band=23,
    thresholds=THRESHOLDS,radius_summaries=radius_summaries,
    sixth_distinct_neighbor_distance_m=dict(all=quantiles(all_six),geometric_band=quantiles(all_six[band])),
    stable_across_adjacent_scales=dict(all=int(stable_any.sum()),geometric_band=int(stable_any[band].sum())),
    stable_and_multiple_source_triplets=dict(all=int(corroborated_any.sum()),geometric_band=int(corroborated_any[band].sum())),
    coverage=dict(triplets_total=32,triplets_with_tracks=sum(r['retained_tracks']>0 for r in overlay_records),
        triplets_with_geometric_band=sum(r['geometric_floor_band']>0 for r in overlay_records),
        unique_train_frames_with_observations=len({i for t in tracks for i in t['train_indices']}),
        geometric_band_xyz_bounds=[xyz[band].min(0).tolist(),xyz[band].max(0).tolist()],
        occupied_XY_cells={str(cell):dict(all=occupied(xyz,cell),geometric_band=occupied(xyz[band],cell)) for cell in (.25,.5,1.)},
        cell_count_is_not_surface_area_or_complete_coverage=True,per_image_80px_cell_counts=occupied_cells),
    overlays=overlay_records,outputs=['coverage_density.png','coverage_density.pdf','neighborhoods.json','neighborhoods.csv','index.html'],
    source_handoff_sha256=sha(HANDOFF/'handoff.json'),normal_valid_written_back=False,new_matching=False,new_triangulation=False,
    limitations=['PCA describes only local shape of existing uncertain3D estimates, not absolute surface correctness.',
        'All diagnostic normals are unoriented and are not a semantic floor label. Source normal_valid remains false.',
        'Single-triplet neighborhoods share pose/calibration error; cross-triplet corroboration is reported separately.',
        'Radius1m may span unrelated surfaces. Threshold and stability failures are not relaxed.',
        'Overlays show only actual observations of each track, not inferred visibility in unrelated frames.',
        'XY occupied cells and image marker footprints are not filled surface area. No plane was fabricated.'],
    resource=dict(cpu_threads_max=3,gpu_used=False,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,elapsed_s=time.monotonic()-START),
    protected_source_sha256=protected,script_sha256=sha(__file__))
for p,h in protected.items():assert sha(p)==h,p
assert rows['resource']['peak_rss_mb']<3072
emit(OUT/'report.json',rows)
cards='\n'.join('<section><h3>Triplet%02d — %d tracks / %d geometric-band</h3><a href="%s"><img loading="lazy" src="%s"></a></section>'%(r['triplet_id']+1,r['retained_tracks'],r['geometric_floor_band'],html.escape(r['plot']),html.escape(r['plot'])) for r in overlay_records)
text='<!doctype html><meta charset="utf-8"><title>Frozen visual-track diagnostics</title><style>body{max-width:1500px;margin:auto;font:16px sans-serif;padding:20px;background:#fafafa}img{width:100%}section{margin:35px 0}pre{white-space:pre-wrap}</style><h1>Frozen300 visual tracks /23 geometric-band points</h1><p>Read-only diagnostics. No new matching, no data integration, no normal_valid promotion. Orange is a geometric height band, not semantic floor.</p><p><a href="report.json">Report</a> · <a href="neighborhoods.csv">Neighborhood CSV</a></p><img src="coverage_density.png">'+cards
(OUT/'index.html').write_text(text)
print(json.dumps({k:rows[k] for k in ('tracks','geometric_floor_band','radius_summaries','sixth_distinct_neighbor_distance_m','stable_across_adjacent_scales','stable_and_multiple_source_triplets','resource')}),flush=True)
