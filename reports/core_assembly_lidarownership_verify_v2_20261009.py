"""Read-only, CPU verification of the frozen full28 LiDAR ownership candidate."""
import sys
from pathlib import Path
import json
import time
import resource
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from assemble_core_gaussians import inspect_ply, sha256, resource_preflight

out = ROOT / 'reports/core_assembly_lidarownership_verified_v2_20261009.json'
if out.exists():
    raise FileExistsError(out)
start = time.monotonic()
preflight = resource_preflight(out, disk_bytes=1024**2, memory_bytes=1024**3)
base = ROOT / 'maps/garage_surface_all28_lidarownership_v2_20261009'
a = json.loads((base/'assembly.json').read_text())
r = json.loads((base/'lidar_ownership_audit.json').read_text())
b = json.loads((ROOT/'maps/garage_surface_all28_noownership_v2_20261009/assembly.json').read_text())
assert a['provenance']['source_index_to_id'] == b['provenance']['source_index_to_id']

def records(path):
    with path.open('rb') as h:
        spec = inspect_ply(h)
    return np.memmap(path, dtype='<u4', mode='r', offset=len(spec['header']),
                     shape=(spec['count'], len(spec['names']))), spec['names']

assert sha256(ROOT/a['full']['path']) == a['full']['sha256']
assert sha256(ROOT/a['full']['provenance']['path']) == a['full']['provenance']['sha256']
full, names = records(ROOT/a['full']['path'])
trace = np.load(ROOT/a['full']['provenance']['path'], mmap_mode='r', allow_pickle=False)
old = np.load(ROOT/b['full']['provenance']['path'], mmap_mode='r', allow_pickle=False)
kept, oldkept, sources, per_source = [], [], [], []
offset, oldoffset = 0, 0
for i, source in enumerate(a['sources']):
    source_path = ROOT/source['path']
    assert sha256(source_path) == source['source_sha256'] == b['sources'][i]['source_sha256']
    raw, snames = records(source_path)
    assert names == snames
    n = source['retained_gaussians']
    tr = trace[offset:offset+n]
    assert np.all(tr['source_index'] == i)
    assert np.all(tr['assigned_core'] == int(source['id'].split('_')[1]))
    assert np.all(tr['measured_support_checked'])
    assert np.all(tr['measured_support_distance_m'] <= .2)
    for begin in range(0,n,32768):
        ids = tr['source_record_index'][begin:begin+32768]
        assert np.array_equal(full[offset+begin:offset+begin+len(ids)], raw[ids])
    mask = np.zeros(len(raw),bool)
    mask[tr['source_record_index']] = True
    assert int(mask.sum()) == n
    on = b['sources'][i]['retained_gaussians']
    omask = np.zeros(len(raw),bool)
    omask[old[oldoffset:oldoffset+on]['source_record_index']] = True
    assert not np.any(mask & ~omask)
    kept.append(mask);oldkept.append(omask);sources.append(raw)
    per_source.append({'source':source['id'],'retained':n,'removed_vs_noownership':int((omask & ~mask).sum())})
    offset += n;oldoffset += on
assert offset == len(full) == len(trace) == a['retained_gaussians']
xyzcols = [names.index(k) for k in ('x','y','z')]
witness_count, removed_survivors, maxdistance = 0, 0, 0.
floor_sets = [set(s['floor_tokens']) for s in r['sources']]
for i, descriptor in enumerate(r['witness_traces']):
    path = ROOT/descriptor['path']
    assert sha256(path) == descriptor['sha256']
    with np.load(path, allow_pickle=False) as t:
        ids=t['rejected_source_record'];wi=t['retained_witness_source_index'];wr=t['retained_witness_record']
        assert not np.any(kept[i][ids])
        removed_survivors += int(oldkept[i][ids].sum())
        deleted=np.zeros(len(sources[i]),bool);deleted[ids]=True
        assert np.array_equal(oldkept[i] & ~kept[i], oldkept[i] & deleted)
        witness_count += len(ids)
        for j in np.unique(wi):
            mask = wi == j
            assert floor_sets[i] & floor_sets[j]
            assert np.all(kept[j][wr[mask]])
            srcxyz = sources[i][ids[mask]][:,xyzcols].view('<f4').astype(np.float64)
            witnessxyz = sources[j][wr[mask]][:,xyzcols].view('<f4').astype(np.float64)
            distance = np.linalg.norm(srcxyz-witnessxyz,axis=1)
            np.testing.assert_allclose(distance,t['witness_distance_m'][mask],atol=1e-12,rtol=0)
            assert np.all(distance<=.1+1e-12)
            maxdistance=max(maxdistance,float(distance.max(initial=0)))
result={'status':'verified_pending_native_evaluation','gpu_used':False,'preflight':preflight,
        'ply':a['full'],'candidate_gaussians':a['candidate_gaussians'],
        'retained_gaussians':len(full),'all_retained_records_byte_exact':True,
        'all_source_sha_match_original_noownership':True,'all_witnesses_in_final_ply':True,
        'all_records_same_source_core':True,'all_same_floor_labels':True,
        'floor_geometry_independently_verified':False,'witnesses':witness_count,
        'ownership_rejected':sum(s['ownership_rejected_gaussians'] for s in a['sources']),
        'support_rejected':sum(s['support_rejected_gaussians'] for s in a['sources']),
        'removed_vs_noownership':removed_survivors,'max_witness_distance_m':maxdistance,
        'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'sources':per_source}
with out.open('x') as h:
    json.dump(result,h,indent=2);h.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('ply','sources','preflight')}))
