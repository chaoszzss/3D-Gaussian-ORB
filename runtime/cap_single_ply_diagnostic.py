"""Static footprint control. Preserve all rows and non-tangent bytes; not trained output."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cap_tangent_ply import log_cap_float32,verify_output
from package_garage import inspect_ply
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=a.input.resolve();out=a.output.resolve()
assert ROOT in src.parents and ROOT in out.parents and src.is_file() and not out.exists()
with src.open('rb') as f:spec=inspect_ply(f)
assert 0<spec['count']<=3_000_000 and src.stat().st_size==len(spec['header'])+spec['count']*spec['stride']
cols=[spec['names'].index(x) for x in ['scale_0','scale_1']];limit=log_cap_float32(.03);step=32768
out.mkdir(parents=True)
dest=out/'gaussians.ply'
with src.open('rb') as f,dest.open('xb') as g:
 assert f.read(len(spec['header']))==spec['header'];g.write(spec['header'])
 for start in range(0,spec['count'],step):
  n=min(step,spec['count']-start);raw=f.read(n*spec['stride']);v=np.frombuffer(raw,dtype='<f4').reshape(n,-1).copy()
  v[:,cols]=np.minimum(v[:,cols],limit);g.write(v.tobytes())
sh,dh,changes=verify_output(src,dest,spec,cols,limit,step)
report={'status':'static_footprint_diagnostic','trained_candidate':False,'high_fidelity_accepted':False,'source':str(src.relative_to(ROOT)),'source_sha256':sh,'output':str(dest.relative_to(ROOT)),'output_sha256':dh,'gaussians':spec['count'],'cap_m':.03,'only_modified_fields':['scale_0','scale_1'],'changed_axes_records':changes,'all_other_bytes_verified_identical':True,'all_records_retained':True}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
