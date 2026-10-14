"""Fixed tangent cap must preserve every non-target bit and row provenance."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import cap_tangent_ply as M
from assemble_core_gaussians import ply_header, PROVENANCE, sha256


@pytest.fixture
def fixture():
    with tempfile.TemporaryDirectory(prefix='tangent_cap_test_v2_', dir=ROOT/'reports') as tmp:
        project = Path(tmp)
        # Scale columns intentionally non-adjacent; third axis interleaved with SH.
        names = ['x','y','z','f_dc_0','scale_2','f_dc_1','scale_0','f_dc_2',
                 'opacity','rot_0','rot_1','scale_1','rot_2','rot_3','f_rest_0']
        names += [f'f_rest_{i}' for i in range(1,9)]
        rows = np.arange(6*len(names),dtype='<f4').reshape(6,-1)/100
        rows[:,names.index('f_dc_0')] = -0.
        rows[:,names.index('rot_0')] = 1
        rows[:,names.index('opacity')] = [-100,-1,-0.,0.,1,100]
        scales = np.array([[.05,.08],[.11999999,.12],[.12000004,.3],[.4,.04],[.2,.4],[.001,.002]])
        for i in range(2):
            rows[:,names.index(f'scale_{i}')] = np.log(scales[:,i])
        rows[:,names.index('scale_2')] = np.log([.01,.2,1.,.015,.8,.009])
        header = ply_header(names,len(rows))
        source = project/'source.ply';source.write_bytes(header+rows.tobytes())
        origins = np.zeros(len(rows),dtype=PROVENANCE)
        origins['source_index'] = [0,1,0,1,0,1]
        origins['source_record_index'] = [37,17,41,29,113,53]
        origins['assigned_core'] = [15,16,15,16,15,16]
        origins['measured_support_distance_m'] = [.01,.02,.03,.1,.07,.15]
        np.save(project/'trace.npy',origins)
        parent = {'full':{'path':'source.ply','sha256':sha256(source),'gaussians':len(rows),
                  'provenance':{'path':'trace.npy','sha256':sha256(project/'trace.npy'),'gaussians':len(rows)}},
                  'provenance':{'source_index_to_id':['block_015','block_016'],'records_modified':False}}
        (project/'parent.json').write_text(json.dumps(parent))
        yield project,names,rows,header,parent


def run(fixture, output='capped', **kwargs):
    project,*_=fixture
    return M.cap_assembly('parent.json',output,cap_m=.12,project=project,**kwargs)


def test_exact_non_target_fields_scale_boundary_provenance_and_order(fixture):
    project,names,before,header,parent=fixture
    source_bytes=(project/'source.ply').read_bytes();origin_bytes=(project/'trace.npy').read_bytes()
    report=run(fixture,chunk_bytes=before.shape[1]*4)
    result=(project/report['full']['path']).read_bytes()
    assert result[:len(header)]==header
    after=np.frombuffer(result[len(header):],dtype='<f4').reshape(before.shape)
    columns=[names.index('scale_0'),names.index('scale_1')]
    other=[i for i in range(len(names)) if i not in columns]
    assert before[:,other].tobytes()==after[:,other].tobytes()
    activated=np.exp(before[:,columns].astype(float))
    changed=activated>.12
    np.testing.assert_allclose(np.exp(after[:,columns].astype(float)),np.minimum(activated,.12),atol=1e-8,rtol=0)
    assert np.all(np.exp(after[:,columns].astype(float))<=.12)
    assert after[:,columns][~changed].tobytes()==before[:,columns][~changed].tobytes()
    assert report['changed_records']==int(changed.any(axis=1).sum())
    assert report['clipped_scale_0']==int(changed[:,0].sum())
    assert report['clipped_scale_1']==int(changed[:,1].sum())
    assert report['both_axes_clipped_records']==int(changed.all(axis=1).sum())
    assert (project/report['full']['provenance']['path']).read_bytes()==origin_bytes
    assert (project/'source.ply').read_bytes()==source_bytes
    assert report['records_deleted']==0
    assert report['provenance']['records_modified'] is True
    assert report['high_fidelity_accepted'] is False
    assert report['on_disk_all_records_verified']==len(before)
    assert sum(s['changed_records'] for s in report['sources'])==report['changed_records']
    assert json.loads((project/'capped/tangent_cap_audit.json').read_text())==report


def test_chunk_independence_and_exact_nearest_scale_quantiles(fixture):
    project,names,before,header,parent=fixture
    one=run(fixture,output='one',chunk_bytes=before.shape[1]*4)
    many=run(fixture,output='many',chunk_bytes=before.nbytes)
    assert one['full']['sha256']==many['full']['sha256']
    assert one['before']==many['before']
    values=np.exp(np.maximum(before[:,names.index('scale_0')],before[:,names.index('scale_1')]).astype(float))
    np.testing.assert_allclose(list(one['before']['max_tangent'].values()),
        np.quantile(values,[0,.5,.9,.95,.99,1],method='nearest'),atol=0,rtol=0)


def test_existing_output_and_preflight_do_not_write(fixture):
    project,*_=fixture
    report=run(fixture,output='preflight',preflight_only=True)
    assert report['status']=='preflight_passed'
    assert not (project/'preflight').exists()
    (project/'existing').mkdir();(project/'existing/sentinel').write_bytes(b'keep')
    with pytest.raises(M.AssemblyError,match='NEW directory'):
        run(fixture,output='existing')
    assert (project/'existing/sentinel').read_bytes()==b'keep'


@pytest.mark.parametrize('bad_cap',[0,-1,float('nan'),float('inf')])
def test_invalid_cap_cannot_publish(fixture,bad_cap):
    project,*_=fixture
    with pytest.raises(M.AssemblyError):
        M.cap_assembly('parent.json','bad',cap_m=bad_cap,project=project)
    assert not (project/'bad').exists()


@pytest.mark.parametrize('field',['sha256','provenance'])
def test_pinned_input_hashes_must_match(fixture,field):
    project,names,rows,header,parent=fixture
    if field=='sha256':parent['full']['sha256']='0'*64
    else:parent['full']['provenance']['sha256']='0'*64
    (project/'parent.json').write_text(json.dumps(parent))
    with pytest.raises(M.AssemblyError,match='SHA differs'):
        run(fixture)
    assert not (project/'capped').exists()


def test_memory_preflight_rejection_leaves_output_absent(fixture,monkeypatch):
    project,*_=fixture
    def fail(*args,**kwargs):
        raise M.AssemblyError('CPU memory preflight failed')
    monkeypatch.setattr(M,'resource_preflight',fail)
    with pytest.raises(M.AssemblyError,match='memory preflight'):
        run(fixture)
    assert not (project/'capped').exists()


def test_verifier_detects_non_target_corruption(fixture):
    project,names,rows,header,parent=fixture
    report=run(fixture)
    output=project/report['full']['path']
    with output.open('r+b') as f:
        f.seek(len(header)+names.index('scale_2')*4)
        f.write(np.float32(0).tobytes())
    with (project/'source.ply').open('rb') as f:
        spec=M.inspect_ply(f)
    with pytest.raises(M.AssemblyError,match='non-tangent attribute'):
        M.verify_output(project/'source.ply',output,spec,[names.index('scale_0'),names.index('scale_1')],
                        M.log_cap_float32(.12),3)
