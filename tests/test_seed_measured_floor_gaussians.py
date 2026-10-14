"""Observed-only seeding: floor/occlusion gates, TRAIN group counts and raw prefix."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import tempfile
import json
import copy
import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import seed_measured_floor_gaussians as M
from assemble_core_gaussians import ply_header,sha256,inspect_ply


@pytest.fixture
def scene():
    with tempfile.TemporaryDirectory(prefix='floorseed_fixture_v2_',dir=ROOT/'reports') as tmp:
        root=Path(tmp);warm=root/'warm';data=root/'data';warm.mkdir();data.mkdir()
        names=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
        names+=['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        raw=np.arange(len(names),dtype='<f4')[None]/100
        raw[0,:3]=[.055,0,0];raw[0,names.index('f_dc_0')]=-0.;raw[0,names.index('opacity')]=2.
        p=warm/'gaussians.ply';p.write_bytes(ply_header(names,1)+raw.tobytes())
        (warm/'manifest.json').write_text(json.dumps({'gaussians':1,'sha256':sha256(p),
            'status':'verified_lossless_raw_original_concatenation'}))
        # Rows 0/1 share a voxel; row2 is near warm; 3 ceiling; 4 wrong foreground;
        # 5 unknown depth; 6 ego; 7 invalid normal; 8 wall; 9 too close in world Z.
        xyz=np.array([[0,0,0],[.005,0,0],[.06,0,0],[0,0,2],[0,0,.3],[1,0,0],
                      [-1,0,0],[0,1,0],[0,-1,0],[0,0,.8],[0,.5,0]],dtype='<f4')
        normals=np.tile([0.,0.,1.],(len(xyz),1)).astype('<f4');normals[8]=[1,0,0];normals[10]=[0,0,-1]
        valid=np.ones(len(xyz),bool);valid[7]=False
        np.savez_compressed(data/'lidar_reference.npz',xyz=xyz,normals=normals,normal_valid=valid,
            source_block_index=np.zeros(len(xyz),np.uint8),source_row_index=np.arange(len(xyz),dtype=np.uint32))
        for name in ('images','masks','depth','depth_masks'):(data/name).mkdir()
        frames=[];teacher=[];assets={}
        for i,group in enumerate([10,10,10,10,20,30]):
            pose=np.eye(4);pose[2,3]=1
            f={'file_path':f'images/{i}.png','mask_path':f'masks/{i}.png','lidar_depth_path':f'depth/{i}.npy',
               'lidar_depth_mask_path':f'depth_masks/{i}.png','group_id':str(group),'camera_id':f'camera_{i%4}',
               'w':8,'h':8,'fl_x':2.,'fl_y':2.,'cx':4.5,'cy':4.5,'transform_matrix':pose.tolist()}
            rgb=np.zeros((8,8,3),np.uint8);rgb[:,:,0 if group==10 else 1 if group==20 else 2]=255
            mask=np.ones((8,8),np.uint8)*255;mask[4,2]=0
            known=np.ones((8,8),np.uint8)*255;known[4,6]=0
            Image.fromarray(rgb).save(data/f['file_path']);Image.fromarray(mask).save(data/f['mask_path'])
            Image.fromarray(known).save(data/f['lidar_depth_mask_path']);np.save(data/f['lidar_depth_path'],np.ones((8,8),np.float32))
            assets[f['file_path']]=sha256(data/f['file_path']);assets[f['mask_path']]=sha256(data/f['mask_path'])
            teacher.append({'file_path':f['file_path'],'depth_sha256':sha256(data/f['lidar_depth_path']),
                            'support_sha256':sha256(data/f['lidar_depth_mask_path'])});frames.append(f)
        held=copy.deepcopy(frames[0]);held['file_path']='images/heldout_DOES_NOT_EXIST.png';held['group_id']='40'
        meta={'orientation_override':'none','frames':frames+[held],
              'train_filenames':[f['file_path'] for f in frames],'val_filenames':[held['file_path']],'test_filenames':[]}
        (data/'transforms.json').write_text(json.dumps(meta))
        prep={'world_frame':'camera_init','units':'metres','copied_asset_hashes':assets,
              'generated_hashes':{'transforms.json':sha256(data/'transforms.json')},
              'teacher_provenance':{'policy':{'version':'surfel_depth_v3','ground_support':True,
                   'world_up':[0.,0.,1.],'foreground_occlusion_tolerance_m':.1},
                   'regenerated_all_frames':True,'reference_sha256':sha256(data/'lidar_reference.npz'),'frames':teacher}}
        (data/'preparation.json').write_text(json.dumps(prep))
        yield root,names,raw,xyz,meta,prep


def test_actual_reference_only_three_groups_color_balanced_and_warm_prefix_exact(scene):
    root,names,raw,xyz,meta,prep=scene
    original=(root/'warm/gaussians.ply').read_bytes()
    r=M.seed_measured_floor('warm/gaussians.ply','data','candidate',project=root,threads=1)
    assert r['counts']['seeds_appended']==2
    with np.load(root/'candidate/seed_provenance.npz') as p:
        assert p['reference_row'].tolist()==[0,10]
        np.testing.assert_array_equal(p['reference_xyz'],xyz[[0,10]])
        assert p['support_train_groups'].tolist()==[3,3]
        assert p['support_train_frames'].tolist()==[6,6]
        np.testing.assert_allclose(p['rgb_srgb'],np.full((2,3),1/3),atol=2e-8,rtol=0)
        assert np.unpackbits(p['train_group_support_bits'],axis=1,bitorder='little').sum(axis=1).tolist()==[3,3]
        assert np.unpackbits(p['train_frame_support_bits'],axis=1,bitorder='little').sum(axis=1).tolist()==[6,6]
        assert np.all(p['nearest_original_warm_center_m']>.02)
        q=p['quaternion_wxyz'];w,x,y,z=q.T
        derived=np.column_stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
        np.testing.assert_allclose(derived,p['normal_world'],atol=1e-7)
        np.testing.assert_allclose(derived,[[0,0,1],[0,0,1]],atol=1e-7)
    with (root/'candidate/gaussians.ply').open('rb') as f:
        spec=inspect_ply(f);payload=f.read()
    assert spec['count']==3
    assert payload[:raw.nbytes]==raw.tobytes()
    out=np.frombuffer(payload,dtype='<f4').reshape(3,-1)
    np.testing.assert_array_equal(out[1:,:3],xyz[[0,10]])
    np.testing.assert_allclose(np.exp(out[1:,[names.index('scale_0'),names.index('scale_1')]]),.02,atol=1e-8)
    np.testing.assert_allclose(np.exp(out[1:,names.index('scale_2')]),.001,atol=1e-9)
    np.testing.assert_allclose(1/(1+np.exp(-out[1:,names.index('opacity')])),.9,atol=1e-7)
    assert np.all(out[1:,[names.index(f'f_rest_{i}') for i in range(45)]]==0)
    assert (root/'warm/gaussians.ply').read_bytes()==original
    assert r['heldout_used_for_seed_geometry_or_color'] is False
    assert not any('heldout_DOES_NOT_EXIST' in key for key in r['inputs'])
    assert r['original_payload_all_bytes_compared'] is True
    assert json.loads((root/'candidate/floor_seed_audit.json').read_text())==r


def test_height_positive_z_egomask_unknown_and_occlusion_gates(scene):
    root,names,raw,xyz,meta,prep=scene
    frame=meta['frames'][0]
    depth=np.ones((8,8),np.float32);known=np.ones((8,8),np.uint8);mask=known.copy()
    known[4,6]=0;mask[4,2]=0
    ids,_,_=M.eligible_observations(xyz.astype(float),frame,depth,known,mask)
    assert 0 in ids and 10 in ids
    assert not set([3,4,5,6,9])&set(ids)
    unknown=np.zeros_like(known)
    assert len(M.eligible_observations(xyz,frame,depth,unknown,mask)[0])==0
    depth[:]=np.nan
    assert len(M.eligible_observations(xyz,frame,depth,known,mask)[0])==0


def test_horizontal_validity_sign_and_thirty_degree_boundary():
    normals=np.array([[0,0,1],[0,0,-2],[0,0,0],[1,0,0],
                      [np.sin(np.deg2rad(29)),0,np.cos(np.deg2rad(29))],
                      [np.sin(np.deg2rad(31)),0,np.cos(np.deg2rad(31))],[0,0,1.]])
    valid=np.ones(7,bool);valid[-1]=False
    ids,_=M.horizontal_reference_rows(np.zeros((7,3)),normals,valid)
    assert ids.tolist()==[0,1,4]
    q,n=M.floor_quaternions(normals[[1,4]])
    np.testing.assert_allclose(np.linalg.norm(q,axis=1),1,atol=1e-7)
    w,x,y,z=q.T
    np.testing.assert_allclose(np.column_stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]),n,atol=1e-7)


def test_group_leakage_and_four_cameras_do_not_form_three_groups(scene):
    root,names,raw,xyz,meta,prep=scene
    bad=copy.deepcopy(meta);bad['frames'][-1]['group_id']='10'
    with pytest.raises(M.AssemblyError,match='group leakage'):M.train_groups(bad)
    one=copy.deepcopy(meta);one['frames']=one['frames'][:4];one['train_filenames']=one['train_filenames'][:4];one['val_filenames']=[]
    with pytest.raises(M.AssemblyError,match='3..65535'):M.train_groups(one)


def test_stable_voxel_selects_actual_lowest_reference_row_never_centroid():
    xyz=np.array([[.001,0,0],[.019,0,0],[.02,0,0],[-.001,0,0]])
    ids=np.array([9,4,8,2]);chosen=M.stable_voxel_rows(xyz,ids)
    assert ids[chosen].tolist()==[2,4,8]
    assert np.array_equal(xyz[chosen],xyz[[3,1,2]])


def test_no_silent_seed_cap():
    M.enforce_seed_limit(500000)
    with pytest.raises(M.AssemblyError,match='refusing to truncate'):M.enforce_seed_limit(500001)


def test_existing_output_and_thread_budget_fail_before_writing(scene):
    root,*_=scene
    (root/'old').mkdir();(root/'old/sentinel').write_bytes(b'keep')
    with pytest.raises(M.AssemblyError,match='NEW directory'):
        M.seed_measured_floor('warm/gaussians.ply','data','old',project=root)
    assert (root/'old/sentinel').read_bytes()==b'keep'
    with pytest.raises(M.AssemblyError,match='1..3'):
        M.seed_measured_floor('warm/gaussians.ply','data','bad',threads=4,project=root)
    assert not (root/'bad').exists()


def test_changed_teacher_and_legacy_policy_rejected(scene):
    root,names,raw,xyz,meta,prep=scene
    np.save(root/'data'/meta['frames'][0]['lidar_depth_path'],np.zeros((8,8),np.float32))
    with pytest.raises(M.AssemblyError,match='TRAIN asset SHA mismatch'):
        M.seed_measured_floor('warm/gaussians.ply','data','bad',project=root)
    assert not (root/'bad').exists()
    prep['teacher_provenance']['policy']['version']='surfel_depth_v2'
    (root/'data/preparation.json').write_text(json.dumps(prep))
    with pytest.raises(M.AssemblyError,match='v3'):
        M.seed_measured_floor('warm/gaussians.ply','data','bad2',project=root)


def test_resource_fail_is_before_any_publication(scene,monkeypatch):
    root,*_=scene
    def fail(*args,**kwargs):raise M.AssemblyError('CPU memory preflight failed')
    monkeypatch.setattr(M,'resource_preflight',fail)
    with pytest.raises(M.AssemblyError,match='memory preflight'):
        M.seed_measured_floor('warm/gaussians.ply','data','bad',project=root)
    assert not (root/'bad').exists()
