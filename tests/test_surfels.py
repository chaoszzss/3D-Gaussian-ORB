import numpy as np
import pytest
from orbvi_gaussian.surfels import estimate_normals,project_surfel_depth,SurfelDepthPolicy


def test_flat_measured_surface_depth_and_unknown_pixels():
    x,y=np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
    xyz=np.column_stack([x.ravel(),y.ravel(),np.full(400,3.)])
    normal,valid=estimate_normals(xyz)
    K=np.array([[20.,0,10],[0,20,10],[0,0,1]])
    depth,mask=project_surfel_depth(xyz,normal,valid,np.eye(4),K,20,20)
    assert mask.any() and not mask.all()
    assert np.allclose(depth[mask],3.)
    assert (depth[~mask]==0).all()


def test_grazing_and_invalid_normal_have_no_depth_support():
    xyz=np.array([[0.,0.,3.],[.3,.4,4.]])
    normals=np.array([[1.,0.,0.],[0,0,1.]])
    K=np.array([[20.,0,10],[0,20,10],[0,0,1]])
    _,mask=project_surfel_depth(xyz,normals,np.array([True,False]),np.eye(4),K,20,20)
    assert not mask.any()


def test_trusted_floor_recovers_measured_grazing_support_without_filling_holes():
    # CV y is world z; floor ray has cos incidence .08, legacy rejects it.
    pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    K=np.array([[100.,0,10],[0,100.,10],[0,0,1.]])
    points=np.array([[0.,-10.,-.8]])
    normals=np.array([[0.,0.,1.]])
    _,legacy=project_surfel_depth(points,normals,[True],pose,K,20,20)
    stats={}
    depth,valid=project_surfel_depth(points,normals,[True],pose,K,20,20,
                                   policy=SurfelDepthPolicy.trusted_ground(),diagnostics=stats)
    assert not legacy.any() and valid.sum()==1 and depth[2,10]==10.
    assert (depth[~valid]==0).all() and stats['trusted_ground_candidates']==1


def test_grazing_plane_correction_has_metric_cap_and_does_not_relax_walls():
    pose=np.eye(4);pose[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    K=np.array([[100.,0,10],[0,100.,10],[0,0,1.]])
    # Off-centre measured ray at v=2.4 would intersect pixel2 at z=9.5;
    # a 0.5m correction is not justified by our .25m local support cap.
    _,valid=project_surfel_depth([[0.,-10.,-.76]],[[0,0,1]],[True],pose,K,20,20,
                               policy=SurfelDepthPolicy.trusted_ground())
    assert not valid.any()
    _,wall=project_surfel_depth([[.8,0,10]],[[1,0,0]],[True],np.eye(4),K,20,20,
                              policy=SurfelDepthPolicy.trusted_ground())
    assert not wall.any()


def test_multilayer_front_cluster_is_measured_and_never_hops_over_occluder():
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    policy=SurfelDepthPolicy.trusted_ground()
    for zs,expected in (([3.,3.05,8.,8.1],3.),([3.,8.,8.1],0.),
                        ([3.,3.14,3.28,8.,8.1],0.)):
        points=np.c_[np.zeros((len(zs),2)),zs]
        depth,valid=project_surfel_depth(points,np.tile([0,0,1.],(len(zs),1)),
            np.ones(len(zs),bool),np.eye(4),K,20,20,policy=policy)
        assert depth[10,10]==expected and bool(valid[10,10])==(expected>0)
    points=np.array([[0,0,3.],[0,0,3.05],[0,0,8.],[0,0,8.1]])
    _,valid=project_surfel_depth(points,np.tile([0,0,1.],(4,1)),[False,False,True,True],
                               np.eye(4),K,20,20,policy=policy)
    assert not valid.any()


def test_multilayer_selection_crosses_chunk_boundaries_and_is_order_invariant():
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    points=np.zeros((65538,3));points[:,2]=-1
    points[[0,65535,65536,65537],2]=[8.,3.,3.05,8.1]
    normals=np.tile([0,0,1.],(len(points),1));known=np.ones(len(points),bool)
    args=(np.eye(4),K,20,20)
    forward=project_surfel_depth(points,normals,known,*args,policy=SurfelDepthPolicy.trusted_ground())
    reverse=project_surfel_depth(points[::-1],normals,known,*args,policy=SurfelDepthPolicy.trusted_ground())
    assert all(np.array_equal(a,b) for a,b in zip(forward,reverse))
    assert forward[0][10,10]==3.


def test_nonunit_and_zero_normals_are_handled_without_fake_support():
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    for normal,expected in (([0.,0.,100.],True),([0.,0.,0.],False),([0.,0.,np.nan],False)):
        depth,valid=project_surfel_depth([[0,0,3]], [normal],[True],np.eye(4),K,20,20)
        assert bool(valid[10,10])==expected
        assert np.isfinite(depth).all()


def test_v3_unknown_foreground_cannot_borrow_background_plane_tolerance():
    """Real v2 counterexample: unknown return at 13m, trusted floor at 13.2m."""
    R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    pose=np.eye(4);pose[:3,:3]=R
    K=np.array([[320.,0,2],[0,320.,0],[0,0,1.]])
    ray=np.array([0.,.1,1.])
    xyz=np.array([ray*13.,ray*13.2])@R.T
    normals=np.array([[0.,0,1],[0.,0,1]])
    for policy in (SurfelDepthPolicy(),SurfelDepthPolicy.trusted_ground(),
                   SurfelDepthPolicy.trusted_ground(front_cluster_min_points=1),
                   SurfelDepthPolicy.trusted_ground_v3(),
                   SurfelDepthPolicy.trusted_ground(multilayer_mode='legacy')):
        assert policy.version=='surfel_depth_v3'
        stats={}
        depth,valid=project_surfel_depth(xyz,normals,[False,True],pose,K,5,40,
                                       policy=policy,diagnostics=stats)
        assert not valid[32,2] and depth[32,2]==0
        if policy.ground_support:
            assert stats['source_occlusion_rejected']==1


def test_v3_nearest_trusted_plane_keeps_own_bounded_grazing_correction():
    # Unlike an unknown occluder, this actual nearest measured point can
    # support its own pixel-centre ray, even when correction > legacy .13m.
    R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    pose=np.eye(4);pose[:3,:3]=R
    K=np.array([[320.,0,2],[0,320.,0],[0,0,1.]])
    # u=2,v=32.49 rounds to (2,32), plane correction = .1990625m.
    xyz=np.array([[0.,13.*32.49/320.,13.]])@R.T
    depth,valid=project_surfel_depth(xyz,[[0,0,1]],[True],pose,K,5,40,
                                   policy=SurfelDepthPolicy.trusted_ground())
    assert valid[32,2]
    np.testing.assert_allclose(depth[32,2],13.*32.49/32.,rtol=0,atol=1e-6)


def test_v3_other_candidate_cannot_expand_foreground_plane_visibility():
    R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    pose=np.eye(4);pose[:3,:3]=R
    K=np.array([[320.,0,2],[0,320.,0],[0,0,1.]])
    # Nearest return has a trusted normal but is grazing this pixel ray.
    # Source at 13.05m is within its .10m source visibility allowance, but
    # its own plane correction cannot borrow nearest-return visibility.
    xyz=np.array([[0.,1.3,13.],[0.,13.05*32.49/320.,13.05]])@R.T
    stats={}
    depth,valid=project_surfel_depth(xyz,[[1,0,0],[0,0,1]],[True,True],pose,K,5,40,
                                   policy=SurfelDepthPolicy.trusted_ground(),diagnostics=stats)
    assert not valid.any() and (depth==0).all()
    assert stats['source_occlusion_rejected']==0 and stats['plane_occlusion_rejected']==1


@pytest.mark.parametrize('offset',[.00001,.05,.08,.1])
@pytest.mark.parametrize('min_points',[1,2])
@pytest.mark.parametrize('normal,known',[
    ([0.,0.,1.],False),([0.,0.,0.],True),([0.,0.,np.nan],True)])
def test_v3_unknown_foreground_blocks_even_close_background(offset,min_points,normal,known):
    R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    pose=np.eye(4);pose[:3,:3]=R
    K=np.array([[320.,0,2],[0,320.,0],[0,0,1.]])
    xyz=np.array([np.array([0.,.1,1.])*z for z in (13.,13.+offset)])@R.T
    stats={}
    depth,valid=project_surfel_depth(xyz,[normal,[0,0,1]],[known,True],pose,K,5,40,
        policy=SurfelDepthPolicy.trusted_ground_v3(front_cluster_min_points=min_points),
        diagnostics=stats)
    assert not valid.any() and (depth==0).all()
    assert stats['source_occlusion_rejected']==1
    assert stats['unknown_foreground_pixels']==1


@pytest.mark.parametrize('offset',[.05,.08,.1])
def test_default_legacy_keeps_historical_close_foreground_tolerance(offset):
    # Legacy explicitly retains its old uncertainty allowance. Strict unknown
    # foreground blocking is opt-in, not a silent rewrite of old baselines.
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    xyz=np.array([[0.,0,13.],[0.,0,13.+offset]])
    args=(xyz,[[0,0,1],[0,0,1]],[False,True],np.eye(4),K,20,20)
    default=project_surfel_depth(*args)
    explicit=project_surfel_depth(*args,policy=SurfelDepthPolicy())
    assert default[1][10,10]
    assert default[0][10,10]==np.float32(13.+offset)
    assert all(np.array_equal(a,b) for a,b in zip(default,explicit))


def test_v3_zero_source_tolerance_does_not_reject_float32_nearest():
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    source_z=13.0000001
    assert source_z>float(np.float32(source_z))
    depth,valid=project_surfel_depth([[0.,0,source_z]],[[0,0,1]],[True],
        np.eye(4),K,20,20,
        policy=SurfelDepthPolicy.trusted_ground_v3(foreground_occlusion_tolerance_m=0))
    assert valid[10,10] and depth[10,10]==np.float32(source_z)


def test_v3_unknown_foreground_blocks_across_chunks_and_input_order():
    K=np.array([[20.,0,10],[0,20,10],[0,0,1.]])
    xyz=np.zeros((65538,3));xyz[:,2]=-1
    xyz[[0,65537],2]=[13.,13.05]
    normals=np.tile([0.,0.,1.],(len(xyz),1))
    known=np.ones(len(xyz),bool);known[0]=False
    args=(np.eye(4),K,20,20)
    forward=project_surfel_depth(xyz,normals,known,*args,policy=SurfelDepthPolicy.trusted_ground_v3())
    reverse=project_surfel_depth(xyz[::-1],normals[::-1],known[::-1],*args,
        policy=SurfelDepthPolicy.trusted_ground_v3())
    assert all(np.array_equal(a,b) for a,b in zip(forward,reverse))
    assert not forward[1].any() and (forward[0]==0).all()


def test_v3_distant_unknown_foreground_does_not_gain_range_based_occlusion_tolerance():
    R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    pose=np.eye(4);pose[:3,:3]=R
    K=np.array([[320.,0,2],[0,320.,0],[0,0,1.]])
    xyz=np.array([np.array([0.,.1,1.])*z for z in (25.,25.2)])@R.T
    policy=SurfelDepthPolicy.trusted_ground_v3()
    assert policy.front_cluster_min_points==1
    depth,valid=project_surfel_depth(xyz,[[0,0,1],[0,0,1]],[False,True],pose,K,5,40,policy=policy)
    assert not valid.any() and (depth==0).all()
