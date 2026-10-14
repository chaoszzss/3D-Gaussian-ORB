"""S0: exact pixel IDs/front layers, before changing depth support policy.

CPU reproduction of the vendored kernel's ndc2pix; no CUDA launch is needed.
CV K uses integer centres; the exported NS K adds .5 to both principal points.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from orbvi_gaussian.inria_surfel_backend import _projection, nerfstudio_c2w_to_cv_w2c
from orbvi_gaussian.points import project_depth
from orbvi_gaussian.surfels import project_surfel_depth


def native_pixels(camera_points, K, width, height, pixel_center):
    points = torch.as_tensor(np.c_[camera_points, np.ones(len(camera_points))], dtype=torch.float64)
    projected = points @ _projection(torch.as_tensor(K, dtype=torch.float64), width, height, pixel_center).T
    ndc = (projected[:, :2] / projected[:, 3:4]).numpy()
    # cuda_rasterizer/auxiliary.h ndc2Pix and forward.cu ndc2pix.
    return ((ndc + 1) * np.array([width, height]) - 1) / 2


def roundtrip_measurements(width, height, principal):
    K = np.array([[width * .5, 0, width / 2 + principal[0]],
                  [0, width * .51, height / 2 + principal[1]], [0, 0, 1.]])
    u, v = np.meshgrid(np.linspace(1, width - 2, 13).astype(int),
                       np.linspace(1, height - 2, 11).astype(int))
    ids = np.c_[u.ravel(), v.ravel()]
    # Exercise either side of the rounding boundary without ambiguous exact ties.
    jitter = np.resize(np.array([-.499, 0., .499]), ids.shape)
    uv = ids + jitter
    z = np.linspace(.5, 30, len(ids))
    points = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0],
                   (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(ids))] * z[:, None]
    ns_K = K.copy(); ns_K[:2, 2] += .5
    cv_uv = native_pixels(points, K, width, height, 0.)
    ns_uv = native_pixels(points, ns_K, width, height, .5)
    np.testing.assert_allclose(cv_uv, uv, atol=5e-13, rtol=0)
    np.testing.assert_allclose(ns_uv, uv, atol=5e-13, rtol=0)
    assert np.array_equal(np.floor(cv_uv + .5).astype(int), ids)
    assert np.array_equal(np.floor(ns_uv + .5).astype(int), ids)
    # A missing principal-point conversion must be detected by this fixture.
    wrong = native_pixels(points, K, width, height, .5)
    assert not np.array_equal(np.floor(wrong + .5).astype(int), ids)
    return dict(tested_indices=int(len(ids)*2), wrong_index_count=int(
        np.any(np.floor(cv_uv+.5).astype(int)!=ids,axis=1).sum()+
        np.any(np.floor(ns_uv+.5).astype(int)!=ids,axis=1).sum()),
        max_coordinate_error=float(max(np.abs(cv_uv-uv).max(),np.abs(ns_uv-uv).max())),
        width=width,height=height,principal_offset=list(principal))


def front_layer_measurements():
    width, height = 17, 13
    K = np.array([[12., 0, 8.], [0, 11., 6.], [0, 0, 1.]])
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    ray = np.c_[(u.ravel() - 8) / 12, (v.ravel() - 6) / 11, np.ones(u.size)]
    points = np.concatenate([ray * 8, ray * 3, ray * 12])
    cv_pose = np.eye(4)
    cv_pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cv_pose[:3, 3] = [13, -7, 2]
    ns_pose = cv_pose @ np.diag([1., -1., -1., 1.])
    w2c = nerfstudio_c2w_to_cv_w2c(torch.tensor(ns_pose)[None])[0].numpy()
    world = points @ cv_pose[:3, :3].T + cv_pose[:3, 3]
    np.testing.assert_allclose(world @ w2c[:3, :3].T + w2c[:3, 3], points, atol=2e-15, rtol=0)
    ns_K = K.copy(); ns_K[:2, 2] += .5
    pixels = np.floor(native_pixels(points, ns_K, width, height, .5) + .5).astype(int)
    native_front = np.full(width * height, np.inf, np.float32)
    np.minimum.at(native_front, pixels[:, 1] * width + pixels[:, 0], points[:, 2].astype(np.float32))
    depth, valid = project_depth(world, cv_pose, K, width, height)
    normals = np.tile([0., 0., 1.], (len(points), 1))
    plane, supported = project_surfel_depth(world, normals, np.ones(len(points), bool), cv_pose, K, width, height)
    assert np.array_equal(depth, native_front.reshape(height, width))
    assert np.array_equal(plane, depth)
    assert np.array_equal(supported, valid) and valid.all()
    reference=native_front.reshape(height,width)
    return dict(tested_values=int(depth.size*2),equal_values=int((depth==reference).sum()+(plane==reference).sum()),
                mask_equal=bool(np.array_equal(supported,valid)),layer_depths_m=[3.,8.,12.])


@pytest.mark.parametrize("width,height", [(17, 13), (640, 540), (1280, 1080)])
@pytest.mark.parametrize("principal", [(0., 0.), (.23, -.37)])
def test_cv_ns_kernel_roundtrip_has_zero_wrong_pixel_ids(width, height, principal):
    assert roundtrip_measurements(width,height,principal)['wrong_index_count']==0


def test_pose_conversion_and_exact_front_layer_closure():
    measured=front_layer_measurements()
    assert measured['equal_values']==measured['tested_values']


if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Run S0 assertions and save actual index/front-layer measurements')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    files=['tests/test_pixel_convention_closure.py','src/orbvi_gaussian/points.py',
           'src/orbvi_gaussian/surfels.py','src/orbvi_gaussian/inria_surfel_backend.py',
           'runtime/inria2dgs_src/cuda_rasterizer/forward.cu',
           'runtime/inria2dgs_src/cuda_rasterizer/auxiliary.h']
    def hashes():
        return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    before=hashes()
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(Path(__file__).resolve())]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    index=[roundtrip_measurements(w,h,p) for w,h in [(17,13),(640,540),(1280,1080)]
           for p in [(0.,0.),(.23,-.37)]]
    front=front_layer_measurements()
    if hashes()!=before:
        raise RuntimeError('S0 source code changed during measurement')
    report=dict(status='passed',wrong_index_count=sum(row['wrong_index_count'] for row in index),
        tested_index_count=sum(row['tested_indices'] for row in index),
        foreground_array_equal_fraction=front['equal_values']/front['tested_values'],
        index_cases=index,foreground_measurement=front,commands=[command,sys.argv],
        source_sha256=before,pytest_returncode=result.returncode,pytest_stdout=result.stdout,
        scope='CPU exact pixel IDs, pose roundtrip and native ndc2pix front-layer oracle; not a CUDA render parity claim')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','wrong_index_count','foreground_array_equal_fraction')}))
