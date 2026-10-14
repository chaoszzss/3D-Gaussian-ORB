#!/usr/bin/env python3
"""Build an isolated research-only AA depth VJP fix; never install/replace original.

Only the AA depth chain in backward.cu changes. Forward remains byte identical.
The copied LICENSE.md and all upstream notices remain in the isolated tree.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'diff_surfel_rasterization_aa_depth_vjp'
EXPECTED = {
    'cuda_rasterizer/forward.cu': '5f99ca4953baf8ce3699e0ded25f9c10ab94f3e3a697ff87e2502a36559cce53',
    'cuda_rasterizer/backward.cu': 'fccec0f96fcf71aee4e412138bb30777913d672f3b2b17e6260dcadb1658489f',
    'setup.py': 'b72115d8346aeac7b4c41623d184cd55da4ce2ea800c5341424b27b1578ee1bd',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_backward(source):
    old = '''\t\t\t\t// // Propagate the gradients of depth
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);'''
    new = '''\t\t\t\t// AA changes the coverage filter, but forward depth is still the
\t\t\t\t// ray-plane intersection. Include its full s(T) chain rule.
\t\t\t\t// Keep the original low-pass coverage/color mean2D VJP above.
\t\t\t\tconst float2 dL_ds = {dL_dz * Tw.x, dL_dz * Tw.y};
\t\t\t\tconst float dsx_pz = dL_ds.x / p.z;
\t\t\t\tconst float dsy_pz = dL_ds.y / p.z;
\t\t\t\tconst float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
\t\t\t\tconst float3 dL_dk = cross(l, dL_dp);
\t\t\t\tconst float3 dL_dl = cross(dL_dp, k);
\t\t\t\tconst float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
\t\t\t\tconst float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
\t\t\t\tconst float3 dL_dTw = {
\t\t\t\t\tpixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * s.x,
\t\t\t\t\tpixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * s.y,
\t\t\t\t\tpixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz};
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu.z);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv.z);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw.z);'''
    if source.count(old) != 1:
        raise ValueError('Require exact original AA block; refusing ambiguous patch')
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    original = ROOT / 'runtime/inria2dgs_src'
    target = ROOT / 'runtime/inria2dgs_aa_depth_vjp_src'
    for relative, digest in EXPECTED.items():
        if sha(original / relative) != digest:
            raise ValueError(f'Original source drift: {relative}')
    if target.exists():
        raise FileExistsError(f'Refuse to overwrite existing candidate: {target}')
    shutil.copytree(original, target, ignore=shutil.ignore_patterns(
        '.git', 'build', '*.egg-info', '__pycache__', '*.so', '*.pyc'))
    (target / 'diff_surfel_rasterization').rename(target / PACKAGE)
    setup = target / 'setup.py'
    setup.write_text(setup.read_text().replace('diff_surfel_rasterization', PACKAGE))
    backward = target / 'cuda_rasterizer/backward.cu'
    backward.write_text(patch_backward(backward.read_text()))
    assert (original / 'cuda_rasterizer/forward.cu').read_bytes() == (target / 'cuda_rasterizer/forward.cu').read_bytes()
    sources = ('cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu',
               'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/auxiliary.h',
               'rasterize_points.cu', 'rasterize_points.h', 'ext.cpp', 'setup.py', 'LICENSE.md')
    manifest = dict(status='PREPARED_GPU_VERIFICATION_PENDING', package=PACKAGE,
        created_utc=datetime.now(timezone.utc).isoformat(),
        original_source_sha256={p: sha(original / p) for p in sources},
        candidate_source_sha256={p: sha(target / p) for p in sources},
        original_provenance_sha256=sha(ROOT / 'runtime/inria2dgs_provenance.json'),
        build_script_sha256=sha(Path(__file__)),
        forward_byte_identical=True, installed_original_replaced=False,
        change='Only complete s(T) chain rule for AA depth; low-pass color/coverage VJP unchanged.',
        limitation='Not scene quality evidence. Red-wall nine rays have zero composed AA weight.')
    output = target / 'aa_depth_vjp_provenance.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.build:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-12.8',
                   TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='2')
        cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
        manifest['build_command'] = cmd
        manifest['build_environment'] = {key: env[key] for key in (
            'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS')}
        result = subprocess.run(cmd, cwd=target, env=env, check=False)
        manifest['build_returncode'] = result.returncode
        if result.returncode == 0:
            binaries = list((target / PACKAGE).glob('_C*.so'))
            assert len(binaries) == 1
            manifest.update(status='BUILT_GPU_VERIFICATION_PENDING',
                candidate_extension=str(binaries[0].relative_to(ROOT)),
                candidate_extension_sha256=sha(binaries[0]))
        output.write_text(json.dumps(manifest, indent=2) + '\n')
        for relative, digest in EXPECTED.items():
            assert sha(original / relative) == digest
        if result.returncode:
            raise SystemExit(result.returncode)
    print(output, flush=True)


if __name__ == '__main__':
    main()
