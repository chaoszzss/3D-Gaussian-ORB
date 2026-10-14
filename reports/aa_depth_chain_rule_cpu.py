"""CPU mathematical audit of the installed 2DGS low-pass branch (no kernel execution)."""
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
DTYPE = torch.float64

def depth_and_support(T, pixel):
    u, v, w = T
    p = torch.linalg.cross(pixel[0] * w - u, pixel[1] * w - v)
    s = p[:2] / p[2]
    depth = torch.dot(s, w[:2]) + w[2]
    xy = torch.stack((u[2], v[2])) / w[2]
    return depth, s.square().sum(), 2 * (xy-pixel).square().sum(), s

rows = []
for tilt in (80., 89., 89.9, 89.94):
    angle = math.radians(tilt)
    sigma, Z, f, cx, cy = .02, 10., 320., 320., 270.
    # Projected centre is .25 pixels left of a fixed integer ray.
    center = torch.tensor([-.25*Z/f, 0., Z], dtype=DTYPE)
    tx = torch.tensor([sigma*math.cos(angle), 0., -sigma*math.sin(angle)], dtype=DTYPE)
    ty = torch.tensor([0., sigma, 0.], dtype=DTYPE)
    basis = torch.stack((tx, ty, center), dim=1)
    K = torch.tensor([[f, 0., cx], [0., f, cy], [0., 0., 1.]], dtype=DTYPE)
    T = (K @ basis).requires_grad_()
    pixel = torch.tensor([320.,270.],dtype=DTYPE)
    z, rho3, rho2, s = depth_and_support(T, pixel)
    exact = torch.autograd.grad(z, T)[0]
    # Installed AA else-branch omits the derivative of s(T).
    recorded = torch.zeros_like(T)
    recorded[2] = torch.cat((s.detach(), s.new_ones(1)))
    numerical = torch.zeros_like(T)
    branches_stable = True
    eps = 1e-6
    for i in range(3):
        for j in range(3):
            plus, minus = T.detach().clone(), T.detach().clone()
            plus[i,j] += eps; minus[i,j] -= eps
            dp, a, b, _ = depth_and_support(plus,pixel)
            dm, c, d, _ = depth_and_support(minus,pixel)
            branches_stable &= bool(a>b and c>d and dp>.2 and dm>.2)
            numerical[i,j] = (dp-dm)/(2*eps)
    rows.append(dict(tilt_degrees=tilt, source_center_z_m=Z, ray_plane_depth_m=z.item(),
                     rho3d=rho3.item(), rho2d=rho2.item(), lowpass_selected=bool(rho2<rho3),
                     finite_difference_branch_stable=branches_stable,
                     exact_gradient=exact.tolist(), installed_else_formula_gradient=recorded.tolist(),
                     finite_difference_gradient=numerical.tolist(),
                     max_abs_exact_vs_finite_difference=float((exact-numerical).abs().max()),
                     relative_norm_error_installed_formula=float(torch.linalg.vector_norm(recorded-exact)/torch.linalg.vector_norm(exact))))
result=dict(status='CPU_CHAIN_RULE_MISMATCH_REPRODUCED_GPU_CONFIRMATION_PENDING', gpu_used=False,
            scope='Mathematical surrogate of transMat derivatives only; not proof of compiled CUDA behavior or scene-wide impact',
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (ROOT/'runtime/inria2dgs_src/cuda_rasterizer/forward.cu', ROOT/'runtime/inria2dgs_src/cuda_rasterizer/backward.cu')},
            cases=rows,
            next_test='GPU finite difference on the actual extension with fixed lowpass/visibility/order branches; then explicit experimental consistent forward/backward variant')
out=ROOT/'reports/aa_depth_chain_rule_cpu.json'
out.write_text(json.dumps(result,indent=2)+'\n')
for row in rows:
    print(row['tilt_degrees'],row['ray_plane_depth_m'],row['rho3d'],row['rho2d'],row['finite_difference_branch_stable'],row['relative_norm_error_installed_formula'])
