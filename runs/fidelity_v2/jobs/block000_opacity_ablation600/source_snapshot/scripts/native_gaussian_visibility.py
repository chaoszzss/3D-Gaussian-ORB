"""Use the actual INRIA preprocessing radius to select inference-active records.

Each native radius is computed independently before binning, so chunking this
geometry pass preserves the exact active set and original record/tie order.
No model file is reduced or modified. Dummy black RGB is never an output map.
"""
import hashlib
import numpy as np


def native_visible_indices(parameters, camera, device, chunk_size=200000, *, backend='original'):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 500000:
        raise ValueError('Native geometry chunk must contain 1..500000 records')
    import torch
    from orbvi_gaussian.inria_surfel_backend import render_surfel, nerfstudio_c2w_to_cv_w2c
    c2w = camera.camera_to_worlds.to(device=device, dtype=torch.float32)
    if len(c2w) != 1:
        raise ValueError('One fixed capture camera is required')
    view = nerfstudio_c2w_to_cv_w2c(c2w)
    K = camera.get_intrinsics_matrices().to(device=device, dtype=torch.float32)
    width, height = int(camera.width.item()), int(camera.height.item())
    selections = []
    count = len(parameters['means'])
    with torch.inference_mode():
        for start in range(0, count, chunk_size):
            end = min(count, start+chunk_size)
            means = torch.as_tensor(parameters['means'][start:end], device=device)
            quats = torch.as_tensor(parameters['quats'][start:end], device=device)
            scales = torch.exp(torch.as_tensor(parameters['scales'][start:end], device=device))
            opacity = torch.sigmoid(torch.as_tensor(parameters['opacities'][start:end], device=device))
            dummy = torch.zeros((end-start,3),dtype=torch.float32,device=device)
            result = render_surfel(means,quats,scales,opacity,dummy,view,K,width,height,backend=backend)
            ids = torch.nonzero(result.info['radii'][0] > 0, as_tuple=False).flatten().cpu().numpy()
            selections.append(ids.astype(np.int64,copy=False)+start)
            del result, means, quats, scales, opacity, dummy
    selected = np.concatenate(selections) if selections else np.empty(0,np.int64)
    if len(selected)>1 and not np.all(selected[1:]>selected[:-1]):
        raise RuntimeError('Native active records lost original stable source order')
    return selected, {'protocol':'Exact original INRIA per-Gaussian radius>0; all source records considered',
        'inria_backend':backend,
        'source_records':count,'active_records':len(selected),'geometry_chunk_records':chunk_size,
        'active_index_sha256':hashlib.sha256(selected.astype('<i8',copy=False).tobytes()).hexdigest()}
