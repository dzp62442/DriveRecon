"""Full-intrinsics OpenCV cameras for the native surfel rasterizer."""

from dataclasses import dataclass
import torch


@dataclass
class RasterCamera:
    height: int
    width: int
    tanfovx: float
    tanfovy: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    center: torch.Tensor


def projection_from_intrinsics(k, shape, znear=0.01, zfar=1e8):
    h, w = shape
    p = k.new_zeros(4, 4, dtype=torch.float32)
    p[0, 0], p[1, 1] = 2 * k[0, 0] / w, 2 * k[1, 1] / h
    p[0, 1] = 2 * k[0, 1] / w
    # CUDA ndc2Pix(x,S) = ((x+1)*S-1)/2; backprojection uses integer pixels.
    p[0, 2], p[1, 2] = (2 * k[0, 2] + 1) / w - 1, (2 * k[1, 2] + 1) / h - 1
    p[2, 2], p[2, 3], p[3, 2] = zfar / (zfar-znear), -zfar*znear / (zfar-znear), 1
    return p


def make_camera(k, c2w, shape, znear=0.01, zfar=1e8):
    with torch.autocast(device_type=k.device.type, enabled=False):
        view = torch.linalg.inv(c2w.float()).T.contiguous()
        projection = projection_from_intrinsics(k.float(), shape, znear, zfar).T
        return RasterCamera(shape[0], shape[1], float(shape[1] / (2 * k[0, 0])),
                            float(shape[0] / (2 * k[1, 1])), view,
                            (view @ projection).contiguous(), c2w[:3, 3].float().contiguous())


def target_cameras(target, cfg):
    shape = target['image'].shape[-2:]
    return [make_camera(k, c2w, shape, cfg['znear'], cfg['zfar'])
            for k, c2w in zip(target['intrinsics_pixel'][0], target['extrinsics'][0])]
