"""OpenCV pixel intrinsics and differentiable metric-depth backprojection."""

import torch


def scale_intrinsics(intrinsics, source_shape, target_shape):
    scaled = intrinsics.clone()
    scaled[..., 0, :] *= target_shape[1] / source_shape[1]
    scaled[..., 1, :] *= target_shape[0] / source_shape[0]
    return scaled


def backproject(depth, intrinsics, c2w, uv_shift=None):
    """Depth [..., H, W] -> world points [..., H, W, 3]; shifts in grid pixels."""
    h, w = depth.shape[-2:]
    with torch.autocast(device_type=depth.device.type, enabled=False):
        depth = depth.float()
        y, x = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=torch.float32),
            torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
        uv = torch.stack((x, y), -1).expand(*depth.shape, 2)
        if uv_shift is not None:
            uv = uv + uv_shift.float()
        pixels = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
        rays = torch.einsum("...ij,...hwj->...hwi", torch.linalg.inv(intrinsics.float()), pixels)
        points = rays * depth.unsqueeze(-1)
        return (torch.einsum("...ij,...hwj->...hwi", c2w[..., :3, :3].float(), points)
                + c2w[..., None, None, :3, 3].float())
