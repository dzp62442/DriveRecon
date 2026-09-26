"""The original DriveRecon network with a single static six-camera reconstruction."""

import torch
from torch import nn

from scene.PointNet import UNet
from scene.gaussian_adapter import Guassian_Adaptor
from scene.geometry import backproject, scale_intrinsics


class StaticDriveRecon(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.view_num = cfg['view_num']
        self.num_frames = cfg['num_frames']
        if self.view_num != 6 or self.num_frames != 1:
            raise ValueError('This experiment requires view_num=6 and num_frames=1')
        self.unet = UNet(**cfg['unet'], view_num=6, num_frames=1, static_geometry=True)
        self.adapter = Guassian_Adaptor(
            cfg['unet']['out_channels'], cfg['num_samples'], cfg['gaussian_scale_min'],
            cfg['gaussian_scale_max'], cfg['seg_num'])
        self.adapter.max_shift = cfg['max_shift']
        dtype_name = cfg.get('parameter_dtype', 'bfloat16')
        if dtype_name not in ('bfloat16', 'float32'):
            raise ValueError('parameter_dtype must be bfloat16 or float32')
        self.unet.to(dtype=getattr(torch, dtype_name))
        self.adapter.to(dtype=getattr(torch, dtype_name))
        self.depth_min, self.depth_max = cfg['depth_min'], cfg['depth_max']
        self.num_samples = cfg['num_samples']
        self.register_buffer('depth_centers', torch.linspace(self.depth_min, self.depth_max, self.num_samples))

    def reconstruct(self, context):
        image = context['image']
        b, v, c, h, w = image.shape
        k = context['intrinsics_pixel'].reshape(b * v, 3, 3)
        c2w = context['extrinsics'].reshape(b * v, 4, 4)
        # The only temporal dimension is implicitly T=1. No labels reach the UNet.
        features, geometry_predictions = self.unet(image.reshape(b * v, c, h, w), k, c2w)
        color, scale, opacity, rotation, logits, residual, seg, uv, motion = self.adapter(features)
        gh, gw = features.shape[-2:]
        logits = logits.float()
        depth = (logits.softmax(-1) * self.depth_centers).sum(-1)
        depth = depth + residual[..., 0].float() * (self.depth_max - self.depth_min) / self.num_samples
        grid_k = scale_intrinsics(k, (h, w), (gh, gw))
        means = backproject(depth, grid_k, c2w, uv.permute(0, 2, 3, 1))
        gaussians = dict(
            means=means.reshape(b, -1, 3).contiguous(),
            color=color.float().reshape(b, -1, 3).contiguous(),
            scale=scale.float().reshape(b, -1, 3).contiguous(),
            rotation=rotation.float().reshape(b, -1, 4).contiguous(),
            opacity=opacity.float().reshape(b, -1, 1).contiguous(),
        )
        # Keep the shared UV/motion head intact; no motion displacement is applied.
        return dict(gaussians=gaussians, depth_logits=logits, depth=depth,
                    segmentation=seg.float(), geometry_depths=geometry_predictions)

    def forward(self, context):
        return self.reconstruct(context)

    def optimizer_groups(self, cfg):
        return [dict(params=self.unet.parameters(), name='unet', lr=cfg['lr'], weight_decay=cfg['weight_decay']),
                dict(params=self.adapter.parameters(), name='adapter', lr=cfg['lr'], weight_decay=cfg['weight_decay'])]


def parameter_counts(model):
    seen, groups, dtypes = set(), {}, {}
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
        owner = '.'.join(name.split('.')[:2]) if name.startswith('adapter.') else name.split('.')[0]
        row = groups.setdefault(owner, dict(trainable=0, frozen=0, total=0))
        row['trainable' if parameter.requires_grad else 'frozen'] += parameter.numel()
        row['total'] += parameter.numel()
    total = {key: sum(row[key] for row in groups.values()) for key in ('trainable', 'frozen', 'total')}
    return dict(**total, modules=groups, parameter_dtypes=dtypes,
                unit='scalar parameters', excludes='evaluation-only LPIPS')
