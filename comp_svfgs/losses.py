"""DriveRecon loss weights/reductions, with Metric3D targets and SVF-GS masking."""

import torch
from torch.nn import functional as F


def depth_classes(depth, minimum=0.1, maximum=400., count=200):
    return ((depth - minimum) / (maximum - minimum) * (count - 1)).round().clamp(0, count-1).long()


def normalized_depth_mse(pred, target, minimum=0.1):
    valid = (target > minimum) & (target < 255.)
    # The condition belongs to the original loss, not to sample acceptance.
    if not valid.any():
        return pred.sum() * 0.
    return F.mse_loss((pred[valid] / 255.).clamp(0, 1), (target[valid] / 255.).clamp(0, 1))


def resize_depth(depth, shape):
    return F.interpolate(depth.reshape(-1, 1, *depth.shape[-2:]).float(), size=shape,
                         mode='bilinear', align_corners=False)[:, 0]


def auxiliary_losses(output, context, cfg, model_cfg):
    depth = output['depth']
    target = resize_depth(context['metric_depth'], depth.shape[-2:])
    valid = target > 0.1
    logits = output['depth_logits']
    classes = depth_classes(target, model_cfg['depth_min'], model_cfg['depth_max'], model_cfg['num_samples'])
    classification = F.cross_entropy(logits[valid], classes[valid]) if valid.any() else logits.sum() * 0.
    label = context['segmentation_label']
    label = F.interpolate(label.reshape(-1, 1, *label.shape[-2:]).float(),
                          size=depth.shape[-2:], mode='nearest')[:, 0].long()
    segmentation = F.cross_entropy(output['segmentation'].permute(0, 3, 1, 2), label)
    geometry = depth.sum() * 0.
    for prediction in output['geometry_depths']:
        target_geo = resize_depth(context['metric_depth'], prediction.shape[-2:])
        geometry = geometry + normalized_depth_mse(prediction, target_geo, minimum=25.5) / 255.
    losses = dict(segmentation=segmentation, depth_class=classification,
                  depth_reg=normalized_depth_mse(depth, target), geometry_aux=geometry)
    total = sum(cfg[name] * value for name, value in losses.items())
    return total, losses


def rgb_loss(prediction, target, mask):
    # Full-image denominator, even where the float loss mask is zero.
    return (prediction * mask.unsqueeze(-3) - target * mask.unsqueeze(-3)).abs().mean()
