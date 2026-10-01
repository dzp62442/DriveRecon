"""SVF-GS pixel support and view reductions, without dataset rejection rules."""

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
import torch

from comp_svfgs.metrics import ImageMetrics, load_lpips, compute_pcc

GROUPS = dict(all_18=slice(0, 18), novel_12=slice(0, 12), input_6=slice(12, 18))
METRIC_NAMES = ('psnr', 'ssim', 'lpips', 'pcc')
MASK_RULES = dict(ssim_win_size=11, ssim_sigma=1.5, ssim_use_sample_covariance=True,
                  lpips_net='vgg', lpips_normalize=True,
                  lpips_rule='gt_fill_invalid_spatial_valid_mean', pcc_rule='valid_group_flatten')


class CrossImageMetrics(ImageMetrics):
    def __init__(self, device, vgg_weights=None, masked=False):
        super().__init__(device, vgg_weights)
        # A separate offline-loaded instance keeps full-image LPIPS exactly unchanged.
        self.spatial_lpips = load_lpips(device, vgg_weights) if masked else None
        if self.spatial_lpips is not None:
            self.spatial_lpips.spatial = True

    @torch.no_grad()
    def __call__(self, ground_truth, predicted, mask=None):
        values = super().__call__(ground_truth, predicted)
        if mask is None:
            return values
        for i in range(len(predicted)):
            valid = mask[i]
            if bool(valid.all()):
                continue
            gt, pred = ground_truth[i], predicted[i]
            squared = (gt.clamp(0, 1)-pred.clamp(0, 1)).square()
            mse = torch.where(valid[None], squared, 0.).sum() / (gt.shape[0]*valid.sum())
            values['psnr'][i] = -10*mse.log10()
            centers = binary_erosion(valid.cpu().numpy(), structure=np.ones((11, 11), dtype=bool))
            _, scores = structural_similarity(gt.cpu().numpy(), pred.cpu().numpy(), win_size=11,
                                               gaussian_weights=True, sigma=1.5, use_sample_covariance=True,
                                               channel_axis=0, data_range=1., full=True)
            # Empty support remains undefined, rather than rejecting or discarding a bin.
            values['ssim'][i] = float(scores[:, centers].mean())
            pred_eval = torch.where(valid[None], pred, gt)
            distance = self.spatial_lpips(gt[None], pred_eval[None], normalize=True)[0, 0]
            values['lpips'][i] = torch.where(valid, distance, 0.).sum()/valid.sum()
        return values


def grouped_metrics(token, scene, gt, pred, reference_depth, rendered_depth, image_metrics,
                    mask=None, pixel_protocol='full_image', mask_manifest_sha256=''):
    values = image_metrics(gt, pred, mask)
    rows = []
    for group, indices in GROUPS.items():
        row = dict(bin_token=token, scene_id=scene, group=group, pixel_protocol=pixel_protocol,
                   mask_manifest_sha256=mask_manifest_sha256)
        row.update({key: float(value[indices].double().mean()) for key, value in values.items()})
        x, y = reference_depth[indices], rendered_depth[indices]
        if mask is not None:
            x, y = x[mask[indices]], y[mask[indices]]
        row['pcc'] = float(compute_pcc(x, y))
        rows.append(row)
    return rows


def summarize_records(rows):
    result = {}
    for group in GROUPS:
        selected = [r for r in rows if r['group'] == group]
        result[group] = dict(num_bins=len(selected), **{
            key: float(np.mean([r[key] for r in selected])) if selected else None for key in METRIC_NAMES})
    return result
