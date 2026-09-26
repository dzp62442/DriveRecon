"""SVF-GS/depthsplat full-image metric definitions, with stateless group PCC."""

from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity


@torch.no_grad()
def compute_psnr(ground_truth, predicted):
    mse = (ground_truth.clamp(0, 1) - predicted.clamp(0, 1)).square().mean(dim=(-3, -2, -1))
    return -10 * mse.log10()


@torch.no_grad()
def compute_ssim(ground_truth, predicted):
    values = [structural_similarity(gt.float().cpu().numpy(), pred.float().cpu().numpy(),
                                    win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0)
              for gt, pred in zip(ground_truth, predicted)]
    return torch.tensor(values, dtype=torch.float32, device=predicted.device)


@torch.no_grad()
def compute_pcc(ground_truth, predicted):
    x, y = ground_truth.float().reshape(-1), predicted.float().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    # No epsilon or invalid-pixel filtering: undefined correlation remains undefined.
    return (torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))).clamp(-1, 1)


def load_lpips(device, vgg_weights=None):
    from lpips import LPIPS
    path = Path(vgg_weights).expanduser() if vgg_weights else Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth'
    if not path.is_file():
        raise FileNotFoundError('LPIPS needs local VGG16 weights before training; set evaluation.lpips_vgg_weights: ' + str(path))
    # pnet_rand avoids any implicit network request. Every VGG feature weight is then loaded strictly.
    metric = LPIPS(net='vgg', pnet_rand=True, verbose=False)
    state = torch.load(path, map_location='cpu', weights_only=True)
    feature_state = {key: state['features.' + key.split('.', 1)[1]] for key in metric.net.state_dict()}
    metric.net.load_state_dict(feature_state, strict=True)
    return metric.requires_grad_(False).eval().to(device)


class ImageMetrics:
    def __init__(self, device, vgg_weights=None):
        self.lpips = load_lpips(device, vgg_weights)

    @torch.no_grad()
    def __call__(self, ground_truth, predicted):
        # One target at a time keeps the evaluation memory independent of the 18-view group size.
        lpips = torch.cat([self.lpips(gt[None], pred[None], normalize=True).reshape(1)
                           for gt, pred in zip(ground_truth, predicted)])
        return dict(psnr=compute_psnr(ground_truth, predicted),
                    ssim=compute_ssim(ground_truth, predicted), lpips=lpips)


def grouped_metrics(token, ground_truth, predicted, relative_depth, rendered_depth, image_metrics):
    values = image_metrics(ground_truth, predicted)
    records = []
    for name, indices in [('all_18', slice(0, 18)), ('novel_12', slice(0, 12))]:
        record = dict(bin_token=token, group=name)
        record.update({key: float(value[indices].mean()) for key, value in values.items()})
        record['pcc'] = float(compute_pcc(relative_depth[indices], rendered_depth[indices]))
        records.append(record)
    return records


def summarize_records(records):
    summary = {}
    for group in ('all_18', 'novel_12'):
        rows = [row for row in records if row['group'] == group]
        summary[group] = dict(num_bins=len(rows), **{
            key: float(np.mean([row[key] for row in rows])) if rows else None
            for key in ('psnr', 'ssim', 'lpips', 'pcc')})
    return summary
