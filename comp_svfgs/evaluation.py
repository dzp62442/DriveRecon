"""Shared mini/total evaluator with full reconstruction timing and two view groups."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from .camera import target_cameras
from .metrics import grouped_metrics, summarize_records
from .model import parameter_counts
from .runtime import atomic_json, move_to_device, preserve_rng, synchronize


def timing_summary(times, skip):
    measured = times[skip:]
    return dict(unit='ms/bin', warmup_bins=min(skip, len(times)), measured_bins=len(measured),
                mean=float(np.mean(measured)) if measured else None,
                median=float(np.median(measured)) if measured else None,
                p95=float(np.percentile(measured, 95)) if measured else None)


class Evaluator:
    def __init__(self, cfg, model, renderer, image_metrics, accelerator):
        self.cfg, self.model, self.renderer = cfg, model, renderer
        self.image_metrics, self.accelerator = image_metrics, accelerator

    @torch.no_grad()
    def __call__(self, loader, split, step, output_dir, checkpoint=None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg, accelerator = self.cfg, self.accelerator
        metadata = dict(split=split, global_step=step, checkpoint=str(checkpoint) if checkpoint else None,
                        image_shape=list(cfg['image_shape']), precision=cfg['precision'],
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        depth_definition='accumulated_center_z', metric_mask='full_image',
                        device=str(accelerator.device), complete=False,
                        gpu=torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == 'cuda' else None)
        counts = parameter_counts(accelerator.unwrap_model(self.model))
        atomic_json(out / 'parameter_counts.json', counts)
        records, times, timing_rows = [], [], []
        fields = ['bin_token', 'group', 'psnr', 'ssim', 'lpips', 'pcc']
        atomic_json(out / 'evaluation_summary.json', dict(metadata, processed_bins=0))
        start = perf_counter()
        was_training = self.model.training
        with preserve_rng(), (out / 'per_bin_metrics.csv').open('w', newline='') as stream:
            self.model.eval()
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            try:
                for index, batch in enumerate(loader):
                    batch = move_to_device(batch, accelerator.device)
                    cameras = target_cameras(batch['target'], cfg['renderer'])
                    synchronize(accelerator.device)
                    before = perf_counter()
                    with accelerator.autocast():
                        output = self.model(batch['context'])
                    synchronize(accelerator.device)
                    duration = (perf_counter() - before) * 1000
                    times.append(duration)
                    token = batch['bin_token'][0]
                    timing_rows.append(dict(bin_token=token, reconstruction_ms=duration,
                                            measured=index >= cfg['evaluation']['time_skip_bins']))
                    gaussians = output['gaussians']
                    rgb = torch.stack([self.renderer.render_view(gaussians, camera) for camera in cameras])
                    depth = torch.stack([self.renderer.render_pcc_depth(gaussians, camera) for camera in cameras])
                    rows = grouped_metrics(token, batch['target']['image'][0].float(), rgb.float(),
                                           batch['target']['rel_depth'][0], depth, self.image_metrics)
                    records.extend(rows)
                    writer.writerows(rows)
                    stream.flush()
                    if cfg['evaluation'].get('save_images', False):
                        image_dir = out / 'images' / token
                        image_dir.mkdir(parents=True, exist_ok=True)
                        for view, image in enumerate(rgb):
                            array = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                            Image.fromarray(array).save(image_dir / f'{view:02d}.png')
                    if (index + 1) % 25 == 0:
                        atomic_json(out / 'evaluation_summary.json', dict(metadata, processed_bins=index+1,
                                                                           groups=summarize_records(records)))
                metadata['complete'] = True
            finally:
                self.model.train(was_training)
                timing = timing_summary(times, cfg['evaluation']['time_skip_bins'])
                summary = dict(metadata, processed_bins=len(times), groups=summarize_records(records),
                               parameter_counts=counts, reconstruction=timing,
                               evaluation_seconds=perf_counter()-start)
                atomic_json(out / 'reconstruction_timing.json', dict(summary=timing, per_bin=timing_rows))
                atomic_json(out / 'evaluation_summary.json', summary)
        return summary
