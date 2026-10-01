"""Independent three-group evaluator; no target data enters reconstruction."""

import csv
from datetime import timedelta
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from comp_svfgs.camera import target_cameras
from comp_svfgs.evaluation import timing_summary
from comp_svfgs.model import parameter_counts
from comp_svfgs.runtime import atomic_json, move_to_device, preserve_rng, synchronize
from .dataset import PROTOCOL, pixel_protocol
from .metrics import MASK_RULES, grouped_metrics, summarize_records


class CrossEvaluator:
    def __init__(self, cfg, model, renderer, image_metrics, accelerator, source=None):
        self.cfg, self.model, self.renderer = cfg, model, renderer
        self.image_metrics, self.accelerator, self.source = image_metrics, accelerator, source

    @torch.no_grad()
    def __call__(self, loader, split, step, output_dir, checkpoint=None):
        cfg, accelerator = self.cfg, self.accelerator
        protocol = pixel_protocol(cfg['evaluation']['eval_use_ego_mask'])
        # Appending here also isolates masked/full-image training mini results.
        out = Path(output_dir) / PROTOCOL / protocol
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(cfg, 'dump'):
            cfg.dump(str(out / 'resolved_config.py'))
        data = loader.dataset
        while hasattr(data, 'dataset'):  # Subset is useful for bounded diagnostics.
            data = data.dataset
        data_meta = data.evaluation_metadata() if hasattr(data, 'evaluation_metadata') else {}
        source = self.source or dict(checkpoint=str(checkpoint), global_step=step,
                                    source_dataset=cfg['dataset']['name'], source_image_shape=list(cfg['image_shape']))
        metadata = dict(dataset=cfg['dataset']['name'], dataset_split='test', split=split,
                        expected_bins=len(loader.dataset), dataset_bins=len(data),
                        evaluation_scope='subset' if len(loader.dataset) != len(data) else 'split',
                        test_range=split, image_shape=list(cfg['image_shape']), global_step=step,
                        checkpoint=str(checkpoint), source=source, zero_shot=cfg.get('zero_shot', False),
                        view_protocol=PROTOCOL, pixel_protocol=protocol, metric_mask=protocol,
                        pcc_reference='metric3d_v2', pcc_role='diagnostic_only',
                        depth_definition='accumulated_center_z', precision=cfg['precision'],
                        cudnn_deterministic=torch.backends.cudnn.deterministic,
                        mask_rules=MASK_RULES if cfg['evaluation']['eval_use_ego_mask'] else None,
                        device=str(accelerator.device), complete=False,
                        gpu=torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == 'cuda' else None)
        metadata.update(data_meta)
        mask_sha = (metadata.get('eval_mask') or {}).get('mask_manifest_sha256', '')
        counts = parameter_counts(accelerator.unwrap_model(self.model))
        atomic_json(out / 'parameter_counts.json', counts)
        atomic_json(out / 'evaluation_summary.json', dict(metadata, processed_bins=0))
        fields = ['bin_token', 'scene_id', 'group', 'psnr', 'ssim', 'lpips', 'pcc',
                  'pixel_protocol', 'mask_manifest_sha256']
        rows, times, timing_rows = [], [], []
        processed, total = 0, len(loader)
        start = perf_counter()
        logging.info('Evaluation [%s/%s] started: %d bins, step=%d, %s', cfg['dataset']['name'], split, total, step, protocol)
        was_training = self.model.training
        with preserve_rng(), (out / 'per_bin_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            self.model.eval()
            try:
                for batch in loader:
                    batch = move_to_device(batch, accelerator.device)
                    cameras = target_cameras(batch['target'], cfg['renderer'])
                    synchronize(accelerator.device)
                    before = perf_counter()
                    with accelerator.autocast():
                        output = self.model(batch['context'])
                    synchronize(accelerator.device)
                    duration = (perf_counter()-before)*1000
                    token, scene = batch['bin_token'][0], batch['scene'][0]
                    gaussians = output['gaussians']
                    rgb = torch.stack([self.renderer.render_view(gaussians, c) for c in cameras])
                    depth = torch.stack([self.renderer.render_pcc_depth(gaussians, c) for c in cameras])
                    mask = batch['target'].get('eval_mask')
                    records = grouped_metrics(token, scene, batch['target']['image'][0].float(), rgb.float(),
                                               batch['target']['metric_depth'][0], depth, self.image_metrics,
                                               None if mask is None else mask[0], protocol, mask_sha)
                    writer.writerows(records)
                    stream.flush()
                    rows.extend(records)
                    times.append(duration)
                    timing_rows.append(dict(bin_token=token, reconstruction_ms=duration,
                                            measured=processed >= cfg['evaluation']['time_skip_bins']))
                    processed += 1
                    if cfg['evaluation']['save_images']:
                        image_dir = out / 'images' / token
                        image_dir.mkdir(parents=True, exist_ok=True)
                        for view, image in enumerate(rgb):
                            pixels = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                            Image.fromarray(pixels).save(image_dir / f'{view:02d}.png')
                    if processed % 25 == 0:
                        atomic_json(out / 'evaluation_summary.json', dict(metadata, processed_bins=processed,
                                                                           groups=summarize_records(rows)))
                    if processed % 100 == 0 or processed == total:
                        elapsed = perf_counter()-start
                        logging.info('Evaluation [%s] %d/%d bins (%.1f%%), elapsed=%s, ETA=%s, speed=%.2f bins/s',
                                     split, processed, total, 100.*processed/total,
                                     timedelta(seconds=int(elapsed)),
                                     timedelta(seconds=int(elapsed/processed*(total-processed))), processed/max(elapsed, 1e-9))
                metadata['complete'] = True
            finally:
                self.model.train(was_training)
                timing = timing_summary(times, cfg['evaluation']['time_skip_bins'])
                summary = dict(metadata, processed_bins=processed, groups=summarize_records(rows),
                               parameter_counts=counts, reconstruction=timing, evaluation_seconds=perf_counter()-start)
                atomic_json(out / 'reconstruction_timing.json', dict(summary=timing, per_bin=timing_rows))
                atomic_json(out / 'evaluation_summary.json', summary)
        logging.info('Evaluation [%s] complete: %d bins in %s; %s', split, processed,
                     timedelta(seconds=int(summary['evaluation_seconds'])), out)
        return summary
