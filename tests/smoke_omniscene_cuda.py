"""Bounded CUDA verification, explicitly NOT a training run or benchmark result.

Reads one train bin and one mini bin, performs two optimizer updates per resolution,
and exercises validation, four metrics, native depth rendering and full-state reload.
Notifications are not instantiated. Outputs stay in the requested smoke work directory.
"""

import argparse
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from mmcv import Config

from comp_svfgs.camera import make_camera
from comp_svfgs.checkpoint import save_checkpoint, load_checkpoint
from comp_svfgs.dataset_omniscene import OmniSceneDataset
from comp_svfgs.evaluation import Evaluator
from comp_svfgs.metrics import ImageMetrics, load_lpips
from comp_svfgs.model import StaticDriveRecon, parameter_counts
from comp_svfgs.renderer import SurfelRenderer
from comp_svfgs.runtime import atomic_json, move_to_device
from comp_svfgs.sampler import ResumableBatchSampler
from comp_svfgs.trainer import train_update, validate


def check_native_depth(renderer, device):
    k = torch.tensor([[80., 0., 63.], [0., 70., 22.], [0., 0., 1.]], device=device)
    camera = make_camera(k, torch.eye(4, device=device), (64, 96))
    gs = dict(means=torch.tensor([[[0., 0., 10.]]], device=device), color=torch.ones(1, 1, 3, device=device),
              scale=torch.full((1, 1, 3), .1, device=device), rotation=torch.tensor([[[1., 0., 0., 0.]]], device=device),
              opacity=torch.full((1, 1, 1), .5, device=device))
    with torch.no_grad():
        alpha = renderer.render_view(gs, camera)[0]
        depth = renderer.render_pcc_depth(gs, camera)
    torch.testing.assert_close(depth, 10*alpha, rtol=1e-5, atol=1e-5)
    peak = int(alpha.argmax())
    assert divmod(peak, 96) == (22, 63), divmod(peak, 96)
    assert depth.max() > 1, 'Depth was clipped to RGB range or nothing was rendered'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', choices=('112x200', '224x400', 'both'), default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    accelerator = Accelerator(mixed_precision='bf16')
    assert accelerator.device.type == 'cuda'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    metrics = ImageMetrics(accelerator.device)
    # Prove our explicit offline VGG loading gives the reference LPIPS model.
    from lpips import LPIPS
    reference_lpips = LPIPS(net='vgg', verbose=False).eval().to(accelerator.device)
    with torch.no_grad():
        x, y = torch.rand(1, 3, 32, 48, device=accelerator.device), torch.rand(1, 3, 32, 48, device=accelerator.device)
        torch.testing.assert_close(metrics.lpips(x, y, normalize=True), reference_lpips(x, y, normalize=True))
    del reference_lpips
    choices = ('112x200', '224x400') if args.resolution == 'both' else (args.resolution,)
    for resolution in choices:
        cfg = Config.fromfile(f'configs/omniscene/{resolution}.py')
        torch.backends.cudnn.deterministic = cfg.deterministic
        torch.backends.cudnn.benchmark = False
        cfg.work_dir = str(out/resolution)
        cfg.training.max_steps = 2
        cfg.evaluation.time_skip_bins = 0
        cfg.feishu.enabled = False
        root = Path(cfg.work_dir)
        root.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(root/'resolved_config.py'))
        torch.manual_seed(0)
        model = StaticDriveRecon(cfg.model)
        optimizer = torch.optim.Adam(model.optimizer_groups(cfg.optimizer), lr=cfg.optimizer.lr,
                                      eps=cfg.optimizer.eps, betas=cfg.optimizer.betas)
        model, optimizer = accelerator.prepare(model, optimizer)
        renderer = SurfelRenderer(cfg.renderer)
        check_native_depth(renderer, accelerator.device)
        train = OmniSceneDataset(cfg.dataset, cfg.image_shape, 'train')
        # Explicit diagnostic subset; the real Dataset and default splits have no cap changes.
        loader = DataLoader(Subset(train, [0]), batch_size=1, generator=torch.Generator().manual_seed(1))
        batch = move_to_device(next(iter(loader)), accelerator.device)
        sampler = ResumableBatchSampler(len(train), cfg.seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model.train()
        loss1 = train_update(model, renderer, batch, optimizer, accelerator, cfg)
        assert all(torch.isfinite(torch.tensor(v)) for v in loss1.values()), loss1
        print(resolution, 'first optimizer update', loss1, flush=True)
        sampler.advance()
        validation = validate(model, renderer, loader, accelerator, cfg)
        state = dict(last_val_step=0, validation_count=0, last_mini_step=0, final_mini_complete=False, complete=False)
        checkpoint = save_checkpoint(root, 1, accelerator.unwrap_model(model), optimizer, sampler, state, cfg)
        model.eval()
        with torch.no_grad(), accelerator.autocast():
            expected = model(batch['context'])['gaussians']['means'].clone()
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        restored = load_checkpoint(checkpoint, accelerator.unwrap_model(model), optimizer, sampler, cfg)
        assert restored['global_step'] == 1 and sampler.cursor == 1
        with torch.no_grad(), accelerator.autocast():
            actual = model(batch['context'])['gaussians']['means']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.train()
        loss2 = train_update(model, renderer, batch, optimizer, accelerator, cfg)
        assert all(torch.isfinite(torch.tensor(v)) for v in loss2.values()), loss2
        sampler.advance()
        checkpoint2 = save_checkpoint(root, 2, accelerator.unwrap_model(model), optimizer, sampler, state, cfg)
        mini = OmniSceneDataset(cfg.dataset, cfg.image_shape, 'mini')
        mini_loader = DataLoader(Subset(mini, [0]), batch_size=1, generator=torch.Generator().manual_seed(2))
        summary = Evaluator(cfg, model, renderer, metrics, accelerator)(mini_loader, 'smoke_mini', 2,
                                                                      root/'evaluation', checkpoint2)
        for group in summary['groups'].values():
            assert all(torch.isfinite(torch.tensor(group[key])) for key in ('psnr', 'ssim', 'lpips', 'pcc'))
        report = dict(resolution=resolution, updates=2, train_bins=1, evaluation_bins=1,
                      first_update=loss1, restored_update=loss2, validation=validation,
                      evaluation=summary, peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                      seconds=time.perf_counter()-started, status='passed', formal_result=False)
        atomic_json(root/'verification.json', report)
        reports.append(report)
        print(resolution, 'PASSED; peak allocated MB', report['peak_allocated_mb'], flush=True)
        del expected, actual, batch, model, optimizer, restored
        accelerator.free_memory()
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(out/'verification.json', dict(formal_result=False, results=reports))


if __name__ == '__main__':
    main()
