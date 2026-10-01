"""PandaSet/DDAD training and zero-shot evaluation, isolated from OmniScene."""

import logging
import os
from pathlib import Path

from cross_dataset.config import load_config


def main(argv=None):
    cfg, print_config = load_config(argv)
    if print_config:
        print(cfg.pretty_text)
        return
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from comp_svfgs.checkpoint import resolve_checkpoint
    from comp_svfgs.model import StaticDriveRecon
    from comp_svfgs.notifications import Notifier
    from comp_svfgs.optimizer import build_optimizer
    from comp_svfgs.renderer import SurfelRenderer
    from comp_svfgs.runtime import atomic_json, preserve_rng
    from comp_svfgs.sampler import ResumableBatchSampler
    from cross_dataset.checkpoint import load_evaluation_checkpoint
    from cross_dataset.dataset import CrossDataset
    from cross_dataset.evaluation import CrossEvaluator
    from cross_dataset.metrics import CrossImageMetrics
    from cross_dataset.trainer import CrossDatasetTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('The native surfel renderer requires CUDA; --print-config is CPU-only')
    accelerator = Accelerator(mixed_precision=cfg.precision, gradient_accumulation_steps=1)
    if accelerator.num_processes != 1:
        raise ValueError('Use one process/GPU so batch size stays 1')
    set_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = False
    os.environ['WANDB_MODE'] = 'offline'
    work_dir = Path(cfg.work_dir)
    model = StaticDriveRecon(cfg.model)
    renderer = SurfelRenderer(cfg.renderer)
    renderer.validate_backend()
    source, checkpoint = None, None
    if cfg.mode == 'test':
        checkpoint = resolve_checkpoint(work_dir, cfg.evaluation.checkpoint)
        if checkpoint is None:
            raise FileNotFoundError('No checkpoint in '+str(work_dir))
        source = load_evaluation_checkpoint(checkpoint, model, cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(work_dir/(cfg.mode+'.log'))]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers, force=True)
    logging.info('Configuration:\n%s', cfg.pretty_text)
    native = renderer.backend_info()
    atomic_json(work_dir/(cfg.mode+'_rasterizer.json'), native)
    logging.info('Rasterizer: %s', native)
    options = dict(num_workers=cfg.data_loader.num_workers, pin_memory=cfg.data_loader.pin_memory)

    def eval_loader(split, test_range='total'):
        dataset = CrossDataset(cfg.dataset, cfg.image_shape, split, test_range,
                               eval_use_ego_mask=split == 'test' and cfg.evaluation.eval_use_ego_mask)
        return DataLoader(dataset, batch_size=1, shuffle=False,
                          generator=torch.Generator().manual_seed(cfg.seed+2), **options)

    with preserve_rng():
        metrics = CrossImageMetrics(accelerator.device, cfg.evaluation.lpips_vgg_weights,
                                    masked=cfg.evaluation.eval_use_ego_mask)
    if cfg.mode == 'test':
        model = accelerator.prepare(model)
        out = work_dir/f'step-{source["global_step"]:08d}'/'test'/cfg.evaluation.test_split
        summary = CrossEvaluator(cfg, model, renderer, metrics, accelerator, source)(
            eval_loader('test', cfg.evaluation.test_split), cfg.evaluation.test_split,
            source['global_step'], out, checkpoint)
        logging.info('Evaluation complete: %s', summary['groups'])
        return summary

    dataset = CrossDataset(cfg.dataset, cfg.image_shape, 'train')
    sampler = ResumableBatchSampler(len(dataset), cfg.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, generator=sampler.loader_generator, **options)
    optimizer = build_optimizer(model.optimizer_groups(cfg.optimizer), cfg.optimizer)
    model, optimizer = accelerator.prepare(model, optimizer)
    evaluator = CrossEvaluator(cfg, model, renderer, metrics, accelerator)
    notifier, tracker = Notifier(cfg.feishu), None
    try:
        if cfg.logging.wandb:
            import wandb
            tracker = wandb.init(project=cfg.logging.project, name=work_dir.name, dir=str(work_dir),
                                 mode='offline', config=cfg.to_dict())
        trainer = CrossDatasetTrainer(cfg, model, optimizer, sampler, loader, eval_loader('val'), eval_loader('test', 'mini'),
                                       renderer, evaluator, accelerator, notifier, tracker)
        trainer.run()
    finally:
        notifier.close()
        if tracker is not None:
            tracker.finish()


if __name__ == '__main__':
    main()
