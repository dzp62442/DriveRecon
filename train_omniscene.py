"""Train/evaluate the static OmniScene experiment; see docs/OmniScene 数据集实验文档.md."""

import argparse
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_config(argv=None):
    from mmcv import Config, DictAction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('train', 'test'))
    parser.add_argument('--work-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--test-split', choices=('mini', 'total'))
    parser.add_argument('--checkpoint', help='For testing: final, latest or a complete checkpoint directory')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    parser.add_argument('--print-config', action='store_true', help='Resolve configuration only; no dataset/model/CUDA access')
    args = parser.parse_args(argv)
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    for key, value in [('mode', args.mode), ('work_dir', args.work_dir), ('dataset.data_root', args.data_root),
                       ('evaluation.test_split', args.test_split), ('evaluation.checkpoint', args.checkpoint)]:
        if value is not None:
            cfg.merge_from_dict({key: value})
    cfg.work_dir = str((ROOT / Path(cfg.work_dir).expanduser()).resolve())
    cfg.dataset.data_root = str((ROOT / Path(cfg.dataset.data_root).expanduser()).resolve())
    cfg.logging.wandb_mode = 'offline'
    if tuple(cfg.image_shape) not in ((112, 200), (224, 400)):
        raise ValueError('OmniScene experiments use exact 112x200 or 224x400 images')
    if cfg.data_loader.batch_size != 1:
        raise ValueError('Train/validation/test batch_size must be 1')
    if cfg.precision not in ('bf16', 'no'):
        raise ValueError('Supported precision: bf16 or no')
    for key in ('max_steps', 'validate_every_steps', 'mini_every_n_validations', 'checkpoint_every_steps', 'log_every_steps'):
        if cfg.training[key] < 1:
            raise ValueError('training.' + key + ' must be positive')
    return cfg, args.print_config


def main(argv=None):
    cfg, print_config = load_config(argv)
    if print_config:
        print(cfg.pretty_text)
        return
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from comp_svfgs.checkpoint import load_checkpoint, resolve_checkpoint
    from comp_svfgs.dataset_omniscene import OmniSceneDataset
    from comp_svfgs.evaluation import Evaluator
    from comp_svfgs.metrics import ImageMetrics
    from comp_svfgs.model import StaticDriveRecon
    from comp_svfgs.notifications import Notifier
    from comp_svfgs.renderer import SurfelRenderer
    from comp_svfgs.runtime import preserve_rng
    from comp_svfgs.sampler import ResumableBatchSampler
    from comp_svfgs.trainer import Trainer

    if not torch.cuda.is_available():
        raise RuntimeError('The native surfel renderer requires CUDA. Use --print-config for CPU-only configuration checks.')
    accelerator = Accelerator(mixed_precision=cfg.precision, gradient_accumulation_steps=1)
    if accelerator.num_processes != 1:
        raise ValueError('Use one process/GPU so the effective batch size stays 1')
    set_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.backends.cudnn.benchmark = False
    os.environ['WANDB_MODE'] = 'offline'
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(work_dir / (cfg.mode + '.log'))])
    logging.info('Configuration:\n%s', cfg.pretty_text)
    model = StaticDriveRecon(cfg.model)
    renderer = SurfelRenderer(cfg.renderer)
    options = dict(num_workers=cfg.data_loader.num_workers, pin_memory=cfg.data_loader.pin_memory)

    def eval_loader(split):
        return DataLoader(OmniSceneDataset(cfg.dataset, cfg.image_shape, split), batch_size=1,
                          shuffle=False, generator=torch.Generator().manual_seed(cfg.seed+2), **options)

    with preserve_rng():
        metrics = ImageMetrics(accelerator.device, cfg.evaluation.lpips_vgg_weights)
    if cfg.mode == 'test':
        model = accelerator.prepare(model)
        path = resolve_checkpoint(work_dir, cfg.evaluation.checkpoint)
        if path is None:
            raise FileNotFoundError('No checkpoint available for evaluation in ' + str(work_dir))
        state = load_checkpoint(path, accelerator.unwrap_model(model), cfg=cfg)
        out = work_dir / 'evaluation' / cfg.evaluation.test_split / f'step-{state["global_step"]:08d}'
        out.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(out / 'resolved_config.py'))
        summary = Evaluator(cfg, model, renderer, metrics, accelerator)(
            eval_loader(cfg.evaluation.test_split), cfg.evaluation.test_split, state['global_step'], out, path)
        logging.info('Evaluation complete: %s', summary)
        return

    optimizer = torch.optim.Adam(model.optimizer_groups(cfg.optimizer), lr=cfg.optimizer.lr,
                                  betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps)
    model, optimizer = accelerator.prepare(model, optimizer)
    train_data = OmniSceneDataset(cfg.dataset, cfg.image_shape, 'train')
    sampler = ResumableBatchSampler(len(train_data), cfg.seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler, generator=sampler.loader_generator, **options)
    evaluator = Evaluator(cfg, model, renderer, metrics, accelerator)
    notifier = Notifier(cfg.feishu)
    tracker = None
    try:
        if cfg.logging.wandb:
            import wandb
            tracker = wandb.init(project=cfg.logging.project, name=work_dir.name, dir=str(work_dir),
                                 mode='offline', config=cfg.to_dict())
        trainer = Trainer(cfg, model, optimizer, sampler, train_loader, eval_loader('val'), eval_loader('mini'),
                          renderer, evaluator, accelerator, notifier, tracker)
        trainer.run()
    finally:
        notifier.close()
        if tracker is not None:
            tracker.finish()


if __name__ == '__main__':
    main()
