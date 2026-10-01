"""Configuration-only CLI; importing this module does not load torch or data."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(argv=None):
    from mmcv import Config, DictAction
    parser = argparse.ArgumentParser(description='DriveRecon PandaSet/DDAD training and zero-shot evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('train', 'test'))
    parser.add_argument('--work-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', help='Complete checkpoint directory, or final/latest for target-domain runs')
    parser.add_argument('--test-split', choices=('total', 'mini'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args(argv)
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    for key, value in [('mode', args.mode), ('work_dir', args.work_dir), ('dataset.data_root', args.data_root),
                       ('evaluation.checkpoint', args.checkpoint), ('evaluation.test_split', args.test_split)]:
        if value is not None:
            cfg.merge_from_dict({key: value})
    for key in ('work_dir',):
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    cfg.dataset.data_root = str((ROOT / Path(cfg.dataset.data_root).expanduser()).resolve())
    selection = cfg.evaluation.checkpoint
    if selection not in ('auto', 'final', 'latest'):
        cfg.evaluation.checkpoint = str((ROOT / Path(selection).expanduser()).resolve())
    cfg.logging.wandb_mode = 'offline'
    if cfg.dataset.name not in ('pandaset', 'ddad') or cfg.dataset.use_dynamic_mask:
        raise ValueError('Use a PandaSet/DDAD configuration with dynamic masks disabled')
    if cfg.mode not in ('train', 'test') or (cfg.zero_shot and cfg.mode != 'test'):
        raise ValueError('Zero-shot configs are evaluation-only; use the dataset config for training')
    if cfg.zero_shot and cfg.evaluation.checkpoint in ('auto', 'final', 'latest'):
        raise ValueError('Zero-shot evaluation needs an explicit source checkpoint directory')
    if tuple(cfg.image_shape) not in ((112, 200), (224, 400)) or cfg.data_loader.batch_size != 1:
        raise ValueError('Use 112x200 or 224x400 and batch_size=1')
    if cfg.precision not in ('bf16', 'no') or cfg.optimizer.type not in ('Adam', 'AdamW'):
        raise ValueError('Unsupported precision or optimizer')
    enabled = cfg.evaluation.eval_use_ego_mask
    if not isinstance(enabled, bool) or (enabled and cfg.dataset.name != 'ddad'):
        raise ValueError('eval_use_ego_mask must be boolean and can only be enabled for DDAD')
    if cfg.evaluation.test_split not in ('mini', 'total'):
        raise ValueError('test_split must be mini or total')
    if cfg.evaluation.time_skip_bins < 0:
        raise ValueError('time_skip_bins must be nonnegative')
    for key in ('max_steps', 'validate_every_steps', 'mini_every_n_validations', 'checkpoint_every_steps', 'log_every_steps'):
        if cfg.training[key] < 1:
            raise ValueError('training.' + key + ' must be positive')
    if min(cfg.dataset.mini_size, cfg.dataset.val_size) < 1:
        raise ValueError('mini_size and val_size must be positive')
    return cfg, args.print_config
