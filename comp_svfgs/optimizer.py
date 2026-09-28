"""Explicit optimizer identity for training, diagnostics and checkpoint recovery."""

import torch


def optimizer_name(optimizer):
    # Accelerate wraps the actual torch optimizer after prepare().
    while hasattr(optimizer, 'optimizer'):
        optimizer = optimizer.optimizer
    return type(optimizer).__name__


def build_optimizer(parameters, cfg):
    options = dict(cfg)
    # Early configs omitted type and used the released Adam optimizer.
    # Never reinterpret a saved config's missing type as the current AdamW default.
    kind = options.pop('type', 'Adam')
    constructors = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
    if kind not in constructors:
        raise ValueError('Unsupported optimizer type: ' + str(kind))
    if 'betas' in options:
        options['betas'] = tuple(options['betas'])
    return constructors[kind](parameters, **options)
