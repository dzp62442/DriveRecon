"""Atomic training snapshots plus a durable journal for validation/mini completion."""

import json
import os
from pathlib import Path
import uuid

import torch

from .runtime import atomic_json, rng_state, restore_rng


def complete_checkpoint(path):
    path = Path(path)
    try:
        with (path / 'complete.json').open() as stream:
            manifest = json.load(stream)
        return (path / 'state.pt').stat().st_size == manifest['state_bytes']
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return False


def resolve_checkpoint(work_dir, selection='auto'):
    root = Path(work_dir)
    if selection in ('auto', 'latest'):
        candidates = [p for p in (root / 'checkpoints').glob('step-*')
                      if p.name[5:].isdigit() and complete_checkpoint(p)]
        return max(candidates, key=lambda p: int(p.name[5:])) if candidates else None
    if selection == 'final':
        with (root / 'final.json').open() as stream:
            selection = json.load(stream)['checkpoint']
        path = root / selection
    else:
        path = Path(selection).expanduser()
    if not complete_checkpoint(path):
        raise FileNotFoundError('Incomplete or missing training checkpoint: ' + str(path))
    return path


def save_checkpoint(work_dir, step, model, optimizer, sampler, events, cfg, scaler=None):
    root = Path(work_dir)
    path = root / 'checkpoints' / f'step-{step:08d}'
    if not complete_checkpoint(path):
        temp = path.with_name('.' + path.name + '-' + uuid.uuid4().hex + '.tmp')
        temp.mkdir(parents=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
                     global_step=step, events=dict(events), rng=rng_state(), config=dict(cfg),
                     scaler=scaler.state_dict() if scaler is not None else None)
        with (temp / 'state.pt').open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(temp / 'events.json', events)
        atomic_json(temp / 'complete.json', dict(step=step, state_bytes=(temp / 'state.pt').stat().st_size))
        # A leftover incomplete directory is never selected as a valid checkpoint.
        if path.exists():
            os.replace(path, path.with_name('.' + path.name + '-incomplete-' + uuid.uuid4().hex))
        os.replace(temp, path)
    atomic_json(root / 'latest.json', dict(checkpoint=str(path.relative_to(root)), step=step))
    return path


def load_checkpoint(path, model, optimizer=None, sampler=None, cfg=None, scaler=None):
    # These are self-owned complete training snapshots, including Python/NumPy RNG state.
    state = torch.load(Path(path) / 'state.pt', map_location='cpu', weights_only=False)
    if cfg is not None:
        for name in ('image_shape', 'model', 'loss', 'optimizer', 'precision', 'deterministic'):
            if state['config'].get(name) != cfg.get(name):
                raise ValueError('Checkpoint configuration differs for ' + name + '; use a separate work_dir')
        if optimizer is not None:
            for name in ('max_steps', 'validate_every_steps', 'mini_every_n_validations'):
                if state['config']['training'][name] != cfg['training'][name]:
                    raise ValueError('Resume schedule differs for ' + name + '; use a separate work_dir')
    model.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if sampler is not None:
        sampler.load_state_dict(state['sampler'])
    if scaler is not None and state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])
    events_path = Path(path) / 'events.json'
    if events_path.is_file():
        with events_path.open() as stream:
            state['events'] = json.load(stream)
    if optimizer is not None:
        restore_rng(state['rng'])
    return state
