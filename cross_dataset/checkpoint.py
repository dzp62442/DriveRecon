"""Model-only evaluation loading and target-domain resume identity."""

from pathlib import Path
import torch

from comp_svfgs.checkpoint import complete_checkpoint


def assert_resume_identity(saved, cfg):
    for name in ('dataset',):
        if saved.get(name) != cfg.get(name):
            raise ValueError('Target-domain resume configuration differs for ' + name + '; use a separate work_dir')
    if saved['evaluation'].get('eval_use_ego_mask') != cfg['evaluation']['eval_use_ego_mask']:
        raise ValueError('Resume mini pixel protocol differs; use a separate work_dir')


def load_evaluation_checkpoint(path, model, cfg):
    path = Path(path).resolve()
    if not complete_checkpoint(path):
        raise FileNotFoundError('Incomplete or missing checkpoint: ' + str(path))
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
    saved = state['config']
    for name in ('model', 'renderer', 'precision'):
        if saved.get(name) != cfg.get(name):
            raise ValueError('Source checkpoint configuration differs for ' + name)
    if tuple(saved['image_shape']) != tuple(cfg['image_shape']):
        raise ValueError('Source checkpoint resolution differs; use matching OmniScene pretrained weights')
    if cfg.get('zero_shot', False):
        if saved['dataset'].get('name', 'omniscene') != 'omniscene':
            raise ValueError('OmniScene zero-shot configs require an OmniScene source checkpoint')
        # Do not put cross-dataset outputs anywhere inside the source training run.
        source_run = path.parent.parent
        output = Path(cfg['work_dir']).resolve()
        if output == source_run or source_run in output.parents:
            raise ValueError('Zero-shot work_dir must be separate from the source training run')
    model.load_state_dict(state['model'], strict=True)
    return dict(global_step=state['global_step'], checkpoint=str(path), source_config=saved,
                source_dataset=saved['dataset'].get('name', 'omniscene'),
                source_image_shape=list(saved['image_shape']))
