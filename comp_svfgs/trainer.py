"""Exact optimizer-step scheduling and resumable validation/mini lifecycle."""

import logging
from pathlib import Path
from time import perf_counter

import torch

from .camera import target_cameras
from .checkpoint import save_checkpoint, load_checkpoint, resolve_checkpoint
from .losses import auxiliary_losses, rgb_loss
from .model import parameter_counts
from .runtime import append_jsonl, atomic_json, move_to_device, preserve_rng


def train_update(model, renderer, batch, optimizer, accelerator, cfg):
    optimizer.zero_grad(set_to_none=True)
    with accelerator.autocast():
        output = model(batch['context'])
    aux, losses = auxiliary_losses(output, batch['context'], cfg['loss'], cfg['model'])
    # Accumulate raster gradients at Gaussian leaves, then backpropagate through the
    # reconstruction network ONCE. This preserves the sum of 18 view losses while
    # retaining only one view's CUDA raster buffers at a time.
    leaves = {key: value.detach().requires_grad_(True) for key, value in output['gaussians'].items()}
    gradients = {key: torch.zeros_like(value) for key, value in leaves.items()}
    photo = 0.
    cameras = target_cameras(batch['target'], cfg['renderer'])
    for i, camera in enumerate(cameras):
        prediction = renderer.render_view(leaves, camera)
        loss = cfg['loss']['rgb'] * rgb_loss(prediction, batch['target']['image'][0, i], batch['target']['loss_mask'][0, i])
        grads = torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True)
        for key, grad in zip(leaves, grads):
            if grad is not None:
                gradients[key].add_(grad.detach())
        photo += float(loss.detach())
    surrogate = aux + sum((value * gradients[key]).sum() for key, value in output['gaussians'].items())
    accelerator.backward(surrogate)
    optimizer.step()
    logged = {key: float(value.detach()) for key, value in losses.items()}
    return dict(logged, rgb=photo, total=photo+float(aux.detach()), lr=optimizer.param_groups[0]['lr'])


@torch.no_grad()
def validate(model, renderer, loader, accelerator, cfg):
    was_training = model.training
    totals, count = {}, 0
    with preserve_rng():
        model.eval()
        try:
            for batch in loader:
                batch = move_to_device(batch, accelerator.device)
                with accelerator.autocast():
                    output = model(batch['context'])
                auxiliary, losses = auxiliary_losses(output, batch['context'], cfg['loss'], cfg['model'])
                photo = 0.
                for i, camera in enumerate(target_cameras(batch['target'], cfg['renderer'])):
                    prediction = renderer.render_view(output['gaussians'], camera)
                    photo += cfg['loss']['rgb'] * float(rgb_loss(prediction, batch['target']['image'][0, i],
                                                               batch['target']['loss_mask'][0, i]))
                values = dict({k: float(v) for k, v in losses.items()}, rgb=photo, total=photo+float(auxiliary))
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + value
                count += 1
        finally:
            model.train(was_training)
    return dict(num_bins=count, losses={key: value/count for key, value in totals.items()})


class Trainer:
    def __init__(self, cfg, model, optimizer, sampler, train_loader, val_loader, mini_loader,
                 renderer, evaluator, accelerator, notifier, tracker=None):
        self.cfg, self.model, self.optimizer, self.sampler = cfg, model, optimizer, sampler
        self.train_loader, self.val_loader, self.mini_loader = train_loader, val_loader, mini_loader
        self.renderer, self.evaluator, self.accelerator, self.notify = renderer, evaluator, accelerator, notifier
        self.tracker = tracker
        self.work_dir = Path(cfg['work_dir'])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.checkpoint = None
        self.events = dict(last_val_step=0, validation_count=0, last_mini_step=0,
                           final_mini_complete=False, complete=False)
        self.start_step, self.started = 0, perf_counter()

    def restore(self):
        selection = self.cfg['training']['resume']
        if selection is False or selection is None:
            if resolve_checkpoint(self.work_dir, 'auto') is not None:
                raise ValueError('resume=False would overwrite an existing run; use a new work_dir')
            return
        path = resolve_checkpoint(self.work_dir, selection)
        if path is not None:
            state = load_checkpoint(path, self.accelerator.unwrap_model(self.model), self.optimizer,
                                    self.sampler, self.cfg, self.accelerator.scaler)
            self.step, self.events, self.checkpoint = state['global_step'], state['events'], path
            logging.info('Resumed complete training state: %s (step %s)', path, self.step)

    def save(self):
        self.checkpoint = save_checkpoint(self.work_dir, self.step, self.accelerator.unwrap_model(self.model),
                                          self.optimizer, self.sampler, self.events, self.cfg, self.accelerator.scaler)

    def persist_events(self):
        atomic_json(self.checkpoint / 'events.json', self.events)

    def log(self, kind, values):
        append_jsonl(self.work_dir / 'metrics.jsonl', dict(kind=kind, step=self.step, **values))
        if self.tracker is not None:
            flattened = {kind + '/' + key: value for key, value in values.items() if isinstance(value, (int, float))}
            self.tracker.log(flattened, step=self.step)

    def run_validation(self):
        result = validate(self.model, self.renderer, self.val_loader, self.accelerator, self.cfg)
        atomic_json(self.work_dir / 'validation' / f'step-{self.step:08d}' / 'summary.json', result)
        self.log('validation', result['losses'])

    def run_mini(self, reason):
        path = self.work_dir / 'evaluation/mini' / f'step-{self.step:08d}' / reason
        summary = self.evaluator(self.mini_loader, 'mini', self.step, path, self.checkpoint)
        flat = {group + '/' + metric: value for group, row in summary['groups'].items()
                for metric, value in row.items() if metric != 'num_bins'}
        self.log('mini_' + reason, dict(flat, reconstruction_ms=summary['reconstruction']['mean']))
        if reason == 'final':
            self.events['final_mini_complete'] = True
        else:
            self.events['last_mini_step'] = self.step
        self.persist_events()
        elapsed = perf_counter()-self.started
        updates = self.step-self.start_step
        eta = (self.cfg['training']['max_steps']-self.step)*elapsed/updates if updates else None
        body = '\n'.join([f'工作目录：{self.work_dir}', f'分辨率：{self.cfg["image_shape"]}',
                          f'step：{self.step}，bin 数：{summary["processed_bins"]}',
                          f'all_18：{summary["groups"]["all_18"]}',
                          f'novel_12：{summary["groups"]["novel_12"]}',
                          f'重建耗时：{summary["reconstruction"]}',
                          f'评估耗时：{summary["evaluation_seconds"]:.1f}s，ETA：{eta}s'])
        self.notify('DriveRecon mini 完成：' + reason, body)

    def due_events(self):
        if not self.step or self.step % self.cfg['training']['validate_every_steps']:
            return
        if self.events['last_val_step'] < self.step:
            self.run_validation()
            self.events['last_val_step'] = self.step
            self.events['validation_count'] += 1
            self.persist_events()
        if (self.events['validation_count'] % self.cfg['training']['mini_every_n_validations'] == 0
                and self.events['last_mini_step'] < self.step):
            self.run_mini('periodic')

    def run(self):
        self.restore()
        if hasattr(self.cfg, 'dump'):
            self.cfg.dump(str(self.work_dir / 'resolved_config.py'))
        maximum = self.cfg['training']['max_steps']
        if self.step > maximum:
            raise ValueError('Checkpoint has already exceeded the configured max_steps')
        if self.events['complete'] and self.step == maximum:
            logging.info('Training and final mini already complete: %s', self.work_dir)
            return
        self.start_step, self.started = self.step, perf_counter()
        counts = parameter_counts(self.accelerator.unwrap_model(self.model))
        atomic_json(self.work_dir / 'parameter_counts.json', counts)
        self.notify('DriveRecon 训练启动', '\n'.join([
            f'工作目录：{self.work_dir}', f'分辨率：{self.cfg["image_shape"]}',
            f'迭代次数：{self.step}/{maximum}', f'恢复来源：{self.checkpoint or "随机初始化"}',
            f'设备：{self.accelerator.device}', f'参数：{counts}']))
        self.due_events()  # Recover events interrupted after the optimizer snapshot.
        self.model.train()
        iterator = iter(self.train_loader)
        while self.step < maximum:
            try:
                batch = next(iterator)
            except StopIteration:
                self.sampler.next_epoch()
                iterator = iter(self.train_loader)
                batch = next(iterator)
            batch = move_to_device(batch, self.accelerator.device)
            values = train_update(self.model, self.renderer, batch, self.optimizer, self.accelerator, self.cfg)
            self.sampler.advance()
            self.step += 1
            self.log('train', values)
            if self.step % self.cfg['training']['log_every_steps'] == 0 or self.step == 1:
                logging.info('step %d/%d loss=%.6f lr=%.6g', self.step, maximum, values['total'], values['lr'])
            validation_due = self.step % self.cfg['training']['validate_every_steps'] == 0
            if validation_due or self.step % self.cfg['training']['checkpoint_every_steps'] == 0:
                self.save()
                self.due_events()
        self.save()
        atomic_json(self.work_dir / 'final.json', dict(checkpoint=str(self.checkpoint.relative_to(self.work_dir)), step=self.step))
        if not self.events['final_mini_complete']:
            self.run_mini('final')
        self.events['complete'] = True
        self.persist_events()
        logging.info('Completed %d optimizer updates and final mini', self.step)
