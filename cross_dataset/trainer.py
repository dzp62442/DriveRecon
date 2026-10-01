"""Reuse the original optimizer loop; adapt only resume identity and mini reporting."""

from time import perf_counter
import torch

from comp_svfgs.checkpoint import resolve_checkpoint
from comp_svfgs.trainer import Trainer
from .checkpoint import assert_resume_identity


class CrossDatasetTrainer(Trainer):
    def restore(self):
        selection = self.cfg['training']['resume']
        if selection is not False and selection is not None:
            path = resolve_checkpoint(self.work_dir, selection)
            if path is not None:
                state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
                assert_resume_identity(state['config'], self.cfg)
                del state
        super().restore()

    def run_mini(self, reason):
        if self.checkpoint_step != self.step:
            self.save()
        path = self.work_dir / 'evaluation/mini' / f'step-{self.step:08d}' / reason
        result = self.evaluator(self.mini_loader, 'mini', self.step, path, self.checkpoint)
        flat = {group+'/'+metric: value for group, row in result['groups'].items()
                for metric, value in row.items() if metric != 'num_bins'}
        self.log('mini_'+reason, dict(flat, reconstruction_ms=result['reconstruction']['mean']))
        if reason == 'final':
            self.events['final_mini_complete'] = True
        else:
            self.events['last_mini_step'] = self.step
        self.persist_events()
        updates, elapsed = self.step-self.start_step, perf_counter()-self.started
        eta = (self.cfg['training']['max_steps']-self.step)*elapsed/updates if updates else None
        body = [f'工作目录：{self.work_dir}', f'数据集：{self.cfg["dataset"]["name"]}',
                f'分辨率：{self.cfg["image_shape"]}，step：{self.step}，bin 数：{result["processed_bins"]}',
                f'像素协议：{result["pixel_protocol"]}，PCC 参考：Metric3D-v2（诊断）']
        body.extend(f'{group}：{value}' for group, value in result['groups'].items())
        body.extend([f'重建耗时：{result["reconstruction"]}', f'评估耗时：{result["evaluation_seconds"]:.1f}s，ETA：{eta}s'])
        self.notify('DriveRecon mini 完成：'+reason, '\n'.join(body))
