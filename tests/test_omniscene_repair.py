"""CPU-only numerical and recovery regressions using tiny synthetic models."""

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from comp_svfgs.checkpoint import load_checkpoint, save_checkpoint
from comp_svfgs.diagnostics import UpdateProbe, prediction_diagnostics
from comp_svfgs.model import StaticDriveRecon
from comp_svfgs.optimizer import build_optimizer, optimizer_name
from comp_svfgs.trainer import Trainer
from tests.omniscene_helpers import make_trainer
from train_omniscene import load_config


def summary(*args):
    return dict(groups={g: dict(psnr=1., ssim=1., lpips=0., pcc=1.) for g in ('all_18', 'novel_12')},
                reconstruction=dict(mean=1.), processed_bins=1, evaluation_seconds=.1)


def sparse_trainer(root, trainer_type=Trainer, evaluator=summary):
    trainer = make_trainer(root, trainer_type, evaluator)
    # The 1:5:10 cadence scales the production 1k:5k:10k schedule down for CPU fixtures.
    trainer.cfg['training'].update(max_steps=11, validate_every_steps=1,
                                   checkpoint_every_steps=5, mini_every_n_validations=10)
    return trainer


class PrecisionAndDiagnosticsTests(unittest.TestCase):
    def test_adam_updates_match_released_training_setup(self):
        # Execute the released setup method without importing its CUDA/data dependencies.
        source = Path(__file__).resolve().parents[1] / 'scene/GS_LRM.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gaussian_LRM')
        setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_setup')
        scope = dict(torch=torch, get_expon_lr_func=lambda **kw: kw)
        exec(compile(ast.Module(body=[setup], type_ignores=[]), str(source), 'exec'), scope)
        cfg, _ = load_config(['--config', 'configs/omniscene/112x200.py', '--print-config'])
        cfg.optimizer.type = 'Adam'  # Audit the release; production retains the approved AdamW exception.
        reference = SimpleNamespace(unet=nn.Linear(2, 2), Guassian_Adaptor=nn.Linear(2, 1))
        scope['training_setup'](reference, SimpleNamespace(
            percent_dense=.01, deformation_lr_delay_mult=.01, position_lr_max_steps=30000))
        model = nn.Module()
        model.unet = copy.deepcopy(reference.unet)
        model.adapter = copy.deepcopy(reference.Guassian_Adaptor)
        optimizer = build_optimizer(StaticDriveRecon.optimizer_groups(model, cfg.optimizer), cfg.optimizer)
        reference_parameters = list(reference.unet.parameters()) + list(reference.Guassian_Adaptor.parameters())
        for step in range(12):
            for left, right in zip(model.parameters(), reference_parameters):
                # Include changing task gradients and a zero-task-gradient update.
                gradient = torch.full_like(left, 0. if step == 4 else (step-6)*.017)
                left.grad, right.grad = gradient.clone(), gradient.clone()
            optimizer.step()
            reference.optimizer.step()
            for left, right in zip(model.parameters(), reference_parameters):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                for key, value in optimizer.state[left].items():
                    torch.testing.assert_close(value, reference.optimizer.state[right][key], rtol=0, atol=0)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], [4e-4, 4e-4])

    def test_both_configs_keep_fp32_parameters_bf16_compute_and_fixed_recipe(self):
        for name in ('112x200', '224x400'):
            cfg, _ = load_config(['--config', 'configs/omniscene/' + name + '.py', '--print-config'])
            self.assertEqual(cfg.model.parameter_dtype, 'float32')
            self.assertEqual(cfg.precision, 'bf16')
            self.assertEqual(cfg.training.checkpoint_every_steps, 5000)
            self.assertEqual(cfg.training.validate_every_steps, 1000)
            self.assertEqual(cfg.training.mini_every_n_validations, 10)
            self.assertEqual(cfg.training.max_steps, 100001)
            self.assertTrue(cfg.diagnostics.enabled)
            self.assertEqual(cfg.optimizer, dict(type='AdamW', lr=4e-4, weight_decay=.05, betas=(.9, .999), eps=1e-15))
            self.assertEqual(cfg.loss, dict(rgb=1., segmentation=1., depth_class=2., depth_reg=2., geometry_aux=2.))

    def test_probe_preserves_update_and_adam_moments_are_fp32(self):
        model = nn.Module()
        model.adapter = nn.Module()
        model.adapter.depth_map_head = nn.Linear(2, 1, bias=False)
        nn.init.ones_(model.adapter.depth_map_head.weight)
        reference = copy.deepcopy(model)
        cfg, _ = load_config(['--config', 'configs/omniscene/112x200.py', '--print-config'])
        optimizer = build_optimizer(model.parameters(), cfg.optimizer)
        reference_optimizer = build_optimizer(reference.parameters(), cfg.optimizer)
        # Autocast must leave parameters, accumulated gradients and Adam moments in FP32.
        for network in (model, reference):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = network.adapter.depth_map_head(torch.ones(1, 2)).float().square().sum()
            loss.backward()
        original_gradient = model.adapter.depth_map_head.weight.grad.clone()
        rng = torch.get_rng_state().clone()
        probe = UpdateProbe(model)
        optimizer.step()
        stats = probe.finish(model, optimizer)
        reference_optimizer.step()
        torch.testing.assert_close(model.adapter.depth_map_head.weight, reference.adapter.depth_map_head.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.adapter.depth_map_head.weight.grad, original_gradient, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        row = stats['updates']['adapter.depth_map_head']
        self.assertEqual(row['changed_fraction'], 1.)
        self.assertEqual(row['gradient_nonfinite_elements'], 0)
        self.assertGreater(row['gradient_l2'], 0.)
        self.assertEqual(stats['precision']['parameter_elements_by_dtype'], {'torch.float32': 2})
        self.assertEqual(stats['precision']['gradient_elements_by_dtype'], {'torch.float32': 2})
        self.assertEqual(stats['optimizer_type'], 'AdamW')
        self.assertGreater(stats['updates']['adapter.depth_map_head']['parameter_abs_mean'], .9)
        for name in ('exp_avg', 'exp_avg_sq'):
            self.assertEqual(stats['precision']['adam_moment_elements_by_dtype'][name], {'torch.float32': 2})

    def test_adamw_decay_is_multiplicative_for_both_original_parameter_groups(self):
        cfg, _ = load_config(['--config', 'configs/omniscene/112x200.py', '--print-config'])
        tiny = nn.Module()
        tiny.unet = nn.GroupNorm(1, 2)
        tiny.adapter = nn.Linear(2, 1, bias=False)
        nn.init.constant_(tiny.adapter.weight, .02)
        groups = StaticDriveRecon.optimizer_groups(tiny, cfg.optimizer)
        optimizer = build_optimizer(groups, cfg.optimizer)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([g['name'] for g in optimizer.param_groups], ['unet', 'adapter'])
        initial = {name: p.detach().clone() for name, p in tiny.named_parameters()}
        for p in tiny.parameters():
            p.grad = torch.zeros_like(p)
        for _ in range(1000):
            optimizer.step()
        factor = (1 - cfg.optimizer.lr * cfg.optimizer.weight_decay) ** 1000
        for name, p in tiny.named_parameters():
            torch.testing.assert_close(p, initial[name] * factor, rtol=5e-5, atol=1e-9)
            # Decay must not enter Adam's adaptive moments.
            self.assertEqual(int(torch.count_nonzero(optimizer.state[p]['exp_avg'])), 0)
            self.assertEqual(int(torch.count_nonzero(optimizer.state[p]['exp_avg_sq'])), 0)

    def test_normalization_scale_collapse_is_visible_despite_nonzero_bias(self):
        model = nn.Module()
        model.unet = nn.Module()
        model.unet.mid_block = nn.Module()
        block = nn.Module()
        block.norm2 = nn.GroupNorm(1, 2)
        model.unet.mid_block.nets = nn.ModuleList([block])
        with torch.no_grad():
            block.norm2.weight.fill_(1e-14)
            block.norm2.bias.fill_(1.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.05)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        probe = UpdateProbe(model)
        optimizer.step()
        stats = probe.finish(model, optimizer)
        row = stats['updates']['unet.mid_block.nets.0.norm2']
        self.assertGreater(row['parameter_abs_mean'], .4)
        self.assertEqual(row['gradient_nonzero_elements'], 0)
        self.assertLess(stats['normalization_scales']['unet.mid_block.nets.0.norm2']['max_abs'], 1e-13)

    def test_legacy_optimizer_selection_and_wrapped_identity(self):
        from types import SimpleNamespace
        p = nn.Parameter(torch.ones(1))
        optimizer = build_optimizer([p], dict(lr=4e-4))
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer_name(SimpleNamespace(optimizer=optimizer)), 'Adam')
        with self.assertRaisesRegex(ValueError, 'Unsupported optimizer'):
            build_optimizer([p], dict(type='typo', lr=4e-4))

    def test_depth_stats_identify_saturation_and_preserve_nonfinite_values(self):
        logits = torch.zeros(6, 2, 2, 4, requires_grad=True)
        logits.data[..., 3] = 2.
        depth = torch.arange(24.).reshape(6, 2, 2).requires_grad_()
        geometry = torch.full((6, 2, 2), 255., requires_grad=True)
        output = dict(depth=depth, depth_logits=logits, geometry_depths=[geometry])
        rng = torch.get_rng_state().clone()
        stats = prediction_diagnostics(output)
        self.assertEqual(stats['geometry_0/at_255_fraction'], 1.)
        self.assertEqual(stats['geometry_0/spatial_std_m'], 0.)
        self.assertEqual(stats['depth_class/unique_count'], 1)
        self.assertEqual(stats['depth_class/dominant_fraction'], 1.)
        self.assertGreater(stats['depth/spatial_std_m'], 0.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(depth.grad)
        output['depth'] = depth.detach().clone()
        output['depth'][0, 0, 0] = float('nan')
        stats = prediction_diagnostics(output)
        self.assertGreater(stats['depth/nonfinite_fraction'], 0.)
        self.assertTrue(torch.isnan(torch.tensor(stats['depth/mean_m'])))

    def test_diagnostics_do_not_change_updates_and_record_all_validation_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = make_trainer(Path(directory)/'reference', evaluator=summary)
            reference.run()
            observed = make_trainer(Path(directory)/'observed', evaluator=summary)
            observed.cfg['diagnostics'] = dict(enabled=True)
            observed.run()
            torch.testing.assert_close(observed.model.weight, reference.model.weight, rtol=0, atol=0)
            for name, value in reference.optimizer.state[reference.model.weight].items():
                torch.testing.assert_close(observed.optimizer.state[observed.model.weight][name], value, rtol=0, atol=0)
            root = observed.work_dir
            self.assertEqual(sorted(p.parent.name for p in (root/'diagnostics').glob('*/update.json')),
                             [f'step-{s:08d}' for s in (1, 2, 4, 5)])
            result = json.loads((root/'validation/step-00000002/summary.json').read_text())
            self.assertEqual(len(result['diagnostics']['per_bin']), 7)
            self.assertEqual(result['diagnostics']['mean_per_bin']['depth_class/dominant_fraction'], 1.)


class SparseCheckpointTests(unittest.TestCase):
    def test_only_scheduled_and_final_weights_with_every_validation_preserved(self):
        events = []
        def evaluator(loader, split, step, path, checkpoint):
            self.assertEqual(checkpoint.name, f'step-{step:08d}')
            events.append((step, path.name))
            return summary()
        with tempfile.TemporaryDirectory() as directory:
            trainer = sparse_trainer(directory, evaluator=evaluator)
            trainer.run()
            self.assertEqual(sorted(p.name for p in (Path(directory)/'checkpoints').glob('step-*')),
                             [f'step-{s:08d}' for s in (5, 10, 11)])
            self.assertEqual(len(list((Path(directory)/'validation').glob('*/summary.json'))), 11)
            self.assertEqual(events, [(10, 'periodic'), (11, 'final')])
            for step in (5, 10):
                journal = json.loads((Path(directory)/f'checkpoints/step-{step:08d}/events.json').read_text())
                self.assertEqual(journal['last_val_step'], step)
                self.assertEqual(journal['validation_count'], step)

    def test_interruptions_between_saves_and_at_mini_replay_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = sparse_trainer(Path(directory)/'reference')
            expected.run()
            for failure in (1, 7, 'periodic', 'final'):
                class Interrupted(Trainer):
                    def run_validation(self):
                        if self.step == failure:
                            raise RuntimeError('injected interruption')
                        return super().run_validation()
                    def run_mini(self, reason):
                        if reason == failure:
                            raise RuntimeError('injected interruption')
                        return super().run_mini(reason)
                root = Path(directory)/str(failure)
                interrupted = sparse_trainer(root, Interrupted)
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    interrupted.run()
                if failure == 7:
                    journal = json.loads((root/'checkpoints/step-00000005/events.json').read_text())
                    self.assertEqual(journal['last_val_step'], 5)
                    self.assertEqual(journal['validation_count'], 5)
                resumed = sparse_trainer(root)
                resumed.run()
                torch.testing.assert_close(resumed.model.weight, expected.model.weight, rtol=0, atol=0)
                self.assertEqual(resumed.events, expected.events)
                self.assertEqual(resumed.step, 11)
                self.assertEqual((resumed.sampler.epoch, resumed.sampler.cursor),
                                 (expected.sampler.epoch, expected.sampler.cursor))
                for name, value in expected.optimizer.state[expected.model.weight].items():
                    torch.testing.assert_close(resumed.optimizer.state[resumed.model.weight][name], value, rtol=0, atol=0)

    def test_legacy_bf16_checkpoint_cannot_silently_resume_fp32_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            old = make_trainer(directory, evaluator=summary)
            old.model.to(torch.bfloat16)
            old.cfg['model']['parameter_dtype'] = 'bfloat16'
            path = save_checkpoint(directory, 0, old.model, old.optimizer, old.sampler, old.events, old.cfg)
            new = make_trainer(directory, evaluator=summary)
            new.cfg['model']['parameter_dtype'] = 'float32'
            before = new.model.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'Checkpoint configuration differs for model'):
                load_checkpoint(path, new.model, new.optimizer, new.sampler, new.cfg)
            torch.testing.assert_close(new.model.weight, before, rtol=0, atol=0)

    def test_old_adam_config_and_runtime_type_cannot_resume_adamw(self):
        with tempfile.TemporaryDirectory() as directory:
            old = make_trainer(directory, evaluator=summary)
            old.cfg['optimizer'].pop('type')
            old.optimizer = build_optimizer(old.model.parameters(), old.cfg['optimizer'])
            with torch.no_grad():
                old.model.weight.fill_(.8)
            path = save_checkpoint(directory, 0, old.model, old.optimizer, old.sampler, old.events, old.cfg)
            new = make_trainer(directory, evaluator=summary, optimizer_type='AdamW')
            before = new.model.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'Checkpoint configuration differs for optimizer'):
                load_checkpoint(path, new.model, new.optimizer, new.sampler, new.cfg)
            # Even without config comparison, serialized runtime identity catches the
            # otherwise state-dict-compatible Adam -> AdamW mismatch before mutating weights.
            with self.assertRaisesRegex(ValueError, 'Checkpoint optimizer type differs'):
                load_checkpoint(path, new.model, new.optimizer)
            torch.testing.assert_close(new.model.weight, before, rtol=0, atol=0)

    def test_adam_and_legacy_adamw_survive_complete_state_reload(self):
        for kind in ('Adam', 'AdamW'):
            with self.subTest(optimizer=kind), tempfile.TemporaryDirectory() as directory:
                reference = make_trainer(directory, evaluator=summary, optimizer_type=kind)
                reference.run()
                resumed = make_trainer(directory, evaluator=summary, optimizer_type=kind)
                state = load_checkpoint(reference.checkpoint, resumed.model, resumed.optimizer,
                                        resumed.sampler, resumed.cfg)
                self.assertEqual(state['optimizer_type'], kind)
                self.assertEqual(optimizer_name(resumed.optimizer), kind)
                torch.testing.assert_close(resumed.model.weight, reference.model.weight, rtol=0, atol=0)
                for key, value in reference.optimizer.state[reference.model.weight].items():
                    torch.testing.assert_close(resumed.optimizer.state[resumed.model.weight][key], value, rtol=0, atol=0)

    def test_completed_adamw_snapshot_cannot_be_relabelled_as_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            old = make_trainer(directory, evaluator=summary, optimizer_type='AdamW')
            old.run()
            new = make_trainer(directory, evaluator=summary, optimizer_type='Adam')
            before = new.model.weight.detach().clone()
            # Reject even evaluation when it is labelled with the wrong training recipe.
            with self.assertRaisesRegex(ValueError, 'configuration differs for optimizer'):
                load_checkpoint(old.checkpoint, new.model, cfg=new.cfg)
            with self.assertRaisesRegex(ValueError, 'optimizer type differs'):
                load_checkpoint(old.checkpoint, new.model, new.optimizer)
            torch.testing.assert_close(new.model.weight, before, rtol=0, atol=0)
            # The saved historical config still permits evaluation without an optimizer.
            load_checkpoint(old.checkpoint, new.model, cfg=old.cfg)
            torch.testing.assert_close(new.model.weight, old.model.weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
