import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from comp_svfgs.checkpoint import save_checkpoint
from comp_svfgs.runtime import rng_state
from cross_dataset.checkpoint import load_evaluation_checkpoint, assert_resume_identity
from cross_dataset.config import load_config
from cross_dataset.dataset import CrossDataset, select_split, PROTOCOL
from cross_dataset.evaluation import CrossEvaluator
from cross_dataset.trainer import CrossDatasetTrainer
from tests.cross_dataset_helpers import prepared_fixture, CrossToyDataset, CheapMetrics
from tests.omniscene_helpers import make_trainer
from train_omniscene import load_config as original_config


class ConfigurationTests(unittest.TestCase):
    def test_all_configs_inherit_original_recipe_and_separate_workdirs(self):
        for size in ('112x200', '224x400'):
            original, _ = original_config(['--config', f'configs/omniscene/{size}.py', '--print-config'])
            for name in ('pandaset', 'ddad'):
                for config in (f'configs/{name}/{size}.py', f'configs/zero_shot/omniscene_to_{name}_{size}.py'):
                    cfg, _ = load_config(['--config', config, '--print-config'])
                    for key in ('model', 'renderer', 'optimizer', 'loss', 'training', 'image_shape', 'precision', 'data_loader'):
                        self.assertEqual(cfg[key], original[key], (config, key))
                    self.assertFalse(cfg.dataset.use_dynamic_mask)
                    self.assertEqual(cfg.evaluation.eval_use_ego_mask, name == 'ddad')
                    self.assertNotEqual(cfg.work_dir, original.work_dir)
                    if cfg.zero_shot:
                        self.assertIn(f'_{size}/checkpoints/step-00100001', cfg.evaluation.checkpoint)

    def test_cli_override_and_zero_shot_is_not_training(self):
        cfg, printing = load_config(['--config', 'configs/zero_shot/omniscene_to_ddad_112x200.py',
                                     '--print-config', '--data-root', '/nonexistent', '--work-dir', '/tmp/unused',
                                     '--cfg-options', 'evaluation.eval_use_ego_mask=False'])
        self.assertTrue(printing)
        self.assertFalse(cfg.evaluation.eval_use_ego_mask)
        self.assertEqual(cfg.dataset.data_root, '/nonexistent')
        with self.assertRaises(ValueError):
            load_config(['--config', 'configs/zero_shot/omniscene_to_ddad_112x200.py', '--mode', 'train'])


class DataTests(unittest.TestCase):
    def test_split_selection_matches_svfgs(self):
        bins = list(range(324))
        self.assertEqual(select_split(bins, 'train'), bins)
        self.assertEqual(select_split(bins, 'test'), bins)
        self.assertEqual(select_split(bins, 'val'), np.linspace(0, 323, 10, dtype=int).tolist())
        self.assertEqual(select_split(bins, 'test', 'mini'), np.linspace(0, 323, 100, dtype=int).tolist())
        self.assertEqual(select_split(['a'], 'test', 'mini'), ['a'])

    def test_view_order_intrinsics_depth_static_labels_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = prepared_fixture(tmp)
            for shape in ((32, 48), (16, 24)):
                data = CrossDataset(cfg, shape, 'test', eval_use_ego_mask=True)[0]
                self.assertEqual(tuple(data['context']['image'].shape), (6, 3, *shape))
                self.assertEqual(tuple(data['target']['image'].shape), (18, 3, *shape))
                torch.testing.assert_close(data['context']['image'], data['target']['image'][12:], rtol=0, atol=0)
                expected = torch.tensor([20+ci*25+f for ci in range(6) for f in (1, 2)]+[20+ci*25 for ci in range(6)]) / 255.
                torch.testing.assert_close(data['target']['image'][:, 0, 0, 0], expected)
                k = data['context']['intrinsics_pixel'][0]
                self.assertEqual(float(k[0, 0]), 30.*shape[1]/48)
                self.assertEqual(float(k[1, 2]), 15.*shape[0]/32)
                # DDAD's common frame changes; camera-local OpenCV axes stay intact.
                torch.testing.assert_close(data['context']['extrinsics'][0], torch.tensor(
                    [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))
                self.assertNotIn('metric_depth', data['context'])
                self.assertNotIn('segmentation_label', data['context'])
                self.assertAlmostEqual(float(data['target']['metric_depth'].mean()), 30., places=4)
                self.assertTrue(bool(data['target']['eval_mask'][12:].all()))
                self.assertFalse(bool(data['target']['eval_mask'][:12, :, -1].any()))
                for split in ('train', 'val'):
                    supervised = CrossDataset(cfg, shape, split)[0]
                    self.assertTrue(bool((supervised['context']['segmentation_label'] == 0).all()))
                    self.assertTrue(bool(supervised['target']['loss_mask'].all()))
                    self.assertNotIn('eval_mask', supervised['target'])

    def test_no_audits_no_masks_when_disabled_no_train_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = prepared_fixture(tmp, train=False)
            # Delete even the mask manifest: unmasked evaluation must never try reading it.
            (Path(tmp)/'processed/ego_masks/vidar_v1/manifest.json').unlink()
            dataset = CrossDataset(cfg, (32, 48), 'test')
            self.assertEqual(len(dataset), 2)
            self.assertNotIn('eval_mask', dataset[0]['target'])
            with self.assertRaises(FileNotFoundError):
                CrossDataset(cfg, (32, 48), 'train')
            # Non-finite depth is passed through, without rejection or bin replacement.
            np.save(Path(tmp)/'processed/0_0.npy', np.full((32, 48), np.nan, np.float32))
            self.assertTrue(bool(torch.isnan(dataset[0]['target']['metric_depth'][12]).all()))


def cross_toy(root, trainer_type=CrossDatasetTrainer):
    trainer = make_trainer(root, trainer_type)
    trainer.cfg.update(dataset=dict(name='ddad'), zero_shot=False)
    trainer.cfg['evaluation']['eval_use_ego_mask'] = False
    trainer.cfg['training'].update(max_steps=11, validate_every_steps=1, checkpoint_every_steps=5,
                                   mini_every_n_validations=10)
    loader = DataLoader(CrossToyDataset(), batch_size=1, generator=torch.Generator().manual_seed(90))
    trainer.mini_loader = loader
    trainer.evaluator = CrossEvaluator(trainer.cfg, trainer.model, trainer.renderer, CheapMetrics(), trainer.accelerator)
    return trainer


class EvaluationAndRecoveryTests(unittest.TestCase):
    def test_three_groups_full_coverage_and_rng_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = cross_toy(tmp)
            before = rng_state()
            state = copy.deepcopy(trainer.model.state_dict())
            result = trainer.evaluator(trainer.mini_loader, 'mini', 0, Path(tmp)/'eval')
            self.assertTrue(result['complete'])
            self.assertEqual(result['processed_bins'], 7)
            self.assertEqual(set(result['groups']), {'all_18', 'novel_12', 'input_6'})
            self.assertEqual(result['reconstruction']['measured_bins'], 6)
            path = Path(tmp)/'eval'/PROTOCOL/'full_image/per_bin_metrics.csv'
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 21)
            self.assertEqual(len({(r['bin_token'], r['group']) for r in rows}), 21)
            torch.testing.assert_close(before['torch'], rng_state()['torch'], rtol=0, atol=0)
            self.assertTrue(trainer.model.training)
            for key, value in state.items():
                torch.testing.assert_close(value, trainer.model.state_dict()[key], rtol=0, atol=0)

    def test_interrupted_evaluation_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = cross_toy(tmp)
            with patch.object(trainer.renderer, 'render_pcc_depth', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    trainer.evaluator(trainer.mini_loader, 'mini', 0, Path(tmp)/'eval')
            summary = json.loads((Path(tmp)/'eval'/PROTOCOL/'full_image/evaluation_summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['processed_bins'], 0)
            self.assertTrue(trainer.model.training)

    def test_target_training_sparse_save_resume_and_final_mini(self):
        class Interrupted(CrossDatasetTrainer):
            def run_validation(self):
                if self.step == 7:
                    raise RuntimeError('injected')
                return super().run_validation()
        with tempfile.TemporaryDirectory() as tmp:
            expected = cross_toy(Path(tmp)/'baseline')
            expected.run()
            first = cross_toy(Path(tmp)/'resume', Interrupted)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                first.run()
            resumed = cross_toy(Path(tmp)/'resume')
            notices = []
            resumed.notify = lambda title, body: notices.append((title, body))
            resumed.run()
            torch.testing.assert_close(expected.model.weight, resumed.model.weight, rtol=0, atol=0)
            self.assertEqual(sorted(p.name for p in (Path(tmp)/'resume/checkpoints').iterdir()),
                             ['step-00000005', 'step-00000010', 'step-00000011'])
            self.assertTrue(resumed.events['complete'])
            self.assertTrue(resumed.events['final_mini_complete'])
            self.assertEqual(resumed.events['validation_count'], 11)
            self.assertEqual(sum('input_6' in body for _, body in notices), 2)
            with self.assertRaises(ValueError):
                cfg = copy.deepcopy(resumed.cfg)
                cfg['dataset']['name'] = 'pandaset'
                assert_resume_identity(resumed.cfg, cfg)

    def test_model_only_load_ignores_target_recipe_but_checks_architecture_and_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = cross_toy(Path(tmp)/'source')
            trainer.cfg['dataset'] = dict(data_root='omniscene')
            ck = save_checkpoint(trainer.work_dir, 0, trainer.model, trainer.optimizer, trainer.sampler,
                                 trainer.events, trainer.cfg)
            cfg = copy.deepcopy(trainer.cfg)
            cfg.update(work_dir=str(Path(tmp)/'eval'), zero_shot=True, dataset=dict(name='ddad'))
            cfg['loss']['segmentation'] = 0.
            original = trainer.model.weight.detach().clone()
            with torch.no_grad():
                trainer.model.weight.add_(1.)
            state = load_evaluation_checkpoint(ck, trainer.model, cfg)
            torch.testing.assert_close(trainer.model.weight, original, rtol=0, atol=0)
            self.assertEqual(state['source_dataset'], 'omniscene')
            self.assertNotIn('optimizer', state)
            cfg['image_shape'] = (224, 400)
            with self.assertRaisesRegex(ValueError, 'resolution'):
                load_evaluation_checkpoint(ck, trainer.model, cfg)
