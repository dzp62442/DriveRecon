import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from comp_svfgs.camera import make_camera, projection_from_intrinsics
from comp_svfgs.checkpoint import resolve_checkpoint, load_checkpoint
from comp_svfgs.dataset_omniscene import OmniSceneDataset, select_split
from comp_svfgs.evaluation import Evaluator
from comp_svfgs.losses import auxiliary_losses, depth_classes, rgb_loss
from comp_svfgs.metrics import compute_pcc, compute_psnr, grouped_metrics
from comp_svfgs.model import StaticDriveRecon, parameter_counts
from comp_svfgs.renderer import SurfelRenderer
from comp_svfgs.runtime import atomic_json, preserve_rng
from comp_svfgs.sampler import ResumableBatchSampler
from comp_svfgs.trainer import Trainer, train_update
from scene.geometry import backproject, scale_intrinsics
from tests.omniscene_helpers import fixture_assets, make_trainer, CPUAccelerator, ToyModel, ToyRenderer, ToyDataset, toy_config

torch.set_num_threads(2)


def cheap_image_metrics(gt, prediction):
    n = len(gt)
    return dict(psnr=compute_psnr(gt, prediction), ssim=torch.ones(n), lpips=torch.zeros(n))


class DataProtocolTests(unittest.TestCase):
    def test_split_selection(self):
        bins = list(range(30080))
        self.assertEqual(select_split(bins, 'total'), bins)
        self.assertEqual(select_split(bins, 'mini'), bins[::14][:2048])
        self.assertEqual(select_split(bins, 'val'), list(range(0, 30000, 3000)))

    def test_camera_order_units_masks_and_no_extra_assets(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = fixture_assets(root)
            item = OmniSceneDataset(cfg, (8, 12), 'train')[0]
            context, target = item['context'], item['target']
            self.assertEqual(tuple(context['image'].shape), (6, 3, 8, 12))
            self.assertEqual(tuple(target['image'].shape), (18, 3, 8, 12))
            torch.testing.assert_close(target['image'][12:], context['image'])
            expected = torch.tensor([i+frame/10 for i in range(6) for frame in (1, 2)])
            torch.testing.assert_close(target['extrinsics'][:12, 0, 3], expected)
            self.assertTrue(torch.all(context['metric_depth'] == 42.))
            self.assertEqual(context['intrinsics_pixel'][0, 0, 0], 10.)
            self.assertTrue(torch.all(target['loss_mask'][12:] == 1.))
            self.assertTrue(((target['loss_mask'][:12] > 0) & (target['loss_mask'][:12] < 1)).any())
            self.assertEqual(set(context['segmentation_label'].unique().tolist()), {0, 1})
            self.assertNotIn('rel_depth', target)
            total = OmniSceneDataset(cfg, (8, 12), 'total')
            self.assertEqual(len(total), 2)
            evaluation = total[0]
            self.assertEqual(evaluation['target']['rel_depth'].shape, (18, 8, 12))
            self.assertNotIn('rel_depth', evaluation['context'])
            self.assertNotIn('metric_depth', evaluation['context'])
            self.assertNotIn('loss_mask', evaluation['target'])

    def test_mask_uses_full_image_denominator(self):
        mask = torch.tensor([[1., 0.], [.5, 0.]])
        loss = rgb_loss(torch.ones(3, 2, 2), torch.zeros(3, 2, 2), mask)
        self.assertEqual(float(loss), .375)


class GeometryAndModelTests(unittest.TestCase):
    def test_full_intrinsics_round_trip(self):
        shape = (14, 22)
        k = torch.tensor([[15., .5, 7.2], [0., 16., 4.3], [0., 0., 1.]])
        pose = torch.tensor([[0., -1., 0., 2.], [1., 0., 0., -3.], [0., 0., 1., 4.], [0., 0., 0., 1.]])
        depth = torch.full(shape, 5., requires_grad=True)
        world = backproject(depth, k, pose)
        camera = make_camera(k, pose, shape)
        homogeneous = torch.cat([world, torch.ones(*shape, 1)], -1)
        clip = homogeneous @ camera.projmatrix
        ndc = clip[..., :2] / clip[..., 3:4]
        pixels = ((ndc+1)*torch.tensor([shape[1], shape[0]])-1)/2
        y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
        torch.testing.assert_close(pixels, torch.stack([x, y], -1).float(), atol=5e-6, rtol=1e-6)
        world.sum().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        scaled = scale_intrinsics(k, (28, 44), shape)
        torch.testing.assert_close(scaled[:2], k[:2] / 2)

    def test_depth_bins_endpoints_and_nearest_center(self):
        centers = torch.linspace(.1, 400., 200)
        torch.testing.assert_close(depth_classes(centers), torch.arange(200))
        self.assertEqual(depth_classes(torch.tensor([-1., 800.])).tolist(), [0, 199])

    def test_static_native_network_shape_and_no_label_dependency(self):
        from mmcv import Config
        cfg = Config.fromfile('configs/omniscene/112x200.py')
        cfg.model.parameter_dtype = 'float32'  # CPU structural test; CUDA smoke uses the BF16 experiment unchanged.
        model = StaticDriveRecon(cfg.model).eval()
        for h, w in [(24, 40), (32, 56)]:
            context = dict(image=torch.rand(1, 6, 3, h, w),
                           intrinsics_pixel=torch.tensor([[30., 0., 10.], [0., 30., 10.], [0., 0., 1.]]).repeat(1, 6, 1, 1),
                           extrinsics=torch.eye(4).repeat(1, 6, 1, 1))
            with torch.no_grad():
                result = model(context)
                changed = model(dict(context, metric_depth=torch.tensor(float('nan')), segmentation_label='unused'))
            self.assertEqual(result['gaussians']['means'].shape, (1, 6*(h//2)*(w//2), 3))
            for key in result['gaussians']:
                torch.testing.assert_close(result['gaussians'][key], changed['gaussians'][key], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(result['gaussians'][key]).all())
        from scene.PointNet import UNet, TCAttention
        legacy = UNet(in_channels=3, out_channels=128)
        self.assertEqual({k: v.shape for k, v in legacy.state_dict().items()},
                         {k: v.shape for k, v in model.unet.state_dict().items()})
        for module in model.modules():
            if isinstance(module, TCAttention):
                self.assertEqual((module.num_frames, module.view_num), (1, 6))
        count = parameter_counts(model)
        self.assertEqual(count['total'], count['trainable']+count['frozen'])

    def test_pcc_renderer_uses_unclamped_center_z(self):
        captured = {}
        class Settings:
            def __init__(self, **kwargs):
                captured.update(kwargs)
        class Rasterizer:
            def __init__(self, raster_settings):
                pass
            def __call__(self, **kwargs):
                captured.update(kwargs)
                color = kwargs['colors_precomp'].mean(0).view(3, 1, 1) * .5
                return color, None, torch.full((7, 1, 1), -100.)
        fake = types.SimpleNamespace(GaussianRasterizationSettings=Settings, GaussianRasterizer=Rasterizer)
        renderer = SurfelRenderer(dict(background=(1., 1., 1.)))
        k = torch.eye(3)
        pose = torch.eye(4)
        pose[2, 3] = 2.
        camera = make_camera(k, pose, (1, 1))
        gs = dict(means=torch.tensor([[[0., 0., 12.]]]), color=torch.ones(1, 1, 3),
                  scale=torch.ones(1, 1, 3), rotation=torch.tensor([[[1., 0., 0., 0.]]]), opacity=torch.ones(1, 1, 1))
        with patch.dict('sys.modules', {'diff_surfel_rasterization': fake}):
            self.assertEqual(float(renderer.render_pcc_depth(gs, camera)), 5.)
        self.assertTrue(torch.all(captured['bg'] == 0.))
        self.assertTrue(torch.all(captured['colors_precomp'] == 10.))


class MetricTests(unittest.TestCase):
    def test_pcc_is_group_flattened_and_stateless(self):
        x = torch.tensor([[0., 1.], [100., 101.]])
        y = torch.tensor([[0., 1.], [-100., -99.]])
        reference = np.corrcoef(x.flatten().numpy(), y.flatten().numpy())[0, 1]
        self.assertAlmostEqual(float(compute_pcc(x, y)), reference, places=6)
        self.assertAlmostEqual(float(compute_pcc(x, x)), 1., places=6)
        self.assertTrue(torch.isnan(compute_pcc(torch.ones(4), torch.ones(4))))

    def test_groups_and_psnr_reduction(self):
        gt = torch.zeros(18, 3, 12, 12)
        pred = torch.cat([torch.full((12, 3, 12, 12), .1), torch.full((6, 3, 12, 12), .5)])
        depth = torch.arange(18*12*12).reshape(18, 12, 12).float()
        rows = grouped_metrics('test', gt, pred, depth, depth*2+7, cheap_image_metrics)
        self.assertAlmostEqual(rows[0]['psnr'], (12*20+6*(-20*np.log10(.5)))/18, places=5)
        self.assertAlmostEqual(rows[1]['psnr'], 20., places=5)
        self.assertAlmostEqual(rows[1]['pcc'], 1., places=6)


class RecoveryTests(unittest.TestCase):
    def test_resume_after_epoch_rollover(self):
        def evaluator(*args):
            return dict(groups={g: dict(psnr=1., ssim=1., lpips=0., pcc=1.) for g in ('all_18', 'novel_12')},
                        reconstruction=dict(mean=1.), processed_bins=1, evaluation_seconds=.1)
        class InterruptAfterRollover(Trainer):
            def run_validation(self):
                if self.step == 8:
                    raise RuntimeError('epoch rollover interruption')
                return super().run_validation()
        with tempfile.TemporaryDirectory() as directory:
            expected = make_trainer(Path(directory)/'expected', evaluator=evaluator)
            expected.cfg['training']['max_steps'] = 10
            expected.run()
            root = Path(directory)/'resumed'
            stopped = make_trainer(root, InterruptAfterRollover, evaluator)
            stopped.cfg['training']['max_steps'] = 10
            with self.assertRaisesRegex(RuntimeError, 'epoch rollover interruption'):
                stopped.run()
            resumed = make_trainer(root, evaluator=evaluator)
            resumed.cfg['training']['max_steps'] = 10
            resumed.run()
            self.assertEqual((resumed.sampler.epoch, resumed.sampler.cursor), (1, 3))
            torch.testing.assert_close(resumed.sampler.order, expected.sampler.order)
            torch.testing.assert_close(resumed.model.weight, expected.model.weight, rtol=0, atol=0)
            self.assertEqual(resumed.events, expected.events)

    def test_prefetch_does_not_advance_consumed_cursor(self):
        sampler = ResumableBatchSampler(10, 1)
        iterator = iter(sampler)
        prefetched = [next(iterator) for _ in range(5)]
        sampler.advance()
        sampler.advance()
        state = sampler.state_dict()
        other = ResumableBatchSampler(10, 2)
        other.load_state_dict(state)
        self.assertEqual(next(iter(other)), prefetched[2])

    def test_chunked_gaussian_gradients_match_direct_view_sum(self):
        from torch.utils.data import DataLoader
        batch = next(iter(DataLoader(ToyDataset(), batch_size=1)))
        model = ToyModel()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        cfg = toy_config('.')
        torch.manual_seed(22)
        train_update(model, ToyRenderer(), batch, optimizer, CPUAccelerator(), cfg)
        torch.manual_seed(22)
        output = reference(batch['context'])
        aux, _ = auxiliary_losses(output, batch['context'], cfg['loss'], cfg['model'])
        camera = make_camera(batch['target']['intrinsics_pixel'][0, 0], batch['target']['extrinsics'][0, 0], (12, 12))
        total = aux
        for i in range(18):
            total = total + rgb_loss(ToyRenderer().render_view(output['gaussians'], camera),
                                    batch['target']['image'][0, i], batch['target']['loss_mask'][0, i])
        total.backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-6, atol=1e-6)

    def test_interruptions_recover_optimizer_rng_and_pending_events(self):
        from comp_svfgs.evaluation import timing_summary
        def summary(*args):
            torch.rand(9)  # Evaluations must not perturb training RNG.
            return dict(groups={name: dict(psnr=1., ssim=1., lpips=0., pcc=1.) for name in ('all_18', 'novel_12')},
                        reconstruction=timing_summary([1., 2.], 1), processed_bins=2, evaluation_seconds=.1)
        def evaluator(*args):
            with preserve_rng():
                return summary(*args)
        with tempfile.TemporaryDirectory() as directory:
            expected = make_trainer(Path(directory)/'reference', evaluator=evaluator)
            expected.run()
            for failure in ('validation', 'periodic', 'final'):
                class Interrupted(Trainer):
                    def run_validation(self):
                        if failure == 'validation':
                            raise RuntimeError('injected interruption')
                        return super().run_validation()
                    def run_mini(self, reason):
                        if reason == failure:
                            raise RuntimeError('injected interruption')
                        return super().run_mini(reason)
                root = Path(directory)/failure
                interrupted = make_trainer(root, Interrupted, evaluator)
                with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                    interrupted.run()
                resumed = make_trainer(root, evaluator=evaluator)
                resumed.run()
                torch.testing.assert_close(resumed.model.weight, expected.model.weight, rtol=0, atol=0)
                self.assertEqual(resumed.step, 5)
                self.assertEqual(resumed.events, expected.events)
                self.assertEqual(resumed.events['validation_count'], 2)
                self.assertEqual(resumed.events['last_mini_step'], 4)
                self.assertTrue(resumed.events['final_mini_complete'])
                for name, value in expected.optimizer.state[expected.model.weight].items():
                    torch.testing.assert_close(resumed.optimizer.state[resumed.model.weight][name], value, rtol=0, atol=0)
                again = make_trainer(root, evaluator=lambda *_: self.fail('Completed final mini must not rerun'))
                again.run()
                self.assertEqual(again.step, 5)
                # A stale pointer and an incomplete newer snapshot never beat the latest complete step.
                atomic_json(root/'latest.json', dict(step=2, checkpoint='checkpoints/step-00000002'))
                (root/'checkpoints/step-99999999').mkdir()
                self.assertEqual(resolve_checkpoint(root).name, 'step-00000005')

    def test_evaluator_writes_both_groups_and_keeps_warmup_quality(self):
        from torch.utils.data import DataLoader, Subset
        with tempfile.TemporaryDirectory() as root:
            model = ToyModel().train()
            cfg = toy_config(root)
            loader = DataLoader(Subset(ToyDataset(), [0, 1]), batch_size=1)
            evaluator = Evaluator(cfg, model, ToyRenderer(), cheap_image_metrics, CPUAccelerator())
            result = evaluator(loader, 'mini', 3, Path(root)/'evaluation')
            self.assertTrue(model.training)
            self.assertTrue(result['complete'])
            self.assertEqual(result['groups']['all_18']['num_bins'], 2)
            self.assertEqual(result['groups']['novel_12']['num_bins'], 2)
            self.assertEqual(result['reconstruction']['measured_bins'], 1)
            self.assertEqual(len((Path(root)/'evaluation/per_bin_metrics.csv').read_text().splitlines()), 5)


if __name__ == '__main__':
    unittest.main()
