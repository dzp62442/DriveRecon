"""Mask support, per-view weighting, and undefined-value regressions on the CPU."""

import unittest
import warnings

import numpy as np
import torch

from comp_svfgs.metrics import compute_pcc
from cross_dataset.metrics import CrossImageMetrics, grouped_metrics, summarize_records


class Distance:
    def __init__(self, spatial=False):
        self.spatial = spatial

    def __call__(self, gt, pred, normalize=True):
        value = (gt-pred).square().mean(1, keepdim=True)
        return value if self.spatial else value.mean((2, 3), keepdim=True)


class MaskMetricTests(unittest.TestCase):
    def setUp(self):
        self.metrics = CrossImageMetrics.__new__(CrossImageMetrics)
        self.metrics.lpips, self.metrics.spatial_lpips = Distance(), Distance(True)
        generator = torch.Generator().manual_seed(31)
        self.gt = torch.rand(18, 3, 32, 48, generator=generator)
        self.pred = (self.gt*.8+.03).clamp(0, 1)
        self.mask = torch.ones(18, 32, 48, dtype=torch.bool)
        self.mask[:12, :, 36:] = False
        self.depth = torch.linspace(10, 80, 18*32*48).reshape(18, 32, 48)

    def test_masked_rgb_and_pcc_ignore_only_excluded_pixels(self):
        baseline = self.metrics(self.gt, self.pred, self.mask)
        changed = self.pred.clone()
        changed[:12, :, :, 36:] = 1.
        other = self.metrics(self.gt, changed, self.mask)
        for key in baseline:
            torch.testing.assert_close(baseline[key], other[key], rtol=0, atol=0)
        changed[:12, :, :, :20] = 0.
        valid_change = self.metrics(self.gt, changed, self.mask)
        for key in baseline:
            self.assertFalse(torch.equal(baseline[key][:12], valid_change[key][:12]))
        # PSNR divides by valid RGB elements, rather than all pixels.
        valid = self.mask[0]
        expected = -10*torch.log10((self.gt[0, :, valid]-self.pred[0, :, valid]).square().mean())
        torch.testing.assert_close(baseline['psnr'][0], expected)
        pred_depth = self.depth.square()
        rows = grouped_metrics('a', 's', self.gt, self.pred, self.depth, pred_depth, self.metrics, self.mask)
        expected_pcc = np.corrcoef(self.depth[self.mask].numpy(), pred_depth[self.mask].numpy())[0, 1]
        self.assertAlmostEqual(rows[0]['pcc'], expected_pcc, places=6)
        pred_depth[~self.mask] = -100000.
        rows2 = grouped_metrics('a', 's', self.gt, self.pred, self.depth, pred_depth, self.metrics, self.mask)
        self.assertEqual([r['pcc'] for r in rows], [r['pcc'] for r in rows2])

    def test_input_six_and_all_ones_use_exact_full_image_path(self):
        full = self.metrics(self.gt, self.pred)
        ones = self.metrics(self.gt, self.pred, torch.ones_like(self.mask))
        masked = self.metrics(self.gt, self.pred, self.mask)
        for key in full:
            torch.testing.assert_close(full[key], ones[key], rtol=0, atol=0)
            torch.testing.assert_close(full[key][12:], masked[key][12:], rtol=0, atol=0)

    def test_view_reduction_not_pooled_pixel_or_group_mean(self):
        rows = grouped_metrics('a', 's', self.gt, self.pred, self.depth, self.depth.square(), self.metrics, self.mask)
        summary = summarize_records(rows)
        for key in ('psnr', 'ssim', 'lpips'):
            self.assertAlmostEqual(summary['all_18'][key], (2*summary['novel_12'][key]+summary['input_6'][key])/3, places=10)
        self.assertEqual([summary[g]['num_bins'] for g in summary], [1, 1, 1])

    def test_undefined_support_propagates_without_aborting_or_skipping(self):
        mask = self.mask.clone()
        mask[0] = False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rows = grouped_metrics('a', 's', self.gt, self.pred, self.depth, torch.ones_like(self.depth), self.metrics, mask)
        self.assertEqual(len(rows), 3)
        self.assertTrue(np.isnan(rows[0]['psnr']))
        self.assertTrue(np.isnan(rows[0]['ssim']))
        self.assertTrue(np.isnan(rows[0]['lpips']))
        self.assertTrue(np.isnan(rows[0]['pcc']))
        self.assertTrue(np.isnan(summarize_records(rows)['all_18']['psnr']))
        self.assertTrue(torch.isnan(compute_pcc(torch.empty(0), torch.empty(0))))
