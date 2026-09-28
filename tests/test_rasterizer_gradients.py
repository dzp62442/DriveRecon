"""CPU checks of the very same AABB VJP helper used by the native CUDA kernel."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from comp_svfgs.checkpoint import load_checkpoint, save_checkpoint
from comp_svfgs.renderer import SurfelRenderer
from tests.omniscene_helpers import make_trainer


class AABBCenterGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler required to test the actual native helper')
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(__file__).resolve().parents[1]
        header = root/'submodules/diff-surfel-rasterization/cuda_rasterizer/aabb_math.h'
        source = Path(cls.directory.name)/'bridge.cpp'
        library = source.with_suffix('.so')
        source.write_text(
            '#include "' + str(header) + '"\n'
            'extern "C" void vjp(const double* t, double c, double x, double y, double* g) '
            '{ aabb_center_vjp(t, c, x, y, g); }\n')
        subprocess.run([compiler, '-std=c++11', '-shared', '-fPIC', '-O2', str(source), '-o', str(library)],
                       check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(library))
        cls.function = cls.library.vjp
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        cls.function.restype = None

    @staticmethod
    def center(t, cutoff=3.):
        q = t.new_tensor([cutoff**2, cutoff**2, -1.])
        return (q*t[:2]*t[2]).sum(-1)/(q*t[2].square()).sum()

    def native_vjp(self, t, cotangent, initial=None):
        source = (ctypes.c_double*9)(*t.detach().flatten().tolist())
        output = (ctypes.c_double*9)(*([0.]*9 if initial is None else initial.flatten().tolist()))
        self.function(source, 9., float(cotangent[0]), float(cotangent[1]), output)
        return torch.tensor(list(output), dtype=torch.float64).reshape(3, 3)

    def test_compiled_helper_matches_autograd_and_finite_difference(self):
        # Tilted disks and nonzero off-diagonal terms exercise the missing terms.
        generator = torch.Generator().manual_seed(7)
        for _ in range(20):
            t = torch.randn(3, 3, generator=generator, dtype=torch.float64)
            t[2, :2] *= .15
            t[2, 2] = 2.5
            t.requires_grad_()
            cotangent = torch.randn(2, generator=generator, dtype=torch.float64)
            expected, = torch.autograd.grad((self.center(t)*cotangent).sum(), t)
            actual = self.native_vjp(t, cotangent)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            numerical = torch.zeros_like(t)
            for i in range(9):
                offset = torch.zeros_like(t)
                offset.flatten()[i] = 1e-5
                numerical.flatten()[i] = ((self.center(t.detach()+offset)-self.center(t.detach()-offset))*cotangent).sum()/2e-5
            torch.testing.assert_close(actual, numerical, rtol=1e-8, atol=1e-9)

    def test_cross_terms_and_existing_gradients_are_preserved(self):
        t = torch.tensor([[2., 3., 4.], [-1., 2., 3.], [.2, .3, 2.]], dtype=torch.float64)
        cotangent = torch.tensor([1., 0.], dtype=torch.float64)
        result = self.native_vjp(t, cotangent)
        torch.testing.assert_close(result[2], torch.tensor([-8.023574, -12.035361, 3.261372], dtype=torch.float64),
                                   rtol=1e-6, atol=1e-6)
        old_denominator = t[2, 0]**2+t[2, 1]**2-t[2, 2]**2
        old_z = -t[0, 2]*(1/old_denominator+2*t[2, 2]**2/old_denominator**2)
        self.assertLess(float(old_z*result[2, 2]), 0.)  # Old derivative even had the opposite sign.
        initial = torch.arange(9, dtype=torch.float64).reshape(3, 3)
        torch.testing.assert_close(self.native_vjp(t, cotangent, initial), initial+result, rtol=0, atol=0)


class RendererVersionTests(unittest.TestCase):
    def test_stale_binary_is_rejected_and_rebuilt_binary_is_accepted(self):
        renderer = SurfelRenderer(dict(aabb_backward_version=1))
        fake = types.SimpleNamespace(_C=types.SimpleNamespace())
        with patch.dict('sys.modules', {'diff_surfel_rasterization': fake}):
            with self.assertRaisesRegex(RuntimeError, 'Rebuild'):
                renderer.validate_backend()
            fake._C.aabb_backward_version = 1
            renderer.validate_backend()

    def test_old_renderer_optimizer_history_cannot_resume_new_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            path = save_checkpoint(directory, 0, trainer.model, trainer.optimizer, trainer.sampler, trainer.events, trainer.cfg)
            trainer.cfg['renderer']['aabb_backward_version'] = 1
            before = trainer.model.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'configuration differs for renderer'):
                load_checkpoint(path, trainer.model, trainer.optimizer, trainer.sampler, trainer.cfg)
            torch.testing.assert_close(trainer.model.weight, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
