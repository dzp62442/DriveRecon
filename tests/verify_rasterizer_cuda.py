"""Bounded native-gradient checks, plus old/new fixed-weight forward comparison.

Run --record before rebuilding the extension, then --compare after rebuilding.
No optimizer updates, notifications, dataset scanning or formal evaluation.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record', type=Path)
    group.add_argument('--compare', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sample', type=Path, help='An existing diagnostic context/target batch')
    args = parser.parse_args()
    # Honor the user's GPU-debugging condition before creating a CUDA context.
    used = int(subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError(f'GPU is occupied ({used} MiB); diagnostic not started')

    import torch
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, _C
    from comp_svfgs.camera import make_camera, target_cameras
    from comp_svfgs.model import StaticDriveRecon
    from comp_svfgs.renderer import SurfelRenderer
    from comp_svfgs.runtime import move_to_device, atomic_json

    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    k = torch.tensor([[40., 0., 16.], [0., 40., 15.], [0., 0., 1.]], device=device)
    camera = make_camera(k, torch.eye(4, device=device), (32, 32))
    settings = GaussianRasterizationSettings(
        image_height=32, image_width=32, tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
        bg=torch.zeros(3, device=device), scale_modifier=1., viewmatrix=camera.viewmatrix,
        projmatrix=camera.projmatrix, sh_degree=0, campos=camera.center, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(settings)
    pixel = (14, 15)
    means = torch.tensor([[0., 0., 2.]], device=device)

    def render(parameters, precomputed):
        kwargs = dict(means3D=means if precomputed else parameters[0],
                      means2D=torch.zeros_like(means), colors_precomp=torch.ones(1, 3, device=device),
                      opacities=torch.full((1, 1), .5, device=device))
        if precomputed:
            kwargs['cov3D_precomp'] = parameters[0]
        else:
            kwargs.update(scales=parameters[1], rotations=torch.nn.functional.normalize(parameters[2], dim=-1))
        return rasterizer(**kwargs)[0]

    cases = {
        'precomputed_transform': (True, [torch.tensor([[.68, .37, 32., .47, .55, 30., .03, .02, 2.]], device=device)]),
        'means_scales_rotation': (False, [means.clone(), torch.tensor([[.012, .018]], device=device),
                                           torch.tensor([[.85, .2, .3, .1]], device=device)]),
    }
    saved = {}
    report = dict(extension_path=_C.__file__, aabb_backward_version=getattr(_C, 'aabb_backward_version', 0),
                  source_sha256=getattr(_C, 'aabb_backward_source_sha256', None),
                  formal_result=False, optimizer_updates=0, cases={})
    for name, (precomputed, parameters) in cases.items():
        for p in parameters:
            p.requires_grad_()
        image = render(parameters, precomputed)
        loss = image[0, pixel[0], pixel[1]]
        analytical = torch.autograd.grad(loss, parameters)
        numerical = []
        for index, p in enumerate(parameters):
            grad = torch.zeros_like(p)
            # Float32 rendering needs a finite difference large enough to exceed roundoff.
            for j in range(p.numel()):
                epsilon = 1e-3
                if not precomputed and (index == 2 or (index == 0 and j == 2)):
                    epsilon = 1e-2  # Weak rotation/z derivatives need a larger float32 signal.
                plus, minus = [v.detach().clone() for v in parameters], [v.detach().clone() for v in parameters]
                plus[index].flatten()[j] += epsilon
                minus[index].flatten()[j] -= epsilon
                a = render(plus, precomputed)[0, pixel[0], pixel[1]]
                b = render(minus, precomputed)[0, pixel[0], pixel[1]]
                grad.flatten()[j] = (a-b)/(2*epsilon)
            numerical.append(grad)
        report['cases'][name] = dict(value=float(loss), analytical=[g.tolist() for g in analytical],
                                    finite_difference=[g.tolist() for g in numerical],
                                    max_absolute_error=max(float((a-b).abs().max()) for a, b in zip(analytical, numerical)))
        saved[name] = image.detach().cpu()
        if precomputed:
            # Prove this fixture uses the 2D low-pass branch that the faulty VJP affects.
            t = parameters[0].detach().reshape(3, 3).double()
            q = t.new_tensor([9., 9., -1.])
            center = (q*t[:2]*t[2]).sum(-1)/(q*t[2].square()).sum()
            cross = torch.linalg.cross(pixel[1]*t[2]-t[0], pixel[0]*t[2]-t[1])
            rho3d = (cross[:2]/cross[2]).square().sum()
            rho2d = 2*(center-center.new_tensor([pixel[1], pixel[0]])).square().sum()
            assert rho2d < rho3d and float(loss) > 0, dict(
                rho2d=float(rho2d), rho3d=float(rho3d), value=float(loss),
                center=center.tolist(), image_max=float(image.max()),
                brightest_pixel=divmod(int(image[0].argmax()), 32), extension=_C.__file__)
            torch.testing.assert_close(loss.double(), .5*torch.exp(-rho2d/2), rtol=1e-4, atol=1e-6)
            report['cases'][name].update(rho2d=float(rho2d), rho3d=float(rho3d))

    if args.checkpoint is not None:
        if args.sample is None:
            raise ValueError('--sample is required with --checkpoint')
        state = torch.load(args.checkpoint/'state.pt', map_location='cpu', mmap=True, weights_only=False)
        model = StaticDriveRecon(state['config']['model']).to(device).eval()
        model.load_state_dict(state['model'])
        batch = move_to_device(torch.load(args.sample, map_location='cpu', weights_only=False)['batch'], device)
        renderer = SurfelRenderer(state['config']['renderer'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch['context'])
        with torch.no_grad():
            image = renderer.render_view(output['gaussians'], target_cameras(batch['target'], state['config']['renderer'])[0])
        saved['checkpoint_image'] = image.cpu()
        saved['checkpoint_gaussians'] = {k: v.cpu() for k, v in output['gaussians'].items()}
        report['checkpoint'] = str(args.checkpoint.resolve())
        report['sample'] = str(args.sample.resolve())

    if args.record is not None:
        if args.record.exists():
            raise FileExistsError(args.record)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.record)
    else:
        baseline = torch.load(args.compare, map_location='cpu', weights_only=False)
        torch.testing.assert_close(saved, baseline, rtol=0, atol=0)
        report['forward_identical'] = True
        assert report['aabb_backward_version'] == 1
        for name, row in report['cases'].items():
            for analytical, numerical in zip(row['analytical'], row['finite_difference']):
                torch.testing.assert_close(torch.tensor(analytical), torch.tensor(numerical), rtol=.025, atol=.0002,
                                           msg=name+' native backward does not match finite differences')
        report['gradient_checks_passed'] = True
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
