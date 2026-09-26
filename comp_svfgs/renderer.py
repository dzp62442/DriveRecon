"""Native 2D Gaussian rendering, including a separate accumulated-center-z pass."""

import torch


class SurfelRenderer:
    def __init__(self, cfg):
        self.cfg = cfg

    def render_view(self, gaussians, camera, depth=False):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        means = gaussians['means'][0].float().contiguous()
        with torch.autocast(device_type=means.device.type, enabled=False):
            if depth:
                # viewmatrix stores transposed w2c. This is center z, not surfel intersection z.
                z = means @ camera.viewmatrix[:3, 2] + camera.viewmatrix[3, 2]
                colors = z[:, None].expand(-1, 3).contiguous()
                background = means.new_zeros(3)
            else:
                colors = gaussians['color'][0].float().contiguous()
                background = means.new_tensor(self.cfg['background'])
            settings = GaussianRasterizationSettings(
                image_height=camera.height, image_width=camera.width,
                tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
                bg=background, scale_modifier=1., viewmatrix=camera.viewmatrix,
                projmatrix=camera.projmatrix, sh_degree=0, campos=camera.center,
                prefiltered=False, debug=False)
            image, _, _ = GaussianRasterizer(raster_settings=settings)(
                means3D=means, means2D=torch.zeros_like(means), shs=None,
                colors_precomp=colors, opacities=gaussians['opacity'][0].float().contiguous(),
                scales=gaussians['scale'][0, :, :2].float().contiguous(),
                rotations=gaussians['rotation'][0].float().contiguous(), cov3D_precomp=None)
            return image.mean(0) if depth else image.clamp(0, 1)

    def render_pcc_depth(self, gaussians, camera):
        # Do not clamp to RGB range or divide by alpha.
        return self.render_view(gaussians, camera, depth=True)
