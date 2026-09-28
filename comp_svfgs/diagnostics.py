"""Read-only training diagnostics; no loss terms, thresholds, or data rejection."""

from collections import Counter
import math

import torch

from .optimizer import optimizer_name


UPDATE_MODULES = (
    'unet.conv_in', 'adapter.depth_map_head', 'adapter.depth_regression_head',
    'unet.down_blocks.3.nets.0.norm1', 'unet.down_blocks.3.nets.0.norm2',
    'unet.mid_block.nets.0.norm1', 'unet.mid_block.nets.0.norm2',
    'unet.mid_block.nets.1.0.position_fusion.0', 'unet.mid_block.nets.1.0.position_fusion.3',
)


def precision_summary(model, optimizer):
    parameters = Counter()
    gradients = Counter()
    moments = {'exp_avg': Counter(), 'exp_avg_sq': Counter()}
    for parameter in model.parameters():
        parameters[str(parameter.dtype)] += parameter.numel()
        if parameter.grad is not None:
            gradients[str(parameter.grad.dtype)] += parameter.grad.numel()
    for state in optimizer.state.values():
        for name, counts in moments.items():
            if name in state:
                counts[str(state[name].dtype)] += state[name].numel()
    return dict(parameter_elements_by_dtype=dict(parameters),
                gradient_elements_by_dtype=dict(gradients),
                adam_moment_elements_by_dtype={k: dict(v) for k, v in moments.items()})


class UpdateProbe:
    """Snapshot small, named heads immediately before one optimizer update."""

    def __init__(self, model):
        self.groups = {name: [] for name in UPDATE_MODULES}
        for name, parameter in model.named_parameters():
            for prefix in self.groups:
                if parameter.requires_grad and name.startswith(prefix + '.'):
                    self.groups[prefix].append((parameter, parameter.detach().clone()))

    @torch.no_grad()
    def finish(self, model, optimizer):
        rows = {}
        for name, parameters in self.groups.items():
            if not parameters:
                continue
            count = changed = grad_count = grad_nonfinite = 0
            grad_nonzero = 0
            max_change, grad_square = 0., 0.
            parameter_abs_sum = parameter_max = gradient_max = 0.
            for parameter, previous in parameters:
                delta = parameter.detach().float() - previous.float()
                count += parameter.numel()
                changed += int(torch.count_nonzero(delta))
                local_max = float(delta.abs().max())
                max_change = max(max_change, local_max) if math.isfinite(local_max) else local_max
                magnitude = parameter.detach().float().abs()
                parameter_abs_sum += float(magnitude.sum())
                local_max = float(magnitude.max())
                parameter_max = max(parameter_max, local_max) if math.isfinite(local_max) else local_max
                if parameter.grad is not None:
                    grad = parameter.grad.detach().float()
                    grad_count += grad.numel()
                    grad_nonfinite += int((~torch.isfinite(grad)).sum())
                    grad_square += float(grad.square().sum())
                    grad_nonzero += int(torch.count_nonzero(grad))
                    local_max = float(grad.abs().max())
                    gradient_max = max(gradient_max, local_max) if math.isfinite(local_max) else local_max
            rows[name] = dict(elements=count, changed_elements=changed, changed_fraction=changed/count,
                              max_abs_change=max_change, gradient_elements=grad_count,
                              gradient_nonfinite_elements=grad_nonfinite, gradient_l2=math.sqrt(grad_square),
                              gradient_nonzero_elements=grad_nonzero, gradient_abs_max=gradient_max,
                              parameter_abs_mean=parameter_abs_sum/count, parameter_abs_max=parameter_max)
        # Norm offsets can remain nonzero after their scales collapse. Report the scale
        # separately so a surviving bias cannot hide a nearly zero normalization weight.
        scales = {}
        for name, module in model.named_modules():
            if name in self.groups and isinstance(module, (torch.nn.GroupNorm, torch.nn.BatchNorm2d)):
                if module.weight is not None:
                    weight = module.weight.detach().float().abs()
                    scales[name] = dict(mean_abs=float(weight.mean()), max_abs=float(weight.max()))
        return dict(optimizer_type=optimizer_name(optimizer), precision=precision_summary(model, optimizer),
                    updates=rows, normalization_scales=scales)


@torch.no_grad()
def prediction_diagnostics(output):
    """Per-bin statistics; spatial std is computed per camera, then averaged."""
    result = {}

    def depth_stats(prefix, value, geometry=False):
        depth = value.detach().float()
        result[prefix + '/mean_m'] = float(depth.mean())
        result[prefix + '/spatial_std_m'] = float(depth.flatten(-2).std(dim=-1, unbiased=False).mean())
        result[prefix + '/min_m'] = float(depth.min())
        result[prefix + '/max_m'] = float(depth.max())
        result[prefix + '/nonfinite_fraction'] = float((~torch.isfinite(depth)).float().mean())
        if geometry:
            result[prefix + '/at_zero_fraction'] = float((depth <= 0.).float().mean())
            result[prefix + '/at_255_fraction'] = float((depth >= 255.).float().mean())

    depth_stats('depth', output['depth'])
    logits = output['depth_logits'].detach()
    histogram = torch.bincount(logits.argmax(-1).reshape(-1), minlength=logits.shape[-1])
    result['depth_class/dominant_fraction'] = float(histogram.max() / histogram.sum())
    result['depth_class/unique_count'] = int(torch.count_nonzero(histogram))
    result['depth_class/nonfinite_logits_fraction'] = float((~torch.isfinite(logits)).float().mean())
    for index, depth in enumerate(output['geometry_depths']):
        depth_stats('geometry_' + str(index), depth, geometry=True)
    return result
