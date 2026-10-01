"""Bounded GPU integration check. All outputs must be under /tmp; no notifications.

112x200 uses the real OmniScene checkpoint; 224x400 checks randomly initialized
network interfaces only. One validation sample may be used for a diagnostic update;
it never creates a target-domain train split or a formal training result.
"""

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if Path('/tmp') not in root.parents:
        raise ValueError('Diagnostic outputs must be inside /tmp')
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if max(float(line) for line in memory.splitlines()) >= 1024:
        raise RuntimeError('GPU memory usage is >=1 GiB; postpone this diagnostic')
    root.mkdir(parents=True, exist_ok=True)
    for variable, folder in [('CUDA_CACHE_PATH', 'cuda_cache'), ('TORCH_EXTENSIONS_DIR', 'extensions')]:
        directory = root/folder
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Subset
    from comp_svfgs.checkpoint import save_checkpoint, load_checkpoint
    from comp_svfgs.model import StaticDriveRecon
    from comp_svfgs.optimizer import build_optimizer
    from comp_svfgs.renderer import SurfelRenderer
    from comp_svfgs.runtime import atomic_json, move_to_device
    from comp_svfgs.sampler import ResumableBatchSampler
    from comp_svfgs.trainer import train_update, validate
    from cross_dataset.config import load_config
    from cross_dataset.checkpoint import load_evaluation_checkpoint
    from cross_dataset.dataset import CrossDataset
    from cross_dataset.evaluation import CrossEvaluator
    from cross_dataset.metrics import CrossImageMetrics

    torch.set_num_threads(4)
    accelerator = Accelerator(mixed_precision='bf16')
    if accelerator.device.type != 'cuda' or accelerator.num_processes != 1:
        raise RuntimeError('This diagnostic needs one CUDA process')
    root.mkdir(parents=True, exist_ok=True)
    metrics = CrossImageMetrics(accelerator.device, masked=True)
    reports = []
    for size in ('112x200', '224x400'):
        for name in ('pandaset', 'ddad'):
            cfg, _ = load_config(['--config', f'configs/{name}/{size}.py', '--work-dir', str(root/name/size)])
            cfg.feishu.enabled = False
            cfg.evaluation.time_skip_bins = 0
            torch.manual_seed(cfg.seed)
            torch.backends.cudnn.deterministic = cfg.deterministic
            torch.backends.cudnn.benchmark = False
            model = StaticDriveRecon(cfg.model)
            source = None
            checkpoint = None
            if size == '112x200':
                source_cfg, _ = load_config(['--config', f'configs/zero_shot/omniscene_to_{name}_{size}.py'])
                checkpoint = source_cfg.evaluation.checkpoint
                source = load_evaluation_checkpoint(checkpoint, model, cfg)
            model = accelerator.prepare(model)
            renderer = SurfelRenderer(cfg.renderer)
            renderer.validate_backend()
            records = {}
            states = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
            for masked in ((False, True) if name == 'ddad' else (False,)):
                cfg.evaluation.eval_use_ego_mask = masked
                dataset = CrossDataset(cfg.dataset, cfg.image_shape, 'test', eval_use_ego_mask=masked)
                loader = DataLoader(Subset(dataset, [0]), batch_size=1, generator=torch.Generator().manual_seed(2))
                result = CrossEvaluator(cfg, model, renderer, metrics, accelerator, source)(
                    loader, 'debug_one_bin', source['global_step'] if source else 0, Path(cfg.work_dir)/'eval', checkpoint)
                assert result['processed_bins'] == 1 and result['complete']
                assert set(result['groups']) == {'all_18', 'novel_12', 'input_6'}
                for row in result['groups'].values():
                    assert all(torch.isfinite(torch.tensor(row[k])) for k in ('psnr', 'ssim', 'lpips', 'pcc'))
                records['masked' if masked else 'full'] = result
            if name == 'ddad':
                assert records['masked']['groups']['input_6'] == records['full']['groups']['input_6']
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), states[key], rtol=0, atol=0)
            del states
            # Test the training adapter and native backward with a clearly labelled diagnostic sample.
            optimizer = build_optimizer(model.optimizer_groups(cfg.optimizer), cfg.optimizer)
            optimizer = accelerator.prepare(optimizer)
            val = CrossDataset(cfg.dataset, cfg.image_shape, 'val')
            loader = DataLoader(Subset(val, [0]), batch_size=1, generator=torch.Generator().manual_seed(3))
            batch = move_to_device(next(iter(loader)), accelerator.device)
            model.train()
            update = train_update(model, renderer, batch, optimizer, accelerator, cfg)
            assert all(torch.isfinite(torch.tensor(v)) for v in update.values())
            validation = validate(model, renderer, loader, accelerator, cfg)
            sampler = ResumableBatchSampler(len(val), cfg.seed)
            sampler.advance()
            events = dict(last_val_step=0, validation_count=0, last_mini_step=0, final_mini_complete=False, complete=False)
            saved = save_checkpoint(cfg.work_dir, 1, accelerator.unwrap_model(model), optimizer, sampler, events, cfg)
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                expected = model(batch['context'])['gaussians']['means'].clone()
            with torch.no_grad():
                next(model.parameters()).add_(1.)
            state = load_checkpoint(saved, accelerator.unwrap_model(model), optimizer, sampler, cfg)
            del state
            with torch.no_grad(), accelerator.autocast():
                actual = model(batch['context'])['gaussians']['means']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reports.append(dict(dataset=name, resolution=size, formal_result=False,
                                initialization='omniscene_pretrained' if source else 'random_interface_check_only',
                                evaluation=records, diagnostic_update=update, validation=validation,
                                checkpoint_roundtrip='passed'))
            atomic_json(root/'verification.json', dict(formal_result=False, complete=False, results=reports))
            print(name, size, 'PASSED', flush=True)
            del model, optimizer, batch, expected, actual
            accelerator.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(root/'verification.json', dict(formal_result=False, complete=True, results=reports))


if __name__ == '__main__':
    main()
