"""Synthetic fixtures: no production data is inspected or filtered by these tests."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from comp_svfgs.dataset_omniscene import CAMERAS
from comp_svfgs.omniscene_io import asset_path
from comp_svfgs.sampler import ResumableBatchSampler
from comp_svfgs.trainer import Trainer


def fixture_assets(root):
    root = Path(root)
    directory = root / 'interp_12Hz_trainval'
    (directory / 'bin_infos_3.2m').mkdir(parents=True)
    sensors = {}
    for camera_index, cam in enumerate(CAMERAS):
        sensors[cam] = []
        for frame in range(3):
            path = root / ('samples' if frame == 0 else 'sweeps') / cam / f'{frame}.jpg'
            k = [[20., 0., 9.], [0., 18., 7.], [0., 0., 1.]]
            for kind, suffix in [('small', '.jpg'), ('param_small', '.json'), ('dptm_small', '_dpt.npy'),
                                 ('dpt_small', '.npy'), ('mask_small', '.png')]:
                asset_path(path, kind, suffix).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((16, 24, 3), 20+camera_index*25+frame, np.uint8)).save(asset_path(path, 'small', '.jpg'))
            asset_path(path, 'param_small', '.json').write_text(json.dumps({'camera_intrinsic': k}))
            np.save(asset_path(path, 'dptm_small', '_dpt.npy'), np.full((16, 24), 42.0, np.float32))
            np.save(asset_path(path, 'dpt_small', '.npy'), np.linspace(1., 3., 16*24, dtype=np.float32).reshape(16, 24))
            mask = np.full((16, 24), 255, np.uint8)
            mask[:, 11:] = 0
            Image.fromarray(mask).save(asset_path(path, 'mask_small', '.png'))
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = camera_index + frame / 10
            sensors[cam].append(dict(data_path=str(path).replace(str(root), '/datasets/nuScenes'), sensor2lidar_transform=pose))
    # Intentionally no LIDAR_TOP, scene mapping, confidence, or sky assets.
    for name in ['one', 'two']:
        with (directory / 'bin_infos_3.2m' / (name+'.pkl')).open('wb') as stream:
            pickle.dump(dict(sensor_info=sensors), stream)
    for split in ['train', 'val']:
        (directory / f'bins_{split}_3.2m.json').write_text(json.dumps(dict(bins=['one', 'two'])))
    return dict(data_root=str(root), data_version='interp_12Hz_trainval', dataset_prefix='/datasets/nuScenes', use_dynamic_mask=True)


class CPUAccelerator:
    device = torch.device('cpu')
    scaler = None

    def autocast(self):
        return nullcontext()

    def backward(self, value):
        value.backward()

    def unwrap_model(self, model):
        return model


class ToyDataset(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        k = torch.tensor([[12., 0., 6.], [0., 12., 6.], [0., 0., 1.]])
        context = dict(image=torch.full((6, 3, 12, 12), index/10.),
                       metric_depth=torch.full((6, 12, 12), 50.),
                       segmentation_label=torch.zeros(6, 12, 12, dtype=torch.long))
        depth = torch.linspace(0, 1, 18*12*12).reshape(18, 12, 12)
        target = dict(image=torch.full((18, 3, 12, 12), 0.1+index/10.),
                      loss_mask=torch.ones(18, 12, 12), rel_depth=depth,
                      intrinsics_pixel=k.repeat(18, 1, 1), extrinsics=torch.eye(4).repeat(18, 1, 1))
        return dict(context=context, target=target, bin_token=str(index), scene=str(index))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, context):
        signal = self.weight + context['image'].mean()*.1 + torch.rand(())*.01
        gaussians = dict(means=signal.expand(1, 24, 3), color=signal.square().expand(1, 24, 3),
                         scale=(signal+1).expand(1, 24, 3), rotation=signal.expand(1, 24, 4),
                         opacity=signal.sigmoid().expand(1, 24, 1))
        logits = torch.stack([signal, -signal, signal*.4]).expand(6, 2, 2, 3)
        return dict(gaussians=gaussians, depth_logits=logits, depth=(signal+30).expand(6, 2, 2),
                    segmentation=logits, geometry_depths=[(signal+60).expand(6, 2, 2)])


class ToyRenderer:
    def render_view(self, gaussians, camera):
        value = sum(v.mean() for v in gaussians.values()).sigmoid()
        return value.expand(3, camera.height, camera.width)

    def render_pcc_depth(self, gaussians, camera):
        return torch.linspace(0, 1, camera.height*camera.width).reshape(camera.height, camera.width)


def toy_config(root):
    return dict(work_dir=str(root), image_shape=(12, 12), precision='no',
                renderer=dict(znear=.01, zfar=1e8, background=(0., 0., 0.)),
                model=dict(depth_min=.1, depth_max=400., num_samples=3),
                loss=dict(rgb=1., segmentation=1., depth_class=2., depth_reg=2., geometry_aux=2.),
                optimizer=dict(lr=.001), evaluation=dict(time_skip_bins=1, save_images=False),
                training=dict(max_steps=5, validate_every_steps=2, mini_every_n_validations=2,
                              checkpoint_every_steps=2, log_every_steps=1, resume='auto'))


def make_trainer(root, trainer_type=Trainer, evaluator=None):
    torch.manual_seed(11)
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    sampler = ResumableBatchSampler(7, 37)
    loader = DataLoader(ToyDataset(), batch_sampler=sampler, generator=sampler.loader_generator)
    evaluation_loader = DataLoader(ToyDataset(), batch_size=1, generator=torch.Generator().manual_seed(90))
    return trainer_type(toy_config(root), model, optimizer, sampler, loader, evaluation_loader, evaluation_loader,
                        ToyRenderer(), evaluator, CPUAccelerator(), lambda *_: None)
