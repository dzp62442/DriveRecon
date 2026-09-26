"""One center bin: six context cameras, twelve novel targets, six input targets."""

import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .omniscene_io import read_view

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def select_split(bins, split):
    if split in ('train', 'total'):
        return bins
    if split == 'val':
        return bins[:30000:3000][:10]
    if split == 'mini':
        return bins[0::14][:2048]
    raise ValueError('Unknown configured split: ' + split)


def stack_views(views):
    return {key: torch.stack([view[key] for view in views]) for key in views[0]}


class OmniSceneDataset(Dataset):
    def __init__(self, cfg, image_shape, split):
        self.root = Path(cfg['data_root']).expanduser()
        self.directory = self.root / cfg['data_version']
        self.prefix = cfg['dataset_prefix']
        self.shape = tuple(image_shape)
        self.split = split
        self.supervision = split in ('train', 'val')
        self.use_dynamic_mask = cfg.get('use_dynamic_mask', True)
        filename = 'bins_train_3.2m.json' if split == 'train' else 'bins_val_3.2m.json'
        with (self.directory / filename).open() as stream:
            self.bin_tokens = select_split(json.load(stream)['bins'], split)

    def __len__(self):
        return len(self.bin_tokens)

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        with (self.directory / 'bin_infos_3.2m' / (token + '.pkl')).open('rb') as stream:
            sensors = pickle.load(stream)['sensor_info']
        kwargs = dict(root=self.root, prefix=self.prefix, shape=self.shape,
                      relative=not self.supervision)
        centers = [read_view(sensors[cam][0], metric=self.supervision,
                             segmentation=self.supervision, **kwargs) for cam in CAMERAS]
        novel = [read_view(sensors[cam][idx], loss_mask=self.supervision and self.use_dynamic_mask,
                           **kwargs) for cam in CAMERAS for idx in (1, 2)]
        target_keys = ('image', 'extrinsics', 'intrinsics', 'intrinsics_pixel')
        if not self.supervision:
            target_keys += ('rel_depth',)
        target_views = []
        for i, view in enumerate(novel + centers):
            target_view = {key: view[key] for key in target_keys}
            if self.supervision:
                target_view['loss_mask'] = (view['loss_mask'] if i < 12 and self.use_dynamic_mask
                                             else torch.ones(self.shape, dtype=torch.float32))
            target_views.append(target_view)
        # DA2 is target-only even when the last six targets duplicate context RGB.
        context = stack_views([{k: v for k, v in view.items() if k != 'rel_depth'} for view in centers])
        return dict(scene=token, bin_token=token, context=context, target=stack_views(target_views))
