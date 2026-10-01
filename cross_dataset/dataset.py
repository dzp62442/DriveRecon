"""Static six-camera reconstruction with the prepared temporal18 target protocol."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from comp_svfgs.dataset_omniscene import CAMERAS, stack_views
from .ego_mask import DDADEgoMasks
from .io import read_view

PROTOCOL = 'novel18_s10_d1p6_min0p1'


def pixel_protocol(enabled):
    return 'ddad_ego_novel12_v1' if enabled else 'full_image'


def select_split(bins, split, test_range='total', mini_size=100, val_size=10):
    if split not in ('train', 'val', 'test') or test_range not in ('total', 'mini'):
        raise ValueError('Use train/val/test and test_range=total/mini')
    size = val_size if split == 'val' else mini_size if split == 'test' and test_range == 'mini' else None
    if size is None:
        return bins
    indices = np.linspace(0, len(bins)-1, min(size, len(bins)), dtype=int)
    return [bins[i] for i in indices]


class CrossDataset(Dataset):
    def __init__(self, cfg, image_shape, split, test_range='total', eval_use_ego_mask=False):
        self.name, self.shape, self.split = cfg['name'], tuple(image_shape), split
        if self.name not in ('pandaset', 'ddad'):
            raise ValueError('CrossDataset supports pandaset and ddad')
        if eval_use_ego_mask and (self.name != 'ddad' or split != 'test'):
            raise ValueError('Ego masks belong only to DDAD metric evaluation')
        self.supervision = split in ('train', 'val')
        self.processed_root = Path(cfg['data_root']).expanduser() / cfg['processed_subdir']
        self.test_range = test_range
        storage = 'train' if split == 'train' else 'test'
        with (self.processed_root / f'bins_{storage}.json').open() as stream:
            bins = json.load(stream)['bins']
        self.bin_tokens = select_split(bins, split, test_range, cfg['mini_size'], cfg['val_size'])
        self.ego_masks = DDADEgoMasks(self.processed_root, self.shape) if eval_use_ego_mask else None

    def __len__(self):
        return len(self.bin_tokens)

    def evaluation_metadata(self):
        return dict(dataset=self.name, dataset_split=self.split, test_range=self.test_range,
                    processed_root=str(self.processed_root), view_protocol=PROTOCOL,
                    pixel_protocol=pixel_protocol(self.ego_masks is not None),
                    eval_mask=self.ego_masks.metadata() if self.ego_masks is not None else None)

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        with (self.processed_root / 'bin_infos' / (token + '.pkl')).open('rb') as stream:
            info = pickle.load(stream)
        center_infos = [info['sensor_info'][cam][0] for cam in CAMERAS]
        novel_infos = [info['sensor_info'][cam][i] for cam in CAMERAS for i in (1, 2)]
        # Centers supply training depth or evaluation reference; neither is a new model input.
        centers = [read_view(v, self.processed_root, self.shape, depth=True) for v in center_infos]
        novel = [read_view(v, self.processed_root, self.shape, depth=not self.supervision) for v in novel_infos]
        camera_keys = ('image', 'intrinsics', 'intrinsics_pixel', 'extrinsics')
        context = stack_views([{k: v[k] for k in camera_keys} for v in centers])
        keys = camera_keys if self.supervision else camera_keys + ('metric_depth',)
        target = stack_views([{k: v[k] for k in keys} for v in novel + centers])
        if self.supervision:
            context['metric_depth'] = torch.stack([v['metric_depth'] for v in centers])
            context['segmentation_label'] = torch.zeros((6, *self.shape), dtype=torch.long)
            target['loss_mask'] = torch.ones((18, *self.shape))
        if self.ego_masks is not None:
            target['eval_mask'] = torch.cat((self.ego_masks.load(info['scene_id'], novel_infos),
                                             torch.ones((6, *self.shape), dtype=torch.bool)))
        return dict(bin_token=token, scene=info['scene_id'], context=context, target=target)
