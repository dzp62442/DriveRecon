"""Small prepared-data fixtures, always created in the caller's temporary directory."""

import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch

from comp_svfgs.dataset_omniscene import CAMERAS
from tests.omniscene_helpers import ToyDataset


def prepared_fixture(root, name='ddad', train=True):
    root = Path(root)
    processed = root/'processed'
    processed.mkdir(parents=True)
    (processed/'bin_infos').mkdir()
    cameras = ('CAMERA_01', 'CAMERA_06', 'CAMERA_05', 'CAMERA_09', 'CAMERA_07', 'CAMERA_08')
    sensors = {}
    for ci, cam in enumerate(CAMERAS):
        sensors[cam] = []
        for frame in range(3):
            stem = f'{ci}_{frame}'
            Image.fromarray(np.full((32, 48, 3), 20+ci*25+frame, np.uint8)).save(processed/(stem+'.png'))
            (processed/(stem+'.json')).write_text(json.dumps(dict(camera_intrinsic=[[30., 0., 23.], [0., 32., 15.], [0., 0., 1.]])))
            np.save(processed/(stem+'.npy'), np.linspace(10., 50., 32*48, dtype=np.float32).reshape(32, 48))
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3], pose[1, 3] = ci, frame
            sensors[cam].append(dict(camera=cameras[ci], data_path=stem+'.png', intrinsic_path=stem+'.json',
                                     depth_path=stem+'.npy', sensor2lidar_transform=pose))
    for token in ('a', 'b'):
        with (processed/'bin_infos'/(token+'.pkl')).open('wb') as stream:
            pickle.dump(dict(scene_id='scene', sensor_info=sensors), stream)
    for split in (('train', 'test') if train else ('test',)):
        (processed/f'bins_{split}.json').write_text(json.dumps(dict(bins=['a', 'b'])))
    mask_root = processed/'ego_masks/vidar_v1'
    mask_root.mkdir(parents=True)
    pixels = np.full((32, 48), 255, np.uint8)
    pixels[:, 36:] = 0
    Image.fromarray(pixels).save(mask_root/'partial.png')
    (mask_root/'manifest.json').write_text(json.dumps(dict(
        scene_camera_to_variant=dict(scene={camera: 'v' for camera in cameras}),
        variants=dict(v=dict(mask_id='m')), masks=dict(m=dict(path='partial.png')))))
    # No raw RGB, selection/quality manifest, confidence, dynamic mask, or depth metadata.
    return dict(name=name, data_root=str(root), processed_subdir='processed', use_dynamic_mask=False,
                mini_size=100, val_size=10)


class CrossToyDataset(ToyDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data['target']['metric_depth'] = data['target'].pop('rel_depth')
        return data


class CheapMetrics:
    def __call__(self, gt, pred, mask=None):
        error = (gt-pred).square().mean((1, 2, 3))
        return dict(psnr=-10*error.log10(), ssim=1-error, lpips=error)
