"""Read existing OmniScene assets on demand; no preprocessing or sample filtering."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def asset_path(image_path, kind, suffix):
    folders = {'samples': 'samples_' + kind, 'sweeps': 'sweeps_' + kind}
    path = Path(*[folders.get(part, part) for part in Path(image_path).parts])
    return path.with_name(path.stem + suffix)


def resize_float(array, shape):
    array = np.asarray(array, dtype=np.float32)
    if tuple(array.shape) != tuple(shape):
        array = np.array(Image.fromarray(array).resize((shape[1], shape[0]), Image.BILINEAR))
    return array


def relative_depth_from_disparity(disp):
    # Deliberately the same conversion as SVF-GS/depthsplat, including its range.
    ratio = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
    depth = 1.0 / np.maximum(disp, disp.max() / ratio)
    return (depth - depth.min()) / (depth.max() - depth.min())


def read_view(info, root, prefix, shape, metric=False, segmentation=False,
              loss_mask=False, relative=False):
    path = str(info['data_path']).replace(str(prefix), str(root), 1)
    with asset_path(path, 'param_small', '.json').open() as stream:
        k = np.asarray(json.load(stream)['camera_intrinsic'], dtype=np.float32)
    with Image.open(asset_path(path, 'small', '.jpg')) as image:
        image = image.convert('RGB')
        scale_h, scale_w = shape[0] / image.height, shape[1] / image.width
        k[0] *= scale_w
        k[1] *= scale_h
        image = np.array(image.resize((shape[1], shape[0])))
    normalized_k = k.copy()
    normalized_k[0] /= shape[1]
    normalized_k[1] /= shape[0]
    result = dict(
        image=torch.from_numpy(image).permute(2, 0, 1).float() / 255,
        extrinsics=torch.as_tensor(np.asarray(info['sensor2lidar_transform']).copy(), dtype=torch.float32),
        intrinsics=torch.from_numpy(normalized_k),
        intrinsics_pixel=torch.from_numpy(k),
    )
    if metric:
        depth = np.load(asset_path(path, 'dptm_small', '_dpt.npy'))
        result['metric_depth'] = torch.from_numpy(resize_float(depth, shape).copy())
    if loss_mask or segmentation:
        with Image.open(asset_path(path, 'mask_small', '.png')) as source_mask:
            source_mask = source_mask.convert('L')
            if loss_mask:
                mask = np.array(source_mask.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32) / 255
                result['loss_mask'] = torch.from_numpy(mask)
            if segmentation:
                mask = np.array(source_mask.resize((shape[1], shape[0]), Image.NEAREST))
                result['segmentation_label'] = torch.from_numpy((mask < 128).astype(np.int64))
    if relative:
        disp = resize_float(np.load(asset_path(path, 'dpt_small', '.npy')), shape)
        result['rel_depth'] = torch.from_numpy(relative_depth_from_disparity(disp).copy())
    return result
