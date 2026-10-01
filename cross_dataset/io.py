"""Read prepared assets on demand, without audits or preprocessing."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def read_view(info, processed_root, shape, depth=False):
    root = Path(processed_root)
    with (root / info['intrinsic_path']).open() as stream:
        param = json.load(stream)
    k = np.asarray(param['camera_intrinsic'], dtype=np.float32).copy()
    height, width = shape
    with Image.open(root / info['data_path']) as source:
        image = source.convert('RGB')
        k[0] *= width / image.width
        k[1] *= height / image.height
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        rgb = np.array(image, dtype=np.float32) / 255.
    normalized = k.copy()
    normalized[0] /= width
    normalized[1] /= height
    result = dict(image=torch.from_numpy(rgb).permute(2, 0, 1),
                  intrinsics_pixel=torch.from_numpy(k), intrinsics=torch.from_numpy(normalized),
                  extrinsics=torch.tensor(np.asarray(info['sensor2lidar_transform']), dtype=torch.float32))
    if depth:
        value = np.load(root / info['depth_path'], allow_pickle=False)
        if value.shape != (height, width):
            value = np.asarray(Image.fromarray(value).resize((width, height), Image.Resampling.BILINEAR))
        result['metric_depth'] = torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
    return result
