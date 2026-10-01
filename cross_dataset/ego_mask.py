"""Read SVF-GS's prepared DDAD masks; hashes record provenance, never gate data."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


class DDADEgoMasks:
    def __init__(self, processed_root, shape):
        self.root = Path(processed_root) / 'ego_masks/vidar_v1'
        raw = (self.root / 'manifest.json').read_bytes()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.shape, self.cache = tuple(shape), {}

    def load(self, scene, infos):
        masks = []
        for info in infos:
            variant_id = self.manifest['scene_camera_to_variant'][scene][info['camera']]
            mask_id = self.manifest['variants'][variant_id]['mask_id']
            if mask_id not in self.cache:
                path = self.root / self.manifest['masks'][mask_id]['path']
                with Image.open(path) as source:
                    resized = source.resize(self.shape[::-1], Image.Resampling.NEAREST)
                    self.cache[mask_id] = torch.from_numpy(np.asarray(resized) == 255)
            masks.append(self.cache[mask_id])
        return torch.stack(masks)

    def metadata(self):
        return dict(mask_manifest_sha256=self.manifest_sha256,
                    source_commit=self.manifest.get('source_commit'), resize='PIL_NEAREST',
                    valid_value=255, invalid_value=0, novel_views=12, input_views_full_image=6)
