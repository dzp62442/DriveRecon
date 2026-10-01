"""DDAD common-frame alignment, without running a model or using CUDA."""

from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from comp_svfgs.camera import target_cameras
from comp_svfgs.dataset_omniscene import CAMERAS
from cross_dataset.dataset import CrossDataset
from scene.geometry import backproject
from tests.cross_dataset_helpers import prepared_fixture


def oriented_fixture(root):
    cfg = prepared_fixture(root)
    path = Path(root) / 'processed/bin_infos/a.pkl'
    with path.open('rb') as stream:
        info = pickle.load(stream)
    # Camera-local right/down/forward expressed in DDAD forward/left/up axes.
    front = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    for ci, camera in enumerate(CAMERAS):
        for frame, sensor in enumerate(info['sensor_info'][camera]):
            yaw, pitch = ci * np.pi / 3 + frame * .04, frame * .03
            rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                           [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
            rx = np.array([[1., 0., 0.], [0., np.cos(pitch), -np.sin(pitch)],
                           [0., np.sin(pitch), np.cos(pitch)]])
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rz @ rx @ front
            pose[:3, 3] = [2. + frame * .7, -3. + ci * .2, 1.5 + frame * .1]
            sensor['sensor2lidar_transform'] = pose
    with path.open('wb') as stream:
        pickle.dump(info, stream)
    centers = [info['sensor_info'][cam][0] for cam in CAMERAS]
    novel = [info['sensor_info'][cam][i] for cam in CAMERAS for i in (1, 2)]
    poses = {side: torch.from_numpy(np.stack([v['sensor2lidar_transform'] for v in views]))
             for side, views in (('context', centers), ('target', novel + centers))}
    return cfg, poses


def model_axes(points):
    return torch.stack((-points[..., 1], points[..., 0], points[..., 2]), dim=-1)


class DDADFrameTests(unittest.TestCase):
    def test_all_loading_modes_align_once_and_preserve_other_data(self):
        modes = (('train', 'total', False), ('val', 'total', False),
                 ('test', 'total', False), ('test', 'total', True),
                 ('test', 'mini', False), ('test', 'mini', True))
        with tempfile.TemporaryDirectory() as tmp:
            cfg, poses = oriented_fixture(tmp)
            for shape in ((112, 200), (224, 400)):
                for split, test_range, masked in modes:
                    with self.subTest(shape=shape, split=split, test_range=test_range, masked=masked):
                        dataset = CrossDataset(cfg, shape, split, test_range, masked)
                        aligned = dataset[0]
                        # The PandaSet branch must retain every prepared pose verbatim.
                        original = CrossDataset(dict(cfg, name='pandaset'), shape, split, test_range)[0]
                        for side in ('context', 'target'):
                            old, new = original[side]['extrinsics'], aligned[side]['extrinsics']
                            torch.testing.assert_close(old, poses[side], rtol=0, atol=0)
                            # Check R and t independently of the loader's matrix constant.
                            torch.testing.assert_close(new[..., :3, :3],
                                                       model_axes(old[..., :3, :3].transpose(-1, -2)).transpose(-1, -2),
                                                       rtol=0, atol=0)
                            torch.testing.assert_close(new[..., :3, 3], model_axes(old[..., :3, 3]), rtol=0, atol=0)
                            torch.testing.assert_close(new[..., 3, :], old[..., 3, :], rtol=0, atol=0)
                            for key in original[side]:
                                if key != 'extrinsics':
                                    torch.testing.assert_close(aligned[side][key], original[side][key], rtol=0, atol=0)
                        # Reusing central views as targets must not apply the transform twice.
                        torch.testing.assert_close(aligned['target']['extrinsics'][12:],
                                                   aligned['context']['extrinsics'], rtol=0, atol=0)
                        front = aligned['context']['extrinsics'][0]
                        torch.testing.assert_close(front[:3, 2], torch.tensor([0., 1., 0.]), rtol=0, atol=0)
                        torch.testing.assert_close(front[:3, 0], torch.tensor([1., 0., 0.]), rtol=0, atol=0)
                        torch.testing.assert_close(front[:3, 3], torch.tensor([3., 2., 1.5]), rtol=0, atol=0)
                        if masked:
                            unmasked = CrossDataset(cfg, shape, split, test_range)[0]
                            for key in unmasked['target']:
                                torch.testing.assert_close(aligned['target'][key], unmasked['target'][key], rtol=0, atol=0)
                            self.assertTrue(bool(aligned['target']['eval_mask'][12:].all()))
                            self.assertFalse(bool(aligned['target']['eval_mask'][:12, :, -1].any()))
                        else:
                            self.assertNotIn('eval_mask', aligned['target'])

    def test_relative_projection_and_derived_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = oriented_fixture(tmp)
            shape = (32, 48)
            aligned = CrossDataset(cfg, shape, 'test')[0]
            original = CrossDataset(dict(cfg, name='pandaset'), shape, 'test')[0]
            old_relative = original['target']['extrinsics'][:, None].inverse() @ original['context']['extrinsics'][None]
            new_relative = aligned['target']['extrinsics'][:, None].inverse() @ aligned['context']['extrinsics'][None]
            torch.testing.assert_close(new_relative, old_relative, atol=3e-6, rtol=1e-5)
            camera_point = torch.tensor([.25, -.5, 8., 1.])
            old_pixel = original['target']['intrinsics_pixel'][:, None] @ (old_relative @ camera_point)[..., :3, None]
            new_pixel = aligned['target']['intrinsics_pixel'][:, None] @ (new_relative @ camera_point)[..., :3, None]
            torch.testing.assert_close(new_pixel / new_pixel[..., 2:3, :],
                                       old_pixel / old_pixel[..., 2:3, :], atol=1e-4, rtol=1e-5)

            # The existing backprojection covers geometric features and Gaussian means,
            # including shifted pixel locations. Depth remains camera-local and metric.
            depth = original['target']['metric_depth'][12:]
            k = original['context']['intrinsics_pixel']
            old_pose, new_pose = original['context']['extrinsics'], aligned['context']['extrinsics']
            shifts = torch.full((*depth.shape, 2), .2)
            for shift in (None, shifts):
                old_xyz = backproject(depth, k, old_pose, shift)
                new_xyz = backproject(depth, k, new_pose, shift)
                torch.testing.assert_close(new_xyz, model_axes(old_xyz), atol=1e-5, rtol=1e-5)
            old_rays = backproject(torch.ones_like(depth), k, old_pose) - old_pose[:, None, None, :3, 3]
            new_rays = backproject(torch.ones_like(depth), k, new_pose) - new_pose[:, None, None, :3, 3]
            torch.testing.assert_close(new_rays, model_axes(old_rays), atol=1e-6, rtol=1e-5)

            cameras = []
            for data in (original, aligned):
                batched = {key: value.unsqueeze(0) for key, value in data['target'].items()}
                cameras.append(target_cameras(batched, dict(znear=.01, zfar=1e8)))
            old_world = original['context']['extrinsics'][0] @ camera_point
            new_world = torch.cat((model_axes(old_world[:3]), old_world[3:]))
            for old_camera, new_camera in zip(*cameras):
                torch.testing.assert_close(new_camera.center, model_axes(old_camera.center), atol=0, rtol=0)
                torch.testing.assert_close(new_world @ new_camera.viewmatrix,
                                           old_world @ old_camera.viewmatrix, atol=3e-6, rtol=1e-5)
                old_clip, new_clip = old_world @ old_camera.projmatrix, new_world @ new_camera.projmatrix
                torch.testing.assert_close(new_clip / new_clip[3], old_clip / old_clip[3], atol=1e-5, rtol=1e-5)

    def test_frame_provenance_is_ddad_only_and_preserves_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = prepared_fixture(tmp)
            pandaset = CrossDataset(dict(cfg, name='pandaset'), (32, 48), 'test').evaluation_metadata()
            self.assertNotIn('camera_frame', pandaset)
            self.assertNotIn('reference_to_model', pandaset)
            for masked in (False, True):
                ddad = CrossDataset(cfg, (32, 48), 'test', eval_use_ego_mask=masked).evaluation_metadata()
                self.assertEqual(ddad['camera_frame'], 'nuscenes_axes_x_right_y_forward_z_up')
                self.assertEqual(ddad['reference_to_model'],
                                 [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
                self.assertEqual(ddad['view_protocol'], pandaset['view_protocol'])
                self.assertEqual(ddad['pixel_protocol'], 'ddad_ego_novel12_v1' if masked else 'full_image')


if __name__ == '__main__':
    unittest.main()
