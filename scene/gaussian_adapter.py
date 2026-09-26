"""Original DriveRecon adapter, shared by Waymo and static OmniScene."""

import torch
from torch import nn
from torch.nn import functional as F
from scene.PointNet import ResnetBlock


class Guassian_Adaptor(nn.Module):
    def __init__(self, out_num, num_samples, gaussian_scale_min, gaussian_scale_max, seg_num):
        super(Guassian_Adaptor, self).__init__()

        self.out_num = out_num
        self.num_samples = num_samples
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.seg_num = seg_num
        self.max_shift = 5


        # RGB
        self.rgb_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 3, kernel_size=1),
            nn.Tanh()  # Apply scaling in forward
        )

        # scale
        self.atri_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 4)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 4)),
            nn.Conv2d(int(self.out_num / 4), 7, kernel_size=1),
        )

        # depth
        self.depth_map_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), self.num_samples, kernel_size=1),
        )

        # position
        self.position_shift_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 5, kernel_size=1),
            nn.Tanh()
        )

        # depth_reg
        self.depth_regression_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 1, kernel_size=1),
            nn.Tanh()
        )

        # seg
        if self.seg_num > 0:
            self.seg_head = nn.Sequential(
                ResnetBlock(self.out_num, int(self.out_num / 2)),
                # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
                nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
                nn.Conv2d(int(self.out_num / 2), self.seg_num, kernel_size=1)
            )


    def forward(self, x):
        _, C, H, W = x.shape

        rgb_output = self.rgb_head(x)
        rgb_output = 0.5 * rgb_output + 0.5  # Scale to range [0, 1]

        atri_output = self.atri_head(x)
        scale_output = self.gaussian_scale_min + 0.1 * (
                    self.gaussian_scale_max - self.gaussian_scale_min) * F.softplus(atri_output[:, 0:3, ...])
        rotation_output = F.normalize(atri_output[:, 3:7, ...], dim=1)
        alpha_output = torch.sigmoid(atri_output[:, -1, ...])

        position_shift = self.position_shift_head(x)
        uv_shift = self.max_shift * position_shift[:, 0:2, ...]
        means_shift = self.max_shift * position_shift[:, 2:, ...]
        depth_map_output = self.depth_map_head(x)

        depth_regression_output = self.depth_regression_head(x)
        if self.seg_num > 0:
            seg_output = self.seg_head(x)
            seg_output = seg_output.permute(0, 2, 3, 1)
        else:
            seg_output = None

        rgb_output = rgb_output.permute(0, 2, 3, 1)
        scale_output = scale_output.permute(0, 2, 3, 1)
        rotation_output = rotation_output.permute(0, 2, 3, 1)
        depth_map_output = depth_map_output.permute(0, 2, 3, 1)
        depth_regression_output = depth_regression_output.permute(0, 2, 3, 1)

        return rgb_output, scale_output, alpha_output, rotation_output, \
               depth_map_output, depth_regression_output, seg_output, \
               uv_shift, means_shift
