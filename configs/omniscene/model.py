model = dict(
    parameter_dtype='bfloat16',  # Original Gaussian_LRM casts both networks to BF16.
    view_num=6, num_frames=1, num_samples=200,
    depth_min=0.1, depth_max=400.0,
    gaussian_scale_min=0.001, gaussian_scale_max=4.0,
    seg_num=3, max_shift=5,
    unet=dict(
        in_channels=3, out_channels=128,
        down_channels=(64, 128, 256, 256),
        down_attention=(False, False, False, False),
        mid_attention=True,
        up_channels=(256, 256, 128), up_attention=(True, True, False),
        layers_per_block=1, skip_scale=0.5 ** 0.5,
    ),
)
renderer = dict(znear=0.01, zfar=1e8, background=(0.0, 0.0, 0.0))
loss = dict(rgb=1.0, segmentation=1.0, depth_class=2.0,
            depth_reg=2.0, geometry_aux=2.0)
