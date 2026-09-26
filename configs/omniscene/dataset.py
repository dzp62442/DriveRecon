dataset = dict(
    data_root='/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene',
    data_version='interp_12Hz_trainval',
    dataset_prefix='/datasets/nuScenes',
    use_dynamic_mask=True,
)
data_loader = dict(batch_size=1, num_workers=0, pin_memory=True)
