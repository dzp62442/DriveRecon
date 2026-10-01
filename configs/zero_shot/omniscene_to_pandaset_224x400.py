_base_ = ['../pandaset/224x400.py']
zero_shot = True
mode = 'test'
work_dir = 'work_dirs/zero_shot/omniscene_224x400_to_pandaset_224x400'
evaluation = dict(checkpoint='work_dirs/omniscene/driverecon_static_t1_224x400/checkpoints/step-00100001')
