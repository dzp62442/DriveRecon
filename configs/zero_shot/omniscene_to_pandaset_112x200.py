_base_ = ['../pandaset/112x200.py']
zero_shot = True
mode = 'test'
work_dir = 'work_dirs/zero_shot/omniscene_112x200_to_pandaset_112x200'
evaluation = dict(checkpoint='work_dirs/omniscene/driverecon_static_t1_112x200/checkpoints/step-00100001')
