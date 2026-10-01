_base_ = ['../omniscene/112x200.py']
zero_shot = False
dataset = dict(_delete_=True, name='pandaset', data_root='datasets/PandaSet',
               processed_subdir='processed', use_dynamic_mask=False, mini_size=100, val_size=10)
work_dir = 'work_dirs/pandaset/driverecon_static_t1_112x200'
evaluation = dict(eval_use_ego_mask=False)
