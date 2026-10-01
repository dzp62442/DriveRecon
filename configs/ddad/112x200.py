_base_ = ['../omniscene/112x200.py']
zero_shot = False
dataset = dict(_delete_=True, name='ddad', data_root='datasets/DDAD',
               processed_subdir='processed', use_dynamic_mask=False, mini_size=100, val_size=10)
work_dir = 'work_dirs/ddad/driverecon_static_t1_112x200'
evaluation = dict(eval_use_ego_mask=True)
