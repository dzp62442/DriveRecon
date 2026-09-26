seed = 0
deterministic = True  # Original train.py setup_seed enables deterministic cuDNN.
mode = 'train'
precision = 'bf16'
optimizer = dict(lr=4e-4, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-15)
training = dict(max_steps=100_001, validate_every_steps=1_000,
                mini_every_n_validations=10, checkpoint_every_steps=1_000,
                log_every_steps=100, resume='auto')
evaluation = dict(test_split='total', checkpoint='final', time_skip_bins=5,
                  save_images=False, lpips_vgg_weights=None)
logging = dict(wandb=False, wandb_mode='offline', project='DriveRecon-OmniScene')
feishu = dict(enabled=True, module_root='~/Libraries')
