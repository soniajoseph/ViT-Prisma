args_eval = {
    'pretrain': {
        'checkpoint_key': 'target_encoder',
        'model_name': 'vit_huge',
        'patch_size': 16,
        'pretrain_folder': '/path/to/pretrained/model/folder',
        'ckp_fname': 'model_checkpoint.pth',
        'tag': 'my_model_tag',
        'use_sdpa': True,
        'use_silu': False,
        'wide_silu': True,
        'uniform_power': False,
        'is_causal': False
    },
    'data': {
        'dataset_name': 'ImageNet',
        'num_classes': 1000,
        'root_path': '/path/to/dataset/root',
        'image_folder': 'images'
    },
    'optimization': {
        'batch_size': 32,
        'num_epochs': 10,
        'wd': 1e-6,
        'start_lr': 1e-4,
        'lr': 1e-3,
        'final_lr': 1e-5,
        'warmup': 0.1,
        'use_bfloat16': False
    },
    'num_probe_blocks': 1,
    'resume_checkpoint': False,
    'tag': 'my_model_tag'
}