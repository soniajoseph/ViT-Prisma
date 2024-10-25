from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

cfg = VisionModelSAERunnerConfig()
print("Config created")

# need to adjust learning rate and L1
# lr = 0.0001
# l1_coefficient = 0.00008
cfg.lr = 0.0001
cfg.l1_coefficient = 0.00008


# to pick hook point, use cfg.hook_point_head_index
cfg.hook_point_layer = 0
cfg.hook_point = "blocks.0.hook_mlp_out"

# You can modify the config hyperparameters however you wish, for example:
cfg.expansion_factor = 16
cfg.d_sae = 1048
cfg.initialization_method = "independent"
cfg.normalize_activations = "null"

cfg.device = "cuda"
cfg.model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90k"
cfg.dataset_path = "/workspace/"
cfg.dataset_train_path = "/workspace/ILSVRC/Data/CLS-LOC/train"
cfg.dataset_val_path = "/workspace/ILSVRC/Data/CLS-LOC/val"
cfg.checkpoint_path ="/workspace/checkpoints"
cfg.wandb_project = f"open_clip_vit_b_32_layer_{cfg.hook_point_layer}_{cfg.hook_point_head_index}"

cfg.train_batch_size = 4096 # tweak store_batch_size instead
cfg.n_batches_in_buffer = 4
cfg.store_batch_size = 256 #cfg.train_batch_size
cfg.d_in = 768
cfg.pretty_print()

trainer = VisionSAETrainer(cfg)

print(cfg.wandb_project)
print(cfg.run_name)
print(cfg.hook_point)
print(type(cfg.hook_point))

sae = trainer.run()