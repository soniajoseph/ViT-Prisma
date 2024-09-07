from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

cfg = VisionModelSAERunnerConfig()
print("Config created")

cfg.model_name = 'openai/clip-vit-base-patch32'
cfg.d_in = 768
cfg.lr = 0.001
cfg.l1_coefficient = 0.00008
cfg.wandb_project = 'openai-clip-vit-base_mlp_out_layer_9_sae_expansion_16'
trainer = VisionSAETrainer(cfg)
sae = trainer.run()
