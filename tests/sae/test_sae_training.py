from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

cfg = VisionModelSAERunnerConfig()
print("Config created")

trainer = VisionSAETrainer(cfg)
sae = trainer.run()
