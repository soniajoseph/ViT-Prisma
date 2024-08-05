from vit_prisma.sae.sae_training_runner import SAETrainingRunner

from vit_prisma.sae.config import VisionModelSAERunnerConfig

cfg = VisionModelSAERunnerConfig()

runner = SAETrainingRunner(cfg)

runner.run()
