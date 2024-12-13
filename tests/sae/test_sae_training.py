from multiprocessing import freeze_support

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.utils.constants import DATA_DIR


def test_train_sae():
    freeze_support()

    cfg = VisionModelSAERunnerConfig()
    print("Config created")

    cfg.model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
    cfg.d_in = 768
    cfg.lr = 0.001
    cfg.l1_coefficient = 0.00008

    cfg.dataset_name = "cifar10"
    cfg.dataset_path = str(DATA_DIR / "cifar-test")
    cfg.dataset_train_path = str(DATA_DIR / "cifar-test")
    cfg.dataset_val_path = str(DATA_DIR / "cifar-test")

    cfg.log_to_wandb = False

    cfg.n_checkpoints = 0
    cfg.n_validation_runs = 0
    cfg.num_epochs = 1
    cfg.total_training_images = 100
    cfg.total_training_tokens = cfg.total_training_images * cfg.context_size

    trainer = VisionSAETrainer(cfg)
    trainer.run()
