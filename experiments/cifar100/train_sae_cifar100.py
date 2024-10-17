from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.utils.constants import DATA_DIR, MODEL_DIR, DEVICE


if __name__ == "__main__":

    cfg = VisionModelSAERunnerConfig(
        num_epochs=3,
        n_checkpoints=5,
        log_to_wandb=True,
        verbose=True,
        wandb_log_frequency=100,
        dataset_path=str(DATA_DIR / "cifar100"),
        dataset_train_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),  # TODO EdS:
        dataset_val_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        checkpoint_path=str(MODEL_DIR / "sae/cifar100/checkpoints"),
        train_batch_size=1024,
        device=DEVICE,
        total_training_images=1000,  # TODO EdS:
        total_training_tokens=1000*50,
        wandb_entity="Stevinson",
        wandb_project="imagenet100",
        sae_path=str(MODEL_DIR / "sae/imagenet/checkpoints")
    )
    trainer = VisionSAETrainer(cfg)
    sae = trainer.run()
