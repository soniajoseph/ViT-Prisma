from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.utils.constants import DATA_DIR, MODEL_DIR, DEVICE, MODEL_CHECKPOINTS_DIR

from experiments.cifar10.cifar10_config import CIFAR10_CONFIG

num_epochs = 10

CIFAR10_SAE_CONFIG = VisionModelSAERunnerConfig(
        num_epochs=num_epochs,
        n_checkpoints=5,
        log_to_wandb=True,
        verbose=True,
        wandb_log_frequency=1,
        dataset_name="imagenet1k",
        dataset_path=str(DATA_DIR / "imagenet1k"),
        checkpoint_path=str(MODEL_DIR / "sae/cifar10/checkpoints/clean-model-clean-data"),
        train_batch_size=2048,
        hook_point_layer=6,
        device=DEVICE,
        total_training_images=40000,
        total_training_tokens=40000*65*num_epochs,
        wandb_entity="Stevinson",
        wandb_project="cifar10",
        model_name="local/cifar10-clean",
        sae_path=str(MODEL_DIR / "sae/cifar10/checkpoints"),
        model_path=str(MODEL_CHECKPOINTS_DIR / "cifar10-clean/model_1987392.pth"),
        vit_model_cfg=CIFAR10_CONFIG,
        d_in=CIFAR10_CONFIG.d_model,
        context_size=65,
        image_size=128,
        l1_coefficient=0.6,
        lr=0.01,
        lr_warm_up_steps=200,
    )


if __name__ == "__main__":
    trainer = VisionSAETrainer(CIFAR10_SAE_CONFIG)
    sae = trainer.run()
