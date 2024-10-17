from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.utils.constants import DATA_DIR, MODEL_DIR, DEVICE, MODEL_CHECKPOINTS_DIR

from experiments.mnist.mnist_config import MNIST_CONFIG

MNIST_SAE_CONFIG = VisionModelSAERunnerConfig(
    num_epochs=10,
    n_checkpoints=5,
    log_to_wandb=True,
    verbose=True,
    wandb_log_frequency=1,
    dataset_name="mnist",
    dataset_path=str(DATA_DIR / "mnist"),
    checkpoint_path=str(MODEL_DIR / "sae/mnist/checkpoints/clean-model-clean-data"),
    train_batch_size=1024,
    hook_point_layer=3,
    device=DEVICE,
    total_training_images=48000,
    total_training_tokens=48000*65*10,
    wandb_entity="Stevinson",
    wandb_project="mnist",
    model_name="local/mnist-clean",
    sae_path=str(MODEL_DIR / "sae/mnist/checkpoints"),
    model_path=str(MODEL_CHECKPOINTS_DIR / "mnist-clean/model_1378816.pth"),
    vit_model_cfg=MNIST_CONFIG,
    d_in=MNIST_CONFIG.d_model,
    context_size=65,
    image_size=128,  # TODO EdS: This is already in the model config
)


if __name__ == "__main__":
    trainer = VisionSAETrainer(MNIST_SAE_CONFIG)
    sae = trainer.run()
