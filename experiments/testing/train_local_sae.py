from experiments.cifar10_experiment import cifar10_model_config
from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from src.vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.sae.sae_utils import load_sae

from vit_prisma.utils.constants import DATA_DIR, MODEL_CHECKPOINTS_DIR, SAE_CHECKPOINTS_DIR, EvaluationContext
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10
from vit_prisma.utils.load_model import load_model


if __name__ == "__main__":

    cfg = VisionModelSAERunnerConfig(
        model_name="local/vit-cifar10",
        hook_point_layer=3,
        d_in=384,
        context_size=65,
        dataset_name="cifar10",
        dataset_path=str(DATA_DIR / "cifar-test"),
        dataset_train_path=str(DATA_DIR / "cifar-test"),
        dataset_val_path=str(DATA_DIR / "cifar-test"),
        log_to_wandb=True,
        wandb_project="adv_superposition",
        checkpoint_path=str(SAE_CHECKPOINTS_DIR / "cifar10/checkpoints"),
        vit_model_cfg=cifar10_model_config,
        model_path=str(MODEL_CHECKPOINTS_DIR / "cifar10-clean/model_4090960.pth"),
        feature_sampling_window=1000,
    )

    trainer = VisionSAETrainer(cfg)
    # Evaluate during training
    sae = trainer.run()

    # Post training evaluation
    model = load_model(cfg)
    sae = load_sae(cfg)
    _, _, test_data = load_cifar_10(cfg.dataset_path)

    evaluator = Evaluator(model, test_data, cfg)
    evaluator.evaluate(sae, context=EvaluationContext.POST_TRAINING)