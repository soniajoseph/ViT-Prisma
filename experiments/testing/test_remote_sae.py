import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

import traceback
import uuid

from torch.utils.data import DataLoader

import wandb

from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.run_wandb_sweep_sae import train
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.sae.sae_utils import load_sae
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.utils.constants import DATA_DIR, MODEL_DIR, EvaluationContext
from vit_prisma.utils.data_utils.loader import load_dataset
from vit_prisma.utils.load_model import load_model

if __name__ == "__main__":

    cfg = VisionModelSAERunnerConfig(
        num_epochs=1,
        n_checkpoints=5,
        # total_training_images=10000 * 2,
        # total_training_tokens=10000 * 2 * 50,  # Images x epochs x tokens
        log_to_wandb=True,
        verbose=True,
        wandb_log_frequency=100,
        dataset_path=str(DATA_DIR / "imagenet"),
        dataset_train_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
        dataset_val_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        checkpoint_path=str(MODEL_DIR / "sae/imagenet/checkpoints"),
        train_batch_size=1024,
        device="cpu",
        total_training_images=1000,
        total_training_tokens=1000*50,
        wandb_entity="Stevinson",
        wandb_project="imagenet100",
        sae_path=str(MODEL_DIR / "sae/imagenet/checkpoints")
    )
    trainer = VisionSAETrainer(cfg)
    sae = trainer.run()


    # cfg.wandb_project = cfg.model_name.replace('/', '-') + "-expansion-" + str(
    # cfg.expansion_factor) + "-layer-" + str(cfg.hook_point_layer)
    # cfg.unique_hash = uuid.uuid4().hex[:8]
    # cfg.run_name = cfg.unique_hash + "-" + cfg.wandb_project
    # wandb.init(project=cfg.wandb_project, name=cfg.run_name, entity="Stevinson")
    #
    cfg.sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints/ff86e305-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16-layer-9/n_images_1300008.pt")
    model = load_model(cfg)
    sae = load_sae(cfg)
    train_data, val_data, val_data_visualize = load_dataset(cfg, visualize=True)  # TODO: I should be using test not validation data here

    evaluator = Evaluator(model, val_data, cfg, visualize_data=val_data_visualize)  # TODO: Do I need the visualize data? Or can I just use the same variable
    evaluator.evaluate(sae, context=EvaluationContext.POST_TRAINING)

    # This is used to create a small subset of the data in l.79 of loader.py for fast testing
    # train_data = SubsetDataset(train_data, 10000)
    # val_data = SubsetDataset(val_data, 1000)