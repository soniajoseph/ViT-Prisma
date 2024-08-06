from vit_prisma.utils.load_model import load_model
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import SparseAutoencoder

import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
import re


class VisionSAETrainer:
    def __init__(self, cfg: VisionModelSAERunnerConfig):
        self.cfg = cfg
        self.model = load_model(cfg.model_class_name, cfg.model_name)
        self.sae = self.initialize_sae()
        self.activations_store = self.initialize_activations_store()
        self.checkpoint_thresholds = self.get_checkpoint_thresholds()

    def initialize_sae(self):
        return SparseAutoencoder(self.cfg)
    
    def load_dataset(self):
        if self.cfg.dataset_name == 'imagenet1k':
            # Imagenet-specific logic
            from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
            from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms, ImageNetValidationDataset
            data_transforms = get_imagenet_transforms()
            imagenet_paths = setup_imagenet_paths(self.cfg.dataset_path)
            train_data = torchvision.datasets.ImageFolder(self.cfg.dataset_train_path, transform=data_transforms)
            val_data = ImageNetValidationDataset(self.cfg.dataset_val_path, 
                                            imagenet_paths['label_strings'], 
                                            imagenet_paths['val_labels'], 
                                            data_transforms
            )
            return train_data, val_data
        else:
            return print("Specify dataset name")

    def initialize_activations_store(self):
        # Initialize activations store here
        activations_loader = VisionActivationsStore(self.cfg, self.model, self.dataset, self.eval_dataset)
        return activations_loader

    def get_checkpoint_thresholds(self):
        if self.cfg.n_checkpoints > 0:
            return list(range(0, self.cfg.total_training_tokens, self.cfg.total_training_tokens // self.cfg.n_checkpoints))[1:]
        return []

    def initialize_training_variables(self):
        num_saes = len(self.sae_group)
        act_freq_scores = [torch.zeros(int(sae.cfg.d_sae), device=sae.cfg.device) for sae in self.sae_group]
        n_forward_passes_since_fired = [torch.zeros(int(sae.cfg.d_sae), device=sae.cfg.device) for sae in self.sae_group]
        n_frac_active_tokens = [0 for _ in range(num_saes)]
        optimizers = [Adam(sae.parameters(), lr=sae.cfg.lr) for sae in self.sae_group]
        schedulers = self.initialize_schedulers(optimizers)
        return act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizers, schedulers

    def initialize_schedulers(self, optimizers):
        return [
            self.get_scheduler(
                sae.cfg.lr_scheduler_name,
                optimizer=opt,
                warm_up_steps=sae.cfg.lr_warm_up_steps,
                training_steps=self.total_training_steps,
                lr_end=sae.cfg.lr / 10,
            )
            for sae, opt in zip(self.sae_group, optimizers)
        ]

    def get_scheduler(self, name, optimizer, warm_up_steps, training_steps, lr_end):
        # Implement scheduler logic here
        pass

    def initialize_geometric_medians(self):
        geometric_medians = {}
        all_layers = self.sae_group.cfg.hook_point_layer
        if not isinstance(all_layers, list):
            all_layers = [all_layers]

        for sae in self.sae_group:
            hyperparams = sae.cfg
            sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
            if hyperparams.b_dec_init_method == "geometric_median":
                layer_acts = self.activations_store.storage_buffer.detach()[:, sae_layer_id, :]
                if sae_layer_id not in geometric_medians:
                    median = self.compute_geometric_median(layer_acts, maxiter=200).median
                    geometric_medians[sae_layer_id] = median
                sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
            elif hyperparams.b_dec_init_method == "mean":
                layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, sae_layer_id, :]
                sae.initialize_b_dec_with_mean(layer_acts)
            sae.train()
        return geometric_medians

    def compute_geometric_median(self, data, maxiter):
        # Implement geometric median computation here
        pass

    def train_step(self, sae, optimizer, scheduler, act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, layer_acts, n_training_steps):
        # Implement single training step for an SAE
        pass

    def log_metrics(self, sae, hyperparams, metrics, n_training_steps):
        if self.cfg.use_wandb and ((n_training_steps + 1) % self.cfg.wandb_log_frequency == 0):
            suffix = self.wandb_log_suffix(self.sae_group.cfg, hyperparams)
            wandb.log(metrics, step=n_training_steps)

    def checkpoint(self, n_training_images, log_feature_sparsity):
        # Implement checkpointing logic
        pass

    def run(self):
        act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizers, schedulers = self.initialize_training_variables()
        geometric_medians = self.initialize_geometric_medians()

        n_training_steps = 0
        n_training_images = 0
        
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")
        
        while n_training_images < self.cfg.total_training_tokens:
            layer_acts = self.activations_store.next_batch()
            n_training_images += self.cfg.batch_size

            for i, sae in enumerate(self.sae_group):
                loss, metrics = self.train_step(sae, optimizers[i], schedulers[i], act_freq_scores[i], n_forward_passes_since_fired[i], n_frac_active_tokens[i], layer_acts, n_training_steps)
                self.log_metrics(sae, sae.cfg, metrics, n_training_steps)

            if self.cfg.n_checkpoints > 0 and n_training_images > self.checkpoint_thresholds[0]:
                self.checkpoint(n_training_images, act_freq_scores)

            n_training_steps += 1
            pbar.update(self.cfg.batch_size)

        self.final_checkpoint(n_training_images, act_freq_scores)
        return self.sae_group

    def final_checkpoint(self, n_training_images, act_freq_scores):
        # Implement final checkpointing logic
        pass

    @staticmethod
    def wandb_log_suffix(group_cfg, sae_cfg):
        # Implement wandb log suffix logic
        pass