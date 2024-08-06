from vit_prisma.utils.load_model import load_model
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.training.activations_store import VisionActivationsStore

from vit_prisma.sae.training.geometric_median import compute_geometric_median
from vit_prisma.sae.training.get_scheduler import get_scheduler

import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
import re

import os

import torchvision

from typing import Any

import uuid

@staticmethod
def wandb_log_suffix(cfg: Any, hyperparams: Any):
# Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix

class VisionSAETrainer:
    def __init__(self, cfg: VisionModelSAERunnerConfig):
        self.cfg = cfg
        self.model = load_model(self.cfg.model_class_name, self.cfg.model_name)
        self.sae = SparseAutoencoder(self.cfg)

        dataset, eval_dataset = self.load_dataset()
        self.activations_store = VisionActivationsStore(self.cfg, self.model, dataset, eval_dataset)

        self.cfg.unique_hash = uuid.uuid4().hex[:8]  # Generate a random 8-character hex string
        self.cfg.run_name = self.cfg.unique_hash + "-" + self.cfg.model_name + "-expansion-" + str(self.cfg.expansion_factor)

        self.checkpoint_thresholds = self.get_checkpoint_thresholds()
        self.setup_checkpoint_path()
    
    def setup_checkpoint_path(self):
        self.cfg.checkpoint_path = f"{self.cfg.checkpoint_path}/{self.cfg.run_name}"
        os.makedirs(self.cfg.checkpoint_path, exist_ok=True)
        print(f"Checkpoint path: {self.cfg.checkpoint_path}") if self.cfg.verbose else None
    
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


    def get_checkpoint_thresholds(self):
        if self.cfg.n_checkpoints > 0:
            return list(range(0, self.cfg.total_training_tokens, self.cfg.total_training_tokens // self.cfg.n_checkpoints))[1:]
        return []

    def initialize_training_variables(self):
        # num_saes = len(self.sae_group)
        act_freq_scores = torch.zeros(int(self.cfg.d_sae), device = self.cfg.device)
        n_forward_passes_since_fired = torch.zeros(int(self.cfg.d_sae), device=self.cfg.device)
        n_frac_active_tokens = 0 
        optimizers = Adam(self.sae.parameters(), lr=self.cfg.lr)
        scheduler = get_scheduler(
                        self.cfg.lr_scheduler_name,
                        optimizer=optimizers,
                        warm_up_steps=self.cfg.lr_warm_up_steps,
                        training_steps=self.total_training_steps,
                        lr_end=self.cfg.lr / 10,
                )       
        return act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizers, scheduler

    def initialize_geometric_medians(self):
        all_layers = self.sae_group.cfg.hook_point_layer
        geometric_medians = {}
        if not isinstance(all_layers, list):
            all_layers = [all_layers]
        hyperparams = self.sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[:, sae_layer_id, :]
            if sae_layer_id not in geometric_medians:
                median = compute_geometric_median(layer_acts, maxiter=200)
                geometric_medians[sae_layer_id] = median
            self.sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, sae_layer_id, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)
        self.sae.train()
        return geometric_medians

    def train_step(self, sparse_autoencoder, optimizer, scheduler, act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, 
                   layer_acts, n_training_steps, n_training_images):
        hyperparams = sparse_autoencoder.cfg
        layer_id = hyperparams.hook_point_layer
        sae_in = layer_acts[:, layer_id, :]
        sparse_autoencoder.train()
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = (
                torch.log10(feature_sparsity + 1e-10).detach().cpu()
            )

            if self.cfg.log_to_wandb:
                suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {
                        f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
                        f"plots/feature_density_line_chart{suffix}": wandb_histogram,
                        f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5)
                        .sum()
                        .item(),
                        f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6)
                        .sum()
                        .item(),
                    },
                    step=n_training_steps,
                )

            act_freq_scores = torch.zeros(
                sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
            )
            n_frac_active_tokens = 0

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_forward_passes_since_fired
            > sparse_autoencoder.cfg.dead_feature_window
        ).bool()

        # Forward and Backward Passes
        (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            ghost_grad_loss,
        ) = sparse_autoencoder(
            sae_in,
            ghost_grad_neuron_mask,
        )
        did_fire = (feature_acts > 0).float().sum(-2) > 0
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            batch_size = sae_in.shape[0]
            n_frac_active_tokens += batch_size 
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if self.cfg.log_to_wandb and ((n_training_steps + 1) % self.cfg.wandb_log_frequency == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]

                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss / total_variance

                suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
                metrics = {
                        # losses
                        f"losses/mse_loss{suffix}": mse_loss.item(),
                        f"losses/l1_loss{suffix}": l1_loss.item()
                        / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
                        f"losses/ghost_grad_loss{suffix}": ghost_grad_loss.item(),
                        f"losses/overall_loss{suffix}": loss.item(),
                        # variance explained
                        f"metrics/explained_variance{suffix}": explained_variance.mean().item(),
                        f"metrics/explained_variance_std{suffix}": explained_variance.std().item(),
                        f"metrics/l0{suffix}": l0.item(),
                        # sparsity
                        f"sparsity/mean_passes_since_fired{suffix}": n_forward_passes_since_fired.mean().item(),
                        f"sparsity/dead_features{suffix}": ghost_grad_neuron_mask.sum().item(),
                        f"details/current_learning_rate{suffix}": current_learning_rate,
                        "details/n_training_images": n_training_images,
                    }
                wandb.log(
                    metrics,
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            if self.cfg.log_to_wandb and (
                (n_training_steps + 1) % (self.cfg.wandb_log_frequency * 10) == 0
            ):
                sparse_autoencoder.eval()
                suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
                #TODO fix for clip!!
                print("eval only set up for classifier models..")
                try: 
                    run_evals_vision(
                        sparse_autoencoder,
                        self.activations_store,
                        model,
                        n_training_steps,
                        suffix=suffix,
                    )
                except:
                    pass
                sparse_autoencoder.train()

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        return loss, metrics, act_freq_scores

    def log_metrics(self, sae, hyperparams, metrics, n_training_steps):
        if self.cfg.use_wandb and ((n_training_steps + 1) % self.cfg.wandb_log_frequency == 0):
            suffix = self.wandb_log_suffix(self.cfg, hyperparams)
            wandb.log(metrics, step=n_training_steps)

    def checkpoint(self, sae, n_training_images, act_freq_scores, n_frac_active_tokens):
        # NOTE fix htis code to not be sae groups anymore
        # path = f"{sae_group.cfg.checkpoint_path}/{n_training_images}_{sae_group.get_name()}.pt"
        path = self.cfg.checkpoint_path + f"/{n_training_images}.pt"
        sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)

        # Save log feature sparsity
        log_feature_sparsity_path = self.cfg.checkpoint_path + f"/n_images_{self.cfg.n_training_images}_log_feature_sparsity.pt"
        feature_sparsity = (
            act_freq_scores / n_frac_active_tokens
        )
        log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
        torch.save(log_feature_sparsity, log_feature_sparsity_path)

        self.checkpoint_thresholds.pop(0)
        if len(self.checkpoint_thresholds) == 0:
            n_checkpoints = 0
        if self.cfg.log_to_wandb:
            self.save_to_wandb(sae, hyperparams, path, log_feature_sparsity_path)

    def save_to_wandb(self, sae, hyperparams, path, log_feature_sparsity_path):
        suffix = self.wandb_log_suffix(sae.cfg, hyperparams)
        name_for_log = re.sub(self.cfg.unique_hash, '_', suffix)
        try:
            model_artifact = wandb.Artifact(
                f"{name_for_log}",
                type="model",
                metadata=dict(sae.cfg.__dict__),
            )
            model_artifact.add_file(path)
            wandb.log_artifact(model_artifact)

            sparsity_artifact = wandb.Artifact(
                f"{name_for_log}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(sae.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)
        except:
            pass
        
    def run(self):

        act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizer, scheduler = self.initialize_training_variables()
        geometric_medians = self.initialize_geometric_medians()

        n_training_steps = 0
        n_training_images = 0
        
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")        
        while n_training_images < self.cfg.total_training_tokens:
            layer_acts = self.activations_store.next_batch()
            n_training_images += self.cfg.batch_size

            # init these here to avoid uninitialized vars
            mse_loss = torch.tensor(0.0)
            l1_loss = torch.tensor(0.0)

            loss, metrics, act_freq_scores = self.train_step(self.sae, optimizer, scheduler, act_freq_scores, n_forward_passes_since_fired, 
                                                             n_frac_active_tokens, layer_acts, n_training_steps, n_training_images)
            self.log_metrics(self.sae, self.sae.cfg, metrics, n_training_steps)

            if self.cfg.n_checkpoints > 0 and n_training_images > self.checkpoint_thresholds[0]:
                self.checkpoint(self.sae, n_training_images, act_freq_scores, n_frac_active_tokens)
                print(f"Checkpoint saved at {n_training_images} images") if self.cfg.verbose else None

            n_training_steps += 1
            pbar.update(self.cfg.batch_size)
            pbar.set_description(
                f"Training SAE: Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, L0: {metrics['l0']:.4f}"
            )
        
        # Final checkpoint
        self.checkpoint(self.sae, n_training_images, act_freq_scores, n_frac_active_tokens)
        print(f"Final checkpoint saved at {n_training_images} images") if self.cfg.verbose else None

        pbar.close()

        return self.sae

    