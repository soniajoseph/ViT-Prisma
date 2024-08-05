'''
Some code based off SAELens by Joseph Bloom:
https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae_training_runner.py
'''

import json
import logging
import os
import signal
from typing import Any, cast

import torch
import torchvision


import wandb
from safetensors.torch import save_file

from vit_prisma.sae.config import VisionModelSAERunnerConfig, HfDataset, print_config_prettily
from vit_prisma.prisma_tools.hooked_root_module import HookedRootModule
from vit_prisma.utils.load_model import load_model
from vit_prisma.sae.sae import SAE_CFG_PATH, SAE_WEIGHTS_PATH, SPARSITY_PATH
from vit_prisma.sae.training.activations_store import ActivationsStore
from vit_prisma.sae.training.geometric_median import compute_geometric_median

from vit_prisma.sae.training.sae_trainer import SAETrainer
from vit_prisma.sae.training.training_sae import TrainingSAE, TrainingSAEConfig




class InterruptedException(Exception):
    pass

def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()


class SAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a Prisma model.
    """

    cfg: VisionModelSAERunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: VisionModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
    ):
        if override_dataset is not None:
            logging.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logging.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg

        if override_model is None:
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=self.cfg.device,
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )
            self.model.to(self.cfg.device)

        else:
            self.model = override_model

        dataset_train, dataset_val = self.setup_datasets()

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            dataset_train = dataset_train,
            dataset_val = dataset_val,
        )

        if self.cfg.from_pretrained_path is not None:
            self.sae = TrainingSAE.load_from_pretrained(
                self.cfg.from_pretrained_path, self.cfg.device
            )
        else:
            self.sae = TrainingSAE(
                TrainingSAEConfig.from_dict(
                    self.cfg.get_training_sae_cfg_dict(),
                )
            )
            self._init_sae_group_b_decs()
    

    def setup_datasets(self):
        if self.cfg.dataset_name == 'imagenet1k': # Feel free to change this for your particular file structure
            from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
            from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms, ImageNetValidationDataset
            data_transforms = get_imagenet_transforms()
            imagenet_paths = setup_imagenet_paths(self.cfg.dataset_path)
            train_data = torchvision.datasets.ImageFolder(self.cfg.dataset_train_path, transform=data_transforms)
            val_data = ImageNetValidationDataset(self.cfg.dataset_val_path, 
                                            imagenet_paths['label_strings'], 
                                            imagenet_paths['val_labels'], 
                                            data_transforms)
        else:
            # raise not implemented
            raise NotImplementedError(f"Dataset {self.cfgdataset_name} not implemented yet")
        
        return train_data, val_data

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        print_config_prettily(self.cfg) if self.cfg.verbose else None
        print("Starting training") if self.cfg.verbose else None
        


        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

    def _compile_if_needed(self):

        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

        if self.cfg.compile_sae:
            if self.cfg.device == "mps":
                backend = "aot_eager"
            else:
                backend = "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            print("interrupted, saving progress")
            checkpoint_name = trainer.n_training_tokens
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            print("done saving")
            raise

        return sae

    # TODO: move this into the SAE trainer or Training SAE class
    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            print("Initializing b_dec with geometric median") if self.cfg.verbose else None
            layer_acts = self.activations_store.storage_buffer_train.detach()[:, 0, :] # (batches x patches) x layers x dimensionality
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            # print if option is met
            print("Initializing b_dec with mean") if self.cfg.verbose else None
            layer_acts = self.activations_store.storage_buffer_train.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore

    def save_checkpoint(
        self,
        trainer: SAETrainer,
        checkpoint_name: int | str,
        wandb_aliases: list[str] | None = None,
    ) -> str:

        checkpoint_path = f"{trainer.cfg.checkpoint_path}/{checkpoint_name}"

        os.makedirs(checkpoint_path, exist_ok=True)

        path = f"{checkpoint_path}"
        os.makedirs(path, exist_ok=True)

        if self.sae.cfg.normalize_sae_decoder:
            self.sae.set_decoder_norm_to_unit_norm()
        self.sae.save_model(path)

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(config, f)

        log_feature_sparsities = {"sparsity": trainer.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if trainer.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            # Avoid wandb saving errors such as:
            #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
            sae_name = self.sae.get_name().replace("/", "__")

            model_artifact = wandb.Artifact(
                sae_name,
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )

            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")

            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{sae_name}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

        return checkpoint_path