import os
import signal
from typing import Any, Optional, cast
from vit_prisma.models.base_vit import HookedViT

import numpy as np
import torch
import argparse
import wandb
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens import __version__

from sae_lens.training.sae_group import SparseAutoencoderDictionary
from sae_lens.training.sparse_autoencoder import (
    SAE_CFG_PATH,
    SAE_WEIGHTS_PATH,
    SPARSITY_PATH,
)
from sae_lens.training.train_sae_on_language_model import (
    SAETrainContext, SAETrainingRunState, TrainSAEGroupOutput,
     get_total_training_tokens, _wandb_log_suffix, _build_train_context,
     _init_sae_group_b_decs, _update_sae_lens_training_version, _train_step, _build_train_step_log_dict,
     TRAINING_RUN_STATE_PATH, SAE_CONTEXT_PATH



)
import re

from sae.vision_evals import run_evals_vision
from sae.vision_config import VisionModelRunnerConfig
from sae.vision_activations_store import VisionActivationsStore
from sae.legacy_load import load_legacy_pt_file
import csv



def train_sae_group_on_vision_model(
    model: HookedRootModule,
    sae_group: SparseAutoencoderDictionary,
    activation_store: VisionActivationsStore,
    train_contexts: Optional[dict[str, SAETrainContext]] = None,
    training_run_state: Optional[SAETrainingRunState] = None,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
    eval_every_n_wandb_logs: int = 100,
    autocast: bool = False,
) -> TrainSAEGroupOutput:
    """
    Trains a group of sparse autoencoders (SAEs) on a vision model's activations.

    Args:
        model (HookedRootModule): The vision model to extract activations from.
        sae_group (SparseAutoencoderDictionary): Group of sparse autoencoders to be trained.
        activation_store (VisionActivationsStore): Store for vision model activations.
        train_contexts (Optional[dict[str, SAETrainContext]]): Optional training contexts for resuming training.
        training_run_state (Optional[SAETrainingRunState]): Optional state for resuming training.
        batch_size (int): Size of the training batch. Default is 1024.
        n_checkpoints (int): Number of checkpoints to save during training. Default is 0.
        feature_sampling_window (int): Number of training steps between resampling features. Default is 1000.
        use_wandb (bool): Whether to use Weights & Biases for logging. Default is False.
        wandb_log_frequency (int): Frequency of logging to Weights & Biases. Default is 50.
        eval_every_n_wandb_logs (int): Frequency of evaluations based on Weights & Biases logs. Default is 100.
        autocast (bool): Whether to use automatic mixed precision. Default is False.

    Returns:
        TrainSAEGroupOutput: Output containing trained SAE group, checkpoint paths, and log feature sparsities.

    Raises:
        ValueError: If train_contexts or training_run_state are None when resuming training.
        InterruptedException: If training is interrupted by a signal.

    """
        
    total_training_tokens = get_total_training_tokens(sae_group=sae_group)
    _update_sae_lens_training_version(sae_group)
    total_training_steps = total_training_tokens // batch_size

    checkpoint_thresholds = []
    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    all_layers = sae_group.cfg.hook_point_layer
    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")

    # not resuming
    if training_run_state is None and train_contexts is None:
        train_contexts = {
            name: _build_train_context(sae, total_training_steps)
            for name, sae in sae_group.autoencoders.items()
        }
        training_run_state = SAETrainingRunState()
        _init_sae_group_b_decs(sae_group, activation_store, all_layers)
    # resuming
    else:
        if train_contexts is None:
            raise ValueError(
                "train_contexts is None, when resuming, pass in training_run_state and train_contexts"
            )
        if training_run_state is None:
            raise ValueError(
                "training_run_state is None, when resuming, pass in training_run_state and train_contexts"
            )
        pbar.update(training_run_state.n_training_tokens)
        training_run_state.set_random_state()

    class InterruptedException(Exception):
        pass

    def interrupt_callback(sig_num: Any, stack_frame: Any):
        raise InterruptedException()

    try:
        # signal handlers (if preempted)
        signal.signal(signal.SIGINT, interrupt_callback)
        signal.signal(signal.SIGTERM, interrupt_callback)

        # Estimate norm scaling factor if necessary
        # TODO(tomMcGrath): this is a bodge and should be moved back inside a class
        if activation_store.normalize_activations:
            print("Estimating activation norm")
            n_batches_for_norm_estimate = int(1e3)
            norms_per_batch = []
            for _ in tqdm(range(n_batches_for_norm_estimate)):
                acts = activation_store.next_batch()
                norms_per_batch.append(acts.norm(dim=-1).mean().item())
            mean_norm = np.mean(norms_per_batch)
            scaling_factor = np.sqrt(activation_store.d_in) / mean_norm
            activation_store.estimated_norm_scaling_factor = scaling_factor

        # Train loop
        while training_run_state.n_training_tokens < total_training_tokens:
            # Do a training step.
            layer_acts = activation_store.next_batch()
            training_run_state.n_training_tokens += batch_size

            mse_losses: list[torch.Tensor] = []
            l1_losses: list[torch.Tensor] = []

            for name, sparse_autoencoder in sae_group.autoencoders.items():
                ctx = train_contexts[name]
                wandb_suffix = _wandb_log_suffix(sae_group.cfg, sparse_autoencoder.cfg)
                step_output = _train_step(
                    sparse_autoencoder=sparse_autoencoder,
                    layer_acts=layer_acts,
                    ctx=ctx,
                    feature_sampling_window=feature_sampling_window,
                    use_wandb=use_wandb,
                    n_training_steps=training_run_state.n_training_steps,
                    all_layers=all_layers,
                    batch_size=batch_size,
                    wandb_suffix=wandb_suffix,
                    autocast=autocast,
                )
                mse_losses.append(step_output.mse_loss)
                l1_losses.append(step_output.l1_loss)

                if use_wandb:
                    with torch.no_grad():
                        if (
                            training_run_state.n_training_steps + 1
                        ) % wandb_log_frequency == 0:
                            wandb.log(
                                _build_train_step_log_dict(
                                    sparse_autoencoder,
                                    step_output,
                                    ctx,
                                    wandb_suffix,
                                    training_run_state.n_training_tokens,
                                ),
                                step=training_run_state.n_training_steps,
                            )

                        # record loss frequently, but not all the time.
                        if (training_run_state.n_training_steps + 1) % (
                            wandb_log_frequency * eval_every_n_wandb_logs
                        ) == 0:
                            sparse_autoencoder.eval()
                             #TODO fix for clip!!
                            try: 
                                run_evals_vision(
                                    sparse_autoencoder,
                                    activation_store,
                                    model,
                                    training_run_state.n_training_steps,
                                    suffix=wandb_suffix,
                                )
                            except:
                                pass
                            sparse_autoencoder.train()

            # checkpoint if at checkpoint frequency
            if (
                checkpoint_thresholds
                and training_run_state.n_training_tokens > checkpoint_thresholds[0]
            ):
                _save_checkpoint(
                    sae_group,
                    train_contexts=train_contexts,
                    training_run_state=training_run_state,
                    checkpoint_name=training_run_state.n_training_tokens,
                )
                checkpoint_thresholds.pop(0)

            ###############

            training_run_state.n_training_steps += 1
            if training_run_state.n_training_steps % 100 == 0:
                pbar.set_description(
                    f"{training_run_state.n_training_steps}| MSE Loss {torch.stack(mse_losses).mean().item():.3f} | L1 {torch.stack(l1_losses).mean().item():.3f}"
                )
            pbar.update(batch_size)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            if (not training_run_state.started_fine_tuning) and (
                training_run_state.n_training_tokens > sae_group.cfg.training_tokens
            ):
                training_run_state.started_fine_tuning = True
                for name, sparse_autoencoder in sae_group.autoencoders.items():
                    ctx = train_contexts[name]
                    # this should turn grads on for the scaling factor and other parameters.
                    ctx.begin_finetuning(sae_group.autoencoders[name])

    except (KeyboardInterrupt, InterruptedException):
        print("interrupted, saving progress")
        checkpoint_name = training_run_state.n_training_tokens
        _save_checkpoint(
            sae_group,
            train_contexts=train_contexts,
            training_run_state=training_run_state,
            checkpoint_name=checkpoint_name,
        )
        print("done saving")
        raise
    # save final sae group to checkpoints folder
    _save_checkpoint(
        sae_group,
        train_contexts=train_contexts,
        training_run_state=training_run_state,
        checkpoint_name=f"final_{training_run_state.n_training_tokens}",
        wandb_aliases=["final_model"],
    )

    log_feature_sparsities = {
        name: ctx.log_feature_sparsity for name, ctx in train_contexts.items()
    }

    return TrainSAEGroupOutput(
        sae_group=sae_group,
        checkpoint_paths=training_run_state.checkpoint_paths,
        log_feature_sparsities=log_feature_sparsities,
    )







# we are not saving activation store unlike saelens.
def _save_checkpoint(
    sae_group: SparseAutoencoderDictionary,
    train_contexts: dict[str, SAETrainContext],
    training_run_state: SAETrainingRunState,
    checkpoint_name: int | str,
    wandb_aliases: list[str] | None = None,
) -> str:
    """
    Saves a checkpoint of the current training state of the sparse autoencoders.

    Args:
        sae_group (SparseAutoencoderDictionary): Group of sparse autoencoders.
        train_contexts (dict[str, SAETrainContext]): Training contexts for each autoencoder.
        training_run_state (SAETrainingRunState): State of the current training run.
        checkpoint_name (int | str): Name or identifier for the checkpoint.
        wandb_aliases (list[str] | None): Optional list of aliases for Weights & Biases logging.

    Returns:
        str: Path to the saved checkpoint.

    """


    checkpoint_path = f"{sae_group.cfg.checkpoint_path}/{checkpoint_name}"
    training_run_state.checkpoint_paths.append(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    training_run_state_path = f"{checkpoint_path}/{TRAINING_RUN_STATE_PATH}"
    training_run_state.save(training_run_state_path)
    name_for_log = re.sub(r'[^a-zA-Z0-9._-]', '_', sae_group.get_name())
    if sae_group.cfg.log_to_wandb:
        training_run_state_artifact = wandb.Artifact(
            f"{name_for_log}_training_run_state",
            type="training_run_state",
            metadata=dict(sae_group.cfg.__dict__),
        )
        training_run_state_artifact.add_file(training_run_state_path)
        # TODO: should these have aliases=wandb_aliases?
        wandb.log_artifact(training_run_state_artifact)


    for name, sae in sae_group.autoencoders.items():

        ctx = train_contexts[name]
        path = f"{checkpoint_path}/{name}"
        os.makedirs(path, exist_ok=True)
        ctx_path = f"{path}/{SAE_CONTEXT_PATH}"
        ctx.save(ctx_path)

        if sae.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)
        log_feature_sparsities = {"sparsity": ctx.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if sae_group.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            model_artifact = wandb.Artifact(
                f"{name_for_log}",
                type="model",
                metadata=dict(sae_group.cfg.__dict__),
            )
            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")
            if sae_group.cfg.log_optimizer_state_to_wandb:
                model_artifact.add_file(ctx_path)
            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{name_for_log}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(sae_group.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

    return checkpoint_path



def get_imagenet_index_to_name(imagenet_path):
    ind_to_name = {}

    with open( os.path.join(imagenet_path, "LOC_synset_mapping.txt" ), 'r') as file:
        # Iterate over each line in the file
        for line_num, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            label = parts[1].split(',')[0]
            ind_to_name[line_num] = label
    return ind_to_name


class ImageNetValidationDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir, imagenet_class_index, validation_labels,  transform=None, return_index=False):
            self.images_dir = images_dir
            self.transform = transform
            self.labels = {}
            self.return_index = return_index


            # load label code to index
            self.label_to_index = {}
    
            with open(imagenet_class_index, 'r') as file:
                # Iterate over each line in the file
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    code = parts[0]
                    self.label_to_index[code] = line_num


            # load image name to label code
            self.image_name_to_label = {}

            # Open the CSV file for reading
            with open(validation_labels, mode='r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.DictReader(csv_file)
                
                # Iterate over each row in the CSV
                for row in csv_reader:
                    # Split the PredictionString by spaces and take the first element
                    first_prediction = row['PredictionString'].split()[0]
                    # Map the ImageId to the first part of the PredictionString
                    self.image_name_to_label[row['ImageId']] = first_prediction



            self.image_names = list(os.listdir(self.images_dir))

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):


            img_path = os.path.join(self.images_dir, self.image_names[idx])
           # print(img_path)
            image = Image.open(img_path).convert('RGB')

            img_name = os.path.basename(os.path.splitext(self.image_names[idx])[0])

            label_i = self.label_to_index[self.image_name_to_label[img_name]]

            if self.transform:
                image = self.transform(image)

            if self.return_index:
                return image, label_i, idx
            else:
                return image, label_i



    
def setup(checkpoint_path,imagenet_path, num_workers=0, legacy_load=False, pretrained_path=None, expansion_factor = 64, num_epochs=2, layers=9, context_size=197, dead_feature_window=5000,d_in=1024, model_name='vit_base_patch32_224', hook_point="blocks.{layer}.mlp.hook_pre", run_name=None):

    # assuming the same structure as here: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    imagenet_train_path = os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/train")
    imagenet_val_path  =os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/val")
    imagenet_val_labels = os.path.join(imagenet_path, "LOC_val_solution.csv")
    imagenet_label_strings = os.path.join(imagenet_path, "LOC_synset_mapping.txt" )

    cfg = VisionModelRunnerConfig(
        #TODO expose more 
    # Data Generating Function (Model + Training Distibuion)
   # model_name = "gpt2-small",
    model_name = model_name, #
    hook_point = hook_point,
    hook_point_layer = layers, # 
    d_in = d_in,# 768,
    #dataset_path = "Skylion007/openwebtext", #
    #is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = expansion_factor, # online weights used 32! 
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constant",
    train_batch_size_tokens = 1024*4,
    context_size = context_size, # TODO should be auto 
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 32,
    training_tokens = int(1_300_000*num_epochs)*context_size,


    store_batch_size = 32, # num images
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=dead_feature_window,
   # dead_feature_threshold = 1e-6, # did not apper to be used in either future, (is it a different method than window?)
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "vit_sae_training", #
    wandb_entity = None,
    wandb_log_frequency=100,
    eval_every_n_wandb_logs = 10, # so every 1000 steps.
    run_name = run_name,
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = checkpoint_path, # #TODO 
    dtype = torch.float32,

    #loading
    from_pretrained_path = pretrained_path
    )


    #TODO support cfg.resume
    if cfg.from_pretrained_path is not None:
        if legacy_load:
            sae_group = load_legacy_pt_file(cfg.from_pretrained_path)
        else:
            sae_group = SparseAutoencoderDictionary.load_from_pretrained(cfg.from_pretrained_path)
    else:
        sae_group = SparseAutoencoderDictionary(cfg)


    model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True)
    model.to(cfg.device)


    import torchvision
    from transformers import CLIPProcessor
    clip_processor = CLIPProcessor.from_pretrained(cfg.model_name)
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        #TODO for clip only 
        torchvision.transforms.Normalize(mean=clip_processor.image_processor.image_mean,
                         std=clip_processor.image_processor.image_std),
    ])

    


    imagenet1k_data = torchvision.datasets.ImageFolder(imagenet_train_path, transform=data_transforms)




    
    imagenet1k_data_val = ImageNetValidationDataset(imagenet_val_path,imagenet_label_strings, imagenet_val_labels ,data_transforms)




    activations_loader = VisionActivationsStore(
        cfg,
        model,
        imagenet1k_data,

        num_workers=num_workers,
        eval_dataset=imagenet1k_data_val,
    )


  


    return cfg , model, activations_loader, sae_group


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", 
                        required=True,
                        help="folder where you will save checkpoints"
                        )
    parser.add_argument('--imagenet_path', required=True, help='folder containing imagenet1k data organized as follows: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description')

    parser.add_argument("--expansion_factor",
                        type=int,
                        default=64)
    parser.add_argument("--context_size",
                        type=int,
                        default=197)
    parser.add_argument("--d_in",
                        type=int,
                        default=1024)
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    parser.add_argument("--model_name",
                        default="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    parser.add_argument("--hook_point",
                        default="blocks.{layer}.mlp.hook_pre")
    parser.add_argument("--layers",
                    type=int,
                    nargs="+",
                    default=[9])
    parser.add_argument("--num_epochs",
                        type=int,
                        default=2)
    parser.add_argument("--dead_feature_window",
                        type=int,
                        default=5000)
    parser.add_argument("--run_name")

    parser.add_argument('--load', type=str, default=None, help='Pretrained SAE path')
    args = parser.parse_args()

    cfg ,model, activations_loader, sae_group = setup(args.checkpoint_path,args.imagenet_path, pretrained_path=args.load, expansion_factor=args.expansion_factor,layers=args.layers, context_size=args.context_size, model_name=args.model_name ,num_epochs=args.num_epochs,dead_feature_window=args.dead_feature_window,num_workers=args.num_workers, d_in=args.d_in, run_name =args.run_name, hook_point=args.hook_point)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name)


    # train SAE
    train_sae_group_on_vision_model(
        model,
        sae_group,
        activations_loader,
        train_contexts=None, #TODO load checkpoints correctly to match saelens v2.1.3 lm_runner!
        training_run_state=None,  #TODO load checkpoints correctly to match saelens v2.1.3 lm_runner!
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size_tokens,
        feature_sampling_window=cfg.feature_sampling_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.eval_every_n_wandb_logs,
        autocast=cfg.autocast,

    )

    if cfg.log_to_wandb:
        wandb.finish()