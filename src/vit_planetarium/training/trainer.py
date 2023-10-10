
# # To do:
# 1. write some really robust training code
# make sure you understand the details of every line
# 2. precache the grokking dataset
# 3. run hyperparam search on grokking dataset on compute canada

import wandb
import torch
import torch.optim as optim
import tqdm
from tqdm.auto import tqdm
from vit_planetarium.training.training_utils import calculate_accuracy, calculate_loss, set_seed
from vit_planetarium.training.training_dictionary import optimizer_dict, loss_function_dict
import os
from torch.utils.data import Dataset, DataLoader

def update_config_with_wandb(config, wandb_config, prefix=""):
    for key, value in vars(config).items():
        wandb_key = f"{prefix}{key}" if prefix else key
        if wandb_key in wandb_config:
            if isinstance(value, type):  # Check if the attribute is a nested class
                update_config_with_wandb(value, wandb_config, prefix=f"{wandb_key}.")
            else:
                setattr(config, key, wandb_config[wandb_key])

def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, type):  # Check if the attribute is a nested class
            result[key] = class_to_dict(value)
        else:
            result[key] = value
    return result

def train(
        model,
        config,
        train_dataset,
        val_dataset,
):

    # Replace config with wandb values if they exist (esp if in hyperparam sweep)
    if config.logging.use_wandb:
        update_config_with_wandb(config, wandb.config)
        print(f"Updated config with wandb values: {config}")
        if config.training.wandb_project_name is None:
            config.training.wandb_project_name = "vit-planetarium"
        config_dict = class_to_dict(config)
        wandb.init(project = config.wandb_project_name, config = config_dict)

    set_seed(config.training.seed)
    model.train()
    model.to(config.training.device)
    optimizer = optimizer_dict[config.training.optimizer_name](model.parameters(), lr = config.training.lr, weight_decay = config.training.weight_decay)
    loss_fn = loss_function_dict[config.training.loss_fn_name]()


    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    steps = 0
    if config.training.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, steps/ config.training.warmup_steps),
        )
    for epoch in tqdm(range(1, config.training.num_epochs + 1)):
        for idx, (images, labels) in tqdm(enumerate(train_loader)):
            if steps % config.logging.log_frequency == 0:
                model.eval()
                logs = {}
                train_loss = calculate_loss(model, train_loader, loss_fn, config.training.device)
                train_acc = calculate_accuracy(model, train_loader, config.training.device)
                test_loss = calculate_loss(model, test_loader, loss_fn, config.training.device)
                test_acc = calculate_accuracy(model, test_loader, config.training.device)
                tqdm.write(f"Steps{steps} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")
                log_dict = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                }
                model.train() # set model back to train mode
        
            images, labels = images.to(config.training.device), labels.to(config.training.device)
            y = model(images)
            loss = loss_fn(y, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm) if config.training.max_grad_norm is not None else None
            optimizer.step()
            scheduler.step() if config.training.warmup_steps > 0 else None
            optimizer.zero_grad()
            tqdm.write(f"Epoch {epoch} | steps{steps} | Loss {loss.item()}") if config.logging.print_every and steps % config.logging.print_every == 0 else None
            torch.save(model.state_dict(), os.path.join(os.path.join(config.saving.parent_dir, config.saving.save_dir), f"model_{steps}.pth")) if steps % config.saving.save_cp_frequency == 0 else None
            if hasattr(config.training, 'max_steps') and config.training.max_steps and steps >= config.training.max_steps:
                break
            steps += 1
    if config.logging.use_wandb:
        wandb.finish()
    return model