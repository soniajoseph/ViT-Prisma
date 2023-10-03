
# # To do:
# 1. write some really robust training code
# make sure you understand the details of every line
# 2. precache the grokking dataset
# 3. run hyperparam search on grokking dataset on compute canada

import wandb
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from vit_planetarium.utils.training_utils import calculate_accuracy, calculate_loss, set_seed
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
    if config.Logging.use_wandb:
        update_config_with_wandb(config, wandb.config)
        print(f"Updated config with wandb values: {config}")

    set_seed(config.Training.seed)
    
    model.train()
    model.to(config.Training.device)

    if config.Logging.use_wandb:
        if config.Training.wandb_project_name is None:
            config.Training.wandb_project_name = "vit-planetarium"
        config_dict = class_to_dict(config)
        wandb.init(project = config.wandb_project_name, config = config_dict)
    
    optimizer = optim.AdamW(model.parameters(), lr = config.Training.lr, weight_decay = config.Training.weight_decay)

    if config.Training.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, steps/ config.Training.warmup_steps),
        )

    train_loader = DataLoader(train_dataset, batch_size=config.Training.batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


    steps = 0
    for epoch in tqdm(range(1, config.Training.num_epochs + 1)):
        for images, labels in tqdm(enumerate(train_loader)):

            if steps % config.Logging.log_frequency == 0:
                model.eval()
                logs = {}
                train_loss = calculate_loss(model, train_loader, hp['loss_function'], config.Training.device)
                train_acc = calculate_accuracy(model, train_loader, config.Training.device)
                test_loss = calculate_loss(model, test_loader, hp['loss_function'], config.Training.device)
                test_acc = calculate_accuracy(model, test_loader, config.Training.device)

                print(f"Steps{steps} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")

                if config.Logging.save_checkpoints:
                    torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_{steps}.pth"))

                log_dict = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                }

                if config.Logging.use_wandb:
                    wandb.log(log_dict, step=steps)

                model.train() # set model back to train mode
                
            images, labels = images.to(config.Training.device), labels.to(config.Training.device)

            loss = model(images)
            loss.backward()

            if config.Training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.Training.max_grad_norm)

            optimizer.step()
            if config.Training.warmup_steps > 0:
                scheduler.step()

            optimizer.zero_grad()

            samples += len(labels)
            
            if config.Training.print_every is not None and steps % config.Training.print_every == 0:
                print(f"Epoch {epoch} | steps{steps} | Loss {loss.item()}")
            
            if config.Training.max_steps is not None and steps>= config.Training.max_steps:
                break

            steps += 1

    if config.Logging.use_wandb:
        wandb.finish()

    return model

