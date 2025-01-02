
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
import time
from tqdm import tqdm

# @TODO DEBUG what the hell is going on with prsima updates not loading in apptainer


def iterate_through_dataset(trainer):
    start = time.time()
    n_training_tokens = 0
    # pbar = tqdm(total=trainer.cfg.total_training_tokens, desc="Testing Activation Storage Inference Times")
    while n_training_tokens < trainer.cfg.total_training_tokens:
        _ = trainer.activations_store.next_batch()
        n_training_tokens += trainer.cfg.train_batch_size
        # pbar.update(trainer.cfg.train_batch_size)
    end = time.time()
    return end - start



cfg = VisionModelSAERunnerConfig.load_config("conf_test.json")
cfg.use_cached_activations = True

trainer = VisionSAETrainer(cfg)

precomputed_time = iterate_through_dataset(trainer)

cfg.use_cached_activations = False
trainer.activations_store = trainer.initialize_activations_store(
    trainer.dataset, trainer.eval_dataset
)
on_demand_calc_time = iterate_through_dataset(trainer)


print(f"On Demand took {on_demand_calc_time:.2f} seconds for {cfg.num_epochs}")
print(f"Precomputed took {precomputed_time:.2f} seconds for {cfg.num_epochs}")