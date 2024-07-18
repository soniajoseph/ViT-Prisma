from sae_lens.training.sae_trainer import SAETrainer
import torch



from sae.vision_evals import run_evals_vision

class VisionSAETrainer(SAETrainer):

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.wandb_log_frequency * self.cfg.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            eval_metrics = run_evals_vision(
                self.sae,
                self.activation_store,
                model=self.model,
                n_training_steps=self.n_training_steps,
                suffix= "",

                # eval_config=self.trainer_eval_config,
                # model_kwargs=self.cfg.model_kwargs,
            ) # change wrapper class

            # # Remove eval metrics that are already logged during training
            # eval_metrics.pop("metrics/explained_variance", None)
            # eval_metrics.pop("metrics/explained_variance_std", None)
            # eval_metrics.pop("metrics/l0", None)
            # eval_metrics.pop("metrics/l1", None)
            # eval_metrics.pop("metrics/mse", None)

            # # Remove metrics that are not useful for wandb logging
            # eval_metrics.pop("metrics/total_tokens_evaluated", None)

            # W_dec_norm_dist = self.sae.W_dec.detach().float().norm(dim=1).cpu().numpy()
            # eval_metrics["weights/W_dec_norms"] = wandb.Histogram(W_dec_norm_dist)  # type: ignore

            # if self.sae.cfg.architecture == "standard":
            #     b_e_dist = self.sae.b_enc.detach().float().cpu().numpy()
            #     eval_metrics["weights/b_e"] = wandb.Histogram(b_e_dist)  # type: ignore
            # elif self.sae.cfg.architecture == "gated":
            #     b_gate_dist = self.sae.b_gate.detach().float().cpu().numpy()
            #     eval_metrics["weights/b_gate"] = wandb.Histogram(b_gate_dist)  # type: ignore
            #     b_mag_dist = self.sae.b_mag.detach().float().cpu().numpy()
            #     eval_metrics["weights/b_mag"] = wandb.Histogram(b_mag_dist)  # type: ignore

            # wandb.log(
            #     eval_metrics,
            #     step=self.n_training_steps,
            # )
            self.sae.train()