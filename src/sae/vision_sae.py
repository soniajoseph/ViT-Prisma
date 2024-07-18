from sae_lens import SAETrainingRunner, TrainingSAE, TrainingSAEConfig
import torch

# import librareis
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    mse_loss: float
    l1_loss: float
    ghost_grad_loss: float
    auxiliary_reconstruction_loss: float = 0.0

class VisionTrainingSAE(TrainingSAE):
    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    # Implements top k poss
    def training_forward_pass(
            self,
            sae_in: torch.Tensor,
            current_l1_coefficient: float,
            dead_neuron_mask: Optional[torch.Tensor] = None,
        ) -> TrainStepOutput:

            # do a forward pass to get SAE out, but we also need the
            # hidden pre.
            feature_acts, _ = self.encode_with_hidden_pre_fn(sae_in)
            sae_out = self.decode(feature_acts)

            # MSE LOSS
            per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
            mse_loss = per_item_mse_loss.sum(dim=-1).mean()

            # GHOST GRADS
            if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:

                # first half of second forward pass
                _, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
                ghost_grad_loss = self.calculate_ghost_grad_loss(
                    x=sae_in,
                    sae_out=sae_out,
                    per_item_mse_loss=per_item_mse_loss,
                    hidden_pre=hidden_pre,
                    dead_neuron_mask=dead_neuron_mask,
                )
            else:
                ghost_grad_loss = 0.0


            if self.cfg.architecture == "gated":
                # Gated SAE Loss Calculation

                # Shared variables
                sae_in_centered = (
                    self.reshape_fn_in(sae_in) - self.b_dec * self.cfg.apply_b_dec_to_input
                )
                pi_gate = sae_in_centered @ self.W_enc + self.b_gate
                pi_gate_act = torch.relu(pi_gate)

                # SFN sparsity loss - summed over the feature dimension and averaged over the batch
                l1_loss = (
                    current_l1_coefficient
                    * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
                )

                # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
                via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
                aux_reconstruction_loss = torch.sum(
                    (via_gate_reconstruction - sae_in) ** 2, dim=-1
                ).mean()

                loss = mse_loss + l1_loss + aux_reconstruction_loss

            elif self.cfg.activation_fn_str =='topk':

                # default SAE sparsity loss
                weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
                sparsity = weighted_feature_acts.norm(
                    p=self.cfg.lp_norm, dim=-1
                )  # sum over the feature dimension

                l1_loss = torch.tensor(0.0) # make small value just for logging purposes
                loss = mse_loss + ghost_grad_loss # there's no l1 loss for top k

                aux_reconstruction_loss = torch.tensor(0.0)

            else:
                # default SAE sparsity loss
                weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
                
                if not self.cfg.l1_loss_wd_norm:
                     sparsity = weighted_feature_acts.norm(
                    p=self.cfg.lp_norm, dim=-1
                )  # sum over the feature dimension
                     
                else:
                    # Calculate the L2 norm of each column of W_dec
                    W_dec_norms = self.W_dec.norm(dim=0)  # This is now a 1D tensor
                    # Calculate the sparsity term
                    sparsity = (feature_acts * W_dec_norms).sum(dim=-1)  # sum over the feature dimension

                    # Calculate the L1 loss (no
                                    

                l1_loss = (current_l1_coefficient * sparsity).mean()
                loss = mse_loss + l1_loss + ghost_grad_loss

                aux_reconstruction_loss = torch.tensor(0.0)



            return TrainStepOutput(
                sae_in=sae_in,
                sae_out=sae_out,
                feature_acts=feature_acts,
                loss=loss,
                mse_loss=mse_loss.item(),
                l1_loss=l1_loss.item(),
                ghost_grad_loss=(
                    ghost_grad_loss.item()
                    if isinstance(ghost_grad_loss, torch.Tensor)
                    else ghost_grad_loss
                ),
                auxiliary_reconstruction_loss=aux_reconstruction_loss.item(),
            )