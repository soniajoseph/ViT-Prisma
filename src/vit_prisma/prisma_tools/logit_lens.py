import numpy as np
import torch
from collections import defaultdict

from vit_prisma.utils.data_utils.imagenet_dict import IMAGENET_DICT
from vit_prisma.utils.data_utils.imagenet_utils import imagenet_index_from_word


def get_patch_logit_directions(cache, all_answers, incl_mid=False, return_labels=True):
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=incl_mid, return_labels=True
    )
    scaled_residual_stack = cache.apply_ln_to_stack(
        accumulated_residual, layer=-1,
    )
    result = torch.einsum('lbpd,od -> lbpo', scaled_residual_stack, all_answers)
    # Rearrange so batches are first
    result = result.permute(1, 2, 0, 3)
    return result, labels

def get_patch_logit_dictionary(patch_logit_directions, batch_idx=0, rank_label=None):
    patch_dictionary = defaultdict(list)
    # Go through laeyrs of one batch
    for patch_idx, patches in enumerate(patch_logit_directions[batch_idx]):
        # Go through every patch and get max prediction
        for logits in patches:
            probs = torch.softmax(logits, dim=-1)
            # Get index of max prediction
            predicted_idx = int(torch.argmax(probs))
            logit = logits[predicted_idx].item()
            predicted_class_name = IMAGENET_DICT[predicted_idx]
            if rank_label:
                # Where is the rank_label in the sorted list?
                rank_index = imagenet_index_from_word(rank_label)
                sorted_list = torch.argsort(probs, descending=True)
                rank = np.where(sorted_list == rank_index)[0][0]
                patch_dictionary[patch_idx].append((logit, predicted_class_name, predicted_idx, rank))
            else:
                patch_dictionary[patch_idx].append((logit, predicted_class_name, predicted_idx))
    return patch_dictionary
