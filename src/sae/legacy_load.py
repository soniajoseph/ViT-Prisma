# adapted from sae-lens legacy loader for our own renaming issues (as prisma used to have a copy of sae-lens code)

import pickle 
import os 
import torch 
from types import SimpleNamespace
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

from sae_lens.training.sae_group import SparseAutoencoderDictionary
class PrismaBackwardsCompatibleUnpickler(pickle.Unpickler):
    """
    An Unpickler that can load files saved before the use of pip install sae-lens
    """

    def find_class(self, module: str, name: str):
        module = module.replace("sae_training", "sae_lens.training")
        module = module.replace("sae.language", "sae_lens.training")
        module = module.replace("sae.hooked_prisma", "sae")
        if name == "SAEGroup":
            name = "SparseAutoencoderDictionary"

        return super().find_class(module, name)


def load_legacy_pt_file(path)->SparseAutoencoderDictionary:

    # Ensure the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file found at specified path: {path}")

    # Load the state dictionary
    if path.endswith(".pt"):
        try:
            # this is hacky, but can't figure out how else to get torch to use our custom unpickler
            fake_pickle = SimpleNamespace()
            fake_pickle.Unpickler = PrismaBackwardsCompatibleUnpickler
            fake_pickle.__name__ = pickle.__name__

            if torch.cuda.is_available():
                group = torch.load(
                    path,
                    pickle_module=fake_pickle,
                )
            else:
                map_loc = "mps" if torch.backends.mps.is_available() else "cpu"
                group = torch.load(
                    path, pickle_module=fake_pickle, map_location=map_loc
                )
                if isinstance(group, dict):
                    group["cfg"].device = map_loc
                else:
                    group.cfg.device = map_loc
        except Exception as e:
            raise IOError(f"Error loading the state dictionary from .pt file: {e}")

    else:
        raise ValueError(
            f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
        )
    print("GOT HERE ALREADY...")


    if not isinstance(group, SparseAutoencoderDictionary):
        raise ValueError("The loaded object is not a valid SAEGroup")

    #print(group.cfg)
    if type(group.autoencoders) == list:
        group.autoencoders = {f"dummy_name_{i}":aut for i,aut in enumerate(group.autoencoders)}
    for k in group.autoencoders.keys():
        group.autoencoders[k].noise_scale = 0
        group.autoencoders[k].activation_fn = torch.nn.ReLU()
        group.autoencoders[k].scaling_factor = torch.ones_like(group.autoencoders[k].b_enc)
        group.autoencoders[k].use_ghost_grads = True


    return group
