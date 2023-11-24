# %%
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn

from vit_prisma.models.base_vit import BaseViT
from vit_prisma.configs.DSpritesConfig import GlobalConfig
from vit_prisma.utils.wandb_utils import update_dataclass_from_dict

# Get the required model from huggingface hub
# And load the checkpoint

REPO_ID = "IamYash/dSprites-medium-AttentionOnly"
# REPO_ID = "IamYash/dSprites-tiny-Attention-and-MLP"

FILENAME = "model_4121600.pth" # pick any from hf

checkpoint = torch.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
config = GlobalConfig()

update_dict = {
    "transformer": {
        "attention_only": True,
        "hidden_dim": 512,
        # 'mlp_dim': 2048, # needed for MLP layer
        "num_heads": 8,
        "num_layers": 4,
    }
}

update_dataclass_from_dict(config, update_dict)
bvit_model = BaseViT(config)

bvit_model.load_state_dict(checkpoint["model_state_dict"])
bvit_model.to("cuda").eval()

# %%
# Optional: Sanity check dummy input for the model
bvit_dummy_input = torch.randn(1, 1, 64, 64).to("cuda")
bvit_model(bvit_dummy_input)

# %%
# Get model layer names 
# See layer names that lucent *should* be able to visualize, but this
# doesn't work for BaseViT except on the first patch embedding layer cause
# thats a conv layer
from lucent.modelzoo.util import get_model_layers

layer_names = get_model_layers(bvit_model)
print("\n".join(layer_names))

# %%
from lucent.optvis import render, param, transform, objectives
from lucent.optvis.transform import pad, jitter, random_scale, random_rotate
from torchvision import transforms
from example_code_utils import neuron, all_layer_names, some_layers, vit_transforms

# For for vizualizing neurons
# Define the parameter to optimize, in our case it's the image

bvit_param_f = lambda: param.image(64, batch=1, channels=1)

# Define the objective i.e. pick a layer and neuron to visualize
obj = neuron("patch_embedding", 500, 100)

# for basevit, this only works for the first patch embedding layer,
# and the patch_embedding_proj layer
# it seems that this is because they are conv layers
# show the image, and you can also save it if you want
list_of_images = render.render_vis(
            bvit_model,
            obj,
            param_f=bvit_param_f,
            transforms=vit_transforms,
            thresholds=(250,),
            show_image=True,
            # save_image=True,
            # image_name= wherever_you_want_to_save,
            preprocess=False, # this is needed for BaseViT
        )
