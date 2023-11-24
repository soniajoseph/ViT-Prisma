# %%
# Load up the model
import timm
import torch

timm_model = timm.create_model("vit_base_patch8_224", pretrained=True)
timm_model.to("cuda").eval()

# %%
# Optional: Sanity check dummy input for the model
timm_dummy_input = torch.randn(1, 3, 224, 224).to(
    "cuda"
)  # Adjust the size and channels as per your model
timm_model(timm_dummy_input)

# %%
# Get model layer names 
# See layer names that lucent *should* be able to visualize, but this
# doesn't work for BaseViT except on the first patch embedding layer cause
# thats a conv layer
from lucent.modelzoo.util import get_model_layers

layer_names = get_model_layers(timm_model)
print("\n".join(layer_names))

# %%
from lucent.optvis import render, param, transform, objectives
from lucent.optvis.transform import pad, jitter, random_scale, random_rotate
from torchvision import transforms
from example_code_utils import neuron, all_layer_names, some_layers, vit_transforms

# For for vizualizing neurons
# Define the parameter to optimize, in our case it's the image

timm_param_f = lambda: param.image(224, fft=True, decorrelate=True, batch=5, channels=3)

# Define the objective i.e. pick a layer and neuron to visualize
obj = neuron("blocks_8_mlp", 500, 100)

# show the image, and you can also save it if you want
list_of_images = render.render_vis(
            timm_model,
            obj,
            param_f=timm_param_f,
            transforms=vit_transforms,
            thresholds=(250,),
            show_image=True,
            # save_image=True,
            # image_name= wherever_you_want_to_save,
            preprocess=False, # this is needed for BaseViT
        )