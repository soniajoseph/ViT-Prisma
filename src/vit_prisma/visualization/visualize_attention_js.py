import numpy as np
import string, random, json
import torch
import os
from typing import List
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from jinja2 import Template

def convert_to_3_channels(image):
    # Check if the image has only one channel (grayscale)
    if image.shape[-1] == 1 or image.ndim == 2:
        # Stack the grayscale image three times along the third axis to make it 3-channel
        image = np.squeeze(image)
        image = np.stack([image, image, image], axis=-1)
    return image

def generate_random_string(length=10):
    '''
    Helper function to generate canvas IDs for javascript figures.
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def prepare_image(image):
    if isinstance(image,torch.Tensor):
        image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype('uint8')
    image = np.transpose(image, (1, 2, 0))
    image = convert_to_3_channels(image)
    return image

def flatten_into_patches(image, patch_size, image_size):
    patches = [image[i:i+patch_size, j:j+patch_size, :] for i in range(0, image_size, patch_size) for j in range(0, image_size, patch_size)]
    flattened_patches = [patch.flatten().tolist() for patch in patches]
    return flattened_patches

def normalize_attn_head(attn_head):
    min_val = np.min(attn_head)
    max_val = np.max(attn_head)
    normalized_attn_head = (attn_head - min_val) / (max_val - min_val)
    return normalized_attn_head


# prep data to send to javascript
class AttentionHeadImageJSInfo:

    def __init__(self, attn_head, image, name="No Name", cls_token=True):

        normalized_ah = normalize_attn_head(attn_head)
        if not cls_token:
            normalized_ah = normalized_ah[1:, 1:]

        image_size = image.shape[-1]
        assert image_size == image.shape[-2], "images are assumed to be square"

        patch_size = int(image_size // np.sqrt(len(normalized_ah) - 1))
        image = prepare_image(image)
        flattened_patches = flatten_into_patches(image, patch_size, image_size)

        self.patches = flattened_patches
        self.image_size = image_size
        self.attn_head = normalized_ah.tolist()
        self.name = name



def plot_javascript(
    list_of_attn_heads: Union[torch.Tensor, List[np.ndarray]], 
    list_of_images: Union[List[np.ndarray], np.ndarray], 
    list_of_names: Optional[Union[torch.Tensor, List[str]]] = None, 
    ATTN_SCALING: int = 8, 
    cls_token: bool = True
) -> str:
    """
    Generates HTML and JavaScript code to visualize attention heads with corresponding images.

    Args:
        list_of_attn_heads (Union[torch.Tensor, List[np.ndarray]]): A tensor of shape (num_heads, num_patches, num_patches)
                                                                   or a list of numpy arrays with the same shape.
        list_of_images (Union[List[np.ndarray], np.ndarray]): A list of images or a single image array, each image with shape 
                                                              (height, width, channels).
        list_of_names (Optional[Union[torch.Tensor, List[str]]], optional): A tensor or a list of names for the attention heads. Default is None.
        ATTN_SCALING (int, optional): Scaling factor for attention visualization. Default is 8.
        cls_token (bool, optional): Whether to include the CLS token. Default is True.

    Returns:
        str: Generated HTML and JavaScript code for visualizing the attention heads with corresponding images.
    """
    # if list of attn heads is tensor
    if type(list_of_attn_heads) == torch.Tensor:
        list_of_attn_heads = [np.array(list_of_attn_heads[i]) for i in range(list_of_attn_heads.shape[0])]
    elif type(list_of_attn_heads) != list:
        list_of_attn_heads = [list_of_attn_heads]

    if type(list_of_images) != list:
        list_of_images = [list_of_images]

    if list_of_names == torch.Tensor:
        
        list_of_names = [str(i) for i in list_of_names.tolist()]
    elif list_of_names is None:
        list_of_names = []
        for i in range(len(list_of_attn_heads)):
            list_of_names.append(f"Attention Head {i+1}")


    assert len(list_of_attn_heads) == len(list_of_images), "Must provide an image for each attention head"
    assert len(list_of_attn_heads) == len(list_of_names), "Must provide a name for each attention head"

    attn_head_image_js_infos = []
    for attn_head, image, name in zip(list_of_attn_heads, list_of_images, list_of_names):
        attn_head_image_js_infos.append(AttentionHeadImageJSInfo(attn_head, image, name=name, cls_token=True))

    attn_heads_json = json.dumps([info.attn_head for info in attn_head_image_js_infos])
    patches_json = json.dumps([info.patches for info in attn_head_image_js_infos])
    image_sizes_json = json.dumps([info.image_size for info in attn_head_image_js_infos])
    names_json = json.dumps([info.name for info in attn_head_image_js_infos])

    html_code = generate_html_and_js_code(attn_heads_json, patches_json,image_sizes_json, names_json, ATTN_SCALING, cls_token=cls_token)
    return html_code
    
def generate_html_and_js_code(attn_heads_json, patches_json, image_sizes_json, names_json, ATTN_SCALING, cls_token=True,  canvas_img_id=None, canvas_attn_id=None):

    if canvas_img_id is None:
        canvas_img_id = generate_random_string()
    if canvas_attn_id is None:
        canvas_attn_id = generate_random_string()

    template_folder = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(template_folder, 'template.html'), 'r') as file:
        template = Template(file.read())

    # we want all the js in one html so we read all files and pass them in.
    with open(os.path.join(template_folder, 'main_visualize.js'), 'r') as file:
        main_js = file.read()
    with open(os.path.join(template_folder, 'patch_to_img.js'), 'r') as file:
        patch_to_img_js = file.read()
    with open(os.path.join(template_folder, 'get_color.js'), 'r') as file:
        get_color_js = file.read()

    html_code = template.render(
        main_visualize_js = main_js,
        patch_to_img_js = patch_to_img_js,
        get_color_js = get_color_js,

        attn_heads_json=attn_heads_json,
        patches_json=patches_json,
        image_sizes_json=image_sizes_json,
        names_json = names_json,

        ATTN_SCALING=ATTN_SCALING,
        cls_token = cls_token,

        canvas_attn_id = canvas_attn_id,
        canvas_img_id = canvas_img_id
    )

    return html_code


# get layer 0 activations from running the model on the sample image.
def _get_layer_0_activations(model, image):
    layer_0_activations = None

    def hook_fn(m,i,o):
        nonlocal layer_0_activations
        layer_0_activations = i[0][0].cpu().numpy()

    handle = model.blocks[0].attn.attn_scores.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(np.expand_dims(image,axis=0)).cuda())
    handle.remove()

    return layer_0_activations

def main_example():
    import webbrowser
    import timm
    import copy
    from vit_prisma.utils.get_activations import timmCustomAttention  # Custom Attention has hook functions

    cur_folder = os.path.dirname(os.path.abspath(__file__))

    # load sample image
    sample_image_0 = np.load(os.path.join(cur_folder, "sample_cifar10_image_0.npy"))
    sample_image_10 = np.load(os.path.join(cur_folder, "sample_cifar10_image_10.npy"))

    # Load original model
    orig_model = timm.create_model('vit_base_patch32_224', pretrained=True)

    # Replace original model's attention with hooked attention
    model = copy.deepcopy(orig_model)
    for idx, block in enumerate(model.blocks):
        model.blocks[idx].attn = timmCustomAttention(dim=768, num_heads=12, qkv_bias=True)

    # Reset attention with pretrained weights from original model.
    model.load_state_dict(orig_model.state_dict())
    model = model.cuda()

    layer_0_sample_image_0_activations = _get_layer_0_activations(model, sample_image_0)

    layer_0_sample_image_10_activations = _get_layer_0_activations(model, sample_image_10)
    # plot attention head 3 and 1 ("corner head")

    attn_head_1 = layer_0_sample_image_0_activations[3]
    attn_head_2 = layer_0_sample_image_10_activations[1]

    html_code = plot_javascript([attn_head_1, attn_head_2], [sample_image_0,sample_image_10], list_of_names=["Generic", "Corner"], ATTN_SCALING=8, cls_token=True)

    temp_path = os.path.join(cur_folder, 'temp.html')
    with open(temp_path, 'w') as f:
        f.write(html_code)
    webbrowser.open('file://' + temp_path)

if __name__ == "__main__":
    main_example()
