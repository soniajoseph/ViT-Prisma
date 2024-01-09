import numpy as np
import string, random, json
import torch
import os

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

def plot_javascript(list_of_attn_heads, image, ATTN_SCALING=20, cls_token=True):

    if type(list_of_attn_heads) != list:
        list_of_attn_heads = [list_of_attn_heads]

    num_patches = len(list_of_attn_heads[0])
    for ah in list_of_attn_heads:
        assert num_patches == len(ah), "all attention heads must be same length"

    image_size = len(image[-1]) 
    patch_size = int(image_size // np.sqrt(num_patches-1)) 
    num_patch_width = image_size // (patch_size-1)
    print("num_patches", num_patches)
    print("patch_size", patch_size)

    canvas_img_id = generate_random_string()
    canvas_attn_id = generate_random_string()
    image = prepare_image(image)
    flattened_patches = flatten_into_patches(image, patch_size, image_size)
    patches_json = json.dumps(flattened_patches)

    list_of_normalized_attn_heads = []
    for ah in list_of_attn_heads:
        normalized_ah = normalize_attn_head(ah)
        if not cls_token:
            normalized_ah = normalized_ah[1:, 1:]
            num_patches = len(normalized_ah)
        list_of_normalized_attn_heads.append(normalized_ah)
    list_of_attn_head_json = json.dumps([ah.tolist() for ah in list_of_normalized_attn_heads])
    html_code = generate_html_and_js_code(patches_json, list_of_attn_head_json, canvas_img_id, canvas_attn_id,
    image_size, patch_size, num_patch_width, num_patches, ATTN_SCALING, cls_token=cls_token)
    return html_code




    
def generate_html_and_js_code(patches_json, list_of_attn_head_json, canvas_img_id, canvas_attn_id,
    image_size, patch_size, num_patch_width, num_patches, ATTN_SCALING, cls_token=True):

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
        canvas_attn_id=canvas_attn_id,
        canvas_img_id=canvas_img_id,
        image_size=image_size,
        patch_size = patch_size,
        num_patch_width = num_patch_width,
        num_patches=num_patches,
        ATTN_SCALING=ATTN_SCALING,
        cls_token = cls_token,
        patches_json = patches_json,
        list_of_attn_head_json = list_of_attn_head_json
    )

    return html_code

def main_example():
    import webbrowser
    import timm
    import copy
    from vit_prisma.utils.get_activations import timmCustomAttention  # Custom Attention has hook functions

    cur_folder = os.path.dirname(os.path.abspath(__file__))

    # load sample image
    sample_image = np.load(os.path.join(cur_folder, "sample_cifar10_image.npy"))

    # Load original model
    orig_model = timm.create_model('vit_base_patch32_224', pretrained=True)

    # Replace original model's attention with hooked attention
    model = copy.deepcopy(orig_model)
    for idx, block in enumerate(model.blocks):
        model.blocks[idx].attn = timmCustomAttention(dim=768, num_heads=12, qkv_bias=True)

    # Reset attention with pretrained weights from original model.
    model.load_state_dict(orig_model.state_dict())

    ## get layer 0 activations from running the model on the sample image.
    layer_0_sample_image_activations = None
    def hook_fn(m,i,o):
        nonlocal layer_0_sample_image_activations
        layer_0_sample_image_activations = i[0][0].cpu().numpy()
    handle = model.blocks[0].attn.attn_scores.register_forward_hook(hook_fn)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(np.expand_dims(sample_image,axis=0)).cuda())
    handle.remove()

    # plot attention head 3 and 1 ("corner head")

    attn_head_1 = layer_0_sample_image_activations[3]

    attn_head_2 = layer_0_sample_image_activations[1]

    html_code = plot_javascript([attn_head_1, attn_head_2], image=sample_image, ATTN_SCALING=8, cls_token=True)

    temp_path =  os.path.join(cur_folder, 'temp.html')
    with open(temp_path, 'w') as f:
        f.write(html_code)
    webbrowser.open('file://' + temp_path)

if __name__ == "__main__":
    main_example()