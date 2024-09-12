import pytest
import torch
import timm
from vit_prisma.models.base_vit import HookedViT
import open_clip


#currently only vit_base_patch16_224 supported (config loading issue)
def test_loading_open_clip():
    TOLERANCE = 1e-5

    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"
    
    random_input = torch.randn(1, 3, 224, 224)


    def get_all_layer_outputs(model, input_tensor):
        layer_outputs = []
        layer_names = []
        
        def hook_fn(module, input, output):
            layer_outputs.append(output)
            layer_names.append(type(module).__name__)

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(input_tensor)

        for hook in hooks:
            hook.remove()

        return layer_outputs, layer_names

    model_name = 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
    og_model, *data = open_clip.create_model_and_transforms(model_name)
    og_model.eval()
    all_outputs, layer_names = get_all_layer_outputs(og_model, random_input)
    og_state_dict = og_model.state_dict()


    hooked_model = HookedViT.from_pretrained('open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False) # in future, do all models
    hooked_model.to(device)
    hooked_model.eval()
    hooked_state_dict = hooked_model.state_dict()

    final_output_hooked, cache = hooked_model.run_with_cache(random_input)
    final_output_og, *data = og_model(random_input)[0]

    for k in cache:
        print(k, cache[k].shape)

    print("all layer names", layer_names)

    for i, (output, name) in enumerate(zip(all_outputs, layer_names)):
        try:
            print(f"Layer {i} ({name}) output shape: {output.shape}")
        except Exception as e:
            print(f"Layer {i} ({name}) output shape: {output[0].shape}")
        if i == 0: 
            output = output.flatten(2).transpose(1, 2)
            hooked_output = cache['hook_embed']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print('post conv activations match')
        elif i == 1:
            hooked_output = cache['hook_full_embed'] # before layer norm
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print('post full embedding activations match')
        elif i == 2:
            hooked_output = cache['hook_ln_pre']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
        elif i == 8: # First GeLU
            hooked_output = cache['blocks.0.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("First post GeLU matches")
        elif i == 118: # Last GeLU
            hooked_output = cache['blocks.11.mlp.hook_post']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("Final post GeLU matches")
        elif i == 124: # Final layer norm
            hooked_output = cache['ln_final']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - output))}"
            print("Final post layer norm matches")
        elif i == 125:
            hooked_output = cache['hook_post_head_pre_normalize']
            assert torch.allclose(hooked_output, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(final_output_hooked - output))}"
            print("Transformer pre final layer norm (for original model) matches")
        elif i == 126: # Final output
            output = output[0]
            assert torch.allclose(final_output_hooked, output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(final_output_hooked - output))}"
            print("Final output matches")

    final_output_hooked, cache = hooked_model.run_with_cache(random_input)
    final_output_og, *data = og_model(random_input)

    print("Final output shapes", final_output_hooked.shape, final_output_og.shape)

    assert torch.allclose(final_output_hooked, final_output_og, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(final_output_hooked - final_output_og))}"
    print("All tests passed!")

test_loading_open_clip()

