from PIL import Image
import requests
from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAModel, VJEPAImageProcessor
from vit_prisma.vjepa_hf.configs import CONFIGS
import yaml


from vit_prisma.models.base_vit import HookedViT


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image



# og_model_name = "vjepa_v1_vit_huge"
# print("Loading hooked model")
# hooked_model = HookedViT.from_pretrained(og_model_name, is_timm=False, is_clip=False, fold_ln=False, center_writing_weights=False) # in future, do all models

# print(hooked_model)

model_name = "vjepa_v1_vit_huge"
config = CONFIGS["v1"]["vit_h"]
print(config)
model_paths = yaml.safe_load(open('paths_cw.yaml'))
model_path = model_paths[model_name]["loc"]

image = prepare_img()
processor = VJEPAImageProcessor(crop_size=config.crop_size)
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times for now
pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W

model = VJEPAModel.from_pretrained(model_path)

# print out config and state dictionary
print(model.config)

outputs = model(pixel_values)

print(outputs)
print("Outputs shape:", outputs[0].shape)