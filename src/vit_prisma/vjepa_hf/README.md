## Huggingface versions of VJEPA

This app contains the code for the Huggingface version of VJEPA, along with conversion scripts to convert our trained checkpoints into the HF format. 

| Model | VJEPA checkpoint | HF Checkpoint (CW) |
| --------------- |  --------------- | --------------- |
| [vjepa_v1_vit_huge](configs.py#L5) | /checkpoint/amaia/video/koustuvs/models/vjepa/v1_public/vith16.pth.tar | /checkpoint/amaia/video/koustuvs/models/vjepa/v1_vit_huge_conv/ |
| [vjepa_v1_vit_huge_384](configs.py#L14) | /checkpoint/amaia/video/koustuvs/models/vjepa/v1_vith16_384/vith16-384.pth.tar | /checkpoint/amaia/video/koustuvs/models/vjepa/v1_vith16_384_hf/ |
| [vjepa_v1.5_vit_g_256](configs.py#L26) | /checkpoint/amaia/video/koustuvs/models/vjepa_v1.5/vm2myt1bin1k-16fpc-vitg16-256px-local-ropev2/latest.pth.tar | /checkpoint/amaia/video/koustuvs/models/vjepa_v1.5/vm2myt1bin1k-16fpc-vitg16-256px-local-ropev2/hf/ |

The model paths are also added in [paths_cw.yaml](paths_cw.yaml), which should be updated subsequently with newer checkpoint conversions.

### Using VJEPA HF in your code 

```python
from PIL import Image
import requests
from app.vjepa_hf.modeling_vjepa import VJEPAModel, VJEPAImageProcessor
from app.vjepa_hf.configs import CONFIGS
import yaml

def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

model_name = "vjepa_v1.5_vit_g_256"
config = CONFIGS["v1.5"]["vit_g_256"]
model_paths = yaml.safe_load(open('paths_cw.yaml'))
model_path = model_paths[model_name]["loc"]

image = prepare_img()
processor = VJEPAImageProcessor(crop_size=config.crop_size)
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times for now
pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # B x C x T x H x W

model = VJEPAModel.from_pretrained(model_path)
outputs = model(pixel_values)
```

### Converting VJEPA checkpoints to Huggingface

Check the [conversion script](convert_vjepa_to_hf.jpy) for details. For instance, to convert v1.5, do the following:

```bash
python -m app.vjepa_hf.convert_vjepa_to_hf \
    --model_name vit_g_256 \
	--ckpt_loc <vjepa path> \
	--outp_loc <outp path> \
	--model_type v1.5 \
	--convert
```

### Test converted checkpoint 

After every conversion, check if the outputs match from vjepa model and the HF model:

```bash
python -m app.vjepa_hf.convert_vjepa_to_hf \
    --model_name vit_g_256 \
	--ckpt_loc <vjepa path> \
	--outp_loc <outp path> \
	--model_type v1.5 \
	--test
```

### Questions?

Contact Koustuv.

