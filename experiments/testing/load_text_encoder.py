import open_clip
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_open_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_xl_s13B_b90k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return model, tokenizer, preprocess



def load_huggingface():
    vanilla_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", do_rescale=False)

    # vanilla_model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M")
    # processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M", do_rescale=False)

load_open_clip()