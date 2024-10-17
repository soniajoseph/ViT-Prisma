import traceback
from io import BytesIO
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer


BASE_DIR = Path(__file__).resolve().parent.parent.parent / "mechanistic_interpretability"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINTS_DIR = MODEL_DIR / "vision/checkpoints"
SAE_CHECKPOINTS_DIR = MODEL_DIR / "sae"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

models = [
    # "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",

    # Already Working
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
    # "open-clip:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    # "open-clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K",
    # "open-clip:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    # "open-clip:timm/vit_base_patch16_clip_224.laion400m_e31",
    # "open-clip:timm/vit_base_patch16_clip_224.laion400m_e32",
    # "open-clip:timm/vit_base_patch16_clip_224.metaclip_2pt5b",
    # "open-clip:timm/vit_base_patch16_clip_224.metaclip_400m",
    # "open-clip:timm/vit_base_patch16_clip_224.openai",
    # "open-clip:timm/vit_base_patch32_clip_224.laion2b_e16",
    # "open-clip:timm/vit_base_patch32_clip_224.laion400m_e31",
    # "open-clip:timm/vit_base_patch32_clip_224.laion400m_e32",
    # "open-clip:timm/vit_base_patch32_clip_224.metaclip_2pt5b",
    # "open-clip:timm/vit_base_patch32_clip_224.metaclip_400m",
    # "open-clip:timm/vit_base_patch32_clip_224.openai",

    # Now load and perform inference
    # "open-clip:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K",
    # "open-clip:laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
    # "open-clip:laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
    # "laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k",
    # "open-clip:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    # "open-clip:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    # "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    # "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    # "open-clip:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    # "open-clip:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    # "open-clip:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    # "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e31",
    # "open-clip:timm/vit_base_patch16_plus_clip_240.laion400m_e32",
    # "open-clip:timm/vit_large_patch14_clip_224.laion400m_e31",
    # "open-clip:timm/vit_large_patch14_clip_224.laion400m_e32",
    # "open-clip:timm/vit_large_patch14_clip_224.metaclip_2pt5b",
    # "open-clip:timm/vit_large_patch14_clip_224.metaclip_400m",
    # "open-clip:timm/vit_large_patch14_clip_224.openai",
    # "open-clip:timm/vit_large_patch14_clip_336.openai",
    # "open-clip:timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m",
    # "open-clip:timm/vit_medium_patch32_clip_224.tinyclip_laion400m",
    # "open-clip:timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m",
    # "open-clip:timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m",
    # "open-clip:timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b",
    # "open-clip:timm/vit_huge_patch14_clip_224.metaclip_2pt5b",
    # "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    # "open-clip:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    # "open-clip:laion/CLIP-ViT-g-14-laion2B-s34B-b88K",

    # Known not working
    # This is actually a key architectural difference in SigLIP compared to standard ViT - it uses attention pooling with learnable queries instead of a class token for aggregating the image features. I apologize for my earlier incorrect assumption about it having a standard class token.
    # "open-clip:timm/ViT-B-16-SigLIP-384",
    # "open-clip:timm/ViT-B-16-SigLIP-512",
    # "open-clip:timm/ViT-B-16-SigLIP-i18n-256",
    # "open-clip:timm/ViT-L-16-SigLIP-256",
    # "open-clip:timm/ViT-L-16-SigLIP-384",
    # "open-clip:timm/ViT-SO400M-14-SigLIP",
    # "open-clip:timm/ViT-SO400M-14-SigLIP-384",
    # "open-clip:timm/ViT-SO400M-16-SigLIP-i18n-256",
    # "open-clip:timm/ViT-B-16-SigLIP",
    # "open-clip:timm/ViT-B-16-SigLIP-256",
    # "open-clip:laion/CoCa-ViT-B-32-laion2B-s13B-b90k",
    # "open-clip:laion/CoCa-ViT-L-14-laion2B-s13B-b90k",
    # "open-clip:laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k",
    # "open-clip:laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k",

    # TODO EdS: Needs a convert_eva_weights similar to convert_timm_weights
    "open-clip:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k",
    # "open-clip:timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k",
    # "open-clip:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144",
    # "open-clip:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
    # "open-clip:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k",
    # "open-clip:timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k",
    # "open-clip:timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k",
]

def run_inference(model, image_url, image_size):
    """
    Run inference on a single ImageNet image using a pre-trained model.

    Args:
        model: PyTorch model (should be pre-trained on ImageNet)
        image_url: URL to an image

    Returns:
        Prediction index and confidence score
    """
    # Download image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Prepare image
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Get prediction
    _, predicted_idx = torch.max(output, 1)
    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx]

    return predicted_idx.item(), confidence.item()


if __name__ == "__main__":

    image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    for name in models:
        print("=====================")
        print(name)
        print("=====================")
        try:
            cfg = VisionModelSAERunnerConfig(
                model_name=name,
                num_epochs=1,
                n_checkpoints=5,
                # total_training_images=10000 * 2,
                # total_training_tokens=10000 * 2 * 50,  # Images x epochs x tokens
                log_to_wandb=False,
                verbose=True,
                wandb_log_frequency=100,
                dataset_path=str(DATA_DIR / "imagenet"),
                dataset_train_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
                dataset_val_path=str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
                checkpoint_path=str(MODEL_DIR / "sae/imagenet/checkpoints"),
                train_batch_size=1024,
                device=DEVICE,
                total_training_images=1000,
                total_training_tokens=1000*50,
                wandb_entity="Stevinson",
                wandb_project="loading",
            )
            trainer = VisionSAETrainer(cfg)
            # sae = trainer.run()

            idx, conf = run_inference(trainer.model, image_url, trainer.model.cfg.image_size)
            print(f"Idx: -----------------------> {idx}")
            # print(f"Predicted class: {idx}")
            # print(f"Confidence: {conf:.2%}")
        except Exception as e:
            print(f"-------------------------------------------")
            print(f"-------------------------------------------\n\n\n")
            print(f"Error: {str(e)}\n{traceback.format_exc()}")
            print(f"\n\n\n-------------------------------------------")
            print(f"-------------------------------------------")
        # sae = trainer.run()

    # cfg.wandb_project = cfg.model_name.replace('/', '-') + "-expansion-" + str(
    # cfg.expansion_factor) + "-layer-" + str(cfg.hook_point_layer)
    # cfg.unique_hash = uuid.uuid4().hex[:8]
    # cfg.run_name = cfg.unique_hash + "-" + cfg.wandb_project
    # wandb.init(project=cfg.wandb_project, name=cfg.run_name, entity="Stevinson")
    #
    # cfg.sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints/ff86e305-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16-layer-9/n_images_1300008.pt")
    # model = load_model(cfg)
    # sae = load_sae(cfg)
    # train_data, val_data, val_data_visualize = load_dataset(cfg, visualize=True)  # TODO: I should be using test not validation data here
    #
    # evaluator = Evaluator(model, val_data, cfg, visualize_data=val_data_visualize)  # TODO: Do I need the visualize data? Or can I just use the same variable
    # evaluator.evaluate(sae, context=EvaluationContext.POST_TRAINING)

    # This is used to create a small subset of the data in l.79 of loader.py for fast testing
    # train_data = SubsetDataset(train_data, 10000)
    # val_data = SubsetDataset(val_data, 1000)