from PIL import Image
import os

import torch
from experiments.utils.loaders.loaders import load_remote_sae_and_model
from vit_prisma.sae.evals.evaluator import Evaluator
from vit_prisma.utils.constants import EvaluationContext, DATA_DIR, MODEL_DIR, DEVICE, BASE_DIR
from vit_prisma.utils.data_utils.loader import load_dataset

from experiments.testing.load_text_encoder import load_open_clip


def main():
    loaded_model, loaded_tokenizer, loaded_preprocess = load_open_clip()  # laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K

    repo_id = "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_mlp_out-l1-1e-05"  # DataComp.XL-s13B-b90K
    sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints") + f"/{repo_id}"
    os.makedirs(sae_path, exist_ok=True)

    overload_cfg = {
        "wandb_log_frequency": 100,
        "dataset_path": str(DATA_DIR / "imagenet"),
        "dataset_train_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/train"),
        "dataset_val_path": str(DATA_DIR / "imagenet/ILSVRC/Data/CLS-LOC/val"),
        "checkpoint_path": str(MODEL_DIR / "sae/imagenet/checkpoints"),
        "sae_path": sae_path,
        "wandb_entity": "Stevinson",
        "wandb_project": "imagenet",
        "log_to_wandb": False,
        "verbose": True,
        "device": DEVICE,
    }

    sae, language, model = load_remote_sae_and_model(repo_id, current_cfg=overload_cfg)

    image = loaded_preprocess(Image.open(BASE_DIR / "tests/test_text_encoder/test_diagram_img.png")).unsqueeze(0)
    text = loaded_tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        loaded_image_features = loaded_model.encode_image(image)
        loaded_text_features = loaded_model.encode_text(text)

        image_features = model(image)
        text_features = language.encode_text(text)

        class_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        loaded_class_probs = (100.0 * loaded_image_features @ loaded_text_features.T).softmax(dim=-1)

    assert (class_probs == torch.tensor([[1., 0., 0.]])).all()
    assert (loaded_class_probs == torch.tensor([[1., 0., 0.]])).all()

    # train, test_data, test_data_visualisation = load_dataset(sae.cfg, visualize=True)
    #
    # evaluator_custom_model = Evaluator(model, None, test_data, sae.cfg, visualize_data=test_data_visualisation)
    # evaluator_custom_model.evaluate(sae, context=EvaluationContext.POST_TRAINING)
    #
    # evaluator_original_model = Evaluator(model, language, test_data, sae.cfg, visualize_data=test_data_visualisation)
    # evaluator_original_model.evaluate(sae, context=EvaluationContext.POST_TRAINING)

#     TODO EdS: Compare the two above evaluation stats


if __name__ == "__main__":
    main()