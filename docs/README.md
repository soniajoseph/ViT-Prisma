# Vision Transformer (ViT) Prisma Library
<div style="display: flex; align-items: center;">
  <img src="assets/images/house.jpg" alt="Logo Image 1" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/prism1.jpg" alt="Logo Image 2" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/house2.jpg" alt="Logo Image 3" width="200"/>
</div>

Welcome to ViT Prisma, the mother of vision and multimodal mechanistic interpretability research.

ViT Prisma is an open-source mechanistic interpretability library for Vision Transformers (ViTs). This library was created by Sonia Joseph.

*Contributors:* [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi) [and more coming soon]

We welcome new contributers. Check out our contributing guidelines [here](CONTRIBUTING.md).

## Installing Repo

To install as an editable repo:
```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```
## How do I use this repo?
Check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).


# What's in the repo?

**Part One:** Mechanistic interpretability tooling, including activation caching, path-patching, and attention head visualization. _Coming soon_

**Part Two:** Open source mini trainsformers (ViT "mice"). _In progress._

**Part Three:** Code to train your own vision transformers.

## Part One: Mechanistic Interpretability Tooling

_Coming soon._

## Part Two: Open Source Mini-Transformers (ViT Mice 🐭)
ViT Mice are the mini-versions of the standard Vision Transformers.  Just as mice are often used in scientific experiments for the nimble size and ease of iteration, ViT Mice serve a similar purpose to provide insights about their larger counterparts. By training these mice on both toy datasets and in-the-wild data, we aim to observe their behaviors in various environments.

**Categories of ViT Mice** 
1. **Toy Data Mice:** Trained on controlled, synthetic datasets to understand specific behaviors or to isolate certain aspects of the learning process.
2. **In-the-Wild Mice:** Trained on naturalistic, real-world data reminiscent of models in-production.

**List of Mice** 

_To train_ (1-4 layer, attention-only, and full-attention versions of each)
* ImageNet-1k Mice
   * Attention-only
     * 1-layer, 2-layer, 3-layer, 4-layer, 5-layer
   * Full-attention
     * 1-layer, 2-layer, 3-layer, 4-layer, 5-layer
* Induction Mice (full and attention-only)
     * Smallest possible full model that can do task on induction dataset
* Modular Arithmetic Mice (full and attention-only)
     * Smallest possible full model that can do task on induction dataset
* [dSprites](https://github.com/google-deepmind/dsprites-dataset) Mice
     * Smallest possible model that can recognize shapes. Checkpoints [here](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/experiments/dSprites_results.md).
     * Smallest possible model that can recognize size
     * Smallest possible model that can recognize position
     * Smallest possible model that can do all of the above with minimal fine-tuning 

**Guidelines for training + uploading models**

Upload your trained models to Huggingface. Follow the [Huggingface guidelines](https://huggingface.co/docs/hub/models-uploading) and also create a model card. Document as much of the training process as possible including loss and accuracy curves, dataset (and order of training data), hyperparameters, optimizer, learning rate schedule, hardware, and other details that may be relevant. Links to the wandb training info are also welcome.

Include frequent checkpoints throughout training, which will help other researchers understand training dynamics.


## Part Three: ViT Training Code 🚀

This repo includes training code to easily train ViTs from scratch or finetune existing models. See our [Usage Guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md) for more information.

## Errors 💰
If you point out a conceptual error in this code (e.g. incorrect implementation of a transformer, not a minor function import), I will send you $5-20 per bug depending on the bug's severity.

# Citation

Please cite this repository when used in papers or research projects.

```
@misc{joseph2023vit,
  author = {Sonia Joseph},
  title = {ViT Prisma: A Mechanistic Interpretability Library for Vision Transformers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/soniajoseph/vit-prisma}}
}
```
