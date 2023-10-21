# Vision Transformer (ViT) Prisma Library
<div style="display: flex; align-items: center;">
  <img src="assets/images/house.jpg" alt="Logo Image 1" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/prism1.jpg" alt="Logo Image 2" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/house2.jpg" alt="Logo Image 3" width="200"/>
</div>


ViT Prisma is an open-source mechanistic interpretability library for Vision Transformers (ViTs). This library was created by Sonia Joseph.

*Contributors:* [coming soon]

We welcome new contributers. Check out our contributing guidelines [here](CONTRIBUTING.md).

## ViT Mice 🐭
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
* dSprites Mice
     * Smallest possible model that can recognize shapes
     * Smallest possible model that can recognize size
     * Smallest possible model that can recognize position
     * Smallest possible model that can do all of the above with minimal fine-tuning 

**Guidelines for training + uploading models**

Upload your trained models to Huggingface. Follow the [Huggingface guidelines](https://huggingface.co/docs/hub/models-uploading) and also create a model card. Document as much of the training process as possible including loss and accuracy curves, dataset (and order of training data), hyperparameters, optimizer, learning rate schedule, hardware, and other details that may be relevant. Links to the wandb training info are also welcome.

Include frequent checkpoints throughout training, which will help other researchers understand training dynamics.

## ViT Prisms 🌈
Our "Prisms" are the interpretability tools for ViTs. By viewing a ViT through different prisms, we can uncover different aspects of its operation, from attention patterns to feature visualizations and more.


## ViT Training Code 🚀

This repo includes training code to easily train ViTs from scratch or finetune existing models.

_To do_
* Add hook functions to all transformers to easily retrieve activations

## Installing Repo

To install as an editable repo:

```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```
## Errors 💰
If you point out a conceptual error in those code (e.g. incorrect implementation of a transformer, not a minor function import), I will send you $5-25 per bug depending on the bug's severity.

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
