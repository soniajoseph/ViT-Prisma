# Vision Transformer (ViT) Prisma Library
<div style="display: flex; align-items: center;">
  <img src="assets/images/house.jpg" alt="Logo Image 1" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/prism1.jpg" alt="Logo Image 2" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/house2.jpg" alt="Logo Image 3" width="200"/>
</div>

ViT Prisma is an open-source mechanistic interpretability library for vision and multimodal models. Currently, the library supports ViTs and CLIP. This library was created by [Sonia Joseph](https://twitter.com/soniajoseph_). ViT Prisma is largely based on TransformerLens by Neel Nanda.

*Contributors:* [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi), [Rob Graham](https://github.com/themachinefan) [and more coming soon]

We welcome new contributors. Check out our contributing guidelines [here](CONTRIBUTING.md) and our [open Issues](https://github.com/soniajoseph/ViT-Prisma/issues).

## Installing Repo

Installing with pip:
```
pip install vit_prisma
```

To install as an editable repo from source:
```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

## How do I use this repo?
Check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).

Check out our tutorial notebooks for using the repo.

1. [Main ViT Demo](https://colab.research.google.com/drive/1TL_BY1huQ4-OTORKbiIg7XfTyUbmyToQ) - Overview of main mechanistic interpretability technique on a ViT, including direct logit attribution, attention head visualization, and activation patching. The activation patching switches the net's prediction from tabby cat to Border collie with a minimum ablation.
2. [Emoji Logit Lens](https://colab.research.google.com/drive/1yAHrEoIgkaVqdWC4GY-GQ46ZCnorkIVo) - Deeper dive into layer- and patch-level predictions with interactive plots.
3. [Interactive Attention Head Tour](https://colab.research.google.com/drive/1P252fCvTHNL_yhqJDeDVOXKCzIgIuAz2) - Deeper dive into the various types of attention heads a ViT contains with interactive JavaScript.


## Available Models



## Training Code

Prisma contains training code to train your own custom ViTs. Training small ViTs can be very useful when isolating specific behaviors in the model.

For training your own models, check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).



<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">


### ImageNet-1k classification training checkpoints

The detailed training logs and metrics can be found [here](https://wandb.ai/vit-prisma/Imagenet/overview?workspace=user-yash-vadi). These models were trained by Yash Vadi.

**Table of Results**

Accuracy `[ <Acc> | <Top5 Acc> ]`

| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.16 \| 0.33             |  0.11 \| 0.25             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.23 \| 0.44             |  0.16 \| 0.34             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-base-Attention-and-MLP) |
| **small**| **3**         | 0.28 \| 0.51            | 0.17 \| 0.35             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-small-Attention-and-MLP) |
| **medium**|**4**         | 0.33 \| 0.56             | 0.17 \| 0.36             | [AttentionOnly](https://huggingface.co/IamYash/ImageNet-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/ImageNet-medium-Attention-and-MLP) |

### dSprites Shape Classification training checkpoints

Original dataset is [here](https://github.com/google-deepmind/dsprites-dataset). 

Full results and training setup are [here](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/experiments/dSprites_results.md). These models were trained by Yash Vadi.

**Table of Results**
| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.535             | 0.459             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.996             | 0.685             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-base-Attention-and-MLP) |
| **small**| **3**         | 1.000             | 0.774             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-small-Attention-and-MLP) |
| **medium**|**4**         | 1.000             | 0.991             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-medium-Attention-and-MLP) |



## Guidelines for training + uploading models

Upload your trained models to Huggingface. Follow the [Huggingface guidelines](https://huggingface.co/docs/hub/models-uploading) and also create a model card. Document as much of the training process as possible including links to loss and accuracy curves on weights and biases, dataset (and order of training data), hyperparameters, optimizer, learning rate schedule, hardware, and other details that may be relevant. 

Include frequent checkpoints throughout training, which will help other researchers understand training dynamics.


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
