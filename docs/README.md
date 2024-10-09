# Vision Transformer (ViT) Prisma Library
<div style="display: flex; align-items: center;">
  <img src="assets/images/house.jpg" alt="Logo Image 1" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/prism1.jpg" alt="Logo Image 2" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/house2.jpg" alt="Logo Image 3" width="200"/>
</div>

For a full introduction, including Open Problems in vision mechanistic interpretability, see the original Less Wrong post [here](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).

ViT Prisma is an open-source mechanistic interpretability library for vision and multimodal models. Currently, the library supports ViTs and CLIP. This library was created by [Sonia Joseph](https://twitter.com/soniajoseph_). ViT Prisma is largely based on [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda.

*Contributors:* [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi), [Rob Graham](https://github.com/themachinefan) [and more coming soon]

We welcome new contributors. Check out our contributing guidelines [here](CONTRIBUTING.md) and our [open Issues](https://github.com/soniajoseph/ViT-Prisma/issues).

## Installing Repo

For the latest version, install the repo from the source. While this version will include the latest developments, they may not be fully tested.

For the tested and stable release, install Prisma as a package.

**Install as a package**
Installing with pip:
```
pip install vit_prisma
```

**Install from source**
To install as an editable repo from source:
```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

## How do I use this repo?
Check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).

Check out our tutorial notebooks for using the repo. You can also check out this [corresponding talk](https://youtu.be/gQbh-RZtsq4?t=0) on some of these techniques.

1. [Main ViT Demo](https://colab.research.google.com/drive/1TL_BY1huQ4-OTORKbiIg7XfTyUbmyToQ) - Overview of main mechanistic interpretability technique on a ViT, including direct logit attribution, attention head visualization, and activation patching. The activation patching switches the net's prediction from tabby cat to Border collie with a minimum ablation.
2. [Emoji Logit Lens](https://colab.research.google.com/drive/1yAHrEoIgkaVqdWC4GY-GQ46ZCnorkIVo) - Deeper dive into layer- and patch-level predictions with interactive plots.
3. [Interactive Attention Head Tour](https://colab.research.google.com/drive/1P252fCvTHNL_yhqJDeDVOXKCzIgIuAz2) - Deeper dive into the various types of attention heads a ViT contains with interactive JavaScript.

## Features

For a full demo of Prisma's features, including the visualizations below with interactivity, check out the demo notebooks above.

### Attention head visualization
<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">

<div style="display: flex; align-items: center;">
  <img src="assets/images/attention head 1.png" alt="Logo Image 1" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 2.png" alt="Logo Image 2" width="250" style="margin-right: 10px;"/>
  <img src="assets/images/attention head 3.png" alt="Logo Image 3" width="250"/>
</div>

### Activation patching
<img src="assets/images/patched head.png" width="400">

### Direct logit attribution
<img src="assets/images/direct logit attribution.png" width="600">

### Emoji logit lens
<div style="display: flex; align-items: center;">
<img src="assets/images/dogit lens 2.png" width="400">
<img src="assets/images/cat toilet segmentation.png" width="400">
<img src="assets/images/child lion segmentation.png" width="400">
<img src="assets/images/cheetah segmentation.png" width="400">

</div>


## Supported Models
* [timm ViTs](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
* [CLIP](https://huggingface.co/docs/transformers/main/en/model_doc/clip)
* Our custom toy models (see below)

## Training Code

Prisma contains training code to train your own custom ViTs. Training small ViTs can be very useful when isolating specific behaviors in the model.

For training your own models, check out [our guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md).

## Custom Models & Checkpoints

### ImageNet-1k classification checkpoints (patch size 32)

This model was trained by Praneet Suresh. All models include training checkpoints, in case you want to analyze training dynamics.

This larger patch size ViT has inspectable attention heads; else the patch size 16 attention heads are too large to easily render in JavaScript.


| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **3**         | 0.22 \| 0.42 |            N/A            | [Attention+MLP](https://huggingface.co/PraneetNeuro/ImageNet-Small-Attention-and-MLP-Patch32) |

### ImageNet-1k classification checkpoints (patch size 16)

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
