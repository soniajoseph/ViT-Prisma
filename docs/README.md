# Vision Transformer (ViT) Prisma Library
<div style="display: flex; align-items: center;">
  <img src="assets/images/house.jpg" alt="Logo Image 1" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/prism1.jpg" alt="Logo Image 2" width="200" style="margin-right: 10px;"/>
  <img src="assets/images/house2.jpg" alt="Logo Image 3" width="200"/>
</div>

ViT Prisma is an open-source mechanistic interpretability library for Vision Transformers (ViTs). This library was created by [Sonia Joseph](https://twitter.com/soniajoseph_).

*Contributors:* [Praneet Suresh](https://github.com/PraneetNeuro), [Yash Vadi](https://github.com/YashVadi) [and more coming soon]

We welcome new contributors. Check out our contributing guidelines [here](CONTRIBUTING.md).

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

**Part One:** Mechanistic interpretability tooling, including activation caching, path-patching, and attention head visualization. _In progress._

**Part Two:** Open source mini transformers (ViT "mice"). _In progress._

**Part Three:** Code to train your own vision transformers.

# Part One: Mechanistic Interpretability Tooling

*Coming soon.*

**Attention head visualization code**
* [Attention Head Demo Notebook](https://colab.research.google.com/drive/1xyNa2ghlALC7SejHNJYmAHc9wBYWUhZJ#scrollTo=MyKK6W1ltsKk)


<img src="https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/assets/images/corner-head.gif" width="300">

# Part Two: Open Source Mini-Transformers (ViT Mice 🐭)
ViT Mice are the mini-versions of the standard Vision Transformers.  Just as mice are used in scientific experiments for their small size, ViT Mice serve a similar purpose to illuminate their larger counterparts. By training these mice on both toy datasets and in-the-wild data, we aim to observe their behaviors in various environments.

**Categories of ViT Mice** 
1. **Toy Data Mice:** Trained on controlled, synthetic datasets to understand specific behaviors or to isolate certain aspects of the learning process.
2. **In-the-Wild Mice:** Trained on naturalistic, real-world data reminiscent of models in-production.
 
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


### To do: to train

1-4 layer, attention-only, and full-attention versions of each.
* ImageNet-1k Mice (reconstruction loss)
   * Attention-only
     * 1-layer, 2-layer, 3-layer, 4-layer, 5-layer
   * Full-attention
     * 1-layer, 2-layer, 3-layer, 4-layer, 5-layer
* dSprites
    * Smallest possible model that can recognize size.
    * Smallest possible model that can recognize position.

## Guidelines for training + uploading models

Upload your trained models to Huggingface. Follow the [Huggingface guidelines](https://huggingface.co/docs/hub/models-uploading) and also create a model card. Document as much of the training process as possible including links to loss and accuracy curves on weights and biases, dataset (and order of training data), hyperparameters, optimizer, learning rate schedule, hardware, and other details that may be relevant. 

Include frequent checkpoints throughout training, which will help other researchers understand training dynamics.

# Part Three: ViT Training Code 🚀

This repo includes training code to easily train ViTs from scratch or finetune existing models. See our [Usage Guide](https://github.com/soniajoseph/ViT-Prisma/blob/main/docs/UsageGuide.md) for more information.

## Configurations 

The Prisma config object houses certain hyperparameters of the models, along with attributes of the training procedure, which can come in handy to track, and design new experiments. 

There are several config objects available in the framework. 

Below are the configs available as part of the framework:
 - InductionConfig
 - CircleConfig
 - MNISTConfig
 - DSpritesConfig

```python
from vit_prisma.configs import <CONFIG_OF_CHOICE>
```
By inspecting one of the configs available in the framework, you can create your own one.

## Base ViT

The base ViT part of the Prisma Project allows instantiating models in a flexible way, where the architecture is defined by the Prisma config object discussed in the previous section.

### Instantiating the Base ViT 

```python
from vit_prisma.models import base_vit
from vit_prisma.configs import InductionConfig

# The InductionConfig is available in the framework itself, which is setup for the induction dataset
config = InductionConfig.GlobalConfig()

model = base_vit.BaseViT(config)
```

### Using pretrained models

```python
from vit_prisma.models.pretrained_model import PretrainedModel
from vit_prisma.configs import InductionConfig

config = InductionConfig.GlobalConfig()

hf_model = PretrainedModel('google/vit-base-patch16-224-in21k', config)

timm_model = PretrainedModel('vit_base_patch32_224', config, is_timm=True)
```

### Datasets

The framework also has a few datasets that can be synthetically generated, and cached. 

Below are the datasets available as part of the framework: 
 - Circle
 - [dsprites](https://github.com/google-deepmind/dsprites-dataset)
 - Induction

You will be able to use any other dataset with the framework, for ease of use, you can instantiate it as a Pytorch dataset object.

 ```python
 from vit_prisma.dataloaders.induction import InductionDataset

 train = InductionDataset('train')

 # If you don't provide a test split to the trainer, the trainer will perform the train/test split for you.
 ```

 ### Training and tracking experiments

 The trainer has built in support for wandb experiment tracking. Make sure to set up wandb on your localhost, and configure the tracking attributes in the Prisma config object.

 ```python
 from vit_prisma.models.base_vit import BaseViT
 from vit_prisma.configs import InductionConfig
 from vit_prisma.training import trainer
 from vit_prisma.dataloaders.induction import InductionDataset

 train_dataset = InductionDataset('train')

 config = InductionConfig.GlobalConfig()

 model_function = BaseViT

 trainer.train(model_function, config, train_dataset)
 ```

# Errors 💰
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
