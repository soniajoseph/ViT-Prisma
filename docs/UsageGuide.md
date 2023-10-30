# Usage Guide

The Prisma Project is a framework that facilitates training Vision Transformers (ViTs) in various configurations, with synthetically generated datasets that would be best suited for mechanstically interpreting them. It also allows loading pretrained models from huggingface and timm which can further be finetuned with datasets using the trainer built into the framework.

## Installing Repo

To install as an editable repo:

```
git clone https://github.com/soniajoseph/ViT-Prisma
cd ViT-Prisma
pip install -e .
```

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
 from vit_prisma.models import base_vit
 from vit_prisma.configs import InductionConfig
 from vit_prisma.training import trainer
 from vit_prisma.dataloaders.induction import InductionDataset

 train_dataset = InductionDataset('train')

 config = InductionConfig.GlobalConfig()

 model = base_vit.BaseViT(config)

 trainer.train(model, config, train_dataset)
 ```
