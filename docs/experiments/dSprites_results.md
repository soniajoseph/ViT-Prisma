## dSprites Results
The dSprites dataset consists of 2D shapes generated procedurally based on six independent latent factors: color, shape, scale, rotation, and the x and y positions of a sprite. Our specific focus was on the task of shape classification within the dSprites dataset, which encompasses a total of three distinct shapes. We train different mice vits for this task and presented the result below.

## Training Details
We conducted experiments with Mice ViTs using the following configurations:

- Number of Attention Layers: Varied from 1 to 4
- Number of Attention Heads: 8
- Training Batch Size: 512
- Image Patch Size: 8
- Learning Rate: 1e-4
- Scheduler Step Size: 200
- Scheduler Gamma: 0.8

For each attention layer setting, we analyzed two variants: an attention-only model and a model combining attention with the MLP module. We didn't apply any dropout or layer normalization for this experiment to make it simple to understand.

## Table of Results
Below table descirbe the accuracy of Mice ViTs with different configuration.
| **Size** | **NumLayers** | **Attention+MLP** | **AttentionOnly** | **Model Link**                              |
|:--------:|:-------------:|:-----------------:|:-----------------:|--------------------------------------------|
| **tiny** | **1**         | 0.535             | 0.459             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-tiny-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-tiny-Attention-and-MLP) |
| **base** | **2**         | 0.996             | 0.685             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-base-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-base-Attention-and-MLP) |
| **small**| **3**         | 1.000             | 0.774             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-small-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-small-Attention-and-MLP) |
| **medium**|**4**         | 1.000             | 0.991             | [AttentionOnly](https://huggingface.co/IamYash/dSprites-medium-AttentionOnly), [Attention+MLP](https://huggingface.co/IamYash/dSprites-medium-Attention-and-MLP) |
