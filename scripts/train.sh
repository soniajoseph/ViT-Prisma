#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128Gb
#SBATCH --time=15:00:00
#SBATCH --output=sbatch_out/grokking_finetune_vit.%A.out
#SBATCH --error=sbatch_err/grokking_finetune_vit.%A.err
#SBATCH --job-name=grokking_finetune_vit

module load libffi
source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate
python /home/mila/s/sonia.joseph/ViT-Planetarium/tests/test_trainer_circle_timm.py