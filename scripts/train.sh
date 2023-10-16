#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --nodelist="cn-g[001-029],cn-i001,cn-d[001-002],cn-d[003-004],cn-e[002-003]"
#SBATCH --mem=128Gb
#SBATCH --time=15:00:00
#SBATCH --output=sbatch_out/transformer_hyperparam.%A.out
#SBATCH --error=sbatch_err/transformer_hyperparam.%A.err
#SBATCH --job-name=transformer_hyperparam
#SBATCH --array=0-3

module load libffi
source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate
wandb agent --count 1 'perceptual-alignment/ViT 1-Variable Sweeps/26kfrd6h'