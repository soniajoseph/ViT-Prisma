#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128Gb
#SBATCH --time=6:00:00
###SBATCH --ntasks=16
#SBATCH --output=../output/get_sampled_images.%A.%a.out
#SBATCH --error=../output/get_sampled_images.%A.%a.err
#SBATCH --job-name=get_sampled_images

# module load anaconda/3
# module load cuda/11.7
# module load libffi

source /home/mila/s/sonia.joseph/env/bin/activate


python main.py --expansion_factor 16 --activation_fn 'relu' --run_name 'vanilla_16' --hook_layer 9