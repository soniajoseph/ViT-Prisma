#!/bin/bash
###SBATCH --array=0-12
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128Gb
#SBATCH --time=3:00:00
###SBATCH --ntasks=16
#SBATCH --output=sbatch_out/run_sae.%A.%a.out
#SBATCH --error=sbatch_err/run_sae.%A.%a.err
#SBATCH --job-name=run_sae

module load anaconda/3
module load cuda/11.7
module load libffi

source /home/mila/s/sonia.joseph/env/bin/activate
python main.py --checkpoint_path /network/scratch/s/sonia.joseph/saes/exp_32 --layers 9 --num_epochs 3 --expansion_factor 128 --imagenet_path /network/scratch/s/sonia.joseph/datasets/kaggle_datasets --hook_point blocks.{layer}.hook_mlp_out --context_size 50 --d_in 512 --model_name wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M --l1_coefficient 0.00008



python main.py --layers 9 --num_epochs 1 --expansion_factor 16 --checkpoint_path /network/scratch/s/sonia.joseph/saes/exp_16 --imagenet_path /network/scratch/s/sonia.joseph/datasets/kaggle_datasets --model_name wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M --context_size 50 --d_in 512 --run_name mlp_16_1e --hook_point blocks.{layer}.hook_mlp_out
