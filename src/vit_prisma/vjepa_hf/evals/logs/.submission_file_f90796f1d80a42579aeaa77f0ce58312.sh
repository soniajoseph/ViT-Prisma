#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_0_log.err
#SBATCH --gres=gpu:volta:8
#SBATCH --job-name=submitit
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_0_log.out
#SBATCH --partition=learnfair
#SBATCH --signal=USR2@90
#SBATCH --time=01:00:00
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_%t_log.out --error /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_%t_log.err /usr/bin/python -u -m submitit.core._submit /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs
