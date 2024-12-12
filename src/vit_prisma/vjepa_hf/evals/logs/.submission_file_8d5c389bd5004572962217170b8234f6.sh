#!/bin/bash

# Parameters
#SBATCH --SLURM-TRES-PER-TASK=cpu:20
#SBATCH --cpus-per-task=20
#SBATCH --error=/private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=5
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_%t_log.out --error /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs/%j_%t_log.err /usr/bin/python -u -m submitit.core._submit /private/home/soniajoseph/ViT-Prisma/src/vit_prisma/vjepa_hf/evals/logs
