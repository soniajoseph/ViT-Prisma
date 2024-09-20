#!/bin/bash
#SBATCH --gres=gpu:a100l
#SBATCH --cpus-per-task=24
##SBATCH --nodelist=cn-g[001-029],cn-i001,cn-d[001-002],cn-d[003-004],cn-e[002-003]
#SBATCH --mem=128Gb
#SBATCH --time=20:00:00
#SBATCH --output=sbatch_out/kadinsky_adapter.%A.out
#SBATCH --error=sbatch_err/kadinsky_adapter.%A.err
#SBATCH --job-name=kadinsky_adapter

module load libffi
source /home/mila/s/sonia.joseph/env/bin/activate

python train_tinyclip_kadinsky_adapter.py --pretrained_checkpoint '/network/scratch/s/sonia.joseph/diffusion/tinyclip_adapter/sq3oe/adapter_checkpoint_200.pth'