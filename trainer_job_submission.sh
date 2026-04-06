#!/bin/bash
#SBATCH --job-name=MAML_5shot_UMI_interpolated
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL

# Load modules
module load cuda12.2

# Activate  env (pick the one you set up)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env


# Optional: make NCCL a bit quieter / resilient
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Run (repo root = directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${REPO_ROOT}/MAML_trainer.py"

