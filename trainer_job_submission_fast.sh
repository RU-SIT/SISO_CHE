#!/bin/bash
#SBATCH --job-name=MAML_fast
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=15:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodelist=g001

module load cuda12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
export PYTHONUNBUFFERED=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${REPO_ROOT}/MAML_trainer.py"
