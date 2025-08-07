#!/bin/bash
#SBATCH --nodelist=BCM-DGX-H100-1
#SBATCH --job-name=minconvexp_experiments
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Define log file name based on Slurm job ID
LOG_FILE="logs/train_${SLURM_JOB_ID}.log"
mkdir -p logs  # ensure logs/ directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

# Activate correct environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate minconvrnns

# Echo the command for traceability
echo "Running command: $@"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

# Run the training script
srun "$@"

echo "Finished at: $(date)"