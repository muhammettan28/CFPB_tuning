#!/bin/bash
#SBATCH --job-name=cfpb-gpu-check
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%j.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
cd "${PROJECT_DIR}"
mkdir -p logs

echo "Hostname: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Python: $(command -v python3 || true)"
python3 --version || true

echo "nvidia-smi path:"
command -v nvidia-smi || true
nvidia-smi || true

echo "CUDA compiler:"
command -v nvcc || true
nvcc --version || true

echo "Loaded modules:"
module list 2>&1 || true

echo "Available CUDA/PyTorch modules:"
module avail cuda 2>&1 || true
module avail pytorch 2>&1 || true
