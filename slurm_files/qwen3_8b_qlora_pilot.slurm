#!/bin/bash
#SBATCH --job-name=cfpb-qwen3-8b
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%j.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
ENV_DIR="${PROJECT_DIR}/.conda-env"
MODEL_DIR="${PROJECT_DIR}/hf_models/Qwen3-8B"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3_8b_qlora_pilot_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs hf_cache
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  echo "Run: bash scripts/setup_truba_env.sh" >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi
python -V

python scripts/train_qlora.py \
  --model "${MODEL_DIR}" \
  --train-file "${PROJECT_DIR}/dataset/splits/train.csv" \
  --validation-file "${PROJECT_DIR}/dataset/splits/validation.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-samples 10000 \
  --max-eval-samples 2000 \
  --train-sampling natural \
  --max-narrative-chars 500 \
  --max-length 512 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --eval-steps 100 \
  --save-steps 100 \
  --seed 42

python scripts/evaluate_adapter.py \
  --model "${MODEL_DIR}" \
  --adapter "${OUTPUT_DIR}/best_adapter" \
  --data-file "${PROJECT_DIR}/dataset/splits/validation.csv" \
  --output-dir "${OUTPUT_DIR}/validation_inference" \
  --max-narrative-chars 500 \
  --max-samples 2000
