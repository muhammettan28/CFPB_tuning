#!/bin/bash
#SBATCH --job-name=cfpb-qwen3-smoke
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%j.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
ENV_DIR="${PROJECT_DIR}/.conda-env"
MODEL_DIR="${PROJECT_DIR}/hf_models/Qwen3-8B"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3_8b_qlora_smoke_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs hf_cache

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing model: ${MODEL_DIR}" >&2
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
  --max-train-samples 64 \
  --max-eval-samples 32 \
  --max-narrative-chars 500 \
  --max-length 512 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --logging-steps 1 \
  --eval-steps 2 \
  --save-steps 2 \
  --seed 42

python scripts/evaluate_adapter.py \
  --model "${MODEL_DIR}" \
  --adapter "${OUTPUT_DIR}/best_adapter" \
  --data-file "${PROJECT_DIR}/dataset/splits/validation.csv" \
  --output-dir "${OUTPUT_DIR}/validation_inference" \
  --max-narrative-chars 500 \
  --max-samples 32

echo "Smoke test completed: ${OUTPUT_DIR}"
