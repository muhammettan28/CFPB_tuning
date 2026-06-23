#!/bin/bash
#SBATCH --job-name=cfpb-qwen3-search
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
ENV_DIR="${PROJECT_DIR}/.conda-env"
MODEL_DIR="${PROJECT_DIR}/hf_models/Qwen3-8B"

EXPERIMENT_NAMES=(
  "E1_bal_lr2e4_r16"
  "E2_bal_lr1e4_r16"
  "E3_bal_lr2e4_r8"
)
LEARNING_RATES=("2e-4" "1e-4" "2e-4")
LORA_RANKS=("16" "16" "8")
LORA_ALPHAS=("32" "32" "16")

INDEX="${SLURM_ARRAY_TASK_ID}"
EXPERIMENT_NAME="${EXPERIMENT_NAMES[$INDEX]}"
LEARNING_RATE="${LEARNING_RATES[$INDEX]}"
LORA_R="${LORA_RANKS[$INDEX]}"
LORA_ALPHA="${LORA_ALPHAS[$INDEX]}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3_8b_search_${EXPERIMENT_NAME}_${SLURM_ARRAY_JOB_ID}_${INDEX}"

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

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Learning rate: ${LEARNING_RATE}"
echo "LoRA r/alpha: ${LORA_R}/${LORA_ALPHA}"
nvidia-smi

python scripts/train_qlora.py \
  --model "${MODEL_DIR}" \
  --train-file "${PROJECT_DIR}/dataset/splits/train.csv" \
  --validation-file "${PROJECT_DIR}/dataset/splits/validation.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-samples 10000 \
  --max-eval-samples 2000 \
  --train-sampling balanced \
  --max-narrative-chars 500 \
  --max-length 512 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --eval-steps 100 \
  --save-steps 100 \
  --seed 42

python scripts/evaluate_adapter.py \
  --model "${MODEL_DIR}" \
  --adapter "${OUTPUT_DIR}/best_adapter" \
  --data-file "${PROJECT_DIR}/dataset/splits/validation.csv" \
  --output-dir "${OUTPUT_DIR}/validation_inference" \
  --max-narrative-chars 500 \
  --max-samples 2000 \
  --seed 42

echo "Completed: ${EXPERIMENT_NAME}"
