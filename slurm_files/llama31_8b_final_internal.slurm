#!/bin/bash
#SBATCH --job-name=cfpb-llama31-int
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
ENV_DIR="${PROJECT_DIR}/.conda-env"
MODEL_DIR="${LLAMA_MODEL_DIR:-${PROJECT_DIR}/hf_models/Llama-3.1-8B-Instruct}"

VARIANTS=(
  "base_zero_shot"
  "fine_tuned_seed42"
  "fine_tuned_seed43"
  "fine_tuned_seed44"
)
SEEDS=("42" "42" "43" "44")

find_adapter() {
  local seed="$1"
  local config_path
  config_path="$(find "${PROJECT_DIR}/outputs" -maxdepth 3 -type f -path "${PROJECT_DIR}/outputs/llama31_8b_fixed_protocol_seed${seed}_*/best_adapter/adapter_config.json" | sort | tail -n 1)"
  if [[ -z "${config_path}" ]]; then
    echo "Missing Llama adapter for seed ${seed}" >&2
    exit 1
  fi
  dirname "${config_path}"
}

INDEX="${SLURM_ARRAY_TASK_ID}"
VARIANT="${VARIANTS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
RUN_DIR="${PROJECT_DIR}/outputs/llama31_8b_final_eval_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${INDEX}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs hf_cache "${RUN_DIR}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing model: ${MODEL_DIR}" >&2
  echo "Set LLAMA_MODEL_DIR if the model is stored in a different directory." >&2
  exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

COMMON_ARGS=(
  --model "${MODEL_DIR}"
  --max-narrative-chars 500
  --max-input-length 512
  --seed "${SEED}"
  --allow-final-eval
)

if [[ "${VARIANT}" == "base_zero_shot" ]]; then
  MODEL_ARGS=(--base-only)
else
  ADAPTER="$(find_adapter "${SEED}")"
  MODEL_ARGS=(--adapter "${ADAPTER}")
  echo "Adapter: ${ADAPTER}"
fi

echo "Llama 3.1 8B internal final evaluation"
echo "Variant: ${VARIANT}"
echo "Seed: ${SEED}"
echo "Run dir: ${RUN_DIR}"
nvidia-smi

python scripts/evaluate_adapter.py \
  "${COMMON_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --data-file "${PROJECT_DIR}/dataset/splits/internal_test.csv" \
  --output-dir "${RUN_DIR}/internal_test"

echo "Completed Llama internal final evaluation: ${VARIANT}"
