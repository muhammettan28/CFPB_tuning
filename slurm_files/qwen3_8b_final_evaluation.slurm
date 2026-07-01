#!/bin/bash
#SBATCH --job-name=cfpb-qwen3-final
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
MODEL_DIR="${PROJECT_DIR}/hf_models/Qwen3-8B"

VARIANTS=(
  "base_zero_shot"
  "fine_tuned_seed42"
  "fine_tuned_seed43"
  "fine_tuned_seed44"
)
ADAPTERS=(
  ""
  "${PROJECT_DIR}/outputs/qwen3_8b_second_stage_S1_nat_lr2e4_r16_2ep_5981133_0/best_adapter"
  "${PROJECT_DIR}/outputs/qwen3_8b_seed_stability_nat_lr2e4_r16_2ep_seed43_5986129_0/best_adapter"
  "${PROJECT_DIR}/outputs/qwen3_8b_seed_stability_nat_lr2e4_r16_2ep_seed44_5986129_1/best_adapter"
)
SEEDS=("42" "42" "43" "44")

INDEX="${SLURM_ARRAY_TASK_ID}"
VARIANT="${VARIANTS[$INDEX]}"
ADAPTER="${ADAPTERS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
RUN_DIR="${PROJECT_DIR}/outputs/qwen3_8b_final_eval_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${INDEX}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs hf_cache "${RUN_DIR}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Missing model: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ "${VARIANT}" != "base_zero_shot" && ! -f "${ADAPTER}/adapter_config.json" ]]; then
  echo "Missing adapter: ${ADAPTER}" >&2
  exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Variant: ${VARIANT}"
echo "Seed: ${SEED}"
echo "Run dir: ${RUN_DIR}"
nvidia-smi

COMMON_ARGS=(
  --model "${MODEL_DIR}"
  --max-narrative-chars 500
  --max-input-length 512
  --seed "${SEED}"
  --allow-final-eval
)

MODEL_ARGS=()
if [[ "${VARIANT}" == "base_zero_shot" ]]; then
  MODEL_ARGS=(--base-only)
else
  MODEL_ARGS=(--adapter "${ADAPTER}")
fi

echo "Evaluating internal_test.csv"
python scripts/evaluate_adapter.py \
  "${COMMON_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  --data-file "${PROJECT_DIR}/dataset/splits/internal_test.csv" \
  --output-dir "${RUN_DIR}/internal_test"

echo "Evaluating legacy subsets"
for DATA_FILE in "${PROJECT_DIR}"/dataset/legacy_subsets/*.csv; do
  DATASET_NAME="$(basename "${DATA_FILE}" .csv)"
  echo "Evaluating ${DATASET_NAME}"
  python scripts/evaluate_adapter.py \
    "${COMMON_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --data-file "${DATA_FILE}" \
    --output-dir "${RUN_DIR}/legacy_subsets/${DATASET_NAME}"
done

echo "Completed final evaluation: ${VARIANT}"
