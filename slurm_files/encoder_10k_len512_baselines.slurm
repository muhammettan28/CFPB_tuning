#!/bin/bash
#SBATCH --job-name=cfpb-enc-10k
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.out
#SBATCH --error=/arf/scratch/muhtan/ccc/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/arf/scratch/muhtan/ccc
ENV_DIR="${PROJECT_DIR}/.conda-env"
OUT_DIR="${PROJECT_DIR}/CLASSIC_TEST/outputs/encoder_baselines_10k_len512"

MODELS=(
  "bert-base-uncased"
  "distilbert-base-uncased"
  "roberta-base"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}" hf_cache

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Encoder baseline model: ${MODEL}"
echo "Output dir: ${OUT_DIR}"
nvidia-smi

python CLASSIC_TEST/run_encoder_baseline.py \
  --train-file "${PROJECT_DIR}/dataset/splits/train.csv" \
  --test-file "${PROJECT_DIR}/dataset/splits/internal_test.csv" \
  --legacy-dir "${PROJECT_DIR}/dataset/legacy_subsets" \
  --out-dir "${OUT_DIR}" \
  --models "${MODEL}" \
  --max-train 10000 \
  --max-chars 500 \
  --max-length 512 \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-5 \
  --seed 42

python CLASSIC_TEST/summarize_encoder_baselines.py \
  --results-dir "${OUT_DIR}" \
  --output "${OUT_DIR}/encoder_summary.csv"

echo "Completed encoder baseline: ${MODEL}"
