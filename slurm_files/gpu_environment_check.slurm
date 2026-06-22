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
ENV_DIR="${PROJECT_DIR}/.conda-env"
cd "${PROJECT_DIR}"
mkdir -p logs

if [[ -x "${ENV_DIR}/bin/python" ]]; then
    export PATH="${ENV_DIR}/bin:${PATH}"
fi

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

echo "C compiler:"
command -v gcc || true
gcc --version | head -n 1 || true

echo "Python development header:"
python3 - <<'PY' || true
import sysconfig
from pathlib import Path

include = Path(sysconfig.get_paths()["include"])
header = include / "Python.h"
print("include:", include)
print("Python.h exists:", header.exists())
PY

echo "CUDA driver library candidates:"
ldconfig -p 2>/dev/null | grep -E 'libcuda\.so' || true
find /usr /lib /lib64 -name 'libcuda.so*' -print 2>/dev/null | head -n 20 || true

echo "Loaded modules:"
module list 2>&1 || true

echo "Available CUDA/PyTorch modules:"
module avail cuda 2>&1 || true
module avail pytorch 2>&1 || true

if [[ -x "${ENV_DIR}/bin/python" ]]; then
    echo "Virtual environment package check:"
    "${ENV_DIR}/bin/python" - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("torch wheel CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("compute capability:", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch check failed:", repr(exc))

try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except Exception as exc:
    import traceback
    print("bitsandbytes check failed:")
    traceback.print_exc()
PY
fi
