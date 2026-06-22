#!/bin/bash
# Create a self-contained Python environment with development headers.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/arf/scratch/muhtan/ccc}
MINIFORGE_DIR="${PROJECT_DIR}/miniforge3"
ENV_DIR="${PROJECT_DIR}/.conda-env"
INSTALLER="${PROJECT_DIR}/Miniforge3-Linux-x86_64.sh"

cd "${PROJECT_DIR}"

if [[ ! -x "${MINIFORGE_DIR}/bin/conda" ]]; then
    echo "Downloading Miniforge..."
    curl -L \
        -o "${INSTALLER}" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "${INSTALLER}" -b -p "${MINIFORGE_DIR}"
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    conda create -y -p "${ENV_DIR}" python=3.10 pip
fi

conda activate "${ENV_DIR}"
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "${PROJECT_DIR}/requirements-truba.txt"
python -m pip freeze > "${PROJECT_DIR}/environment.freeze.txt"

python - <<'PY'
import sysconfig
from pathlib import Path

header = Path(sysconfig.get_paths()["include"]) / "Python.h"
print("Python:", __import__("sys").version)
print("Python.h:", header)
print("Python.h exists:", header.exists())
if not header.exists():
    raise SystemExit("Python.h is missing from the Conda environment")
PY

echo "Environment ready: ${ENV_DIR}"
