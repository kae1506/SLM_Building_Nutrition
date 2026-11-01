#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${PROJECT_ROOT}/.venv"
source "${PROJECT_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${PROJECT_ROOT}"

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY

echo "Environment ready. Start with: ./run_short_pilot.sh"
