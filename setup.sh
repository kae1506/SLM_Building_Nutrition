#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-setup}"

python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
}

environment_ready() {
  [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]] || return 1
  python_supported "${PROJECT_ROOT}/.venv/bin/python" || return 1
  "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import olm, recipe_dapt, torch, transformers' >/dev/null 2>&1
  "${PROJECT_ROOT}/.venv/bin/python" -m recipe_dapt.smoke_test --quick \
    >/dev/null 2>&1
}

if [[ "${MODE}" == "--check" ]]; then
  environment_ready
  exit $?
fi
if [[ "${MODE}" != "setup" ]]; then
  echo "Usage: ./setup.sh [--check]" >&2
  exit 2
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]] && \
   python_supported "${PROJECT_ROOT}/.venv/bin/python"; then
  echo "Reusing compatible environment: ${PROJECT_ROOT}/.venv"
else
  if [[ -e "${PROJECT_ROOT}/.venv" ]]; then
    BACKUP="${PROJECT_ROOT}/.venv-incompatible-$(date +%Y%m%d%H%M%S)"
    mv "${PROJECT_ROOT}/.venv" "${BACKUP}"
    echo "Moved the incompatible environment to: ${BACKUP}"
  fi

  PYTHON_CANDIDATE=""
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! python_supported "${PYTHON_BIN}"; then
      echo "PYTHON_BIN must point to Python 3.10, 3.11, or 3.12." >&2
      exit 1
    fi
    PYTHON_CANDIDATE="${PYTHON_BIN}"
  else
    for candidate in python3.12 python3.11 python3.10; do
      if command -v "${candidate}" >/dev/null 2>&1 && python_supported "${candidate}"; then
        PYTHON_CANDIDATE="$(command -v "${candidate}")"
        break
      fi
    done
  fi

  if [[ -n "${PYTHON_CANDIDATE}" ]]; then
    "${PYTHON_CANDIDATE}" -m venv "${PROJECT_ROOT}/.venv"
  elif command -v conda >/dev/null 2>&1; then
    echo "Creating a project-local Python 3.12 environment with conda."
    conda create -y -p "${PROJECT_ROOT}/.venv" python=3.12 pip
  else
    BOOTSTRAP_PYTHON="$(command -v python3 || true)"
    if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
      echo "Python 3 and conda are unavailable; install Python 3.12 first." >&2
      exit 1
    fi
    echo "Installing uv for a user-local Python 3.12 runtime."
    "${BOOTSTRAP_PYTHON}" -m pip install --user "uv>=0.9,<1"
    UV_BIN="$("${BOOTSTRAP_PYTHON}" -c 'import site; print(site.getuserbase())')/bin/uv"
    if [[ ! -x "${UV_BIN}" ]]; then
      UV_BIN="$(command -v uv || true)"
    fi
    if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
      echo "uv installation did not provide an executable." >&2
      exit 1
    fi
    "${UV_BIN}" venv --python 3.12 "${PROJECT_ROOT}/.venv"
  fi
fi

VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip wheel setuptools
"${VENV_PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
"${VENV_PYTHON}" -m pip install -e "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m recipe_dapt.smoke_test

"${VENV_PYTHON}" - <<'PY'
import sys
import torch
import transformers
import olm

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"olm={olm.__file__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY

echo "Environment and OLM bridge smoke test passed. Start with: ./run_short_pilot.sh"
