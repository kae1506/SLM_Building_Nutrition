#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "No environment found; running setup first." >&2
  "${PROJECT_ROOT}/setup.sh"
fi
source "${PROJECT_ROOT}/.venv/bin/activate"
if [[ -n "${DAPT_CACHE_ROOT:-}" ]]; then
  mkdir -p "${DAPT_CACHE_ROOT}"
  export HF_HOME="${HF_HOME:-${DAPT_CACHE_ROOT}/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${DAPT_CACHE_ROOT}/torch}"
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m recipe_dapt.pipeline --config "${PROJECT_ROOT}/configs/first_run.yaml" "$@"
