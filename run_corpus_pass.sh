#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! "${PROJECT_ROOT}/setup.sh" --check; then
  echo "Environment missing, incompatible, or stale; running setup first." >&2
  "${PROJECT_ROOT}/setup.sh"
fi
if [[ -n "${DAPT_CACHE_ROOT:-}" ]]; then
  mkdir -p "${DAPT_CACHE_ROOT}"
  export HF_HOME="${HF_HOME:-${DAPT_CACHE_ROOT}/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${DAPT_CACHE_ROOT}/torch}"
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PROJECT_ROOT}/.venv/bin/python" -m recipe_dapt.pipeline \
  --config "${PROJECT_ROOT}/configs/corpus_pass.yaml" "$@"
