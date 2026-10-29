#!/usr/bin/env bash
# Chạy một script Python bằng venv của dự án: bash setup/run.sh <script.py> [args...]
set -euo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0
export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error DATASETS_VERBOSITY=error
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv/bin/python -u "$@"
