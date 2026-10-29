#!/usr/bin/env bash
# Bước E: chạy một model ngoài bằng .venv-baselines, OFFLINE (model đã tải sẵn, ghim revision). Dùng: baselines_v1_2_ext.sh <khóa> checks|wiring|dev [--force]
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONHASHSEED=0
export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
L=results/baselines_v1_2/logs; mkdir -p $L
K=$1; S=$2; shift 2
timeout ${TMO:-14400} .venv-baselines/bin/python -u -X faulthandler scripts/baselines_v1_2/run_external.py "$K" "$S" "$@" > $L/${K}_${S}.txt 2>&1
rc=$?
echo "$K $S $* exit=$rc"; grep -v "Warning\|warn(" $L/${K}_${S}.txt | tail -6
exit $rc
