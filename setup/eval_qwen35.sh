#!/usr/bin/env bash
# Chấm Qwen/Qwen3.5-2B gốc trên dev_v0 bằng đúng scripts/eval_base.py (batch 1), thứ tự thường và đảo, có timeout.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
B=results/baselines
bash setup/run.sh scripts/baselines/prep_qwen35.py
for mode in normal reverse; do
  extra=""; [ $mode = reverse ] && extra="--reverse"
  timeout 2400 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3.5-2B $B/eval/$mode --name Qwen3.5-2B \
    --rows $B/dev_v0_ntok_null.jsonl $extra > $B/eval/log_${mode}_Qwen3.5-2B.txt 2>&1
  echo "$mode exit=$? $(tail -1 $B/eval/log_${mode}_Qwen3.5-2B.txt)"
done
