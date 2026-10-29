#!/usr/bin/env bash
# Chấm cả hai model rồi tính chỉ số, cho từng lượt được truyền vào (mặc định: run1 run2).
set -euo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
for run in "${@:-run1 run2}"; do
  for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B; do
    echo "== $run $m"
    timeout 1800 bash setup/run.sh -X faulthandler scripts/eval_base.py "$m" "results/$run" > "results/log_${run}_${m#Qwen/}.txt" 2>&1
    tail -1 "results/log_${run}_${m#Qwen/}.txt"
  done
  bash setup/run.sh scripts/report.py "results/$run" > "results/log_${run}_report.txt" 2>&1
done
echo ALL_DONE
