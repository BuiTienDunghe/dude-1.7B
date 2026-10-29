#!/usr/bin/env bash
# Chạy thử bộ khôi phục dấu trên 20 dòng dev_v1 không dấu (10 đầu + 10 state dài nhất). Không ghi gì vào data/bench_v1.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/restore/pilot; mkdir -p $R
for n in vit5 small; do
  timeout 1800 bash setup/run.sh scripts/restore/restore_dev.py $n $R --pilot 20 > $R/run_$n.log 2>&1
  echo "$n exit=$? $(tail -1 $R/run_$n.log | cut -c1-200)"
  timeout 600 bash setup/run.sh scripts/restore/check_restore.py $R $n 2>&1 | tail -n 17 | cut -c1-220
done
