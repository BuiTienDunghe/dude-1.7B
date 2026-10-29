#!/usr/bin/env bash
# Chạy lại cổng G2 sau khi đổi khóa sắp xếp trong batch (độ dài bản có dấu).
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_strip
LOG=$R/gates.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "G2 rerun after sort-key change"
for p in 0 0.3; do
  timeout 1200 bash setup/run.sh scripts/train_lora.py Qwen/Qwen3-1.7B 1 $R/gates/dump_p$p --loss full_vocab --strip_p $p \
    --strip_state_only_frac 0.5 --dump_batches 200 > $R/gates/g2_p$p.log 2>&1
  say "G2 dump p=$p exit=$? $(tail -1 $R/gates/g2_p$p.log)"
done
bash setup/run.sh scripts/ft_strip/compare_dumps.py 2>&1 | tee -a "$LOG"
