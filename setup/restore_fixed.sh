#!/usr/bin/env bash
# Kiểm độ nhạy (sau các lượt chính): dựng bản "đã sửa" từ đầu ra thô, kiểm, chấm 3 model. Không ghi gì vào data/bench_v1.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/restore; ROWS=$R/rows
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; NEW=results/ft_strip/train/Qwen3-1.7B-s1/adapter
LOG=$R/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
for n in vit5 small; do
  timeout 1200 bash setup/run.sh scripts/restore/make_fixed.py $ROWS $n > $ROWS/fixed_$n.log 2>&1
  say "fixed $n exit=$? $(tail -1 $ROWS/fixed_$n.log | cut -c1-200)"
  timeout 1200 bash setup/run.sh scripts/restore/check_restore.py $ROWS ${n}_fixed > $ROWS/check_${n}_fixed.log 2>&1
  say "check ${n}_fixed exit=$? $(grep '^{' $ROWS/check_${n}_fixed.log | tail -1 | cut -c1-200)"
done
for n in vit5 small; do
  F=$ROWS/dev_v1_vi_restored_${n}_fixed.jsonl; E=$R/eval/${n}_fixed
  for pair in "control-s1 $CTRL" "strip-s1 $NEW"; do
    set -- $pair
    timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 --adapter $2 --rows $F \
      > $R/eval/log_${n}_fixed_$1.txt 2>&1
    say "eval ${n}_fixed $1 exit=$? $(tail -1 $R/eval/log_${n}_fixed_$1.txt | cut -c1-150)"
  done
  timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $F $E > $R/eval/log_${n}_fixed_decider.txt 2>&1
  say "eval ${n}_fixed decider exit=$? $(grep -v unauth $R/eval/log_${n}_fixed_decider.txt | tail -2 | tr '\n' ' ' | cut -c1-200)"
done
timeout 1800 bash setup/run.sh scripts/restore/report_restore.py > $R/report_run.log 2>&1
say "report exit=$? $(tail -3 $R/report_run.log | tr '\n' ' ' | cut -c1-250)"
say PIPELINE_FIXED_DONE
