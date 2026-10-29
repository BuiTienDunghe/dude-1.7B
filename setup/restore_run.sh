#!/usr/bin/env bash
# Phần A: cổng G1 -> khôi phục dấu dev_v1 không dấu (vit5, small) -> kiểm bất biến/chất lượng/độ trễ -> chấm 3 model trên bản đã khôi phục.
# Chạy tuần tự trên GPU. Không ghi gì vào data/bench_v1; không mở test 1.0. Mọi bước có timeout.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/restore; ROWS=$R/rows; mkdir -p $R/gates $ROWS $R/eval
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; NEW=results/ft_strip/train/Qwen3-1.7B-s1/adapter
LOG=$R/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "G1 start"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/gates/g1 --name Qwen3-1.7B-fullvocab-s1 \
  --adapter $CTRL --rows data/bench_v1/dev_v1.jsonl > $R/gates/g1.log 2>&1
say "G1 eval exit=$?"
sha256sum results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl $R/gates/g1/Qwen3-1.7B-fullvocab-s1/preds.jsonl | tee -a "$LOG"
if cmp -s results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl $R/gates/g1/Qwen3-1.7B-fullvocab-s1/preds.jsonl; then say "G1 PASS byte-identical"; else say "G1 FAIL -> STOP"; exit 2; fi
for n in vit5 small; do
  timeout 14400 bash setup/run.sh -X faulthandler scripts/restore/restore_dev.py $n $ROWS > $ROWS/run_$n.log 2>&1
  say "restore $n exit=$? $(tail -1 $ROWS/run_$n.log | cut -c1-200)"
  timeout 1200 bash setup/run.sh scripts/restore/check_restore.py $ROWS $n > $ROWS/check_$n.log 2>&1
  say "check $n exit=$? $(grep '^{' $ROWS/check_$n.log | tail -1 | cut -c1-200)"
done
for n in vit5 small; do
  F=$ROWS/dev_v1_vi_restored_$n.jsonl
  for pair in "control-s1 $CTRL" "strip-s1 $NEW"; do
    set -- $pair
    timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/eval/$n --name $1 --adapter $2 --rows $F \
      > $R/eval/log_${n}_$1.txt 2>&1
    say "eval $n $1 exit=$? $(tail -1 $R/eval/log_${n}_$1.txt | cut -c1-150)"
  done
  timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $F $R/eval/$n > $R/eval/log_${n}_decider.txt 2>&1
  say "eval $n decider exit=$? $(grep -v unauth $R/eval/log_${n}_decider.txt | tail -2 | tr '\n' ' ' | cut -c1-200)"
done
say PIPELINE_RESTORE_DONE
