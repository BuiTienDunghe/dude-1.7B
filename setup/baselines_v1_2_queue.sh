#!/usr/bin/env bash
# Bước E: hàng đợi GPU tuần tự (không chạy chồng để số độ trễ batch 1 so được). CHỈ DEV 1.2; không test, không train.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
B=results/baselines_v1_2; LOG=$B/run_queue.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
X=setup/baselines_v1_2_ext.sh
for k in quyet-tiny quyet-small laya-vi laya-multilingual quyet-medium; do bash $X $k wiring > /dev/null; say "wiring $k exit=$? $(tail -1 $B/logs/${k}_wiring.txt | cut -c1-300)"; done
bash $X quyet-medium dev > /dev/null; say "dev quyet-medium exit=$?"
for k in quyet-tiny quyet-small laya-vi laya-multilingual; do bash $X $k dev --force > /dev/null 2>&1; say "dev $k (force) exit=$? $(grep -c tin_cay $B/eval/normal/${k}-*/timing.json 2>/dev/null | tr '
' ' ')"; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
E=$B/eval; RR=$B/eval/rerun; mkdir -p $RR/normal $RR/reverse
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E/reverse --name Qwen3-1.7B --rows data/bench_v1_2/dev_v1_2.jsonl --reverse \
  > $E/log_reverse_Qwen3-1.7B.txt 2>&1; say "base reverse exit=$?"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $RR/reverse --name Qwen3-1.7B-fullvocab-s1 --adapter $CTRL \
  --rows data/bench_v1_2/dev_v1_2.jsonl --reverse > $RR/log_reverse_fullvocab.txt 2>&1; say "rerun fullvocab reverse exit=$?"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3.5-2B $RR/normal --name Qwen3.5-2B --rows $B/dev_v1_2_ntok_null.jsonl \
  > $RR/log_normal_Qwen3.5-2B.txt 2>&1; say "rerun qwen35 normal exit=$?"
timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $B/dev_v1_2_choice_reversed.jsonl $RR/reverse/decider > $RR/log_reverse_decider.txt 2>&1
say "rerun decider reverse exit=$?"
say QUEUE_DONE
