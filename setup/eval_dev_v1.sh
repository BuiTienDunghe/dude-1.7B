#!/usr/bin/env bash
# Chấm ba model trên dev_v1 (batch 1, có timeout). Không chạm tập test.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/dev_v1; D=data/bench_v1/dev_v1.jsonl
mkdir -p $R/eval
LOG=$R/run.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
timeout 1800 bash setup/run.sh scripts/bench/eval_decider_v1.py $D $R/eval --checks > $R/eval/log_decider_checks.txt 2>&1
say "decider checks exit=$? $(grep -o 'max_abs_diff[^,}]*' $R/eval/log_decider_checks.txt | tr '\n' ' ')"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/eval --name Qwen3-1.7B --rows $D > $R/eval/log_base.txt 2>&1
say "base exit=$? $(tail -1 $R/eval/log_base.txt)"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/eval --name Qwen3-1.7B-fullvocab-s1 \
  --adapter results/ft_loss/train/Qwen3-1.7B-s1/adapter --rows $D > $R/eval/log_fullvocab.txt 2>&1
say "fullvocab exit=$? $(tail -1 $R/eval/log_fullvocab.txt)"
timeout 7200 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $D $R/eval > $R/eval/log_decider.txt 2>&1
say "decider exit=$? $(grep -v unauth $R/eval/log_decider.txt | tail -2 | tr '\n' ' ')"
say EVAL_DEV_V1_DONE
