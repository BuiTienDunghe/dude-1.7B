#!/usr/bin/env bash
# Kiểm câu hiển nhiên (2 thứ tự) cho Qwen3.5-2B (eval_base.py) và decider-2b (adapter), rồi chấm decider-2b trên dev_v0.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
B=results/baselines
LOG=$B/run.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
for mode in normal reverse; do
  extra=""; [ $mode = reverse ] && extra="--reverse"
  timeout 1200 bash setup/run.sh scripts/eval_base.py Qwen/Qwen3.5-2B $B/obvious/$mode --name Qwen3.5-2B --rows $B/obvious_v0.jsonl $extra \
    > $B/obvious/log_${mode}_qwen35.txt 2>&1
  say "obvious qwen35 $mode exit=$?"
done
timeout 1800 bash setup/run.sh -X faulthandler scripts/baselines/eval_decider.py $B/obvious_v0.jsonl $B/obvious > $B/obvious/log_decider.txt 2>&1
say "obvious decider exit=$? $(grep -v unauth $B/obvious/log_decider.txt | tail -4 | tr '\n' ' ')"
timeout 7200 bash setup/run.sh -X faulthandler scripts/baselines/eval_decider.py $B/dev_v0_ntok_null.jsonl $B/eval > $B/eval/log_decider_dev.txt 2>&1
say "dev decider exit=$? $(grep -v unauth $B/eval/log_decider_dev.txt | tail -4 | tr '\n' ' ')"
say BASELINES_RUN_DONE
