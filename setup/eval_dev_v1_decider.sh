#!/usr/bin/env bash
# Kiểm lại adapter decider (noul/score) rồi chấm decider-2b trên dev_v1.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/dev_v1; D=data/bench_v1/dev_v1.jsonl
LOG=$R/run.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
bash setup/run.sh scripts/bench/make_obvious_typed.py
timeout 1800 bash setup/run.sh scripts/bench/eval_decider_v1.py $D $R/eval --checks > $R/eval/log_decider_checks.txt 2>&1
say "decider checks exit=$? $(grep -o 'max_abs_diff[^,}]*' $R/eval/log_decider_checks.txt | tr '\n' ' ')"
timeout 1200 bash setup/run.sh scripts/bench/eval_decider_v1.py $R/obvious_typed.jsonl $R/obvious_typed_eval > $R/eval/log_decider_obvious.txt 2>&1
say "decider obvious exit=$? $(grep -v unauth $R/eval/log_decider_obvious.txt | tail -2 | tr '\n' ' ')"
timeout 7200 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $D $R/eval > $R/eval/log_decider.txt 2>&1
say "decider exit=$? $(grep -v unauth $R/eval/log_decider.txt | tail -2 | tr '\n' ' ')"
say DECIDER_DEV_V1_DONE
