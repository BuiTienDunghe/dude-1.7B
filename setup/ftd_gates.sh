#!/usr/bin/env bash
# Bước 1 (kiểm khả thi, không train): G2 -> G1 -> G3. Mọi bước có timeout; log ra file.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_decider; mkdir -p $R/gates
LOG=$R/gates.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
step=${1:-all}
if [ "$step" = all ] || [ "$step" = g2 ]; then
  timeout 1800 bash setup/run.sh -X faulthandler scripts/ft_decider/g2_check.py > $R/gates/g2.log 2>&1
  say "G2 exit=$? $(tail -1 $R/gates/g2.log | cut -c1-600)"
fi
if [ "$step" = all ] || [ "$step" = g1 ]; then
  timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py data/bench_v1/dev_v1.jsonl $R/gates/g1 > $R/gates/g1.log 2>&1
  say "G1 eval exit=$?"
  sha256sum results/dev_v1/eval/decider-2b-text/preds.jsonl $R/gates/g1/decider-2b-text/preds.jsonl | tee -a "$LOG"
  if cmp -s results/dev_v1/eval/decider-2b-text/preds.jsonl $R/gates/g1/decider-2b-text/preds.jsonl; then say "G1 PASS byte-identical (text)"; else say "G1 FAIL (text)"; fi
  if cmp -s results/dev_v1/eval/decider-2b-keyed/preds.jsonl $R/gates/g1/decider-2b-keyed/preds.jsonl; then say "G1 keyed byte-identical"; else say "G1 keyed differs"; fi
fi
if [ "$step" = all ] || [ "$step" = g3 ]; then
  timeout 7200 bash setup/run.sh -X faulthandler scripts/ft_decider/train_decider.py 1 $R/gates/g3 --strip_p 0.3 --strip_state_only_frac 0.5 \
    --val_nodiac --g3 12 ${2:-} ${3:-} > $R/gates/g3.log 2>&1
  say "G3 exit=$? $(tail -1 $R/gates/g3.log | cut -c1-700)"
fi
