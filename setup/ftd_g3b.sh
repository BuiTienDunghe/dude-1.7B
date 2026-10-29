#!/usr/bin/env bash
# G3 lặp lại với ngân sách token: $1 = token_budget
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_decider; O=$R/gates/g3_tb$1; mkdir -p $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 7200 bash setup/run.sh -X faulthandler scripts/ft_decider/train_decider.py 1 $O --strip_p 0.3 --strip_state_only_frac 0.5 \
  --val_nodiac --g3 12 --token_budget $1 > $O/g3.log 2>&1
echo "$(date +%H:%M:%S) G3 tb=$1 exit=$? $(grep -E '^worst|^\{' $O/g3.log | tail -3 | cut -c1-900)" | tee -a $R/gates.log
