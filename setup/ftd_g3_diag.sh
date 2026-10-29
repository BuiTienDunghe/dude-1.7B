#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_decider
bash setup/ftd_gates.sh g3
timeout 1800 bash setup/run.sh -X faulthandler scripts/ft_decider/diag_loss.py > $R/gates/diag_loss.log 2>&1
echo "$(date +%H:%M:%S) diag exit=$? $(tail -1 $R/gates/diag_loss.log | cut -c1-900)" | tee -a $R/gates.log
