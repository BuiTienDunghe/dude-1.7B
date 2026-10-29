#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 1800 bash setup/run.sh scripts/bench11/sensitivity.py > results/dev_v1_1/sensitivity_run.log 2>&1
echo "exit=$?"; tail -3 results/dev_v1_1/sensitivity_run.log | cut -c1-600
