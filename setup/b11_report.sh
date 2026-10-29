#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 3600 bash setup/run.sh scripts/bench11/report_dev_v1_1.py > results/dev_v1_1/report_run.log 2>&1
echo "exit=$?"; tail -3 results/dev_v1_1/report_run.log | cut -c1-400
