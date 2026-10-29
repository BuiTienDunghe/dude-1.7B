#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 1800 nice -n 10 bash setup/run.sh scripts/restore/report_restore.py > results/restore/report_run.log 2>&1
echo "exit=$?"; tail -14 results/restore/report_run.log | cut -c1-220
