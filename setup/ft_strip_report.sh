#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 1800 bash setup/run.sh scripts/ft_strip/report_strip.py > results/ft_strip/report_run.log 2>&1
echo "exit=$?"
tail -5 results/ft_strip/report_run.log | cut -c1-200
