#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 2400 nice -n 19 bash setup/run.sh scripts/survey/tasksource_mt_pool.py > results/train_v1_survey/tasksource_pool.log 2>&1
echo "exit=$?"; tail -8 results/train_v1_survey/tasksource_pool.log | cut -c1-300
