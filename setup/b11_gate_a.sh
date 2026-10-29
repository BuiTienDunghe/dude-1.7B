#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
mkdir -p results/dev_v1_1
timeout 1800 bash setup/run.sh scripts/bench11/gate_a.py > results/dev_v1_1/gate_a.log 2>&1
echo "exit=$?"; tail -4 results/dev_v1_1/gate_a.log | cut -c1-900
