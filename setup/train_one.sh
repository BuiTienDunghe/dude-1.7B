#!/usr/bin/env bash
# Train một model với một seed: bash setup/train_one.sh <Qwen/...> <seed> [lr]
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
m="$1"; seed="$2"; lr="${3:-2e-4}"
out="results/ft/train/${m#Qwen/}-s${seed}"
mkdir -p "$out"
timeout 18000 bash setup/run.sh -X faulthandler scripts/train_lora.py "$m" "$seed" "$out" --lr "$lr" > "$out/stdout.txt" 2>&1
echo "train ${m} s${seed} exit=$?"
