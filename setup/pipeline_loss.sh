#!/usr/bin/env bash
# Kiểm chứng loss mới: train Qwen3-1.7B seed 1 với --loss full_vocab (mọi thứ khác như bước 3), có cổng letter_mass ở 20% bước;
# nếu train xong thì chấm normal / reverse / val bằng đúng eval_base.py (batch 1, có timeout) và fit temperature.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_loss
LOG=$R/pipeline.log
mkdir -p $R/train/Qwen3-1.7B-s1 $R/eval $R/temp
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "pipeline_loss start"
timeout 18000 bash setup/run.sh -X faulthandler scripts/train_lora.py Qwen/Qwen3-1.7B 1 $R/train/Qwen3-1.7B-s1 \
  --loss full_vocab --monitor_frac 0.1 --gate_step_frac 0.2 --gate_letter_mass 0.5 > $R/train/Qwen3-1.7B-s1/stdout.txt 2>&1
rc=$?
say "train exit=$rc"
if [ $rc -ne 0 ]; then say "STOP (train exit $rc; 3 = gate letter_mass)"; exit $rc; fi
A=$R/train/Qwen3-1.7B-s1/adapter
ev() {  # ev <subdir> <name> [extra args...]
  local sub=$1 name=$2; shift 2
  timeout 1800 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B "$R/eval/$sub" --name "$name" "$@" \
    > "$R/eval/log_${sub}_${name}.txt" 2>&1
  say "eval $sub $name exit=$? $(tail -1 $R/eval/log_${sub}_${name}.txt)"
}
ev normal Qwen3-1.7B-fullvocab-s1 --adapter $A
ev reverse Qwen3-1.7B-fullvocab-s1 --adapter $A --reverse
ev val Qwen3-1.7B-fullvocab-s1 --adapter $A --rows data/val_v0.jsonl
bash setup/run.sh scripts/fit_temperature.py $R/eval/val/Qwen3-1.7B-fullvocab-s1/preds.jsonl $R/temp/Qwen3-1.7B-fullvocab-s1.json >> "$LOG" 2>&1
say PIPELINE_LOSS_DONE
