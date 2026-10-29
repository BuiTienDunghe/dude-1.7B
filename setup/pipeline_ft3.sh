#!/usr/bin/env bash
# Seed 2 cho cả hai model (chạy thêm ngoài quy tắc 1 giờ, để đo nhiễu giữa các seed): train, chấm normal/reverse/val, fit T.
# Công thức giữ nguyên, chỉ đổi seed. Báo cáo được cập nhật riêng sau khi xong.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
LOG=results/ft/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "pipeline3 (seed 2) start"
for m in Qwen3-0.6B Qwen3-1.7B; do
  [ -f results/ft/train/$m-s2/summary.json ] || say "$(bash setup/train_one.sh Qwen/$m 2)"
done
ev() {  # ev <model_id> <subdir> <name> [extra args...]; bỏ qua nếu đã có preds
  local m=$1 sub=$2 name=$3; shift 3
  [ -f "results/ft/eval/$sub/$name/preds.jsonl" ] && { say "skip $sub $name (exists)"; return; }
  timeout 1800 bash setup/run.sh -X faulthandler scripts/eval_base.py "$m" "results/ft/eval/$sub" --name "$name" "$@" \
    > "results/ft/eval/log_${sub}_${name}.txt" 2>&1
  say "eval $sub $name exit=$? $(tail -1 results/ft/eval/log_${sub}_${name}.txt)"
}
for m in Qwen3-0.6B Qwen3-1.7B; do
  a=results/ft/train/$m-s2/adapter
  [ -d "$a" ] || { say "missing adapter $a"; continue; }
  ev Qwen/$m normal $m-ft-s2 --adapter $a
  ev Qwen/$m reverse $m-ft-s2 --adapter $a --reverse
  ev Qwen/$m val $m-ft-s2 --adapter $a --rows data/val_v0.jsonl
  bash setup/run.sh scripts/fit_temperature.py "results/ft/eval/val/$m-ft-s2/preds.jsonl" "results/ft/temp/$m-ft-s2.json" >> "$LOG" 2>&1
done
say PIPELINE3_DONE
