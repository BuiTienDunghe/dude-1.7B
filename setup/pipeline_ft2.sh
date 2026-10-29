#!/usr/bin/env bash
# Phần tiếp của pipeline_ft.sh: chờ nó xong, train 1.7B, chấm những lượt còn thiếu (bỏ qua lượt đã có), fit T, báo cáo.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
SEEDS="1"
LOG=results/ft/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

while pgrep -f '[p]ipeline_ft.sh' > /dev/null; do sleep 60; done
say "pipeline2 start"
rm -rf results/ft/train/Qwen3-1.7B-s1   # chỉ chứa stdout của lần gọi timeout lỗi
for s in $SEEDS; do
  [ -f results/ft/train/Qwen3-1.7B-s$s/summary.json ] || say "$(bash setup/train_one.sh Qwen/Qwen3-1.7B $s)"
done

ev() {  # ev <model_id> <subdir> <name> [extra args...]; bỏ qua nếu đã có preds
  local m=$1 sub=$2 name=$3; shift 3
  [ -f "results/ft/eval/$sub/$name/preds.jsonl" ] && { say "skip $sub $name (exists)"; return; }
  timeout 1800 bash setup/run.sh -X faulthandler scripts/eval_base.py "$m" "results/ft/eval/$sub" --name "$name" "$@" \
    > "results/ft/eval/log_${sub}_${name}.txt" 2>&1
  say "eval $sub $name exit=$? $(tail -1 results/ft/eval/log_${sub}_${name}.txt)"
}
for m in Qwen3-0.6B Qwen3-1.7B; do
  ev Qwen/$m reverse $m --reverse
  ev Qwen/$m val $m --rows data/val_v0.jsonl
  for s in $SEEDS; do
    a=results/ft/train/$m-s$s/adapter
    [ -d "$a" ] || { say "missing adapter $a"; continue; }
    ev Qwen/$m normal $m-ft-s$s --adapter $a
    ev Qwen/$m reverse $m-ft-s$s --adapter $a --reverse
    ev Qwen/$m val $m-ft-s$s --adapter $a --rows data/val_v0.jsonl
  done
done
for d in results/ft/eval/val/*/; do
  n=$(basename "$d")
  bash setup/run.sh scripts/fit_temperature.py "$d/preds.jsonl" "results/ft/temp/$n.json" >> "$LOG" 2>&1
done
bash setup/run.sh scripts/report_ft.py > results/ft/log_report_ft.txt 2>&1
say "report exit=$?"
say PIPELINE2_DONE
