#!/usr/bin/env bash
# Bench 1.1, mục E: cổng (chấm lại trên dòng không đổi) -> chấm lại dòng đổi chữ -> ghép preds dev_v1_1. Không mở test.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/dev_v1_1; RS=$R/rows; E=$R/eval_rescore; M=$R/eval; mkdir -p $E $M $R/gates
LOG=$R/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; STRIP=results/ft_strip/train/Qwen3-1.7B-s1/adapter; FTD=results/ft_decider/train/decider-2b-s1/adapter
bash setup/run.sh scripts/bench11/split_rows.py >> "$LOG" 2>&1
# cổng: adapter đối chứng trên mọi dòng không đổi -> từng dòng preds phải trùng bản cũ
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/gates/e_unchanged --name control --adapter $CTRL --rows $RS/unchanged.jsonl > $R/gates/e_unchanged.log 2>&1
say "gate E eval exit=$?"
bash setup/run.sh - <<'PY' 2>&1 | tee -a "$LOG"
import json
old = {json.loads(l)["id"]: l.rstrip("\n") for l in open("results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl", encoding="utf-8")}
new = [l.rstrip("\n") for l in open("results/dev_v1_1/gates/e_unchanged/control/preds.jsonl", encoding="utf-8")]
same = sum(old[json.loads(l)["id"]] == l for l in new)
print("GATE_E", "PASS" if same == len(new) else "FAIL", f"{same}/{len(new)} preds lines byte-identical to dev_v1")
PY
grep -q "GATE_E PASS" "$LOG" || { say "STOP gate E failed"; exit 2; }
ev_dec() {  # ev_dec <name> [adapter]
  timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $RS/rescore.jsonl $E/$1 ${2:+--adapter $2} > $E/log_$1.txt 2>&1
  say "eval $1 exit=$? $(grep -v unauth $E/log_$1.txt | tail -2 | tr '\n' ' ' | cut -c1-200)"
}
ev_base() {  # ev_base <name> [adapter]
  timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 ${2:+--adapter $2} --rows $RS/rescore.jsonl > $E/log_$1.txt 2>&1
  say "eval $1 exit=$? $(tail -1 $E/log_$1.txt | cut -c1-150)"
}
merge() { bash setup/run.sh scripts/bench11/merge_preds.py $1 $2 $3 2>&1 | tail -1 | tee -a "$LOG"; }
ev_dec decider
ev_base control $CTRL
merge results/dev_v1/eval/decider-2b-text/preds.jsonl $E/decider/decider-2b-text/preds.jsonl $M/decider-2b-text/preds.jsonl
merge results/dev_v1/eval/decider-2b-keyed/preds.jsonl $E/decider/decider-2b-keyed/preds.jsonl $M/decider-2b-keyed/preds.jsonl
merge results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl $E/control/preds.jsonl $M/Qwen3-1.7B-fullvocab-s1/preds.jsonl
say PRIMARY_DONE
ev_base base
ev_base strip $STRIP
ev_dec ftd $FTD
merge results/dev_v1/eval/Qwen3-1.7B/preds.jsonl $E/base/preds.jsonl $M/Qwen3-1.7B/preds.jsonl
merge results/ft_strip/eval/normal/strip-s1/preds.jsonl $E/strip/preds.jsonl $M/strip-s1/preds.jsonl
merge results/ft_decider/eval/dev_v1_ep2/decider-2b-text/preds.jsonl $E/ftd/decider-2b-text/preds.jsonl $M/decider-ft-ep2-text/preds.jsonl
merge results/ft_decider/eval/dev_v1_ep2/decider-2b-keyed/preds.jsonl $E/ftd/decider-2b-keyed/preds.jsonl $M/decider-ft-ep2-keyed/preds.jsonl
say EVAL_V1_1_DONE
