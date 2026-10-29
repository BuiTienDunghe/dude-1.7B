#!/usr/bin/env bash
# Bước E, phần chạy trong .venv chính (không cài gì mới): Qwen3.5-2B gốc (eval_base, thứ tự thường + đảo, câu hiển nhiên trước),
# decider-2b thứ tự đảo (chỉ choice), và lượt đảo cho hai mốc dự án (Qwen3-1.7B gốc, adapter full_vocab). CHỈ DEV 1.2; không test, không train.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # mọi model ở đây đã có trong cache
B=results/baselines_v1_2; E=$B/eval; mkdir -p $E/normal $E/reverse $B/obvious
LOG=$B/run_main.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
bash setup/run.sh scripts/baselines_v1_2/prep_dev_v1_2.py | tee -a "$LOG" || exit 1
for mode in normal reverse; do
  extra=""; [ $mode = reverse ] && extra="--reverse"
  timeout 1200 bash setup/run.sh scripts/eval_base.py Qwen/Qwen3.5-2B $B/obvious/$mode --name Qwen3.5-2B --rows results/baselines/obvious_v0.jsonl $extra \
    > $B/obvious/log_${mode}_qwen35.txt 2>&1
  say "obvious qwen35 $mode exit=$?"
done
for mode in normal reverse; do
  extra=""; [ $mode = reverse ] && extra="--reverse"
  timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3.5-2B $E/$mode --name Qwen3.5-2B --rows $B/dev_v1_2_ntok_null.jsonl $extra \
    > $E/log_${mode}_Qwen3.5-2B.txt 2>&1
  say "dev qwen35 $mode exit=$? $(tail -1 $E/log_${mode}_Qwen3.5-2B.txt)"
done
timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $B/dev_v1_2_choice_reversed.jsonl $E/reverse/decider > $E/log_reverse_decider.txt 2>&1
say "dev decider reverse exit=$? $(grep -v unauth $E/log_reverse_decider.txt | tail -2 | tr '\n' ' ')"
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E/reverse --name Qwen3-1.7B --rows data/bench_v1_2/dev_v1_2.jsonl --reverse \
  > $E/log_reverse_Qwen3-1.7B.txt 2>&1
say "dev qwen3-1.7B base reverse exit=$?"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E/reverse --name Qwen3-1.7B-fullvocab-s1 --adapter $CTRL --rows data/bench_v1_2/dev_v1_2.jsonl --reverse \
  > $E/log_reverse_fullvocab.txt 2>&1
say "dev fullvocab reverse exit=$?"
say MAIN_DONE
