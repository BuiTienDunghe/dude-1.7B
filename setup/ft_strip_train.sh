#!/usr/bin/env bash
# Một lần train duy nhất: Qwen3-1.7B seed 1, train_v0, LoRA cũ, --loss full_vocab, 2 epoch, + bỏ dấu ngẫu nhiên (p 0,30; nửa chỉ state).
# Theo dõi và cổng letter_mass như lần trước; ghi loss val gốc và val bỏ dấu mỗi epoch. Sau đó chấm trên dev_v1 (không test 1.0).
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_strip; D=data/bench_v1/dev_v1.jsonl; DIAG=$R/diag/dev_v1_vi_state_only_nodiac.jsonl
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; NEW=$R/train/Qwen3-1.7B-s1/adapter
mkdir -p $R/train/Qwen3-1.7B-s1 $R/eval $R/temp
LOG=$R/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "train start"
timeout 18000 bash setup/run.sh -X faulthandler scripts/train_lora.py Qwen/Qwen3-1.7B 1 $R/train/Qwen3-1.7B-s1 \
  --loss full_vocab --monitor_frac 0.1 --gate_step_frac 0.2 --gate_letter_mass 0.5 \
  --strip_p 0.3 --strip_state_only_frac 0.5 --val_nodiac > $R/train/Qwen3-1.7B-s1/stdout.txt 2>&1
rc=$?; say "train exit=$rc"
[ $rc -ne 0 ] && { say "STOP (train exit $rc; 3 = gate letter_mass)"; exit $rc; }
ev() {  # ev <subdir> <name> <adapter> [extra...]
  local sub=$1 name=$2 ad=$3; shift 3
  timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/eval/$sub --name $name --adapter $ad "$@" \
    > $R/eval/log_${sub}_${name}.txt 2>&1
  say "eval $sub $name exit=$? $(tail -1 $R/eval/log_${sub}_${name}.txt)"
}
ev normal strip-s1 $NEW --rows $D
ev reverse strip-s1 $NEW --rows $D --reverse
ev val strip-s1 $NEW --rows data/val_v0.jsonl
ev reverse control-s1 $CTRL --rows $D --reverse
ev diag strip-s1 $NEW --rows $DIAG
ev diag control-s1 $CTRL --rows $DIAG
timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $DIAG $R/eval/diag > $R/eval/log_diag_decider.txt 2>&1
say "eval diag decider exit=$? $(grep -v unauth $R/eval/log_diag_decider.txt | tail -2 | tr '\n' ' ')"
bash setup/run.sh scripts/fit_temperature.py $R/eval/val/strip-s1/preds.jsonl $R/temp/strip-s1.json >> "$LOG" 2>&1
say PIPELINE_STRIP_DONE
