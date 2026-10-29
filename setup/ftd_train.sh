#!/usr/bin/env bash
# MỘT lần train từ decider-2b (seed 1, train_v0, LoRA r16/α32/0,05, 2 epoch, batch 32 = 8 × 4, lr 2e-4, warmup 3% + cosine,
# bỏ dấu như strip_v0), rồi chấm trên dev_v1 (epoch 2 chính thức, epoch 1 để xem) và trên bản khôi phục vit5. Không mở test 1.0.
# $1 = phương án loss (A | C | N) do người dùng chọn.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
V=${1:?loss variant}
R=results/ft_decider; T=$R/train/decider-2b-s1; mkdir -p $T $R/eval
LOG=$R/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "train start variant=$V"
timeout 36000 bash setup/run.sh -X faulthandler scripts/ft_decider/train_decider.py 1 $T --loss_variant $V --strip_p 0.3 --strip_state_only_frac 0.5 \
  --val_nodiac --monitor_frac 0.1 --gate_step_frac 0.2 --gate_mass 0.5 --token_budget 16384 > $T/stdout.txt 2>&1
rc=$?; say "train exit=$rc $(tail -1 $T/stdout.txt | cut -c1-300)"
[ $rc -ne 0 ] && { say "STOP (train exit $rc; 3 = cổng mass)"; exit $rc; }
ev() {  # ev <out_subdir> <rows> <adapter>
  timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $2 $R/eval/$1 --adapter $3 > $R/eval/log_$1.txt 2>&1
  say "eval $1 exit=$? $(grep -v unauth $R/eval/log_$1.txt | tail -2 | tr '\n' ' ' | cut -c1-250)"
}
ev dev_v1_ep2 data/bench_v1/dev_v1.jsonl $T/adapter
ev restored_vit5 results/restore/rows/dev_v1_vi_restored_vit5.jsonl $T/adapter
ev dev_v1_ep1 data/bench_v1/dev_v1.jsonl $T/adapter_epoch1
timeout 1800 bash setup/run.sh scripts/ft_decider/report_ft_decider.py > $R/report_run.log 2>&1
say "report exit=$? $(tail -2 $R/report_run.log | tr '\n' ' ' | cut -c1-300)"
say PIPELINE_FTD_DONE
