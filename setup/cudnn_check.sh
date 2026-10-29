#!/usr/bin/env bash
# Kiểm cuDNN SDPA: trạng thái backend, đường chấm (so từng byte), đường train (3 chế độ). Không train, không đổi .venv.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
C=results/cudnn_check
mkdir -p $C
LOG=$C/run.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
timeout 1200 bash setup/run.sh scripts/cudnn_check/backend_state.py > $C/log_backend_state.txt 2>&1
say "backend_state exit=$?"
timeout 1800 bash setup/run.sh -X faulthandler scripts/cudnn_check/eval_nocudnn.py Qwen/Qwen3-1.7B $C/eval/normal \
  --name Qwen3-1.7B-fullvocab-s1 --adapter results/ft_loss/train/Qwen3-1.7B-s1/adapter > $C/log_eval_nocudnn.txt 2>&1
say "eval_nocudnn exit=$? $(head -1 $C/log_eval_nocudnn.txt)"
sha256sum results/ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl $C/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl | tee -a "$LOG"
if cmp -s results/ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl $C/eval/normal/Qwen3-1.7B-fullvocab-s1/preds.jsonl; then
  say "EVAL BYTE-IDENTICAL"
else
  say "EVAL DIFFERS"
fi
timeout 1800 bash setup/run.sh -X faulthandler scripts/cudnn_check/train_path_check.py > $C/log_train_path.txt 2>&1
say "train_path exit=$?"
say CUDNN_CHECK_DONE
