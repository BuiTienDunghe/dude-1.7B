#!/usr/bin/env bash
# Train thử C, chấm theo thứ tự điều phối đổi lại (26/10): (a) bản cuối trên dev, (b) bản cuối trên P2, P3, P4, Pa, Pb, P5, P1′.
# step484 và step969 đã chấm dev (step969: có preds, thiếu timing.json vì hàng đợi cũ bị dừng trong lượt đo thông lượng). step1454 chấm SAU,
# bằng: .venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter results/train_v1/train/C-Qwen3-1.7B-s1/adapter_step1454 --name C-s1-step1454 --files dev
# Batch 1, cùng script. Không train, không test. Chạy (WSL): bash scripts/train_v1/run_C_final.sh
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=results/train_v1/train/C-Qwen3-1.7B-s1
L=results/probe_v1/eval/logs
.venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter $OUT/adapter --name C-s1 --files dev,p2,p3,p4,pa,pb,p5,p1b > $L/C-s1.txt 2>&1
echo "C 100% exit=$?"
echo C_FINAL_DONE
