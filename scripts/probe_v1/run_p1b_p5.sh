#!/usr/bin/env bash
# Chấm P1′ (p1b) và P5 cho 4 model mốc (M0, M1, decider-2b, Quyet-Medium), sau khi người dùng duyệt 7 chữ (quyet_dinh_chat_pv1b.json, 26/10).
# Batch 1. CHỈ DEV; không test, không train. Chạy: bash scripts/probe_v1/run_p1b_p5.sh  (từ WSL)
cd /mnt/c/Users/dungbt06/Fine-tune
L=results/probe_v1/eval/logs; mkdir -p $L
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
for m in M0 M1 decider-2b quyet-medium; do
  timeout 5400 .venv/bin/python -u -X faulthandler scripts/probe_v1/score_probe.py $m --files p1b,p5 > $L/${m}_p1b_p5.txt 2>&1
  echo "$m exit=$?"
done
echo P1B_P5_DONE
