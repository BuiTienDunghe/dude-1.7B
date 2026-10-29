#!/usr/bin/env bash
# Chấm mốc probe_v1 (26/10): M0, M1, decider-2b, Quyet-Medium trên dev 1.3 + P2 (model M) + P3 + P4 + Pa + Pb. Batch 1. CHỈ DEV; không test, không train.
# P1′ / P5 chấm sau khi người dùng duyệt chữ. Chạy: bash scripts/probe_v1/run_moc.sh  (từ WSL)
cd /mnt/c/Users/dungbt06/Fine-tune
L=results/probe_v1/eval/logs; mkdir -p $L
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
for m in M0 M1; do
  timeout 5400 .venv/bin/python -u -X faulthandler scripts/probe_v1/score_probe.py $m --files dev,p2,p3,p4,pa,pb > $L/$m.txt 2>&1
  echo "$m exit=$?"
done
timeout 7200 .venv/bin/python -u -X faulthandler scripts/probe_v1/score_probe.py decider-2b --files dev,p3,p4,pa,pb > $L/decider-2b.txt 2>&1
echo "decider-2b exit=$?"
timeout 7200 .venv/bin/python -u -X faulthandler scripts/probe_v1/score_probe.py quyet-medium --files dev,p3,p4,pa,pb > $L/quyet-medium.txt 2>&1
echo "quyet-medium exit=$?"
echo MOC_DONE
