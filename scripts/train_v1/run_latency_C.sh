#!/usr/bin/env bash
# Đo lại độ trễ batch 1 trên dev 1.3 (điều phối đồng ý 26/10): M1 và C@100% xen kẽ M1, C, M1, C, liền nhau trên GPU trống.
# CHỈ ĐO THỜI GIAN: dùng chính scripts/eval_base.py (cùng đường đo) nhưng ghi vào results/train_v1/do_tre_lai/, không thay preds nào.
# Trước mỗi lượt ghi nvidia-smi (tiến trình đang dùng GPU, bộ nhớ đã dùng). Không train, không test. Chạy (WSL): bash scripts/train_v1/run_latency_C.sh
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
O=results/train_v1/do_tre_lai
mkdir -p $O
DEV=data/bench_v1_3/dev_v1_3.jsonl
i=0
for m in M1 C M1 C C25; do   # C25 = C@25% (adapter_step484), lượt thêm theo điều phối
  i=$((i+1))
  case $m in
    M1) AD=results/ft_loss/train/Qwen3-1.7B-s1/adapter ;;
    C) AD=results/train_v1/train/C-Qwen3-1.7B-s1/adapter ;;
    C25) AD=results/train_v1/train/C-Qwen3-1.7B-s1/adapter_step484 ;;
  esac
  { date -Is; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu --format=csv;
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; nvidia-smi; } > $O/nvidia_smi_lan${i}_$m.txt 2>&1
  .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $O --rows $DEV --name lan${i}_$m --adapter $AD > $O/log_lan${i}_$m.txt 2>&1
  echo "lan$i $m exit=$? $(python3 -c "import json;print(round(json.load(open('$O/lan${i}_$m/timing.json'))['batch1_median_latency_ms'],2))" 2>/dev/null)"
done
echo LATENCY_DONE
