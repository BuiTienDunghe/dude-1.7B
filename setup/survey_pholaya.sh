#!/usr/bin/env bash
# Đếm nguồn PhoLaya-base cho khảo sát train_v1; chạy nice để không làm nhiễu số đo độ trễ của lượt khôi phục trên GPU.
cd /mnt/c/Users/dungbt06/Fine-tune
mkdir -p results/train_v1_survey/raw/pholaya
timeout 1800 nice -n 19 bash setup/run.sh scripts/survey/pholaya_counts.py "$@" > results/train_v1_survey/pholaya_counts.log 2>&1
echo "exit=$?"; tail -8 results/train_v1_survey/pholaya_counts.log | cut -c1-250
