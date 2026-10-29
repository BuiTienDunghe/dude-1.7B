#!/usr/bin/env bash
# Lính gác nhiệt GPU (ngưỡng người dùng đặt 28/10): đọc nhiệt độ mỗi 30 giây, ghi results/G0/logs/temp.log.
# > 85°C: dừng an toàn hàng đợi G0 (g0_queue*.sh) và lượt GPU đang chạy (eval_base / latency_g0 / train_lora) bằng SIGTERM, ghi cờ
# results/G0/logs/QUA_NHIET.flag (lượt đó phải bỏ, chạy lại sau), rồi đợi GPU < 70°C, ghi "DA_NGUOI" và thoát (worker báo điều phối trước khi chạy tiếp).
# Chạy (WSL, nền): bash setup/g0_temp_guard.sh
L=/mnt/c/Users/dungbt06/Fine-tune/results/G0/logs; mkdir -p $L; T=$L/temp.log
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  echo "$(date -Is) $t" >> $T
  if [ "$t" -gt 85 ]; then
    echo "$(date -Is) QUA_NHIET $t°C: dừng hàng đợi và lượt GPU" | tee -a $T > $L/QUA_NHIET.flag
    pkill -TERM -f "setup/g0_queue" ; sleep 1
    pkill -TERM -f "scripts/eval_base.py|scripts/latency_g0.py|scripts/train_lora.py"
    while true; do
      sleep 30; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
      echo "$(date -Is) dang_nguoi $t" >> $T
      [ "$t" -lt 70 ] && { echo "$(date -Is) DA_NGUOI $t°C" | tee -a $T >> $L/QUA_NHIET.flag; exit 0; }
    done
  fi
  sleep 30
done
