#!/usr/bin/env bash
# Lính gác nhiệt GPU, bản 2 (29/10). Luật nhiệt giữ nguyên bản 1 (setup/g0_temp_guard.sh): đọc nhiệt mỗi 30 giây, ghi results/G0/logs/temp.log;
# > 85°C: dừng hàng đợi G0 và lượt GPU đang chạy (eval_base / latency_g0 / train_lora) bằng SIGTERM, ghi cờ QUA_NHIET.flag, đợi < 70°C,
# ghi "DA_NGUOI" và thoát mã 0 (worker báo điều phối trước khi chạy tiếp).
# Thêm so với bản 1:
#   - stderr và dòng thoát (trap EXIT / tín hiệu) ghi vào results/G0/logs/temp_guard.err;
#   - dòng KHOI_DONG (pid, boot_id, giờ boot của WSL VM) mỗi lần chạy, để thấy VM có bị khởi động lại không;
#   - đọc nhiệt lỗi (không phải số) ghi LOI_DOC_NHIET, không so sánh;
#   - 3 lần đọc liền >= 83°C (khoảng 1 phút) ghi một dòng CANH_BAO_83 vào temp.log (mỗi đợt nóng một lần); worker theo dõi dòng này để báo.
# Nguyên nhân hai lần thoát mã 1 ngày 29/10 (14:45:46, 15:17:02): cả WSL VM bị tắt có trật tự từ phía Windows (wsl --shutdown của một session
# dọn ổ đĩa), không phải lỗi script. Vì vậy chạy bản này qua setup/g0_temp_guard_tram.ps1 (vòng tự khởi động lại phía Windows).
# Chạy trực tiếp (WSL): bash setup/g0_temp_guard_v2.sh
L=/mnt/c/Users/dungbt06/Fine-tune/results/G0/logs; mkdir -p $L; T=$L/temp.log
exec 2>>$L/temp_guard.err
trap 'rc=$?; echo "$(date -Is) linh_gac_v2 pid $$ thoat ma $rc" >> $L/temp_guard.err' EXIT
trap 'echo "$(date -Is) linh_gac_v2 pid $$ nhan tin hieu dung (TERM/HUP/INT)" >> $L/temp_guard.err; exit 143' TERM HUP INT
echo "$(date -Is) KHOI_DONG linh_gac_v2 pid=$$ boot_id=$(cat /proc/sys/kernel/random/boot_id) wsl_boot=$(uptime -s | tr ' ' T)" >> $T
c83=0
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  if ! [[ "$t" =~ ^[0-9]+$ ]]; then
    echo "$(date -Is) LOI_DOC_NHIET '$t'" >> $T; sleep 30; continue
  fi
  echo "$(date -Is) $t" >> $T
  if [ "$t" -ge 83 ]; then c83=$((c83 + 1)); else c83=0; fi
  [ "$c83" -eq 3 ] && echo "$(date -Is) CANH_BAO_83 ${t}°C: 3 lần đọc liền >= 83°C" >> $T
  if [ "$t" -gt 85 ]; then
    echo "$(date -Is) QUA_NHIET $t°C: dừng hàng đợi và lượt GPU" | tee -a $T > $L/QUA_NHIET.flag
    pkill -TERM -f "setup/g0_queue" ; sleep 1
    pkill -TERM -f "scripts/eval_base.py|scripts/latency_g0.py|scripts/train_lora.py"
    while true; do
      sleep 30; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
      echo "$(date -Is) dang_nguoi $t" >> $T
      [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -lt 70 ] && { echo "$(date -Is) DA_NGUOI $t°C" | tee -a $T >> $L/QUA_NHIET.flag; exit 0; }
    done
  fi
  sleep 30
done
