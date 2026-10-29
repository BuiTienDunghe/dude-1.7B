#!/usr/bin/env bash
# Điều phối 02:27: bỏ bước 12 (4B trên toàn dev) của hàng đợi B, chuyển sang hàng đợi C (tập con 1.000 dòng).
# Đợi dòng "XONG 11" trong queue.log rồi dừng hàng đợi B NGAY (lúc đó B đang kiểm GPU rảnh, chưa nạp model), sau đó chạy hàng đợi C
# với thời gian GPU đã cộng dồn. Chạy (WSL, nền): bash setup/g0_switch_to_c.sh
cd /mnt/c/Users/dungbt06/Fine-tune
Q=results/G0/logs/queue.log
until grep -q "XONG 11 " $Q; do sleep 2; done
pkill -TERM -f "bash /mnt/c/Users/dungbt06/Fine-tune/setup/g0_queue_b.sh"
sleep 2
pkill -TERM -f "scripts/latency_g0.py Qwen/Qwen3.5-4B --revision" 2>/dev/null
S=$(grep "XONG 11 " $Q | tail -1 | sed -E 's/.*cộng dồn ([0-9]+)s.*/\1/')
echo "$(date -Is) Đã dừng hàng đợi B sau bước 11 (bỏ bước 12 toàn dev); chuyển sang hàng đợi C, cộng dồn ${S}s" >> $Q
exec bash setup/g0_queue_c.sh "13 14" "$S"
