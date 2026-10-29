#!/usr/bin/env bash
# Hàng đợi GPU phần 2 của G0 (sau khi tải xong Qwen3.5-2B-Base, Qwen3.5-4B): G0b (bước 3, 4), G0c 2B-Base (bước 8), rồi G0a đo độ trễ (9, 11, 12).
# Giữ nguyên quy tắc của setup/g0_queue.sh (GPU rảnh trước mỗi bước, đợi 5 phút nếu bận, ghi log, cộng dồn thời gian GPU, dừng nếu vượt 2,5 giờ).
# Thêm cho đo độ trễ (điều phối 28/10): trong lúc đo có "lính gác" quét nvidia-smi phía Windows mỗi 5 giây; thấy bất kỳ tiến trình Windows nào
# (ngoài PID 4 = tiến trình System ứng với WSL), hoặc lượt đo đè lên khung 03:00–03:05 (eval đêm của dự án khác), thì lượt đo bị gắn cờ,
# kết quả được giữ lại với tên *_BO_*.json và đo lại (tối đa 2 lần). Không train, không test.
# Chạy (WSL): bash setup/g0_queue_b.sh ["danh sách bước"] [giây GPU đã dùng ở phần 1]
set -u
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=results/G0; L=$G/logs; mkdir -p $L $G/nvsmi $G/latency
Q=$L/queue.log
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
ONLY="${1:-3 4 8 9 11 12}"
GPU_S=${2:-0}
REV17=$(.venv/bin/python -c "import json;print(json.load(open('config/models.json'))['Qwen/Qwen3-1.7B'])")
REV35=$(.venv/bin/python -c "import json;print(json.load(open('config/models.json'))['Qwen/Qwen3.5-2B'])")
REVB=${REVB:-$(.venv/bin/python -c "import json;print(json.load(open('$G/env/models_g0.json'))['Qwen/Qwen3.5-2B-Base']['revision'])" 2>/dev/null)}
REV4=${REV4:-$(.venv/bin/python -c "import json;print(json.load(open('$G/env/models_g0.json'))['Qwen/Qwen3.5-4B']['revision'])" 2>/dev/null)}
TPL="Qwen/Qwen3.5-2B@$REV35"
ADC=results/train_v1/train/C-Qwen3-1.7B-s1/adapter
log() { echo "$(date -Is) $*" | tee -a $Q; }
win_apps() { $WSMI --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -E '^4, ' | tr '\n' ';'; }

avoid_window() {
  local hm=$(date +%H%M)
  if [ "$hm" -ge 0250 ] && [ "$hm" -lt 0306 ]; then
    log "TRÁNH KHUNG 02:50–03:06 trước $1: đợi tới 03:06"
    while [ "$(date +%H%M)" -lt 0306 ]; do sleep 30; done
  fi
}
gpu_idle() {   # điều kiện "GPU rảnh" đổi 02:05 (điều phối chọn b): chỉ số tải bị kẹt 100% sau khi dừng PID 4025 (công suất 27 W, VRAM 0)
  # → rảnh = không tiến trình compute (Windows và WSL) + VRAM < 500 MiB + công suất < 40 W liên tục 10 giây
  while true; do
    apps=$(win_apps); lapps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ';')
    okp=1; pmax=0; mmax=0
    for i in $(seq 10); do
      read pw mem < <(nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' ')
      p=${pw%.*}; [ "$p" -gt "$pmax" ] && pmax=$p; [ "$mem" -gt "$mmax" ] && mmax=$mem; sleep 1
    done
    if [ -z "$apps" ] && [ -z "$lapps" ] && [ "$mmax" -lt 500 ] && [ "$pmax" -lt 40 ]; then
      log "GPU rảnh trước $1 (tiến trình: không; VRAM max ${mmax} MiB; công suất max ${pmax} W trong 10s)"; return 0; fi
    log "GPU BẬN trước $1: win=[$apps] wsl=[$lapps] VRAM max ${mmax} MiB công suất max ${pmax} W → đợi 5 phút"
    sleep 300
  done
}
vram_ok() {   # $1 = GB cần
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( (tot - used) / 1024 ))
    if [ "$free" -ge "$1" ]; then log "VRAM trống ${free} GB >= $1 GB"; return 0; fi
    log "VRAM trống ${free} GB < $1 GB → đợi 5 phút"; sleep 300
  done
}
run() {   # $1 = số bước, $2 = tên, $3 = GB VRAM cần, $4 = 1 nếu là đo độ trễ, $5... = lệnh
  local n=$1 name=$2 need=$3 islat=$4; shift 4
  case " $ONLY " in *" $n "*) ;; *) return 0 ;; esac
  local tries=0
  while true; do
    if [ $GPU_S -gt 9000 ]; then log "DỪNG: thời gian GPU cộng dồn $GPU_S s > 2,5 giờ"; exit 9; fi
    avoid_window "$name"; gpu_idle "$name"; vram_ok "$need"
    tc=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)   # ngưỡng nhiệt người dùng đặt 28/10
    if [ "$tc" -gt 85 ] || [ -f $L/QUA_NHIET.flag ]; then log "NHIỆT ĐỘ $tc°C hoặc có cờ QUA_NHIET trước $name: DỪNG hàng đợi"; exit 8; fi
    log "Nhiệt độ trước $name: $tc°C"
    $WSMI > $G/nvsmi/${n}_${name}_truoc_${tries}.txt 2>&1
    local flag=$L/${n}_${name}_canh_${tries}.txt; : > $flag
    local wpid=""
    if [ "$islat" = 1 ]; then
      ( while true; do a=$(win_apps); hm=$(date +%H%M)
          [ -n "$a" ] && echo "$(date -Is) TIẾN TRÌNH WINDOWS: $a" >> $flag
          [ "$hm" -ge 0300 ] && [ "$hm" -lt 0305 ] && echo "$(date -Is) KHUNG 03:00–03:05" >> $flag
          echo "$(date -Is) $(nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,temperature.gpu,memory.used --format=csv,noheader)" >> $L/${n}_${name}_xung_${tries}.txt
          sleep 5; done ) &
      wpid=$!
    fi
    local t0=$(date +%s); log "BẮT ĐẦU $n $name (lần $tries)"
    "$@" > $L/${n}_${name}_${tries}.txt 2>&1
    local rc=$? t1=$(date +%s)
    [ -n "$wpid" ] && kill $wpid 2>/dev/null
    GPU_S=$((GPU_S + t1 - t0))
    $WSMI > $G/nvsmi/${n}_${name}_sau_${tries}.txt 2>&1
    log "XONG $n $name exit=$rc $((t1 - t0))s (cộng dồn ${GPU_S}s)"
    if [ "$islat" = 1 ] && [ -s $flag ]; then
      log "LƯỢT ĐO $name BỊ GẮN CỜ ($(head -1 $flag)) → bỏ, đo lại"
      for f in $G/latency/latency_${name}.json; do [ -f $f ] && mv $f ${f%.json}_BO_${tries}.json; done
      tries=$((tries + 1)); [ $tries -ge 3 ] && { log "ĐÃ ĐO LẠI 2 LẦN VẪN BỊ GẮN CỜ: $name"; return 0; }
      continue
    fi
    return 0
  done
}

run 3 g0b_2B_Base 6 0 .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B-Base $G/eval/dev --rows $G/rows/dev_ntok_null.jsonl --name Qwen3.5-2B-Base --revision $REVB --chat_template_from $TPL
run 4 g0b_4B 11 0 .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-4B $G/eval/dev --rows $G/rows/dev_ntok_null.jsonl --name Qwen3.5-4B --revision $REV4
run 8 g0c_2B_Base_EN 6 0 .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B-Base $G/eval/g0c --rows $G/rows/dev_vi_orig_ntok_null.jsonl --name Qwen3.5-2B-Base_khungEN --frame en --revision $REVB --chat_template_from $TPL
run 9 C-s1 6 1 .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3-1.7B --revision $REV17 --adapter $ADC --name C-s1 --out_dir $G/latency
# Chốt sau C-s1 (điều phối 02:05): trung vị eager lệch > 15% so với 32,6 ms (đo 26/10 trên GPU trống) → dừng phần độ trễ, báo bất thường
case " $ONLY " in *" 9 "*)
  med=$(.venv/bin/python -c "import json;print(json.load(open('$G/latency/latency_C-s1.json'))['eager']['trung_binh_cac_lan']['trung_vi'])" 2>/dev/null || echo 0)
  ok=$(.venv/bin/python -c "print(int(abs($med-32.6)/32.6<=0.15))")
  log "Chốt C-s1: trung vị eager ${med} ms so với 32,6 ms → $([ "$ok" = 1 ] && echo 'trong 15%, đo tiếp' || echo 'LỆCH QUÁ 15%: DỪNG phần độ trễ')"
  [ "$ok" = 1 ] || exit 7 ;;
esac
run 11 Qwen3.5-2B-Base_khongFLA 6 1 .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3.5-2B-Base --revision $REVB --chat_template_from $TPL --name Qwen3.5-2B-Base_khongFLA --out_dir $G/latency
run 12 Qwen3.5-4B_khongFLA 11 1 .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3.5-4B --revision $REV4 --name Qwen3.5-4B_khongFLA --out_dir $G/latency --repeats 1
log "G0_QUEUE_B_DONE cộng dồn GPU ${GPU_S}s"
