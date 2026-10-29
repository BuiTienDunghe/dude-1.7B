#!/usr/bin/env bash
# Hàng đợi GPU của bước 0 + G0 (BRIEF_buoc0_G0.md). KHÔNG train (chỉ một forward+backward đo VRAM), KHÔNG test.
# Trước MỖI bước: GPU phải rảnh = không tiến trình nào phía Windows dùng GPU (nvidia-smi.exe; bỏ qua PID 4 là tiến trình System ứng với WSL)
# và tải GPU < 10% trong 10 giây; bận thì đợi 5 phút, thử lại, ghi log. Cộng dồn thời gian GPU; vượt 2,5 giờ thì dừng.
# Chạy (WSL): bash setup/g0_queue.sh ["danh sách bước", mặc định tất cả]
set -u
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=results/G0; L=$G/logs; mkdir -p $L $G/nvsmi
Q=$L/queue.log
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
ONLY="${1:-1 2 3 4 5 6 7 8 9 11 12}"   # danh sách bước cần chạy
REV17=$(.venv/bin/python -c "import json;print(json.load(open('config/models.json'))['Qwen/Qwen3-1.7B'])")
REV35=$(.venv/bin/python -c "import json;print(json.load(open('config/models.json'))['Qwen/Qwen3.5-2B'])")
REVB=$(.venv/bin/python -c "import json;print(json.load(open('$G/env/models_g0.json'))['Qwen/Qwen3.5-2B-Base']['revision'])" 2>/dev/null)
REV4=$(.venv/bin/python -c "import json;print(json.load(open('$G/env/models_g0.json'))['Qwen/Qwen3.5-4B']['revision'])" 2>/dev/null)
TPL="Qwen/Qwen3.5-2B@$REV35"
DEV=/mnt/c/Users/dungbt06/Fine-tune/data/bench_v1_3/dev_v1_3.jsonl
ADC=results/train_v1/train/C-Qwen3-1.7B-s1/adapter
GPU_S=0
log() { echo "$(date -Is) $*" | tee -a $Q; }

gpu_idle() {   # $1 = tên bước
  while true; do
    apps=$($WSMI --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -E '^4, ' | tr '\n' ';')
    util=0
    for i in $(seq 10); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1); [ "$u" -gt "$util" ] && util=$u; sleep 1; done
    if [ -z "$apps" ] && [ "$util" -lt 10 ]; then log "GPU rảnh trước $1 (tải max 10s = $util%)"; return 0; fi
    log "GPU BẬN trước $1: tiến trình=[$apps] tải max 10s=$util% → đợi 5 phút"
    sleep 300
  done
}
avoid_window() {   # tác vụ eval đêm của dự án khác chạy trên GPU lúc 03:00 (1–3 phút): không bắt đầu bước nào trong 02:50–03:06
  local hm=$(date +%H%M)
  if [ "$hm" -ge 0250 ] && [ "$hm" -lt 0306 ]; then
    log "TRÁNH KHUNG 02:50–03:06 trước $1: đợi tới 03:06"
    while [ "$(date +%H%M)" -lt 0306 ]; do sleep 30; done
  fi
}
step() {   # $1 = số bước, $2 = tên, $3... = lệnh
  local n=$1 name=$2; shift 2
  case " $ONLY " in *" $n "*) ;; *) return 0 ;; esac
  if [ $GPU_S -gt 9000 ]; then log "DỪNG: thời gian GPU cộng dồn $GPU_S s > 2,5 giờ"; exit 9; fi
  avoid_window "$name"
  gpu_idle "$name"
  $WSMI > $G/nvsmi/${n}_${name}_truoc.txt 2>&1
  local t0=$(date +%s)
  log "BẮT ĐẦU $n $name"
  "$@" > $L/${n}_${name}.txt 2>&1
  local rc=$? t1=$(date +%s)
  GPU_S=$((GPU_S + t1 - t0))
  $WSMI > $G/nvsmi/${n}_${name}_sau.txt 2>&1
  log "XONG $n $name exit=$rc $((t1 - t0))s (cộng dồn ${GPU_S}s)"
}

# ---- bước 0
step 1 rescore_C .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B /mnt/c/Users/dungbt06/Fine-tune/$G/rescore_C --rows $DEV --name C-s1 --adapter $ADC
for TB in 8192 16384 32768; do
  step 2 vram_tb$TB .venv/bin/python -u scripts/train_lora.py Qwen/Qwen3-1.7B 1 $G/buoc0/vram --train_file results/train_v1/components/C.jsonl --vram_probe 2048 --token_budget $TB
done
# ---- G0b: chấm dev (khung dự án, batch 1)
step 3 g0b_2B_Base .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B-Base $G/eval/dev --rows $G/rows/dev_ntok_null.jsonl --name Qwen3.5-2B-Base --revision $REVB --chat_template_from $TPL
step 4 g0b_4B .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-4B $G/eval/dev --rows $G/rows/dev_ntok_null.jsonl --name Qwen3.5-4B --revision $REV4
# ---- G0c: khung EN so với khung VI trên dòng VI có dấu (cùng môi trường cho cùng model)
step 5 g0c_17B_EN .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $G/eval/g0c --rows $G/rows/dev_vi_orig_ntok_null.jsonl --name Qwen3-1.7B_khungEN --frame en
step 6 g0c_35_2B_VI .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B $G/eval/g0c --rows $G/rows/dev_vi_orig_ntok_null.jsonl --name Qwen3.5-2B_khungVI --frame vi
step 7 g0c_35_2B_EN .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B $G/eval/g0c --rows $G/rows/dev_vi_orig_ntok_null.jsonl --name Qwen3.5-2B_khungEN --frame en
step 8 g0c_2B_Base_EN .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3.5-2B-Base $G/eval/g0c --rows $G/rows/dev_vi_orig_ntok_null.jsonl --name Qwen3.5-2B-Base_khungEN --frame en --revision $REVB --chat_template_from $TPL
# ---- G0a: độ trễ
step 9 lat_C .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3-1.7B --revision $REV17 --adapter $ADC --name C-s1 --out_dir $G/latency
# FLA 0.6.0 cài được nhưng KHÔNG chạy được (triton cần trình biên dịch C, WSL không có, sudo cần mật khẩu): theo brief, ghi lỗi, đo không FLA,
# không vá. .venv-g0 vì vậy không dùng cho Qwen3.5 (fla import được nên transformers chọn nó rồi lỗi lúc chạy); mọi bước Qwen3.5 chạy trong .venv.
step 11 lat_2B_Base_noFLA .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3.5-2B-Base --revision $REVB --chat_template_from $TPL --name Qwen3.5-2B-Base_khongFLA --out_dir $G/latency
step 12 lat_4B .venv/bin/python -u scripts/latency_g0.py Qwen/Qwen3.5-4B --revision $REV4 --name Qwen3.5-4B_khongFLA --out_dir $G/latency --repeats 1
log "G0_QUEUE_DONE cộng dồn GPU ${GPU_S}s"
