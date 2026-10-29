#!/usr/bin/env bash
# BẬC 1 của 2b (results/BRIEF_2b_bac1_train.md; người dùng duyệt 28/10 khoảng 10:50): đo VRAM, train ĐÚNG 1 lần 2b-s1, chấm. Không đụng test.
# Cấu hình chép scripts/train_v1/run_train_C.sh; chỉ đổi file train (train_2b_v3.jsonl, sha ea604a01…), thư mục đầu ra và phần chia batch
# theo số đo VRAM (reserved <= 14 GB → --micro_batch 8 như C; ngược lại --token_budget 9216, đỉnh bằng C; gradient tương đương Σ loss / 32).
# Ngưỡng dừng: lính gác nhiệt (setup/g0_temp_guard.sh) chạy nền, > 85°C thì nó dừng train_lora / eval_base và ghi QUA_NHIET.flag; script này
# kiểm cờ trước mỗi bước. Ngân sách GPU của brief 5 giờ (18.000 giây, cộng dồn ở $OUT/gpu_giay.txt): train chạy dưới `timeout` để còn
# 40 phút cho chấm; bước chấm chỉ chạy khi còn ngân sách.
# Chạy (WSL), CHỈ sau khi điều phối báo "GPU SẴN":
#   bash scripts/train_2b/run_2b_bac1.sh vram [T]        # đo VRAM (mặc định T = 1664)
#   bash scripts/train_2b/run_2b_bac1.sh train mb8|tb9216 # train + chấm
set -u
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=results/train_2b/train/2b-Qwen3-1.7B-s1
TRAIN=results/train_2b/train_2b_v3.jsonl
VAL=data/val_v0.jsonl
ADC=results/train_v1/train/C-Qwen3-1.7B-s1/adapter
FLAG=results/G0/logs/QUA_NHIET.flag
WSMI=/mnt/c/Windows/System32/nvidia-smi.exe
BUDGET=18000
mkdir -p $OUT results/train_2b/eval/logs
GS=$OUT/gpu_giay.txt; [ -f $GS ] || echo 0 > $GS
log() { echo "$(date -Is) $*" | tee -a $OUT/hang_doi.log; }
used() { cat $GS; }
add_used() { echo $(( $(used) + $1 )) > $GS; }
check_flag() { if [ -f $FLAG ]; then log "QUÁ NHIỆT (cờ $FLAG): DỪNG"; exit 4; fi; }
win_apps() { $WSMI --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v -E '^4, ' | tr '\n' ';'; }
gpu_idle() {   # tiêu chí cũ: không tiến trình compute (Windows, WSL) + VRAM < 500 MiB + công suất < 40 W liên tục 10 giây
  local apps lapps pmax mmax pw mem p
  apps=$(win_apps); lapps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ';'); pmax=0; mmax=0
  for i in $(seq 10); do
    read pw mem < <(nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' ')
    p=${pw%.*}; [ "$p" -gt "$pmax" ] && pmax=$p; [ "$mem" -gt "$mmax" ] && mmax=$mem; sleep 1
  done
  if [ -z "$apps" ] && [ -z "$lapps" ] && [ "$mmax" -lt 500 ] && [ "$pmax" -lt 40 ]; then
    log "GPU rảnh trước $1 (VRAM max ${mmax} MiB, công suất max ${pmax} W trong 10 giây)"; return 0; fi
  log "GPU BẬN trước $1: win=[$apps] wsl=[$lapps] VRAM max ${mmax} MiB công suất max ${pmax} W → DỪNG, báo điều phối"; exit 5
}
timed() {   # $1 = tên bước; còn lại = lệnh. Cộng dồn giây GPU.
  local name=$1; shift; check_flag; local t0=$(date +%s)
  "$@"; local rc=$?
  local dt=$(( $(date +%s) - t0 )); add_used $dt
  log "XONG $name exit=$rc ${dt}s (cộng dồn $(used)s / $BUDGET s)"; check_flag; return $rc
}
[ "$(sha256sum $TRAIN | cut -c1-8)" = "ea604a01" ] || { log "SAI SHA $TRAIN: DỪNG"; exit 2; }
MODE=${1:?vram|train}
if [ "$MODE" = vram ]; then
  T=${2:-1664}
  gpu_idle "vram_probe $T"
  timed "vram_probe_$T" .venv/bin/python -u scripts/train_lora.py Qwen/Qwen3-1.7B 1 $OUT --micro_batch 8 --effective_batch 32 --loss full_vocab \
      --train_file $TRAIN --val_file $VAL --vram_probe $T
  cat $OUT/vram_probe_*_0.json          # tên file theo T thực (<= T yêu cầu)
  exit 0
fi
[ "$MODE" = train ] || { echo "MODE sai"; exit 2; }
SPLIT=${2:?mb8|tb9216}
case $SPLIT in mb8) BATCH=(--micro_batch 8) ;; tb9216) BATCH=(--micro_batch 8 --token_budget 9216) ;; *) echo "SPLIT sai"; exit 2 ;; esac
if [ -e "$OUT/adapter" ] || [ -e "$OUT/train_log.jsonl" ]; then log "ĐÃ CÓ $OUT/adapter hoặc train_log: không ghi đè, không train lần 2"; exit 2; fi
ARGS=(Qwen/Qwen3-1.7B 1 $OUT --lr 2e-4 --epochs 2 "${BATCH[@]}" --effective_batch 32 --warmup_frac 0.03 --loss full_vocab
      --monitor_frac 0.0645 --monitor_rows 200 --gate_step_frac 0.129 --gate_letter_mass 0.5
      --train_file $TRAIN --val_file $VAL --save_fracs 0.25,0.5,0.75)
.venv/bin/python - "$OUT" "$TRAIN" "$VAL" "${ARGS[*]}" "$SPLIT" <<'EOF'
import glob, hashlib, json, sys, time
out, train, val, args, split = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = {"luot": "BẬC 1 của 2b: 2b-s1 (train đúng 1 lần)", "cho_phep": "người dùng 28/10 khoảng 10:50 ('được, đồng ý duyệt…'), results/BRIEF_2b_bac1_train.md",
     "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lenh": "scripts/train_lora.py " + args, "chia_batch": split,
     "vram_probe": {p: json.load(open(p)) for p in sorted(glob.glob(f"{out}/vram_probe_*.json"))},
     "vram_probe_sai_lech": "brief ghi --vram_probe 1664 (ước của Fable); đo ở 1752 = prompt dài nhất thật của train_2b_v3 (render_prompt, "
                            "prep_bac1.json); điều phối đồng ý 28/10. Luật: reserved <= 14 GB → micro_batch 8, ngược lại token_budget 9216",
     "train_file": train, "train_sha256": sha(train), "val_file": val, "val_sha256": sha(val),
     "train_lora_py_sha256": sha("scripts/train_lora.py"), "common_py_sha256": sha("scripts/common.py"),
     "prompt_frames_sha256": sha("config/prompt_frames.json"), "models_json": json.load(open("config/models.json"))["Qwen/Qwen3-1.7B"],
     "cau_hinh_tham_chieu": "scripts/train_v1/run_train_C.sh (C-s1); chỉ đổi file train, thư mục đầu ra, chia batch",
     "tran_state": json.load(open("results/train_2b/train/prep_bac1.json")),
     "ghi_chu": "strip_p 0, val_v0, cổng letter_mass cùng tỉ lệ bước như C (monitor 0,0645, cổng 0,129); KHÔNG chạy gì trên test"}
open(f"{out}/manifest_train.json", "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
print("manifest ok", m["train_sha256"][:12], split)
EOF
gpu_idle "train 2b-s1"
LEFT=$(( BUDGET - $(used) - 2400 ))
log "BẮT ĐẦU train 2b-s1 ($SPLIT), giới hạn ${LEFT}s"
timed train_2b-s1 timeout --signal=TERM $LEFT .venv/bin/python -u -X faulthandler scripts/train_lora.py "${ARGS[@]}" > $OUT/stdout.txt 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "TRAIN_FAIL rc=$rc (3 = cổng letter_mass trượt; 124 = hết ngân sách): DỪNG, không chấm"; exit $rc; fi
L=results/probe_v1/eval/logs
left() { [ $(( BUDGET - $(used) )) -gt 300 ] || { log "HẾT NGÂN SÁCH GPU trước $1: DỪNG"; exit 6; }; }
left score_2b-s1
timed score_2b-s1 .venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter $OUT/adapter --name 2b-s1 --files dev,p2,p3,p4,pa,pb,p1b,p5 > $L/2b-s1.txt 2>&1
E=results/train_2b/eval
for who in 2b-s1 C-s1; do
  ad=$OUT/adapter; [ $who = C-s1 ] && ad=$ADC
  left proxy_$who
  timed proxy_$who .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $E/proxy --rows data/proxy_v1/proxy_v1.jsonl --name $who --adapter $ad > $E/logs/proxy_$who.txt 2>&1
  left valkc_$who
  timed valkc_$who .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $E/val_kc_v2 --rows $E/_rows/val_kc_v2_eval.jsonl --name $who --adapter $ad > $E/logs/valkc_$who.txt 2>&1
done
log "BAC1_DONE (GPU cộng dồn $(used)s)"
