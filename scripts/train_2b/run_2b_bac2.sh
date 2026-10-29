#!/usr/bin/env bash
# BẬC 2 của 2b (results/BRIEF_2b_bac2_train.md; người dùng duyệt 29/10 khoảng 01:20): train ĐÚNG 1 lần 2b-s2, chấm. Không đụng test, không C-s2.
# Chép scripts/train_2b/run_2b_bac1.sh; CHỈ đổi seed (1 → 2) và thư mục đầu ra. Giữ micro_batch 8 (token_budget mặc định 0) như s1, không đo
# VRAM lại (dữ liệu giống s1, s1 đỉnh 10,25 GB), cùng biến môi trường (PYTHONHASHSEED=0), cùng --save_fracs 0.25,0.5,0.75.
# Trần state 2.048: train_2b_v3.jsonl giữ sha ea604a01… như s1, nên dùng lại kết quả kiểm của prep_2b_bac1.py (results/train_2b/train/prep_bac1.json:
# state max 1.560 token, 0 dòng > 2.048, prompt max 1.752) — script kiểm sha và ghi vào manifest.
# Ngưỡng dừng: lính gác nhiệt (setup/g0_temp_guard.sh) chạy nền, > 85°C thì nó dừng train_lora / eval_base và ghi QUA_NHIET.flag; script này
# kiểm cờ trước mỗi bước. Ngân sách GPU 5 giờ (18.000 giây, cộng dồn ở $OUT/gpu_giay.txt): train chạy dưới `timeout` để còn 40 phút cho chấm.
# Chấm: score_probe 2b-s2 (8 file), proxy_v1 và val_kc_v2 cho 2b-s2 (C-s1 và 2b-s1 đã chấm ở bậc 1). Checkpoint 25/50/75% chỉ lưu.
# Sau train ghi thêm nhiệt độ tối đa trong khung train vào summary.json (khóa "nhiet_do_toi_da_c", lấy từ results/G0/logs/temp.log).
# Chạy (WSL), CHỈ sau khi điều phối báo "GPU SẴN":
#   bash scripts/train_2b/run_2b_bac2.sh
set -u
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=2
OUT=results/train_2b/train/2b-Qwen3-1.7B-s2
S1=results/train_2b/train/2b-Qwen3-1.7B-s1
TRAIN=results/train_2b/train_2b_v3.jsonl
VAL=data/val_v0.jsonl
FLAG=results/G0/logs/QUA_NHIET.flag
TEMP=results/G0/logs/temp.log
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
if [ -e "$OUT/adapter" ] || [ -e "$OUT/train_log.jsonl" ]; then log "ĐÃ CÓ $OUT/adapter hoặc train_log: không ghi đè, không train lần 2"; exit 2; fi
ARGS=(Qwen/Qwen3-1.7B $SEED $OUT --lr 2e-4 --epochs 2 --micro_batch 8 --effective_batch 32 --warmup_frac 0.03 --loss full_vocab
      --monitor_frac 0.0645 --monitor_rows 200 --gate_step_frac 0.129 --gate_letter_mass 0.5
      --train_file $TRAIN --val_file $VAL --save_fracs 0.25,0.5,0.75)
.venv/bin/python - "$OUT" "$TRAIN" "$VAL" "${ARGS[*]}" "$S1" <<'EOF'
import hashlib, json, sys, time
out, train, val, args, s1 = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m1 = json.load(open(f"{s1}/manifest_train.json", encoding="utf-8"))
prep = json.load(open("results/train_2b/train/prep_bac1.json", encoding="utf-8"))
assert prep["train_sha256"] == sha(train) and prep["so_dong_state_gt_2048"] == 0, "trần state: prep_bac1 không khớp file train"
m = {"luot": "BẬC 2 của 2b: 2b-s2 (train đúng 1 lần, seed 2)", "cho_phep": "người dùng 29/10 khoảng 01:20 ('được, chạy 2b-s2 đi, nhớ giữ nhiệt độ dưới ngưỡng'), results/BRIEF_2b_bac2_train.md",
     "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lenh": "scripts/train_lora.py " + args, "chia_batch": "mb8 (như s1; không đo VRAM lại theo brief)",
     "lenh_s1": m1["lenh"], "khac_s1": "chỉ seed (1 → 2) và thư mục đầu ra",
     "train_file": train, "train_sha256": sha(train), "val_file": val, "val_sha256": sha(val),
     "train_lora_py_sha256": sha("scripts/train_lora.py"), "common_py_sha256": sha("scripts/common.py"),
     "prompt_frames_sha256": sha("config/prompt_frames.json"), "models_json": json.load(open("config/models.json"))["Qwen/Qwen3-1.7B"],
     "tran_state": {"nguon": "results/train_2b/train/prep_bac1.json (cùng sha file train)", **prep},
     "ghi_chu": "strip_p 0, val_v0, cổng letter_mass cùng tỉ lệ bước như s1/C; KHÔNG chạy gì trên test; không C-s2"}
for k in ("train_lora_py_sha256", "common_py_sha256", "prompt_frames_sha256", "val_sha256"):
    assert m[k] == m1[k], f"{k} khác s1"
assert m["lenh"].replace(" 2 results/train_2b/train/2b-Qwen3-1.7B-s2 ", " 1 results/train_2b/train/2b-Qwen3-1.7B-s1 ") == m1["lenh"], "lệnh khác s1 ngoài seed/thư mục"
open(f"{out}/manifest_train.json", "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
print("manifest ok", m["train_sha256"][:12])
EOF
[ $? -eq 0 ] || { log "MANIFEST/KIỂM KHÔNG KHỚP s1: DỪNG"; exit 2; }
gpu_idle "train 2b-s2"
LEFT=$(( BUDGET - $(used) - 2400 ))
T0=$(date -Is)
log "BẮT ĐẦU train 2b-s2 (mb8), giới hạn ${LEFT}s"
timed train_2b-s2 timeout --signal=TERM $LEFT .venv/bin/python -u -X faulthandler scripts/train_lora.py "${ARGS[@]}" > $OUT/stdout.txt 2>&1
rc=$?
T1=$(date -Is)
if [ -f $OUT/summary.json ]; then
  .venv/bin/python - "$OUT/summary.json" "$TEMP" "$T0" "$T1" <<'EOF'
import json, sys
f, temp, t0, t1 = sys.argv[1:]
v = [int(l.split()[1]) for l in open(temp) if len(l.split()) == 2 and t0[:19] <= l.split()[0][:19] <= t1[:19]]
s = json.load(open(f, encoding="utf-8"))
s["nhiet_do_toi_da_c"] = {"gia_tri": max(v) if v else None, "so_lan_do": len(v), "khung": [t0, t1], "nguon": temp, "ghi_boi": "run_2b_bac2.sh sau train"}
open(f, "w", encoding="utf-8").write(json.dumps(s, ensure_ascii=False, indent=2))
print("nhiet do toi da", s["nhiet_do_toi_da_c"]["gia_tri"])
EOF
fi
if [ $rc -ne 0 ]; then log "TRAIN_FAIL rc=$rc (3 = cổng letter_mass trượt; 124 = hết ngân sách): DỪNG, không chấm"; exit $rc; fi
L=results/probe_v1/eval/logs
left() { [ $(( BUDGET - $(used) )) -gt 300 ] || { log "HẾT NGÂN SÁCH GPU trước $1: DỪNG"; exit 6; }; }
left score_2b-s2
timed score_2b-s2 .venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter $OUT/adapter --name 2b-s2 --files dev,p2,p3,p4,pa,pb,p1b,p5 > $L/2b-s2.txt 2>&1
E=results/train_2b/eval
left proxy_2b-s2
timed proxy_2b-s2 .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $E/proxy --rows data/proxy_v1/proxy_v1.jsonl --name 2b-s2 --adapter $OUT/adapter > $E/logs/proxy_2b-s2.txt 2>&1
left valkc_2b-s2
timed valkc_2b-s2 .venv/bin/python -u scripts/eval_base.py Qwen/Qwen3-1.7B $E/val_kc_v2 --rows $E/_rows/val_kc_v2_eval.jsonl --name 2b-s2 --adapter $OUT/adapter > $E/logs/valkc_2b-s2.txt 2>&1
log "BAC2_DONE (GPU cộng dồn $(used)s)"
