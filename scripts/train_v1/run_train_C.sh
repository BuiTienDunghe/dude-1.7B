#!/usr/bin/env bash
# TRAIN THỬ bản C (người dùng cho phép 26/10, "Từ 1.7B gốc"; điều phối ghi PLAN_decisions). KHÔNG phải train chính thức (2 seed). Không đụng test.
# Công thức = M1 (results/ft_loss/train/Qwen3-1.7B-s1/summary.json): Qwen/Qwen3-1.7B @70d244cc, LoRA r16/α32/0,05 mọi lớp proj, lr 2e-4,
# batch hiệu dụng 32 (micro 8 × accum 4), warmup 3% + cosine, 2 epoch, seed 1, loss full_vocab, strip_p 0 (như M1), theo dõi letter_mass mỗi
# 125 bước, cổng 0,5 ở bước 250 (bước tuyệt đối như M1: 0,0645 × 1.938 = 125; 0,129 × 1.938 = 250). Chỉ đổi: --train_file C.jsonl;
# val giữ val_v0 (như M1, để val_loss so được); thêm --save_fracs 0.25,0.5,0.75 (bước 484, 969, 1454).
# Train xong (exit 0): chấm 4 mốc trên dev 1.3; mốc 100% thêm mọi phép thử probe_v1. Cổng trượt (exit 3) hay lỗi: DỪNG, không chấm.
# Chạy (WSL): bash scripts/train_v1/run_train_C.sh
set -u
cd /mnt/c/Users/dungbt06/Fine-tune
export PYTHONHASHSEED=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=results/train_v1/train/C-Qwen3-1.7B-s1
TRAIN=results/train_v1/components/C.jsonl
VAL=data/val_v0.jsonl
if [ -e "$OUT/adapter" ] || [ -e "$OUT/train_log.jsonl" ]; then echo "ĐÃ CÓ $OUT: không ghi đè"; exit 2; fi
mkdir -p $OUT
ARGS=(Qwen/Qwen3-1.7B 1 $OUT --lr 2e-4 --epochs 2 --micro_batch 8 --effective_batch 32 --warmup_frac 0.03 --loss full_vocab
      --monitor_frac 0.0645 --monitor_rows 200 --gate_step_frac 0.129 --gate_letter_mass 0.5
      --train_file $TRAIN --val_file $VAL --save_fracs 0.25,0.5,0.75)
.venv/bin/python - "$OUT" "$TRAIN" "$VAL" "${ARGS[*]}" <<'EOF'
import hashlib, json, sys, time, subprocess
out, train, val, args = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = {"luot": "TRAIN THỬ bản C (không phải train chính thức)", "cho_phep": "người dùng 26/10 ('Từ 1.7B gốc'), điều phối ghi PLAN_decisions",
     "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "lenh": "scripts/train_lora.py " + args,
     "train_file": train, "train_sha256": sha(train), "val_file": val, "val_sha256": sha(val),
     "train_lora_py_sha256": sha("scripts/train_lora.py"), "common_py_sha256": sha("scripts/common.py"),
     "prompt_frames_sha256": sha("config/prompt_frames.json"), "models_json": json.load(open("config/models.json"))["Qwen/Qwen3-1.7B"],
     "cong_thuc_tham_chieu": "results/ft_loss/train/Qwen3-1.7B-s1/summary.json (M1)",
     "ghi_chu": "strip_p 0 như M1; val_v0 như M1; cổng letter_mass ở bước 250 (tuyệt đối) như M1; KHÔNG chạy gì trên test"}
open(f"{out}/manifest_train.json", "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
print("manifest ok", m["train_sha256"][:12])
EOF
.venv/bin/python -u -X faulthandler scripts/train_lora.py "${ARGS[@]}" > $OUT/stdout.txt 2>&1
rc=$?
echo "TRAIN exit=$rc"
if [ $rc -ne 0 ]; then echo "TRAIN_FAIL rc=$rc: DỪNG, không chấm"; exit $rc; fi
L=results/probe_v1/eval/logs
for s in 484 969 1454; do
  .venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter $OUT/adapter_step$s --name C-s1-step$s --files dev > $L/C-s1-step$s.txt 2>&1
  echo "C step$s dev exit=$?"
done
.venv/bin/python -u scripts/probe_v1/score_probe.py M2 --adapter $OUT/adapter --name C-s1 --files dev,p2,p3,p4,pa,pb,p1b,p5 > $L/C-s1.txt 2>&1
echo "C 100% exit=$?"
echo TRAIN_C_DONE
