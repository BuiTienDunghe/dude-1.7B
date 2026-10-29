"""Chấm MỘT model trên dev 1.3 và các phép thử probe_v1 (P1-P5). VIẾT SẴN, CHƯA CHẠY. Chỉ dev; không bao giờ đọc test; không train.
Không tự chấm: chỉ gọi đúng các đường chấm đã có của dự án, batch 1:
  M0/M1/M2 (Qwen3-1.7B, có hoặc không LoRA): scripts/eval_base.py --rows <file> --name <tên> [--adapter DIR]  (.venv; lượt chính batch 1)
  decider-2b                                 : scripts/bench/eval_decider_v1.py <file> <out>               (.venv; ra decider-2b-text và -keyed)
  quyet-medium                               : scripts/bench_v1_3/score_rows_ext.py quyet-medium <file> <out> (.venv-baselines, HF_HUB_OFFLINE=1)
Chạy trong WSL từ gốc repo:
  .venv/bin/python scripts/probe_v1/score_probe.py <model> [--adapter DIR] [--name TÊN] [--files dev,p2,p3,p4,pa,pb,p1b,p5] [--dry]
  <model>: M0 | M1 | M2 | decider-2b | quyet-medium. M1 mặc định dùng adapter của model dự án (results/ft_loss/train/Qwen3-1.7B-s1/adapter);
  M2 bắt buộc --adapter (ví dụ adapter_step<N> của train_lora.py --save_fracs để vẽ đường xu hướng).
  p2 = dev đảo thứ tự lựa chọn (eval_base --reverse; chỉ model M). --dry: in lệnh, không chạy.
Kiểm trước khi chạy: sha dev 1.3; sha từng file phép thử khớp manifest_probe_v1.json (file đã đóng băng); tên file và id không có "test".
Phép (theo quyết định 26/10): dev, p2 (đảo), p3, p4, pa (lựa chọn bỏ dấu), pb (nhãn tiếng Anh) không cần duyệt; p1b (P1′), p5 chỉ có khi người dùng
  đã duyệt chữ (build_p1_p5.py). pa câu noul với decider / Quyet: hai model này trả lời noul bằng đường có/không riêng, không nhìn chữ lựa chọn,
  nên đổi chữ Có/Không không có nghĩa; script bỏ các dòng đó khi chấm R (ghi trong run log).
Ra: results/probe_v1/eval/<phép>/<tên>/preds.jsonl (Quyet: .../<phép>/normal/<tên>/preds.jsonl) và results/probe_v1/eval/run_log.jsonl."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PR = ROOT / "results/probe_v1"
DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
FILES = {"dev": DEV, "p1b": PR / "p1b_doi_chu_nhan.jsonl", "p2": DEV, "p3": PR / "p3_bo_van_ban.jsonl", "p4": PR / "p4_van_ban_khong_lien_quan.jsonl",
         "p5": PR / "p5_doi_cau_hoi.jsonl", "pa": PR / "pa_lua_chon_khong_dau.jsonl", "pb": PR / "pb_nhan_tieng_anh.jsonl"}
NOUL_BO_CHO_R = ("pa",)   # phép chỉ đổi chữ lựa chọn: câu noul của decider / Quyet không nhìn chữ lựa chọn nên bỏ
M_ADAPTER = {"M0": None, "M1": ROOT / "results/ft_loss/train/Qwen3-1.7B-s1/adapter"}
DEFAULT_NAME = {"M0": "Qwen3-1.7B", "M1": "Qwen3-1.7B-fullvocab-s1 (dự án)"}

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["M0", "M1", "M2", "decider-2b", "quyet-medium"])
ap.add_argument("--adapter", type=Path)
ap.add_argument("--name")
ap.add_argument("--files", default="dev,p2,p3,p4,pa,pb")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(DEV) == DEV_SHA, "dev 1.3 sai sha"
man = json.loads((PR / "manifest_probe_v1.json").read_text(encoding="utf-8"))
frozen = {**man.get("files", {}), **man.get("P1_P5", {}).get("files", {}), **man.get("Pa_Pb", {}).get("files", {})}
is_m = a.model in ("M0", "M1", "M2")
adapter = a.adapter or M_ADAPTER.get(a.model)
if a.model == "M2":
    assert a.adapter, "M2 cần --adapter"
name = a.name or DEFAULT_NAME.get(a.model) or (f"M2-{a.adapter.parent.name}-{a.adapter.name}" if a.model == "M2" else a.model)
env = {**os.environ, "PYTHONHASHSEED": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1", "TRANSFORMERS_VERBOSITY": "error",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
tmp = PR / "eval" / "_rows"
log = PR / "eval" / "run_log.jsonl"
loi = []
for key in [k.strip() for k in a.files.split(",") if k.strip()]:
    f = FILES[key]
    if key == "p2" and not is_m:
        print("bỏ p2: chỉ chấm cho model M (decider / Quyet đã có hai thứ tự ở bước E)")
        continue
    if not f.exists():
        print(f"bỏ {key}: chưa có file {f.name} (P1′/P5 chờ người dùng duyệt chữ?)")
        continue
    assert "test" not in f.name.lower()
    if f != DEV:
        assert frozen.get(f.name) == sha(f), f"{f.name} không khớp sha trong manifest_probe_v1.json"
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    assert not any(":test:" in r["id"] for r in rows)
    bo = 0
    if key in NOUL_BO_CHO_R and not is_m:
        keep = [r for r in rows if r["kind"] != "noul"]
        bo = len(rows) - len(keep)
        tmp.mkdir(parents=True, exist_ok=True)
        f = tmp / f"{key}_khong_noul.jsonl"
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    out = PR / "eval" / key
    if is_m:
        cmd = [".venv/bin/python", "-u", "scripts/eval_base.py", "Qwen/Qwen3-1.7B", str(out), "--rows", str(f), "--name", name]
        if adapter:
            cmd += ["--adapter", str(adapter)]
        if key == "p2":
            cmd += ["--reverse"]
    elif a.model == "decider-2b":
        cmd = [".venv/bin/python", "-u", "scripts/bench/eval_decider_v1.py", str(f), str(out)]
    else:
        cmd = [".venv-baselines/bin/python", "-u", "scripts/bench_v1_3/score_rows_ext.py", "quyet-medium", str(f), str(out)]
    print(" ".join(cmd))
    if a.dry:
        continue
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    rec = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": a.model, "name": name, "adapter": str(adapter) if adapter else None,
           "phep": key, "file": str(f.relative_to(ROOT)), "rows_sha256": sha(f), "so_dong": len(rows), "bo_dong_noul_cho_R": bo,
           "exit": rc, "giay": round(time.time() - t0, 1)}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rc != 0:                                   # một phép lỗi (vd. state rỗng của P3 với model R) không chặn các phép sau
        loi.append(f"{key}: exit {rc}")
if loi:
    sys.exit("; ".join(loi))
