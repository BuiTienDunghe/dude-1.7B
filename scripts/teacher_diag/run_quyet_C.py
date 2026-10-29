"""(B) của results/BRIEF_chan_doan_teacher.md: Quyet-Medium (adapter đã kiểm ở bước E, scripts/baselines_v1_2/adapters.py, quyet-medium-text) chấm các dòng
choice / noul của results/train_v1/components/C.jsonl, ĐÚNG văn bản trong file (bản có dấu, không chạy trên bản bỏ dấu). KHÔNG train; không đụng dev/test.
Chạy bằng .venv-baselines, offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1), batch 1 (adapter chấm từng câu). probs theo đúng thứ tự options của
dòng C; câu noul ánh xạ Có/Không như adapter (noul_map). Ghi tăng dần, chạy lại thì tiếp từ id chưa có (dừng vì nhiệt độ cũng không mất kết quả).
Ra results/teacher_diag/quyet_C.jsonl (+ manifest_quyet_C.json khi xong).
Dùng: .venv-baselines/bin/python -u scripts/teacher_diag/run_quyet_C.py"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines_v1_2"))
import adapters as A  # noqa: E402

assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "phải chạy offline"
C = ROOT / "results/train_v1/components/C.jsonl"
C_SHA = "005ca3d547aa2f9e2f356a8f75eb7f3f17addba1cc4d74e3a7b44a7b39c86e89"
OUT = ROOT / "results/teacher_diag"; OUT.mkdir(parents=True, exist_ok=True)
F = OUT / "quyet_C.jsonl"
h = hashlib.sha256(C.read_bytes()).hexdigest()
assert h == C_SHA, h
rows = [r for r in (json.loads(l) for l in open(C, encoding="utf-8")) if r["kind"] in ("choice", "noul")]
done = set()
if F.exists():
    done = {json.loads(l)["id"] for l in open(F, encoding="utf-8") if l.strip()}
repo = "chinhnc/Quyet-1.0-Medium"
snap = Path(json.load(open(ROOT / "results/baselines_v1_2/downloads.json", encoding="utf-8"))[repo]["snapshot"])
ad = A.Quyet(snap, "text")
t0 = time.time(); n_new = 0
with open(F, "a", encoding="utf-8", newline="\n") as fo:
    for r in rows:
        if r["id"] in done:
            continue
        s = ad.score(r)
        k = len(r["options"]); p = s.get("probs")
        gold = int(r["target_argmax"])
        rec = {"id": r["id"], "dataset": r.get("dataset"), "source": r.get("source"), "text_lang": r.get("text_lang"), "kind": r["kind"], "k": k,
               "gold_argmax": gold, "gold_target": r["target"], "probs": p, "truncated": bool(s.get("truncated")), "rejected": bool(s.get("rejected")),
               "note": s.get("note"), "n_ctx_tokens": s.get("n_ctx_tokens")}
        if p is not None:
            pred = max(range(k), key=lambda j: p[j])
            ent = -sum(x * math.log(x) for x in p if x > 0) / math.log(k) if k > 1 else 0.0
            rec.update({"pred": pred, "agree": pred == gold, "p_gold": p[gold], "p_max": p[pred], "entropy_norm": ent})
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
        n_new += 1
        if n_new % 1000 == 0:
            print(f"{len(done) + n_new}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
man = {"dau_vao": str(C.relative_to(ROOT)), "sha256": C_SHA, "so_dong_choice_noul": len(rows), "teacher": repo, "snapshot": str(snap),
       "revision": snap.name, "adapter": "scripts/baselines_v1_2/adapters.py Quyet, mapping text (đã kiểm ở bước E)", "batch": 1,
       "giay_lan_chay_nay": time.time() - t0, "so_dong_moi_lan_nay": n_new, "ket_qua": str(F.relative_to(ROOT)), "sha256_ket_qua": hashlib.sha256(F.read_bytes()).hexdigest()}
(OUT / "manifest_quyet_C.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("XONG", json.dumps(man, ensure_ascii=False))
