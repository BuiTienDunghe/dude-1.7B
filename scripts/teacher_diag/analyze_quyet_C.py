"""Phân tích (B) của results/BRIEF_chan_doan_teacher.md trên results/teacher_diag/quyet_C.jsonl. Không chạy model. Thư viện chuẩn.
Chỉ số, tính theo toàn bộ, theo text_lang, theo từng dataset có >= 100 dòng, theo nhóm dạng bài phần VI:
  A   = tỉ lệ argmax teacher trùng gold_argmax (target_argmax của C);
  S   = trung bình (1 − p_teacher(gold)) trên các dòng TRÙNG (đích 0,6·teacher + 0,4·nhãn lệch nhãn 0,6·S);
  D90 = tỉ lệ dòng teacher BẤT ĐỒNG với nhãn mà teacher tự tin > 0,9;
  entropy chuẩn hóa trung bình; số dòng bị loại (cắt / từ chối).
Cách đọc (chốt trước trong brief, trên phần VI): S <= 0,05 và A >= 0,95 → "làm mềm nhãn"; S >= 0,10 hoặc A <= 0,90 → "teacher mang thông tin khác
nhãn đáng kể"; còn lại → "chưa rõ". Chỉ báo số và áp bảng, không quyết bước tiếp.
Mẫu soát mù: 60 dòng VI bất đồng tự tin (p > 0,9), phân tầng theo nhóm dạng bài (chia đều, nhóm thiếu thì dồn cho nhóm khác), seed 20261028,
cột tách riêng nhan_goc / dap_an_teacher để điều phối che khi soát. Ra results/teacher_diag/{chan_doan.json, mau_soat_mu.jsonl}.
Dùng: python scripts/teacher_diag/analyze_quyet_C.py"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/teacher_diag"
R = [json.loads(l) for l in open(D / "quyet_C.jsonl", encoding="utf-8") if l.strip()]
C = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / "results/train_v1/components/C.jsonl", encoding="utf-8"))}
LETTERS = "ABCDEFGHIJ"
GROUPS = [("doc_hieu", ("vsmrc", "race", "quail", "cosmos_qa", "dream", "exams")),
          ("kiem_chung_nli", ("vitaminc", "nli_fever", "health_fact", "vianli", "vinli")),
          ("y_dinh", ("tourism", "multidogo", "banking77")),
          ("nhan_qua", ("e-care", "headline_cause")),
          ("chu_de", ("vntc", "vsfc/topic"))]


def group(r):
    s = f"{r.get('source') or ''} {r.get('dataset') or ''}".lower()
    for g, keys in GROUPS:
        if any(k in s for k in keys):
            return g
    return "con_lai"


def metrics(rs):
    ok = [r for r in rs if r.get("probs") is not None]
    ag = [r for r in ok if r["agree"]]
    if not ok:
        return {"n": len(rs), "bi_loai": len(rs)}
    return {"n": len(rs), "bi_loai": len(rs) - len(ok), "cat": sum(r["truncated"] for r in rs), "tu_choi": sum(r["rejected"] for r in rs),
            "A": round(len(ag) / len(ok), 4), "S": round(sum(1 - r["p_gold"] for r in ag) / max(1, len(ag)), 4),
            "D90": round(sum((not r["agree"]) and r["p_max"] > 0.9 for r in ok) / len(ok), 4),
            "entropy_norm_tb": round(sum(r["entropy_norm"] for r in ok) / len(ok), 4)}


def reading(m):
    if m.get("S") is None:
        return "không đủ dữ liệu"
    if m["S"] <= 0.05 and m["A"] >= 0.95:
        return "Chưng cất trên C gần như chỉ là làm mềm nhãn → đề xuất bỏ 2a, đi thẳng 2b"
    if m["S"] >= 0.10 or m["A"] <= 0.90:
        return "Teacher mang thông tin khác nhãn đáng kể → 2a đáng train" + ("; A thấp: phải soát mù trước" if m["A"] <= 0.90 else "")
    return "Chưa rõ: điều phối cùng Fable cân nhắc, người dùng quyết"


res = {"so_dong": len(R), "toan_bo": metrics(R), "theo_text_lang": {}, "theo_dataset_ge_100": {}, "VI_theo_nhom": {}, "theo_kind": {}}
by = defaultdict(list)
for r in R:
    by[r.get("text_lang")].append(r)
res["theo_text_lang"] = {k: metrics(v) for k, v in sorted(by.items(), key=lambda x: str(x[0]))}
by = defaultdict(list)
for r in R:
    by[r.get("source") or r.get("dataset")].append(r)
res["theo_dataset_ge_100"] = {k: metrics(v) for k, v in sorted(by.items()) if len(v) >= 100}
vi = [r for r in R if r.get("text_lang") == "vi"]
by = defaultdict(list)
for r in vi:
    by[group(r)].append(r)
res["VI_theo_nhom"] = {k: metrics(v) for k, v in sorted(by.items())}
res["theo_kind"] = {k: metrics([r for r in R if r["kind"] == k]) for k in ("choice", "noul")}
mvi = metrics(vi)
res["VI"] = mvi
res["cach_doc_VI"] = reading(mvi)

# mẫu soát mù
pool = defaultdict(list)
for r in vi:
    if r.get("probs") is not None and not r["agree"] and r["p_max"] > 0.9:
        pool[group(r)].append(r)
rng = random.Random(20261028)
for g in pool:
    pool[g].sort(key=lambda r: r["id"]); rng.shuffle(pool[g])
quota = {g: 0 for g in pool}
left = 60
while left > 0 and any(quota[g] < len(pool[g]) for g in pool):
    for g in sorted(pool):
        if left and quota[g] < len(pool[g]):
            quota[g] += 1; left -= 1
mau = []
for g in sorted(pool):
    for r in pool[g][:quota[g]]:
        c = C[r["id"]]
        mau.append({"ma": None, "id": r["id"], "nhom": g, "dataset": r.get("source") or r.get("dataset"), "kind": r["kind"],
                    "state": c["state"], "question": c["question"], "options": [f"{LETTERS[i]}. {o}" for i, o in enumerate(c["options"])],
                    "nhan_goc": f"{LETTERS[r['gold_argmax']]}. {c['options'][r['gold_argmax']]}",
                    "dap_an_teacher": f"{LETTERS[r['pred']]}. {c['options'][r['pred']]}", "p_teacher": round(r["p_max"], 4)})
rng.shuffle(mau)
for i, m in enumerate(mau):
    m["ma"] = f"S{i + 1:03d}"
with open(D / "mau_soat_mu.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for m in mau:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
res["mau_soat_mu"] = {"so_dong": len(mau), "theo_nhom": dict(Counter(m["nhom"] for m in mau)), "kho_bat_dong_tu_tin_VI": {g: len(v) for g, v in pool.items()}}
(D / "chan_doan.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("so_dong", "toan_bo", "VI", "cach_doc_VI", "VI_theo_nhom", "mau_soat_mu")}, ensure_ascii=False, indent=1))
