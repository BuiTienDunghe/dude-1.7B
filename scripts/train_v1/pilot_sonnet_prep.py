"""Dịch thử 200 dòng phần B bằng subagent Sonnet (người dùng duyệt 19/10): CHUẨN BỊ (không dịch).
- Mẫu: 200 dòng của to_translate/part_b_to_translate.jsonl, seed 20261019, 40 dòng mỗi dạng bài (dạng bài lấy từ part_b_meta.jsonl).
- Chia 5 lô 40 dòng (trộn dạng bài theo thứ tự id đã xáo) cho người dịch: chỉ id/state/question/options, không nhãn/target.
- Đầu vào bảng thuật ngữ: câu hỏi và lựa chọn của mẫu mà (a) lặp >= 5 lần trong toàn bộ file dịch (chuoi_lap_lai.json) hoặc (b) có dạng mã (chữ thường,
  số, gạch dưới, không dấu cách) hoặc (c) nhãn cố định toàn chữ hoa; kèm số lần xuất hiện.
Ra results/train_v1/translated_sonnet/pilot/: ids.json, src_200.jsonl, lo/lo_{k}_en.jsonl, thuat_ngu_dau_vao.json. Chỉ thư viện chuẩn."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/train_v1"
P = T / "translated_sonnet/pilot"
SEED, PER, LOT = 20261019, 40, 40
src = [json.loads(l) for l in open(T / "to_translate/part_b_to_translate.jsonl", encoding="utf-8") if l.strip()]
meta = {json.loads(l)["id"]: json.loads(l) for l in open(T / "part_b_meta.jsonl", encoding="utf-8") if l.strip()}
by = defaultdict(list)
for r in src:
    by[meta[r["id"]]["dang_bai"]].append(r["id"])
rng = random.Random(SEED)
ids = {g: sorted(rng.sample(sorted(v), PER)) for g, v in sorted(by.items())}
allids = [i for g in sorted(ids) for i in ids[g]]
rng.shuffle(allids)
pos = {r["id"]: r for r in src}
sample = [pos[i] for i in allids]
(P / "lo").mkdir(parents=True, exist_ok=True)
(P / "ids.json").write_text(json.dumps({"seed": SEED, "theo_dang_bai": ids, "thu_tu": allids}, ensure_ascii=False, indent=1), encoding="utf-8")
with open(P / "src_200.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k in range(0, len(sample), LOT):
    with open(P / f"lo/lo_{k // LOT + 1}_en.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in sample[k:k + LOT]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep = json.load(open(T / "to_translate/chuoi_lap_lai.json", encoding="utf-8"))
full_q, full_o = Counter(r["question"] for r in src), Counter(o for r in src for o in r["options"])
code = re.compile(r"^[a-z0-9_]+$")
upper = re.compile(r"^[A-Z][A-Z _]+$")
terms = {}
for r in sample:
    for kind, t in [("cau_hoi", r["question"])] + [("lua_chon", o) for o in r["options"]]:
        n = (full_q if kind == "cau_hoi" else full_o)[t]
        why = [w for w, ok in (("lap_lai", n >= 5), ("dang_ma", bool(code.match(t))), ("nhan_co_dinh", bool(upper.match(t)))) if ok]
        if why and t not in terms:
            terms[t] = {"loai": kind, "so_lan_toan_file": n, "ly_do": why,
                        "nguon": sorted({meta[r["id"]]["source"] for r in sample if t == r["question"] or t in r["options"]})}
(P / "thuat_ngu_dau_vao.json").write_text(json.dumps(terms, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"dong": len(sample), "lo": (len(sample) + LOT - 1) // LOT, "theo_dang_bai": {g: len(v) for g, v in ids.items()}, "thuat_ngu": len(terms),
                  "thuat_ngu_theo_loai": dict(Counter(v["loai"] for v in terms.values()))}, ensure_ascii=False))
