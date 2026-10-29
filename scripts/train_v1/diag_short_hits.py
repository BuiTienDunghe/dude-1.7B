"""Chẩn đoán (không ghi dữ liệu train): với các dòng train_v0 bị chỉ mục mở rộng bắt (results/train_v1/v0_leak.json), tìm đoạn của bench 1.3 / Belebele /
MASSIVE nằm trong văn bản của dòng đó, để phân biệt rò rỉ thật với cụm từ chung. Ra results/train_v1/diag_v0_hits.json."""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, read_jsonl  # noqa: E402
from decontam import normalize  # noqa: E402

OUT = RESULTS / "train_v1"
leak = json.load(open(OUT / "v0_leak.json", encoding="utf-8"))
units = {}
for f in ("dev_v1_3.jsonl", "test_v1.3.jsonl"):
    for r in read_jsonl(DATA / "bench_v1_3" / f):
        for t in [r["state"], r["question"]] + list(r["options"]):
            n = normalize(t)
            if 4 <= len(n.split()) <= 12:
                units.setdefault(n, f"bench13:{f[:4]}:{r['source']}")
for lang in ("eng_Latn", "vie_Latn"):
    df = pd.read_json(OUT / f"raw_refs/facebook__belebele/data/{lang}.jsonl", lines=True)
    for c in ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"):
        for t in df[c]:
            n = normalize(t)
            if 4 <= len(n.split()) <= 12:
                units.setdefault(n, f"belebele:{c}")
rows = {r["id"]: r for r in read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl")}
out, phr = [], Counter()
for h in leak["hits"]:
    if h["kind"] != "contain_short":
        out.append({**h, "doan": None}); continue
    r = rows[h["id"]]
    txt = " " + normalize(" ".join([r["state"], r["question"] or ""] + r["options"])) + " "
    m = [(u, t) for u, t in units.items() if f" {u} " in txt]
    m.sort(key=lambda x: -len(x[0]))
    out.append({**h, "doan": m[0][0] if m else None, "doan_tu": m[0][1] if m else None})
    if m:
        phr[m[0][0]] += 1
(OUT / "diag_v0_hits.json").write_text(json.dumps({"hits": out, "doan_hay_gap": phr.most_common(40)}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(phr.most_common(40), ensure_ascii=False))
