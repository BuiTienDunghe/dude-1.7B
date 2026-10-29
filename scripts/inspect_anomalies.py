"""Soi các điểm bất thường: câu có khối xác suất trên chữ cái thấp, và độ lệch vị trí chữ A theo ngôn ngữ."""
import sys
from collections import Counter
from pathlib import Path

from common import DATA, read_jsonl

run = Path(sys.argv[1])
rows = {r["id"]: r for r in read_jsonl(DATA / "dev_v0.jsonl")}
for m in ["Qwen3-0.6B", "Qwen3-1.7B"]:
    P = read_jsonl(run / m / "preds.jsonl")
    low = [p for p in P if p["letter_mass"] < 0.5]
    print(f"== {m}: rows with letter_mass<0.5: {len(low)}  by src/lang {Counter((p['source'], p['lang']) for p in low)}")
    print("   top tokens there:", Counter(p["top_token"] for p in low).most_common(8))
    for p in sorted(low, key=lambda p: p["letter_mass"])[:4]:
        r = rows[p["id"]]
        print(f"   {p['id']} mass={p['letter_mass']:.3f} top={p['top_token']!r} state={r['state'][:90]!r}")
    # khi đáp án đúng KHÔNG phải A, tỉ lệ model chọn A
    for src in ["belebele", "massive", "xcopa"]:
        for lang in ["en", "vi"]:
            S = [p for p in P if p["source"] == src and p["lang"] == lang and p["target"] != 0]
            print(f"   {src}/{lang}: P(pred=A | gold!=A) = {sum(p['pred'] == 0 for p in S) / len(S):.3f}  (chance ~{1 / S[0]['k']:.3f})")
