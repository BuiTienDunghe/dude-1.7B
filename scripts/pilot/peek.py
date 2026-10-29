import csv, json, sys
import pyarrow.parquet as pq
R = "/mnt/c/Users/dungbt06/Fine-tune/results/bench_candidates/pilot/raw"
for f in ("vi.vi", "en.en"):
    t = pq.read_table(f"{R}/mlqa/{f}.validation.parquet").to_pylist()
    print(f, len(t), list(t[0].keys()))
    r = t[0]; print({k: (str(v)[:160]) for k, v in r.items()})
    print("contexts", len({x["context"] for x in t})); globals()[f.replace(".","_")]=t
vi={x["id"]:x for x in vi_vi}; en={x["id"]:x for x in en_en}; print("id overlap", len(set(vi)&set(en)))
rows = list(csv.DictReader(open(f"{R}/gpiqa/nonparallel_vie_latn.tsv", encoding="utf-8"), delimiter="\t"))
print("gpiqa", len(rows), list(rows[0].keys()))
from collections import Counter
print(Counter(r["label"] for r in rows), Counter(r["llm_used"] for r in rows), Counter(r.get("approx_cultural_score") for r in rows).most_common(6))
print({k: v[:80] for k, v in rows[0].items()})
