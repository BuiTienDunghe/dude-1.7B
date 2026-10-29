"""So preds decider-2b trên dev_v1 (dòng song song, có dấu) với preds của bước baseline trên cùng id. In tối đa 5 ví dụ khác."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, read_jsonl  # noqa: E402

rows = {r["id"]: r for r in read_jsonl(DATA / "bench_v1/dev_v1.jsonl")}
for m in ("text", "keyed"):
    base = {p["id"]: p for p in read_jsonl(RESULTS / f"baselines/eval/normal/decider-2b-{m}/preds.jsonl")}
    p = RESULTS / f"dev_v1/eval/decider-2b-{m}/preds.jsonl"
    if not p.exists():
        print(m, "no dev_v1 preds yet"); continue
    new = {x["id"]: x for x in read_jsonl(p)}
    ids = [i for i in base if i in new]
    dmax = [max(abs(a - b) for a, b in zip(base[i]["probs"], new[i]["probs"])) for i in ids]
    flips = [i for i in ids if base[i]["pred"] != new[i]["pred"]]
    print(f"{m}: common {len(ids)} | identical probs {sum(d == 0 for d in dmax)} | max|Δp| {max(dmax):.4f} | pred flips {len(flips)}")
    print("  flips by source/lang:", Counter((rows[i]["source"], rows[i]["lang"]) for i in flips).most_common())
    print("  mapping field in new:", Counter(new[i]["mapping"] for i in ids).most_common(3), "| k equal:", all(base[i]["k"] == new[i]["k"] for i in ids),
          "| target equal:", all(base[i]["target"] == new[i]["target"] for i in ids))
    big = sorted(ids, key=lambda i: -max(abs(a - b) for a, b in zip(base[i]["probs"], new[i]["probs"])))[:5]
    for i in big:
        print("  ", i[:60], "base", [round(x, 3) for x in base[i]["probs"]], "new", [round(x, 3) for x in new[i]["probs"]],
              "n_ctx", base[i].get("n_ctx_tokens"), new[i].get("n_ctx_tokens"))
