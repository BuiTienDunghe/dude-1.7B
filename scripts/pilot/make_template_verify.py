"""Ghép templates_in + templates_out (lượt 1) -> templates_verify_in.jsonl cho lượt soi lại."""
import json
from pathlib import Path
A = Path(__file__).resolve().parents[2] / "results/bench_candidates/pilot2/audit"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
inp = read(A / "templates_in.jsonl"); out = {x["khoa"]: x for x in read(A / "templates_out.jsonl")}
assert set(out) == {x["khoa"] for x in inp} and len(out) == len(inp) == 65, (len(out), len(inp))
with open(A / "templates_verify_in.jsonl", "w", encoding="utf-8") as f:
    for x in inp:
        f.write(json.dumps({**x, "lan_1": out[x["khoa"]]}, ensure_ascii=False) + "\n")
print(json.dumps({"n": len(inp), "lan_1": {k: sum(1 for v in out.values() if v["ket_luan"] == k) for k in ("OK", "SUA")}}))
