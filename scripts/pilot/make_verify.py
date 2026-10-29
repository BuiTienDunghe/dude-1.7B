"""Dựng lô soi lại (lượt 2) từ kết quả quét out_{mlqa,gpiqa}.jsonl: mọi câu verdict khác OK. Ra audit/verify_in.jsonl."""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "results" / "bench_candidates" / "pilot"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = {r["id"]: r for r in read(P / "pilot_rows.jsonl")}
out, n = [], {}
for src in ("mlqa", "gpiqa"):
    scan = read(P / "audit" / f"out_{src}.jsonl")
    ids = {r["pair_id"] for r in rows.values() if r["source"] == src}
    assert {x["audit_id"] for x in scan} == ids and len(scan) == len(ids), (src, len(scan))
    flagged = [x for x in scan if x["verdict"] != "OK"]
    n[src] = len(flagged)
    for x in flagged:
        side = lambda r: {"state": r["state"], "question": r["question"], "options": r["options"], "gold_index": r["target"]}
        out.append({"audit_id": x["audit_id"], "source": src, "vi": side(rows[x["audit_id"] + ":vi"]), "en": side(rows[x["audit_id"] + ":en"]),
                    "phat_hien_lan_1": x})
with open(P / "audit" / "verify_in.jsonl", "w", encoding="utf-8") as f:
    for u in out:
        f.write(json.dumps(u, ensure_ascii=False) + "\n")
print(json.dumps(n))
