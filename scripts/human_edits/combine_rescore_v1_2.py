"""Gộp preds chấm lại của dev 1.2 thử: dòng đổi byte so với 1.1 lấy preds của lần chấm r2 (nếu dòng đó đổi so với bản thử trước) hoặc của lần
chấm r1 (bản thử trước, đã lưu ở prev_r1/eval_rescore). Ghi ra eval_rescore/<model>/preds.jsonl để report_trial_v1_2.py dùng như cũ."""
import json
_PROV = ("vi_provenance", "vi_revised_fields", "gold_provenance")
def _core(l):
    """Dòng JSONL bỏ các trường provenance (để so với 1.1 / bản thử trước)."""
    d = json.loads(l)
    return json.dumps({k: v for k, v in d.items() if k not in _PROV}, ensure_ascii=False)
from pathlib import Path
T = Path(__file__).resolve().parents[2] / "results/bench_v1_2_trial"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
SUBS = ["decider/decider-2b-text", "decider/decider-2b-keyed", "ftd/decider-2b-text", "ftd/decider-2b-keyed", "control", "base", "strip"]
old = {json.loads(l)["id"]: l for l in open(T.parents[1] / "data/bench_v1_1/dev_v1_1.jsonl", encoding="utf-8") if l.strip()}
new = [l for l in open(T / "dev_v1_2_trial.jsonl", encoding="utf-8") if l.strip()]
need = [json.loads(l)["id"] for l in new if _core(old[json.loads(l)["id"]]) != _core(l)]
r2ids = {x["id"] for x in read(T / "rescore/rescore_dev_r2.jsonl")}
out = {}
for s in SUBS:
    p1 = {x["id"]: x for x in read(T / "prev_r1/eval_rescore" / s / "preds.jsonl")}
    p2 = {x["id"]: x for x in read(T / "eval_rescore_r2" / s / "preds.jsonl")}
    assert r2ids <= set(p2), s
    rows = [p2[i] if i in r2ids else p1[i] for i in need]
    (T / "eval_rescore" / s).mkdir(parents=True, exist_ok=True)
    with open(T / "eval_rescore" / s / "preds.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    out[s] = {"r2": sum(i in r2ids for i in need), "r1": sum(i not in r2ids for i in need)}
print(json.dumps({"need": len(need), "per_model": out["control"]}))
