"""Ghép preds dev_v1_1 = preds cũ (dev_v1) cho dòng không đổi + preds mới (chấm lại) cho dòng chấm lại, theo thứ tự dev_v1_1.
Dùng: merge_preds.py <preds_cũ> <preds_mới_chấm_lại> <ra>. Kiểm: dòng dùng lại phải trùng từng byte nội dung với dev_v1; đủ id."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS  # noqa: E402
old_p = {json.loads(l)["id"]: l.rstrip("\n") for l in open(sys.argv[1], encoding="utf-8")}
new_p = {json.loads(l)["id"]: l.rstrip("\n") for l in open(sys.argv[2], encoding="utf-8")}
keep = {json.loads(l)["id"] for l in open(RESULTS / "dev_v1_1/rows/unchanged.jsonl", encoding="utf-8")}
resc = {json.loads(l)["id"] for l in open(RESULTS / "dev_v1_1/rows/rescore.jsonl", encoding="utf-8")}
assert set(new_p) == resc, "preds chấm lại không khớp tập dòng chấm lại"
out = []
for l in open(DATA / "bench_v1_1/dev_v1_1.jsonl", encoding="utf-8"):
    i = json.loads(l)["id"]
    out.append(old_p[i] if i in keep else new_p[i])
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(json.dumps({"out": sys.argv[3], "rows": len(out), "reused": len(keep), "rescored": len(resc)}))
