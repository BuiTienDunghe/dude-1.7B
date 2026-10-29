"""Chia dev_v1_1 thành dòng dùng lại preds cũ và dòng phải chấm lại.
- Chấm lại: mọi dòng MASSIVE (có dấu, không dấu, tiếng Anh) và mọi dòng ViFactCheck/ViWikiFC, cùng bất kỳ dòng nào khác nội dung với dev_v1.
- Dùng lại: dòng Belebele/SIB-200/XCOPA mà nội dung (JSON từng dòng) trùng từng byte với dev_v1, kể cả dòng không dấu.
Ra: results/dev_v1_1/rows/{rescore.jsonl, unchanged.jsonl, split.json}
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS  # noqa: E402
old = {json.loads(l)["id"]: l.rstrip("\n") for l in open(DATA / "bench_v1/dev_v1.jsonl", encoding="utf-8")}
new = [l.rstrip("\n") for l in open(DATA / "bench_v1_1/dev_v1_1.jsonl", encoding="utf-8")]
O = RESULTS / "dev_v1_1/rows"
resc, keep, st = [], [], {"rescore": {}, "unchanged": {}}
for l in new:
    r = json.loads(l)
    same = old.get(r["id"]) == l
    if r["source"] in ("massive", "vifactcheck", "viwikifc") or not same:
        assert r["source"] in ("massive", "vifactcheck", "viwikifc"), ("dòng đổi ngoài nguồn dự kiến", r["id"])
        resc.append(l); st["rescore"][r["source"]] = st["rescore"].get(r["source"], 0) + 1
    else:
        keep.append(l); st["unchanged"][r["source"]] = st["unchanged"].get(r["source"], 0) + 1
(O / "rescore.jsonl").write_text("\n".join(resc) + "\n", encoding="utf-8")
(O / "unchanged.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
st["massive_rows_identical_to_v1_0_but_rescored"] = sum(1 for l in resc if json.loads(l)["source"] == "massive" and old.get(json.loads(l)["id"]) == l)
(O / "split.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
print(json.dumps(st))
