"""Chứng minh recast_v0.ts_row / pl_row trùng code dựng train_v0: dựng lại MỌI dòng train_v0 + val_v0 từ nguồn (tasksource @ revision v0, PhoLaya @ revision v0)
và so từng trường (val: lựa chọn đã xáo một lần lúc dựng, nên so theo cặp lựa chọn-target). Ra: results/train_v1/verify_recast_v0.json."""
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_json, read_jsonl  # noqa: E402
from recast_v0 import pl_row, ts_row  # noqa: E402

man = load_json(DATA / "manifest_train_v0.json")
TS = (man["sources"]["tasksource"]["repo"], man["sources"]["tasksource"]["revision"])
PL = (man["sources"]["pholaya"]["repo"], man["sources"]["pholaya"]["revision"])
F = ["id", "source", "dataset", "lang", "text_lang", "kind", "state", "question", "license_use", "variant"]
rows = read_jsonl(DATA / "train_v0.jsonl") + read_jsonl(DATA / "val_v0.jsonl")
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
pos = {f"ts:{i}": k for k, i in enumerate(ts["id"])}
pl = {}
for ds in ("vinli", "vsfc", "visfd"):
    with open(hf_hub_download(PL[0], f"sources/{ds}/train.jsonl", repo_type="dataset", revision=PL[1]), encoding="utf-8") as f:
        for n, line in enumerate(f):
            pl[f"pl:{ds}:{n}"] = (json.loads(line), ds, n)
res = Counter()
for r in rows:
    new = ts_row(ts[pos[r["id"]]]) if r["dataset"] == "tasksource" else pl_row(*pl[r["id"]])
    same = all(new[k] == r[k] for k in F)
    if r["split"] == "train":
        same &= new["options"] == r["options"] and new["target"] == r["target"]
    else:
        same &= sorted(zip(new["options"], new["target"])) == sorted(zip(r["options"], r["target"]))
    res[f"{r['split']}:{r['dataset'].split('/')[0]}:{'trung' if same else 'KHAC'}"] += 1
out = {"rows": len(rows), "ket_qua": dict(res), "tat_ca_trung": all(k.endswith("trung") for k in res)}
(RESULTS / "train_v1" / "verify_recast_v0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
