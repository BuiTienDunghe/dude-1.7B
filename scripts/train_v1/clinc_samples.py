"""Lấy 5 câu mẫu thật mỗi nhãn cho 9 nhãn clinc cần rà (chỉ đọc; tasksource @ revision train_v0, nguồn clinc_oos/plus, dòng choice có đáp án = nhãn).
Seed 20261020. Ra: results/train_v1/translated_sonnet/full_prep/clinc_mau_9_nhan.json."""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, load_json  # noqa: E402

LAB = ["gas", "taxes", "travel_alert", "vaccines", "w2", "direct_deposit", "user_name", "international_visa", "reminder"]
man = load_json(DATA / "manifest_train_v0.json")
TS = (man["sources"]["tasksource"]["repo"], man["sources"]["tasksource"]["revision"])
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
idx = [k for k, s in enumerate(ts["source"]) if s == "clinc_oos/plus"]
pool = {l: [] for l in LAB}
for k in idx:
    x = ts[k]
    if x["kind"] != "choice" or not x["options"]:
        continue
    lab = x["options"][max(range(len(x["target"])), key=lambda j: x["target"][j])]
    if lab in pool and x["state"] not in pool[lab]:
        pool[lab].append(x["state"])
rng = random.Random(20261020)
out = {"nguon": f"{TS[0]}@{TS[1]} clinc_oos/plus (split train)", "nhan": {l: rng.sample(v, min(5, len(v))) for l, v in pool.items()},
       "so_cau_co": {l: len(v) for l, v in pool.items()}}
(RESULTS / "train_v1/translated_sonnet/full_prep/clinc_mau_9_nhan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
