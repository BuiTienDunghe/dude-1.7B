"""Fit một temperature T trên val nội bộ (không dùng dev_v0).

Dùng: fit_temperature.py <val_preds.jsonl> <out.json>
val_preds.jsonl là đầu ra của eval_base.py --rows data/val_v0.jsonl (có logit tại k chữ cái theo thứ tự đã lưu).
T tối thiểu hoá cross-entropy trung bình giữa softmax(logits/T) và phân phối target (mềm) của val.
Áp lên dev: softmax(logits/T) trên k chữ cái = p^(1/T) / sum p^(1/T), nên không cần chấm lại.
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from common import DATA, read_jsonl

preds = read_jsonl(sys.argv[1])
val = {r["id"]: r for r in read_jsonl(DATA / "val_v0.jsonl")}
assert len(preds) == len(val)
items = [(np.array(p["logits"], dtype=np.float64), np.array(val[p["id"]]["target"])) for p in preds]
for (lg, t), p in zip(items, preds):
    assert len(lg) == len(t) == p["k"]


def nll(log_t):
    T = np.exp(log_t)
    return float(np.mean([-(t * log_softmax(lg / T)).sum() for lg, t in items]))


res = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-6})
T = float(np.exp(res.x))
out = {"T": T, "val_ce_T1": nll(0.0), "val_ce_T": nll(res.x), "n_val": len(items), "preds": sys.argv[1]}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print(json.dumps(out))
