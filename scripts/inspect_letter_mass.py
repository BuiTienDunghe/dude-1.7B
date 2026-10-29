"""Token đứng đầu (top_token) và letter_mass của từng điều kiện, để hiểu xác suất ngoài các chữ cái đi đâu."""
from collections import Counter

import numpy as np

from common import RESULTS, read_jsonl

paths = {"0.6B gốc": RESULTS / "base_rerun/Qwen3-0.6B", "0.6B ft": RESULTS / "ft/eval/normal/Qwen3-0.6B-ft-s1",
         "1.7B gốc": RESULTS / "base_rerun/Qwen3-1.7B", "1.7B ft": RESULTS / "ft/eval/normal/Qwen3-1.7B-ft-s1"}
for name, p in paths.items():
    P = read_jsonl(p / "preds.jsonl")
    m = np.array([x["letter_mass"] for x in P])
    print(f"{name}: letter_mass median {np.median(m):.3f}  <0.5: {(m < 0.5).sum()}  top_token: {Counter(repr(x['top_token']) for x in P).most_common(5)}")
    low = [x for x in P if x["letter_mass"] < 0.5]
    if low:
        acc_low = np.mean([x["correct"] for x in low]); acc_hi = np.mean([x["correct"] for x in P if x["letter_mass"] >= 0.5])
        print(f"   acc on rows <0.5: {acc_low:.3f}  on rows >=0.5: {acc_hi:.3f}")
