"""Hàm thống kê dùng chung cho báo cáo sau fine-tune (cùng định nghĩa với report.py của bước dev_v0)."""
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

B = 2000


def brier(p, t):
    v = np.asarray(p, dtype=np.float64).copy()
    v[t] -= 1.0
    return float((v ** 2).sum())


def temper(p, T):
    """softmax(logits/T) trên k chữ cái, tính từ xác suất: p^(1/T) chuẩn hoá lại."""
    lp = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-300, None)) / T
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


def cluster_index(ids, rows):
    """Gom câu theo state (Belebele: theo đoạn văn/link; nguồn khác: mỗi câu một state)."""
    groups = defaultdict(list)
    for j, i in enumerate(ids):
        groups[rows[i]["cluster"]].append(j)
    return [np.array(g) for _, g in sorted(groups.items())]


def boot(per_source, seed):
    """per_source: list of (values, clusters). Lấy mẫu lại cụm trong từng nguồn; trả về mảng [B, n_src]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((B, len(per_source)))
    for s, (vals, cl) in enumerate(per_source):
        sums = np.array([vals[c].sum() for c in cl]); cnts = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        out[:, s] = sums[pick].sum(1) / cnts[pick].sum(1)
    return out


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def mcnemar(a, b):
    """McNemar chính xác (binomial trên các cặp bất đồng). a, b: mảng bool đúng/sai trên cùng câu."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    n = n10 + n01
    return {"a_only": n10, "b_only": n01, "p": 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)}
