"""Bảng cổng mới của V-e (điều phối 29/10; results/huong_di_2710/FABLE_truoc_train_2b_prime.md mục 1–2). Chỉ CPU, chỉ đọc jsonl.
Khối = dòng kiểm chứng VI (KC + 2d + 2e + ViANLI khung), như is_fc của build_2b_prime.py. cover = |từ nhận định ∩ từ state| / |từ nhận định|
(kc_util.wset, cả state), như Fable. Chế độ: có dấu / không dấu / cả hai (cờ "nodiac").
  1a. P(nhãn | dải cover), dòng choice (Xác nhận / Bác bỏ / NEI, %), dải [0; 0,5), [0,5; 0,7), [0,7; 0,9), ≥ 0,9.
  1b. Bác bỏ choice cover < 0,7 theo nguồn (2d, 2e, ViANLI, KC) × chế độ.
  1c. Noul: P("Có" | dải cover).
  Probe riêng ViANLI (cổng mới (i)): đặc trưng phan_tich_bacbo, LR numpy, 5-fold phân tầng theo seed (đúng cv5 của build_2b_prime.py), trên dòng
  choice có dấu Bác bỏ / NEI của khối ViANLI. Đạt khi acc ≤ đa số + 10 điểm.
Dùng: bash setup/run.sh -I scripts/train_2b_prime/bang_ve.py --train <train.jsonl> --ten <nhãn> [--out <json>]"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train_2b"))
from kc_util import name_spans, normalize, num_spans, overlap, read_jsonl, sentences, wset  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--train", type=Path, required=True)
ap.add_argument("--ten", required=True)
ap.add_argument("--out", type=Path)
a = ap.parse_args()
SEED = 20261029
LABS = ("supported", "refuted", "nei")
MK = ("Nhận định: ", "Phát biểu: ", "Nhan dinh: ", "Phat bieu: ")
NEG_P = re.compile(r"\b(không|chưa|chẳng|chả|khong|chua|chang|cha)\b", re.I)
YES = {"Có", "Co"}
BANDS = ((0, 0.5, "[0;0,5)"), (0.5, 0.7, "[0,5;0,7)"), (0.7, 0.9, "[0,7;0,9)"), (0.9, 9, ">=0,9"))
fl = lambda r: set(r.get("flags") or [])
dia = lambda r: "nodiac" not in fl(r)
is_fc = lambda r: r["text_lang"] == "vi" and (r["source"].startswith("kc_") or r["source"] == "vianli_khung" or bool({"2d", "2e"} & fl(r)))


def claim_of(r):
    return next(r["question"].split(mk, 1)[1] for mk in MK if mk in r["question"])


def src(r):
    return "ViANLI" if r["source"] == "vianli_khung" else "KC" if r["source"].startswith("kc_") else "2d" if "2d" in fl(r) else "2e"


rows = [r for r in read_jsonl(a.train) if is_fc(r)]
for r in rows:
    c = claim_of(r)
    r["_cover"] = len(wset(c) & wset(r["state"])) / max(1, len(wset(c)))
MODES = {"co_dau": dia, "khong_dau": lambda r: not dia(r), "ca_hai": lambda r: True}


def band(x):
    return next(t for lo, hi, t in BANDS if lo <= x < hi)


res = {"train": str(a.train), "ten": a.ten, "1a": {}, "1b": {}, "1c": {}}
for m, f in MODES.items():
    ch = [r for r in rows if r["kind"] == "choice" and f(r)]
    o = {}
    for _, _, t in BANDS:
        g = [r for r in ch if band(r["_cover"]) == t]
        o[t] = {"n": len(g), **{l: round(100 * sum(r["gold_label"] == l for r in g) / max(1, len(g))) for l in LABS}}
    res["1a"][m] = o
    no = [r for r in rows if r["kind"] == "noul" and f(r)]
    res["1c"][m] = {t: {"n": len(g), "co": round(100 * sum(r["options"][r["target_argmax"]] in YES for r in g) / max(1, len(g)))}
                    for _, _, t in BANDS for g in [[r for r in no if band(r["_cover"]) == t]]}
for s in ("2d", "2e", "ViANLI", "KC"):
    res["1b"][s] = {m: sum(r["kind"] == "choice" and r["gold_label"] == "refuted" and r["_cover"] < 0.7 and src(r) == s and f(r) for r in rows)
                    for m, f in MODES.items()}
res["1b"]["tong"] = {m: sum(v[m] for v in res["1b"].values()) for m in MODES}
res["thanh_phan_cover_lt_0_7_co_dau"] = {l: sum(r["kind"] == "choice" and dia(r) and r["gold_label"] == l and r["_cover"] < 0.7 for r in rows) for l in LABS}


# ------------------------------------------------------------------ probe riêng ViANLI (cv5 như build_2b_prime.py)
def feats(claim, st):
    ss = sentences(st) or [st]
    best = max(ss, key=lambda s: overlap(claim, s))
    nst = normalize(st)
    nums = [v for _, _, v, _ in num_spans(claim)]
    names = [x for _, _, x, _ in name_spans(claim)]
    return {"max_ov": overlap(claim, best), "cover": len(wset(claim) & wset(st)) / max(1, len(wset(claim))),
            "phu_dinh_nd": bool(NEG_P.search(claim)), "lech_phu_dinh": bool(NEG_P.search(claim)) != bool(NEG_P.search(best)),
            "so_la": any(normalize(v) not in nst for v in nums), "ten_la": any(normalize(x) not in nst for x in names), "do_dai": len(claim.split())}


FK = ("max_ov", "cover", "phu_dinh_nd", "lech_phu_dinh", "so_la", "ten_la", "do_dai")


def fit(fs):
    X = np.array([[float(f[k]) for k in FK] for f, _ in fs]); y = np.array([1 if l == "refuted" else 0 for _, l in fs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Z @ w)); w -= 0.1 * (Z.T @ (p - y) / len(y) + 1e-3 * np.r_[0, w[1:]])
    return lambda f: float(1 / (1 + np.exp(-(np.r_[1, (np.array([float(f[k]) for k in FK]) - mu) / sd] @ w))))


def cv5(fs):
    fs = [x for x in fs if x[1] in ("refuted", "nei")]
    by = defaultdict(list)
    for i, (_, l) in enumerate(fs):
        by[l].append(i)
    fold = {}
    for l, idx in by.items():
        random.Random(SEED).shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % 5
    accs = []
    for k in range(5):
        tr = [fs[i] for i in range(len(fs)) if fold[i] != k]; te = [fs[i] for i in range(len(fs)) if fold[i] == k]
        s = fit(tr)
        accs.append(sum((s(f) >= 0.5) == (l == "refuted") for f, l in te) / len(te))
    c = Counter(l for _, l in fs)
    return {"n": dict(c), "da_so": round(max(c.values()) / len(fs), 4), "acc_5fold": round(float(np.mean(accs)), 4)}


va = [r for r in rows if r["source"] == "vianli_khung" and r["kind"] == "choice" and dia(r)]
if va:
    p = cv5([(feats(claim_of(r), r["state"]), r["gold_label"]) for r in va])
    p["vuot_da_so_diem"] = round(100 * (p["acc_5fold"] - p["da_so"]), 2); p["dat"] = p["acc_5fold"] <= p["da_so"] + 0.10
    res["probe_rieng_vianli"] = p
out = a.out or ROOT / f"results/train_2b_prime/tam/bang_1abc_{a.ten}.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
