"""Train thử bản C: so chỉ tiêu đặt trước (results/BRIEF_train_chinh_thuc.md mục 2) và đường xu hướng 4 mốc (25/50/75/100%) trên DEV 1.3.
Không chấm model, chỉ đọc preds đã có (results/probe_v1/eval/dev/<tên>/preds.jsonl, timing.json). Không đọc test. Thư viện chuẩn + numpy.

Chỉ tiêu (điểm = TRUNG BÌNH ĐỀU 6 nguồn, như bảng mục 1 của brief; kèm điểm trên mọi câu để đối chiếu):
  VI có dấu >= 84,0 và CI 95% của chênh macro so với M1 không chứa 0; VI không dấu >= 61,0; EN >= 87,0; trung vị độ trễ batch 1 trên dev <= 40 ms;
  theo nguồn (VI có dấu): ViFactCheck, MASSIVE tăng có McNemar p < 0,05; Belebele, ViWikiFC, XCOPA không tụt.
CI: bootstrap theo cụm (trường cluster), rút lại cụm RIÊNG TỪNG NGUỒN (giữ cơ cấu macro), 1.000 lần, seed cố định. McNemar chính xác theo cặp câu.
Dùng: python scripts/train_v1/trend_C.py [--out results/train_v1/trend_C.json]"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import SEED, read_jsonl, sha256_file  # noqa: E402

DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
EV = ROOT / "results/probe_v1/eval/dev"
MODELS = {"M0": "Qwen3-1.7B", "M1": "Qwen3-1.7B-fullvocab-s1 (dự án)", "C@25%": "C-s1-step484", "C@50%": "C-s1-step969",
          "C@75%": "C-s1-step1454", "C@100%": "C-s1", "R1": "decider-2b-text", "R2": "normal/quyet-medium-text"}
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
SETS = {"VI có dấu": ("vi", "orig"), "VI không dấu": ("vi", "nodiac"), "EN": ("en", "orig")}
N_BOOT = 1000

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=ROOT / "results/train_v1/trend_C.json")
a = ap.parse_args()
assert sha256_file(DEV) == DEV_SHA
dev = read_jsonl(DEV)
P, LAT = {}, {}
for code, name in MODELS.items():
    f = EV / name / "preds.jsonl"
    if f.exists():
        P[code] = {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}
        t = EV / name / "timing.json"
        if t.exists():
            LAT[code] = json.loads(t.read_text(encoding="utf-8")).get("batch1_median_latency_ms")
rng = np.random.default_rng(SEED)


def rows(lang, var, src=None):
    return [r for r in dev if r["lang"] == lang and r["variant"] == var and (src is None or r["source"] == src) and (lang == "en" or r["source"] in SOURCES)]


def acc(code, rs):
    return sum(P[code][r["id"]]["correct"] for r in rs) / len(rs)


def macro(code, lang, var):
    v = [acc(code, rows(lang, var, s)) for s in SOURCES if rows(lang, var, s)]
    return sum(v) / len(v)


def boot_macro_diff(c2, c1, lang, var):
    """CI 95% cho macro(c2) − macro(c1): rút lại cụm có hoàn lại riêng trong từng nguồn. Seed cố định cho TỪNG phép so (không phụ thuộc
    có bao nhiêu model khác được chấm), nên CI của một cặp không đổi khi thêm mốc."""
    rng = np.random.default_rng(SEED)
    per = []
    for s in SOURCES:
        rs = rows(lang, var, s)
        if not rs:
            continue
        cl = defaultdict(lambda: [0.0, 0])
        for r in rs:
            cl[r["cluster"]][0] += int(P[c2][r["id"]]["correct"]) - int(P[c1][r["id"]]["correct"]); cl[r["cluster"]][1] += 1
        S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
        idx = rng.integers(0, len(S), size=(N_BOOT, len(S)))
        per.append(S[idx].sum(1) / N[idx].sum(1))
    v = np.sort(np.mean(per, axis=0))
    return [round(100 * float(v[int(0.025 * N_BOOT)]), 2), round(100 * float(v[int(0.975 * N_BOOT) - 1]), 2)]


def mcnemar(c2, c1, rs):
    b = sum(P[c1][r["id"]]["correct"] and not P[c2][r["id"]]["correct"] for r in rs)
    c = sum(P[c2][r["id"]]["correct"] and not P[c1][r["id"]]["correct"] for r in rs)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"dung_thanh_sai": b, "sai_thanh_dung": c, "p": round(p, 4)}


res = {"CANH_BAO": "Train THỬ bản C; dev 1.3; không phải điểm benchmark cuối; không chạy gì trên test.", "models": {c: MODELS[c] for c in P},
       "do_tre_batch1_trung_vi_ms": LAT, "diem": {}, "theo_nguon_VI_co_dau": {}, "chenh_so_voi_M1": {}}
for code in P:
    res["diem"][code] = {k: {"macro": round(100 * macro(code, *lv), 2), "tren_moi_cau": round(100 * acc(code, rows(*lv)), 2)} for k, lv in SETS.items()}
    res["theo_nguon_VI_co_dau"][code] = {s: round(100 * acc(code, rows("vi", "orig", s)), 2) for s in SOURCES}
for code in [c for c in P if c.startswith("C@")]:
    e = {}
    for k, lv in SETS.items():
        e[k] = {"chenh_macro": round(res["diem"][code][k]["macro"] - res["diem"]["M1"][k]["macro"], 2), "ci95": boot_macro_diff(code, "M1", *lv)}
    e["theo_nguon_VI_co_dau"] = {s: {"chenh": round(res["theo_nguon_VI_co_dau"][code][s] - res["theo_nguon_VI_co_dau"]["M1"][s], 2),
                                     **mcnemar(code, "M1", rows("vi", "orig", s))} for s in SOURCES}
    res["chenh_so_voi_M1"][code] = e
if "C@100%" in P:
    d, ch = res["diem"]["C@100%"], res["chenh_so_voi_M1"]["C@100%"]
    ns = ch["theo_nguon_VI_co_dau"]
    res["chi_tieu_C@100%"] = {
        "VI có dấu >= 84,0": {"gia_tri": d["VI có dấu"]["macro"], "dat": d["VI có dấu"]["macro"] >= 84.0},
        "CI chênh VI có dấu so với M1 không chứa 0": {"ci95": ch["VI có dấu"]["ci95"], "dat": ch["VI có dấu"]["ci95"][0] > 0},
        "VI không dấu >= 61,0": {"gia_tri": d["VI không dấu"]["macro"], "dat": d["VI không dấu"]["macro"] >= 61.0},
        "EN >= 87,0": {"gia_tri": d["EN"]["macro"], "dat": d["EN"]["macro"] >= 87.0},
        "độ trễ <= 40 ms": {"gia_tri": LAT.get("C@100%"), "dat": LAT.get("C@100%") is not None and LAT["C@100%"] <= 40},
        "ViFactCheck, MASSIVE tăng (McNemar p < 0,05)": {s: {"chenh": ns[s]["chenh"], "p": ns[s]["p"], "dat": ns[s]["chenh"] > 0 and ns[s]["p"] < 0.05}
                                                          for s in ("vifactcheck", "massive")},
        "Belebele, ViWikiFC, XCOPA không tụt": {s: {"chenh": ns[s]["chenh"], "p": ns[s]["p"], "khong_tut": ns[s]["chenh"] >= 0}
                                                 for s in ("belebele", "viwikifc", "xcopa")}}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("diem", "do_tre_batch1_trung_vi_ms")}, ensure_ascii=False))
