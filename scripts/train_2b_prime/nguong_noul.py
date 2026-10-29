"""Bước A của 2b′ (results/BRIEF_2b_prime_du_lieu.md mục 1; P1 trong results/huong_di_2710/FABLE_nguyen_nhan_2b_va_huong_sua.md mục 6).
Ngưỡng noul hậu xử lý trên preds đã có; không chấm model, không sửa đường chấm argmax (score_probe), không đọc test.

Quy tắc: dòng noul đoán "Có" khi p(Có) ≥ t, với p(Có) = probs[Có] / (probs[Có] + probs[Không]).
Fit: trên proxy_v1, dòng noul có dấu (ViWikiFC + ViFactCheck gộp), t = 0,05 … 0,95 bước 0,01; t* = accuracy cao nhất, hòa thì lấy t gần 0,5
nhất. Fit riêng từng adapter. Báo trên dev 1.3 (không fit trên dev): noul theo nguồn, có dấu và không dấu (không dấu dùng cùng t* có dấu),
trước và sau; VI macro (6 nguồn, trung bình đều) có dấu và không dấu, trước và sau. Với C-s1, 2b-s1, 2b-s2 đây chỉ là dòng chỉ báo (hậu kiểm).
Ra results/train_2b_prime/P1_nguong_noul.json và .md (bảng để chép vào báo cáo).
Dùng: python -X utf8 -I scripts/train_2b_prime/nguong_noul.py [--models C-s1 2b-s1 2b-s2]"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
PROXY = ROOT / "data/proxy_v1/proxy_v1.jsonl"
EV_DEV = ROOT / "results/probe_v1/eval/dev"
EV_PX = ROOT / "results/train_2b/eval/proxy"
OUT = ROOT / "results/train_2b_prime"
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
FC = ("vifactcheck", "viwikifc")
YES, NO = {"Có", "Co"}, {"Không", "Khong"}
TS = [round(0.05 + 0.01 * i, 2) for i in range(91)]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["C-s1", "2b-s1", "2b-s2"])
a = ap.parse_args()


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert sha(DEV) == DEV_SHA
dev, px = rj(DEV), rj(PROXY)


def p_yes(r, pr):
    iy = next(i for i, o in enumerate(r["options"]) if o in YES)
    ino = next(i for i, o in enumerate(r["options"]) if o in NO)
    return pr["probs"][iy] / (pr["probs"][iy] + pr["probs"][ino])


def ok(r, pr, t):   # t=None: đúng theo argmax (giữ nguyên preds)
    if r["kind"] != "noul" or t is None:
        return int(pr["correct"])
    return int((p_yes(r, pr) >= t) == (r["options"][r["target"]] in YES))


def acc(rs, P, t):
    return sum(ok(r, P[r["id"]], t) for r in rs) / len(rs)


def ti_le_co(rs, P, t):
    return sum((p_yes(r, P[r["id"]]) >= t) if t is not None else (r["options"][P[r["id"]]["pred"]] in YES) for r in rs) / len(rs)


def sel(rows, var, src=None, kind=None):
    return [r for r in rows if r["lang"] == "vi" and r["variant"] == var and r["source"] in SOURCES
            and (src is None or r["source"] == src) and (kind is None or r["kind"] == kind)]


def macro(P, var, t):
    return 100 * sum(acc(sel(dev, var, s), P, t) for s in SOURCES) / len(SOURCES)


res = {"quy_tac": "Có khi p(Có)/(p(Có)+p(Không)) ≥ t; fit trên proxy_v1 noul có dấu (ViWikiFC+ViFactCheck), t 0,05–0,95 bước 0,01, hòa → gần 0,5",
       "sha": {"dev": DEV_SHA, "proxy_v1": sha(PROXY)}, "mo_hinh": {}}
for m in a.models:
    PP = {x["id"]: x for x in rj(EV_PX / m / "preds.jsonl")}
    PD = {x["id"]: x for x in rj(EV_DEV / m / "preds.jsonl")}
    fit = sel(px, "orig", kind="noul")
    assert all(r["source"] in FC for r in fit) and all(r["id"] in PP for r in fit)
    curve = {t: acc(fit, PP, t) for t in TS}
    best = max(curve.values())
    t_star = min((t for t in TS if curve[t] == best), key=lambda t: (abs(t - 0.5), t))
    o = {"t_sao": t_star, "proxy_fit": {"n": len(fit), "acc_argmax": round(100 * acc(fit, PP, None), 2), "acc_t0.5": round(100 * curve[0.5], 2),
                                        "acc_t_sao": round(100 * best, 2),
                                        "cac_t_cung_acc_max": [t for t in TS if curve[t] == best],
                                        "theo_nguon_t_sao": {s: round(100 * acc([r for r in fit if r["source"] == s], PP, t_star), 2) for s in FC},
                                        "nodiac_argmax": round(100 * acc(sel(px, "nodiac", kind="noul"), PP, None), 2),
                                        "nodiac_t_sao": round(100 * acc(sel(px, "nodiac", kind="noul"), PP, t_star), 2)},
         "dev": {}}
    for var in ("orig", "nodiac"):
        d = {}
        for s in FC:
            rs = sel(dev, var, s, "noul")
            d[f"noul_{s}"] = {"n": len(rs), "truoc": round(100 * acc(rs, PD, None), 2), "sau": round(100 * acc(rs, PD, t_star), 2),
                              "ti_le_Co_truoc": round(100 * ti_le_co(rs, PD, None), 1), "ti_le_Co_sau": round(100 * ti_le_co(rs, PD, t_star), 1),
                              "ti_le_Co_gold": round(100 * sum(r["options"][r["target"]] in YES for r in rs) / len(rs), 1)}
            src = sel(dev, var, s)
            d[f"nguon_{s}"] = {"n": len(src), "truoc": round(100 * acc(src, PD, None), 2), "sau": round(100 * acc(src, PD, t_star), 2)}
        d["VI_macro"] = {"truoc": round(macro(PD, var, None), 2), "sau": round(macro(PD, var, t_star), 2)}
        d["VI_macro"]["chenh"] = round(d["VI_macro"]["sau"] - d["VI_macro"]["truoc"], 2)
        o["dev"][var] = d
    res["mo_hinh"][m] = o

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "P1_nguong_noul.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

f = lambda x: f"{x:.2f}".replace(".", ",")
L = ["| Mô hình | t* | proxy noul có dấu: argmax → t* | dev noul ViFactCheck có dấu | dev noul ViWikiFC có dấu | dev noul ViFactCheck không dấu "
     "| dev noul ViWikiFC không dấu | VI macro có dấu | VI macro không dấu |", "|---|---|---|---|---|---|---|---|---|"]
for m, o in res["mo_hinh"].items():
    d = o["dev"]
    cell = lambda x: f"{f(x['truoc'])} → {f(x['sau'])}"
    L.append(f"| {m} | {f(o['t_sao'])} | {f(o['proxy_fit']['acc_argmax'])} → {f(o['proxy_fit']['acc_t_sao'])} "
             f"| {cell(d['orig']['noul_vifactcheck'])} | {cell(d['orig']['noul_viwikifc'])} | {cell(d['nodiac']['noul_vifactcheck'])} "
             f"| {cell(d['nodiac']['noul_viwikifc'])} | {cell(d['orig']['VI_macro'])} | {cell(d['nodiac']['VI_macro'])} |")
L += ["", "Tỉ lệ đoán \"Có\" trên dev (%, trước → sau; gold):", "", "| Mô hình | ViFactCheck có dấu | ViWikiFC có dấu | ViFactCheck không dấu | ViWikiFC không dấu |",
      "|---|---|---|---|---|"]
for m, o in res["mo_hinh"].items():
    d = o["dev"]
    cell = lambda x: f"{x['ti_le_Co_truoc']:.1f} → {x['ti_le_Co_sau']:.1f} ({x['ti_le_Co_gold']:.1f})".replace(".", ",")
    L.append(f"| {m} | {cell(d['orig']['noul_vifactcheck'])} | {cell(d['orig']['noul_viwikifc'])} | {cell(d['nodiac']['noul_vifactcheck'])} "
             f"| {cell(d['nodiac']['noul_viwikifc'])} |")
(OUT / "P1_nguong_noul.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
print(json.dumps({m: {"t_sao": o["t_sao"], "cac_t": o["proxy_fit"]["cac_t_cung_acc_max"], "proxy": o["proxy_fit"]} for m, o in res["mo_hinh"].items()},
                 ensure_ascii=False))
