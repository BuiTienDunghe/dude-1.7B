"""Bảng T1–T4, G1–G5 của results/BRIEF_2b_bac1_train.md mục 5 (luật: results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md mục 6 + mục 9).
Chỉ đọc preds đã có, không chấm model, không đọc test. Thư viện chuẩn + numpy. So model --m với mốc --base (mặc định C-s1).

Định nghĩa (dev 1.3, VI có dấu, câu kiểm chứng choice của ViFactCheck + ViWikiFC, trừ khi ghi khác):
  T1  P4 có dấu, dòng choice: tỉ lệ chọn lựa chọn NEI, từng nguồn (>= 85% cả hai). Kèm theo nhãn gốc của nhận định (orig_gold_label).
  T2  recall NEI gộp hai nguồn (n = 75): >= 0,50 đạt; 0,42–0,50 vùng xám; < 0,42 "không chuyển". McNemar so với base.
  T3a precision NEI gộp = số câu đoán NEI đúng / số câu đoán NEI (>= 0,70).
  T3b recall Xác nhận + Bác bỏ của ViWikiFC (n = 120) (>= 0,78). McNemar so với base.
  T4  VI macro có dấu, trung bình đều 6 nguồn như trend_C.py (>= 79,65). Bootstrap theo cụm 2.000 lần cho hiệu macro (m − base).
  G1  macro không dấu (>= 62,47); G2 macro EN (>= 87,20).
  G3  Pa ViWikiFC choice: acc(Pa) − acc(dev cùng câu) (tụt <= 10 điểm), CI bootstrap theo cụm.
  G4  P5 kiểm chứng (choice + Có/Không, có dấu): mức đổi acc(P5) − acc(dev cùng câu); đạt khi mức đổi của m >= mức đổi của base − 3 điểm.
  G5  SIB-200 và Belebele có dấu, từng nguồn: m >= base − 5 điểm.
Chỉ báo: recall Xác nhận gộp (T3 cũ), recall từng nhãn từng nguồn, điểm từng nguồn (có dấu, không dấu), pb, p1b, proxy kiểm chứng, val_kc_v2.
Dùng: python scripts/train_2b/bang_tg.py --m 2b-s1 [--base C-s1] [--out results/train_2b/bang_tg_2b-s1.json]"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import SEED, read_jsonl, sha256_file, strip_diacritics  # noqa: E402

DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
DEV_SHA = "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
EV = ROOT / "results/probe_v1/eval"
EV2 = ROOT / "results/train_2b/eval"
PR = ROOT / "results/probe_v1"
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
FC = ("vifactcheck", "viwikifc")
NEI = "Văn bản không đủ thông tin để kết luận"
NEI_ALL = {NEI, strip_diacritics(NEI)}

ap = argparse.ArgumentParser()
ap.add_argument("--m", required=True)
ap.add_argument("--base", default="C-s1")
ap.add_argument("--out", type=Path)
a = ap.parse_args()
assert sha256_file(DEV) == DEV_SHA
dev = read_jsonl(DEV)
DEVI = {r["id"]: r for r in dev}


def preds(phep, name):
    f = EV / phep / name / "preds.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else None


def rows_of(phep):
    f = {"p4": "p4_van_ban_khong_lien_quan", "pa": "pa_lua_chon_khong_dau", "p5": "p5_doi_cau_hoi", "pb": "pb_nhan_tieng_anh",
         "p1b": "p1b_doi_chu_nhan"}[phep]
    return read_jsonl(PR / f"{f}.jsonl")


M = {n: {ph: preds(ph, n) for ph in ("dev", "p4", "pa", "p5", "pb", "p1b")} for n in (a.m, a.base)}
for n in M:
    assert M[n]["dev"] is not None, f"thiếu preds dev của {n}"


def vi(lang="vi", var="orig", src=None, kind=None):
    return [r for r in dev if r["lang"] == lang and r["variant"] == var and (src is None or r["source"] == src) and (kind is None or r["kind"] == kind)
            and (lang == "en" or r["source"] in SOURCES)]


def acc(n, rs, ph="dev"):
    return sum(M[n][ph][r["id"]]["correct"] for r in rs) / len(rs)


def macro(n, lang, var):
    v = [acc(n, vi(lang, var, s)) for s in SOURCES if vi(lang, var, s)]
    return 100 * sum(v) / len(v)


def chose(n, r, ph="dev"):
    return r["options"][M[n][ph][r["id"]]["pred"]]


def mcnemar(ok_m, ok_b):
    b = sum(x and not y for x, y in zip(ok_b, ok_m)); c = sum(y and not x for x, y in zip(ok_b, ok_m)); k = b + c
    p = 1.0 if k == 0 else min(1.0, 2 * sum(math.comb(k, i) for i in range(min(b, c) + 1)) / 2 ** k)
    return {"base_dung_m_sai": b, "base_sai_m_dung": c, "p": round(p, 4)}


def boot_macro(lang, var, n_boot=2000):
    rng = np.random.default_rng(SEED); per = []
    for s in SOURCES:
        rs = vi(lang, var, s)
        if not rs:
            continue
        cl = defaultdict(lambda: [0.0, 0])
        for r in rs:
            cl[r["cluster"]][0] += int(M[a.m]["dev"][r["id"]]["correct"]) - int(M[a.base]["dev"][r["id"]]["correct"]); cl[r["cluster"]][1] += 1
        S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
        idx = rng.integers(0, len(S), size=(n_boot, len(S))); per.append(S[idx].sum(1) / N[idx].sum(1))
    v = np.sort(np.mean(per, axis=0))
    return [round(100 * float(v[int(0.025 * n_boot)]), 2), round(100 * float(v[int(0.975 * n_boot) - 1]), 2)]


def boot_paired(n, prow, ph, n_boot=1000):
    """CI 95% cho acc(phép) − acc(dev cùng câu), rút lại theo cụm."""
    rng = np.random.default_rng(SEED)
    cl = defaultdict(lambda: [0.0, 0])
    for r in prow:
        cl[r["cluster"]][0] += int(M[n][ph][r["id"]]["correct"]) - int(M[n]["dev"][r["orig_id"]]["correct"]); cl[r["cluster"]][1] += 1
    S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
    idx = rng.integers(0, len(S), size=(n_boot, len(S))); v = np.sort(S[idx].sum(1) / N[idx].sum(1))
    return [round(100 * float(v[int(0.025 * n_boot)]), 1), round(100 * float(v[int(0.975 * n_boot) - 1]), 1)]


def block(n):
    o = {}
    fc_ch = [r for s in FC for r in vi(src=s, kind="choice")]
    nei = [r for r in fc_ch if r["gold_label"] == "nei"]
    o["T2_recall_NEI_gop"] = {"gia_tri": round(acc(n, nei), 4), "n": len(nei)}
    guess = [r for r in fc_ch if chose(n, r) in NEI_ALL]
    o["T3a_precision_NEI_gop"] = {"gia_tri": round(sum(r["gold_label"] == "nei" for r in guess) / max(1, len(guess)), 4),
                                  "dung": sum(r["gold_label"] == "nei" for r in guess), "doan_NEI": len(guess)}
    sr = [r for r in vi(src="viwikifc", kind="choice") if r["gold_label"] in ("supported", "refuted")]
    o["T3b_recall_XN_BB_ViWikiFC"] = {"gia_tri": round(acc(n, sr), 4), "n": len(sr)}
    o["T4_VI_macro"] = round(macro(n, "vi", "orig"), 2)
    o["G1_khong_dau_macro"] = round(macro(n, "vi", "nodiac"), 2)
    o["G2_EN_macro"] = round(macro(n, "en", "orig"), 2)
    o["G5"] = {s: round(100 * acc(n, vi(src=s)), 1) for s in ("sib200", "belebele")}
    if M[n]["p4"]:
        p4 = [r for r in rows_of("p4") if r["kind"] == "choice" and r["variant"] == "orig"]
        o["T1_P4"] = {s: round(100 * sum(chose(n, r, "p4") in NEI_ALL for r in p4 if r["source"] == s) / sum(r["source"] == s for r in p4), 1) for s in FC}
        o["T1_P4_theo_nhan_goc"] = {f"{s}|{g}": round(100 * sum(chose(n, r, "p4") in NEI_ALL for r in p4 if r["source"] == s and r["orig_gold_label"] == g)
                                                     / max(1, sum(r["source"] == s and r["orig_gold_label"] == g for r in p4)), 1)
                                    for s in FC for g in ("supported", "refuted", "nei")}
    if M[n]["pa"]:
        pa = [r for r in rows_of("pa") if r["source"] == "viwikifc" and r["kind"] == "choice"]
        d = 100 * (sum(M[n]["pa"][r["id"]]["correct"] for r in pa) - sum(M[n]["dev"][r["orig_id"]]["correct"] for r in pa)) / len(pa)
        o["G3_Pa_ViWikiFC_choice"] = {"muc_doi": round(d, 1), "ci95": boot_paired(n, pa, "pa"), "n": len(pa)}
    if M[n]["p5"]:
        p5 = [r for r in rows_of("p5") if r["source"] in FC and r["variant"] == "orig"]
        d = 100 * (sum(M[n]["p5"][r["id"]]["correct"] for r in p5) - sum(M[n]["dev"][r["orig_id"]]["correct"] for r in p5)) / len(p5)
        o["G4_P5_kiem_chung"] = {"muc_doi": round(d, 1), "n": len(p5)}
    for ph in ("pb", "p1b"):
        if M[n][ph]:
            rs = [r for r in rows_of(ph) if r.get("orig_id") in DEVI]
            o[f"chi_bao_{ph}"] = {"acc": round(100 * sum(M[n][ph][r["id"]]["correct"] for r in rs) / len(rs), 1),
                                  "muc_doi_so_voi_dev": round(100 * (sum(M[n][ph][r["id"]]["correct"] for r in rs)
                                                                     - sum(M[n]["dev"][r["orig_id"]]["correct"] for r in rs)) / len(rs), 1), "n": len(rs)}
    xn = [r for r in fc_ch if r["gold_label"] == "supported"]
    o["chi_bao_recall_XN_gop"] = round(acc(n, xn), 4)
    o["chi_bao_recall_tung_nguon"] = {f"{s}|{g}": round(acc(n, [r for r in vi(src=s, kind="choice") if r["gold_label"] == g]), 4)
                                      for s in FC for g in ("supported", "refuted", "nei")}
    o["chi_bao_diem_tung_nguon"] = {f"{s}|{v}": round(100 * acc(n, vi("vi", v, s)), 1) for s in SOURCES for v in ("orig", "nodiac")}
    for tag, f in (("proxy_v1", EV2 / "proxy" / n / "preds.jsonl"), ("val_kc_v2", EV2 / "val_kc_v2" / n / "preds.jsonl")):
        if f.exists():
            P = [json.loads(l) for l in open(f, encoding="utf-8")]
            if tag == "proxy_v1":
                P = [p for p in P if p["source"] in FC]
            o[f"chi_bao_{tag}"] = {"acc": round(100 * sum(p["correct"] for p in P) / len(P), 1), "n": len(P)}
    return o


B, Mm = block(a.base), block(a.m)
fc_ch = [r for s in FC for r in vi(src=s, kind="choice")]
nei = [r for r in fc_ch if r["gold_label"] == "nei"]
sr = [r for r in vi(src="viwikifc", kind="choice") if r["gold_label"] in ("supported", "refuted")]
ok = lambda n, rs: [bool(M[n]["dev"][r["id"]]["correct"]) for r in rs]
t2 = Mm["T2_recall_NEI_gop"]["gia_tri"]
bang = {
    "T1": {"gia_tri": Mm.get("T1_P4"), "dat": bool(Mm.get("T1_P4")) and all(v >= 85 for v in Mm["T1_P4"].values()), "base": B.get("T1_P4")},
    "T2": {"gia_tri": t2, "doc": "đạt" if t2 >= 0.50 else ("vùng xám" if t2 >= 0.42 else "không chuyển"), "base": B["T2_recall_NEI_gop"]["gia_tri"],
           "mcnemar": mcnemar(ok(a.m, nei), ok(a.base, nei))},
    "T3a": {"gia_tri": Mm["T3a_precision_NEI_gop"], "dat": Mm["T3a_precision_NEI_gop"]["gia_tri"] >= 0.70, "base": B["T3a_precision_NEI_gop"]},
    "T3b": {"gia_tri": Mm["T3b_recall_XN_BB_ViWikiFC"]["gia_tri"], "dat": Mm["T3b_recall_XN_BB_ViWikiFC"]["gia_tri"] >= 0.78,
            "base": B["T3b_recall_XN_BB_ViWikiFC"]["gia_tri"], "mcnemar": mcnemar(ok(a.m, sr), ok(a.base, sr))},
    "T4": {"gia_tri": Mm["T4_VI_macro"], "dat": Mm["T4_VI_macro"] >= 79.65, "base": B["T4_VI_macro"],
           "hieu": round(Mm["T4_VI_macro"] - B["T4_VI_macro"], 2), "ci95_bootstrap_cum_2000": boot_macro("vi", "orig")},
    "G1": {"gia_tri": Mm["G1_khong_dau_macro"], "dat": Mm["G1_khong_dau_macro"] >= 62.47, "base": B["G1_khong_dau_macro"]},
    "G2": {"gia_tri": Mm["G2_EN_macro"], "dat": Mm["G2_EN_macro"] >= 87.20, "base": B["G2_EN_macro"]},
    "G3": {"gia_tri": Mm.get("G3_Pa_ViWikiFC_choice"), "dat": bool(Mm.get("G3_Pa_ViWikiFC_choice")) and Mm["G3_Pa_ViWikiFC_choice"]["muc_doi"] >= -10,
           "base": B.get("G3_Pa_ViWikiFC_choice")},
    "G4": {"gia_tri": Mm.get("G4_P5_kiem_chung"), "base": B.get("G4_P5_kiem_chung"),
           "dat": bool(Mm.get("G4_P5_kiem_chung")) and Mm["G4_P5_kiem_chung"]["muc_doi"] >= B["G4_P5_kiem_chung"]["muc_doi"] - 3},
    "G5": {"gia_tri": Mm["G5"], "base": B["G5"], "dat": all(Mm["G5"][s] >= B["G5"][s] - 5 for s in Mm["G5"])},
}
res = {"m": a.m, "base": a.base, "bang": bang, "chi_tiet_m": Mm, "chi_tiet_base": B}
out = a.out or ROOT / f"results/train_2b/bang_tg_{a.m}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(bang, ensure_ascii=False, indent=1))
