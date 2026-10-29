"""Báo cáo bậc 2 của 2b (results/BRIEF_2b_bac2_train.md mục 4; luật: results/huong_di_2710/PHAN_TICH_TRUOC_TRAIN_2b.md mục 9–11).
Chỉ đọc preds đã có, không chấm model, không đọc test. Thư viện chuẩn + numpy. Định nghĩa T1–T4, G1–G5 như scripts/train_2b/bang_tg.py.

Ba cột cho mỗi luật, đều so với --base (C-s1):
  s2  = --s2 một mình; tb = trung bình --s1 và --s2; s1 = --s1 (đối chiếu).
Trung bình hai seed: với luật dựa trên accuracy, mỗi câu lấy trung bình "đúng" của hai seed (bằng trung bình hai giá trị); T1 và T3a
(dựa trên lựa chọn) lấy trung bình hai giá trị từng seed, T3a báo thêm dạng gộp (tổng đúng / tổng đoán NEI). Bootstrap theo cụm dùng cùng
cách: mỗi câu là trung bình đúng của các seed.
Biên theo số câu (mục 4.2): T3b = số câu đúng − sàn 94/120; T3a = số lần đoán NEI đúng phải mất (số đoán giữ nguyên), hoặc số lần đoán NEI sai
phải thêm, để precision < 0,70; G1 = số câu không dấu của từng nguồn phải sai thêm để macro < 62,47.
Chỉ báo (a)–(d) mục 10, nhiễu seed, chỉ báo khác như bậc 1. Thời gian / VRAM / val loss lấy từ summary.json; nhiệt độ tối đa lấy từ
results/G0/logs/temp.log trong khung "BẮT ĐẦU train" → "XONG train_" của hang_doi.log; giây GPU từ gpu_giay.txt.
Ra results/train_2b/bang_tg_bac2.json và results/train_2b/bang_tg_bac2.md (bảng để chép vào REPORT_2b_bac2.md).
Dùng: python -I scripts/train_2b/bang_tg_bac2.py [--s1 2b-s1] [--s2 2b-s2] [--base C-s1] [--out_json …] [--out_md …]"""
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
TR = ROOT / "results/train_2b/train"
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
FC = ("vifactcheck", "viwikifc")
NEI = "Văn bản không đủ thông tin để kết luận"
NEI_ALL = {NEI, strip_diacritics(NEI)}
YES = {"Có", "Co"}
TH = {"T1": 85, "T2": 0.50, "T2_xam": 0.42, "T3a": 0.70, "T3b": 0.78, "T3b_san": 94, "T4": 79.65, "G1": 62.47, "G2": 87.20, "G3": -10, "G4": 3, "G5": 5}
DAI_C = "dải checkpoint trong một lần chạy C (fable_2b/g4_out.txt): macro 80,17–81,65; XN+BB ViWikiFC 95–106/120; precision NEI 0,76–0,89"

ap = argparse.ArgumentParser()
ap.add_argument("--s1", default="2b-s1")
ap.add_argument("--s2", default="2b-s2")
ap.add_argument("--base", default="C-s1")
ap.add_argument("--out_json", type=Path, default=ROOT / "results/train_2b/bang_tg_bac2.json")
ap.add_argument("--out_md", type=Path, default=ROOT / "results/train_2b/bang_tg_bac2.md")
a = ap.parse_args()
assert sha256_file(DEV) == DEV_SHA
dev = read_jsonl(DEV)
DEVI = {r["id"]: r for r in dev}
NAMES = (a.base, a.s1, a.s2)
COLS = {"s2": (a.s2,), "tb": (a.s1, a.s2), "s1": (a.s1,)}
BASE = (a.base,)


def load(phep, name):
    f = EV / phep / name / "preds.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else None


M = {n: {ph: load(ph, n) for ph in ("dev", "p4", "pa", "p5", "pb", "p1b")} for n in NAMES}
for n in NAMES:
    for ph in M[n]:
        assert M[n][ph] is not None, f"thiếu preds {ph} của {n}"
PROBE = {ph: read_jsonl(PR / f"{f}.jsonl") for ph, f in {"p4": "p4_van_ban_khong_lien_quan", "pa": "pa_lua_chon_khong_dau", "p5": "p5_doi_cau_hoi",
                                                          "pb": "pb_nhan_tieng_anh", "p1b": "p1b_doi_chu_nhan"}.items()}


def vi(lang="vi", var="orig", src=None, kind=None):
    return [r for r in dev if r["lang"] == lang and r["variant"] == var and (src is None or r["source"] == src) and (kind is None or r["kind"] == kind)
            and (lang == "en" or r["source"] in SOURCES)]


def c(G, rid, ph="dev"):   # "đúng" trung bình qua các seed của G
    return sum(int(M[n][ph][rid]["correct"]) for n in G) / len(G)


def acc(G, rs, ph="dev"):
    return sum(c(G, r["id"], ph) for r in rs) / len(rs)


def macro(G, lang, var):
    v = [acc(G, vi(lang, var, s)) for s in SOURCES if vi(lang, var, s)]
    return 100 * sum(v) / len(v)


def chose(n, r, ph="dev"):
    return r["options"][M[n][ph][r["id"]]["pred"]]


def mcnemar(ok_m, ok_b):
    b = sum(x and not y for x, y in zip(ok_b, ok_m)); k = b + sum(y and not x for x, y in zip(ok_b, ok_m)); cc = k - b
    p = 1.0 if k == 0 else min(1.0, 2 * sum(math.comb(k, i) for i in range(min(b, cc) + 1)) / 2 ** k)
    return {"base_dung_m_sai": b, "base_sai_m_dung": cc, "p": round(p, 4)}


def boot_macro(G, lang="vi", var="orig", n_boot=2000):
    rng = np.random.default_rng(SEED); per = []
    for s in SOURCES:
        rs = vi(lang, var, s)
        if not rs:
            continue
        cl = defaultdict(lambda: [0.0, 0])
        for r in rs:
            cl[r["cluster"]][0] += c(G, r["id"]) - c(BASE, r["id"]); cl[r["cluster"]][1] += 1
        S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
        idx = rng.integers(0, len(S), size=(n_boot, len(S))); per.append(S[idx].sum(1) / N[idx].sum(1))
    v = np.sort(np.mean(per, axis=0))
    return [round(100 * float(v[int(0.025 * n_boot)]), 2), round(100 * float(v[int(0.975 * n_boot) - 1]), 2)]


def boot_paired(G, prow, ph, n_boot=1000):
    rng = np.random.default_rng(SEED)
    cl = defaultdict(lambda: [0.0, 0])
    for r in prow:
        cl[r["cluster"]][0] += c(G, r["id"], ph) - c(G, r["orig_id"]); cl[r["cluster"]][1] += 1
    S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
    idx = rng.integers(0, len(S), size=(n_boot, len(S))); v = np.sort(S[idx].sum(1) / N[idx].sum(1))
    return [round(100 * float(v[int(0.025 * n_boot)]), 1), round(100 * float(v[int(0.975 * n_boot) - 1]), 1)]


fc_ch = [r for s in FC for r in vi(src=s, kind="choice")]
nei = [r for r in fc_ch if r["gold_label"] == "nei"]
sr = [r for r in vi(src="viwikifc", kind="choice") if r["gold_label"] in ("supported", "refuted")]
p4 = [r for r in PROBE["p4"] if r["kind"] == "choice" and r["variant"] == "orig"]
pa = [r for r in PROBE["pa"] if r["source"] == "viwikifc" and r["kind"] == "choice"]
p5 = [r for r in PROBE["p5"] if r["source"] in FC and r["variant"] == "orig"]


def t3a_one(n):
    guess = [r for r in fc_ch if chose(n, r) in NEI_ALL]
    return sum(r["gold_label"] == "nei" for r in guess), len(guess)


def block(G):
    o = {}
    t1 = [{s: 100 * sum(chose(n, r, "p4") in NEI_ALL for r in p4 if r["source"] == s) / sum(r["source"] == s for r in p4) for s in FC} for n in G]
    o["T1"] = {s: round(np.mean([x[s] for x in t1]), 1) for s in FC}
    o["T1_theo_nhan_goc"] = {f"{s}|{g}": round(np.mean([100 * sum(chose(n, r, "p4") in NEI_ALL for r in p4 if r["source"] == s and r["orig_gold_label"] == g)
                                                        / max(1, sum(r["source"] == s and r["orig_gold_label"] == g for r in p4)) for n in G]), 1)
                             for s in FC for g in ("supported", "refuted", "nei")}
    o["T2"] = round(acc(G, nei), 4)
    cg = [t3a_one(n) for n in G]
    o["T3a"] = {"gia_tri": round(float(np.mean([x / y for x, y in cg])), 4), "dung": float(np.mean([x for x, _ in cg])), "doan_NEI": float(np.mean([y for _, y in cg])),
                "gop": round(sum(x for x, _ in cg) / sum(y for _, y in cg), 4)}
    o["T3b"] = {"gia_tri": round(acc(G, sr), 4), "dung": round(acc(G, sr) * len(sr), 1), "n": len(sr)}
    o["T4"] = round(macro(G, "vi", "orig"), 2)
    o["G1"] = round(macro(G, "vi", "nodiac"), 2)
    o["G2"] = round(macro(G, "en", "orig"), 2)
    o["G3"] = {"muc_doi": round(100 * (sum(c(G, r["id"], "pa") for r in pa) - sum(c(G, r["orig_id"]) for r in pa)) / len(pa), 1), "ci95": boot_paired(G, pa, "pa"), "n": len(pa)}
    o["G4"] = {"muc_doi": round(100 * (sum(c(G, r["id"], "p5") for r in p5) - sum(c(G, r["orig_id"]) for r in p5)) / len(p5), 1), "n": len(p5)}
    o["G5"] = {s: round(100 * acc(G, vi(src=s)), 1) for s in ("sib200", "belebele")}
    return o


def gates(o, b):
    t2 = o["T2"]
    return {"T1": all(v >= TH["T1"] for v in o["T1"].values()),
            "T2": "đạt" if t2 >= TH["T2"] else ("vùng xám" if t2 >= TH["T2_xam"] else "không chuyển"),
            "T3a": o["T3a"]["gia_tri"] >= TH["T3a"], "T3b": o["T3b"]["gia_tri"] >= TH["T3b"], "T4": o["T4"] >= TH["T4"],
            "G1": o["G1"] >= TH["G1"], "G2": o["G2"] >= TH["G2"], "G3": o["G3"]["muc_doi"] >= TH["G3"],
            "G4": o["G4"]["muc_doi"] >= b["G4"]["muc_doi"] - TH["G4"], "G5": all(o["G5"][s] >= b["G5"][s] - TH["G5"] for s in o["G5"])}


def margins(G, o):
    d, g = o["T3a"]["dung"], o["T3a"]["doan_NEI"]
    mat = next(k for k in range(0, 1000) if (d - k) / g < TH["T3a"])
    them = next(k for k in range(0, 1000) if d / (g + k) < TH["T3a"])
    g1 = {}
    for s in SOURCES:
        n_s = len(vi("vi", "nodiac", s)); unit = 100 / (len(SOURCES) * n_s)
        g1[s] = {"n": n_s, "diem_moi_cau": round(unit, 3), "cau_sai_them_de_truot": math.floor((o["G1"] - TH["G1"]) / unit + 1e-9) + 1}
    return {"T3b": {"dung": o["T3b"]["dung"], "san": TH["T3b_san"], "bien_cau": round(o["T3b"]["dung"] - TH["T3b_san"], 1)},
            "T3a": {"dung": d, "doan_NEI": g, "mat_doan_dung_de_truot": mat, "them_doan_sai_de_truot": them},
            "G1": {"gia_tri": o["G1"], "bien_diem": round(o["G1"] - TH["G1"], 2), "theo_nguon": g1}}


B = block(BASE)
R = {k: block(G) for k, G in COLS.items()}
GT = {k: gates(R[k], B) for k in R}
BI = {k: margins(COLS[k], R[k]) for k in ("s2", "tb")}
STAT = {"T4_hieu": {k: {"hieu": round(R[k]["T4"] - B["T4"], 2), "ci95_bootstrap_cum_2000": boot_macro(COLS[k])} for k in R},
        "mcnemar_s2_C": {"T2": mcnemar([bool(M[a.s2]["dev"][r["id"]]["correct"]) for r in nei], [bool(M[a.base]["dev"][r["id"]]["correct"]) for r in nei]),
                         "T3b": mcnemar([bool(M[a.s2]["dev"][r["id"]]["correct"]) for r in sr], [bool(M[a.base]["dev"][r["id"]]["correct"]) for r in sr])}}

# chỉ báo mục 10
ALLG = {"C": BASE, "s1": (a.s1,), "s2": (a.s2,), "tb": (a.s1, a.s2)}
ind_a = {f"{s}|{v}": {k: round(acc(G, [r for r in vi("vi", v, s, "noul") if r["options"][r["target"]] in YES]), 4) for k, G in ALLG.items()}
         | {"n": len([r for r in vi("vi", v, s, "noul") if r["options"][r["target"]] in YES])} for s in FC for v in ("orig", "nodiac")}
wk_nd = vi("vi", "nodiac", "viwikifc", "choice")
ind_b = {k: {"doan_NEI": sum(chose(n, r) in NEI_ALL for r in wk_nd), "n": len(wk_nd)} for k, n in (("C", a.base), ("s1", a.s1), ("s2", a.s2))}
ind_b["gold_NEI"] = sum(r["gold_label"] == "nei" for r in wk_nd)
ref = [r for r in vi(src="viwikifc", kind="choice") if r["gold_label"] == "refuted"]
ind_c = {k: round(acc(G, ref), 4) for k, G in ALLG.items()} | {"n": len(ref)}
dd = {"VI_macro": (macro((a.s2,), "vi", "orig"), macro((a.s1,), "vi", "orig")), "khong_dau_macro": (macro((a.s2,), "vi", "nodiac"), macro((a.s1,), "vi", "nodiac")),
      "EN_macro": (macro((a.s2,), "en", "orig"), macro((a.s1,), "en", "orig"))}
for s in FC:
    for v in ("orig", "nodiac"):
        dd[f"{s}|{v}"] = (100 * acc((a.s2,), vi("vi", v, s)), 100 * acc((a.s1,), vi("vi", v, s)))
ind_d = {k: {"s2": round(x, 2), "s1": round(y, 2), "s2_tru_s1": round(x - y, 2), "abs": round(abs(x - y), 2)} for k, (x, y) in dd.items()}

# nhiễu seed: từng nguồn × chế độ
noise = {}
for lang, var, tag in (("vi", "orig", "có dấu"), ("vi", "nodiac", "không dấu"), ("en", "orig", "EN")):
    for s in SOURCES:
        rs = vi(lang, var, s)
        if not rs:
            continue
        diff = sum(M[a.s1]["dev"][r["id"]]["pred"] != M[a.s2]["dev"][r["id"]]["pred"] for r in rs)
        x1, x2 = 100 * acc((a.s1,), rs), 100 * acc((a.s2,), rs)
        noise[f"{s}|{tag}"] = {"n": len(rs), "tra_loi_khac": diff, "ti_le_khac": round(diff / len(rs), 3), "s1": round(x1, 1), "s2": round(x2, 1), "s2_tru_s1": round(x2 - x1, 1)}

# chỉ báo khác
other = {}
for ph in ("pb", "p1b"):
    rs = [r for r in PROBE[ph] if r.get("orig_id") in DEVI]
    other[ph] = {k: {"acc": round(100 * acc(G, rs, ph), 1), "muc_doi_so_voi_dev": round(100 * (acc(G, rs, ph) - sum(c(G, r["orig_id"]) for r in rs) / len(rs)), 1)}
                 for k, G in ALLG.items()} | {"n": len(rs)}
prow = {r["id"]: r for r in read_jsonl(ROOT / "data/proxy_v1/proxy_v1.jsonl")}
for tag, d in (("proxy_kiem_chung", "proxy"), ("val_kc_v2", "val_kc_v2")):
    other[tag] = {}
    for k, n in (("C", a.base), ("s1", a.s1), ("s2", a.s2)):
        f = EV2 / d / n / "preds.jsonl"
        if not f.exists():
            other[tag][k] = None; continue
        P = [json.loads(l) for l in open(f, encoding="utf-8")]
        if tag == "proxy_kiem_chung":
            other[tag][k] = {v: {"acc": round(100 * float(np.mean([p["correct"] for p in P if p["source"] in FC and prow[p["id"]]["variant"] == v])), 1),
                                 "n": sum(p["source"] in FC and prow[p["id"]]["variant"] == v for p in P)} for v in ("orig", "nodiac")}
        else:
            other[tag][k] = {"acc": round(100 * float(np.mean([p["correct"] for p in P])), 1), "n": len(P)}


def train_info(n):
    d = TR / f"2b-Qwen3-1.7B-{n.split('-')[-1]}"
    if not (d / "summary.json").exists():
        return None
    s = json.load(open(d / "summary.json", encoding="utf-8"))
    hl = open(d / "hang_doi.log", encoding="utf-8").read().splitlines()
    t0 = next((l.split()[0] for l in hl if "BẮT ĐẦU train" in l), None); t1 = next((l.split()[0] for l in hl if "XONG train_" in l), None)
    temps = [int(l.split()[1]) for l in open(ROOT / "results/G0/logs/temp.log") if len(l.split()) == 2 and t0 and t1 and t0[:19] <= l.split()[0][:19] <= t1[:19]]
    return {"val_loss_by_epoch": [round(x, 4) for x in s["val_loss_by_epoch"]], "steps": s["steps"], "wall_time_s": round(s["wall_time_s"]),
            "peak_vram_gb": round(s["peak_vram_gb"], 2), "nhiet_do_toi_da_train_c": max(temps) if temps else None, "so_lan_do_nhiet": len(temps),
            "gpu_giay_cong_don": int(open(d / "gpu_giay.txt").read().strip())}


TI = {k: train_info(n) for k, n in (("s1", a.s1), ("s2", a.s2))}
res = {"s1": a.s1, "s2": a.s2, "base": a.base, "nguong": TH, "base_gia_tri": B, "cot": R, "dat": GT, "bien_cau": BI, "thong_ke": STAT,
       "chi_bao_a_recall_Co": ind_a, "chi_bao_b_doan_NEI_viwikifc_khong_dau": ind_b, "chi_bao_c_recall_BacBo_viwikifc": ind_c, "chi_bao_d_s2_s1": ind_d,
       "nhieu_seed": noise, "chi_bao_khac": other, "train": TI, "dai_checkpoint_C": DAI_C}
a.out_json.parent.mkdir(parents=True, exist_ok=True)
a.out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------- bảng markdown ----------
f2 = lambda x, k=2: f"{x:.{k}f}".replace(".", ",")
ok = lambda v: ("đạt" if v else "trượt") if isinstance(v, bool) else v


def show(code, o):
    if code == "T1":
        return " / ".join(f2(o["T1"][s], 1) for s in FC)
    if code == "T2":
        return f2(o["T2"], 3)
    if code == "T3a":
        return f"{f2(o['T3a']['gia_tri'], 3)} ({f2(o['T3a']['dung'], 1).rstrip('0').rstrip(',')}/{f2(o['T3a']['doan_NEI'], 1).rstrip('0').rstrip(',')})"
    if code == "T3b":
        return f"{f2(o['T3b']['gia_tri'], 3)} ({f2(o['T3b']['dung'], 1).rstrip('0').rstrip(',')}/{o['T3b']['n']})"
    if code in ("T4", "G1", "G2"):
        return f2(o[code])
    if code == "G3":
        return f"{f2(o['G3']['muc_doi'], 1)} [{f2(o['G3']['ci95'][0], 1)}; {f2(o['G3']['ci95'][1], 1)}]"
    if code == "G4":
        return f2(o["G4"]["muc_doi"], 1)
    return " / ".join(f2(o["G5"][s], 1) for s in ("sib200", "belebele"))


NG = {"T1": "≥ 85% cả hai", "T2": "≥ 0,50 (xám 0,42–0,50)", "T3a": "≥ 0,70", "T3b": "≥ 0,78 (94/120)", "T4": "≥ 79,65", "G1": "≥ 62,47", "G2": "≥ 87,20",
      "G3": "tụt ≤ 10", "G4": f"≥ C − 3 = {f2(B['G4']['muc_doi'] - 3, 1)}", "G5": f"≥ C − 5 ({f2(B['G5']['sib200'] - 5, 1)} / {f2(B['G5']['belebele'] - 5, 1)})"}
L = ["| Mã | Ngưỡng | C-s1 | 2b-s2 | Kết quả s2 | TB 2 seed | Kết quả TB | 2b-s1 (đối chiếu) |", "|---|---|---|---|---|---|---|---|"]
for code in NG:
    L.append(f"| {code} | {NG[code]} | {show(code, B)} | {show(code, R['s2'])} | {ok(GT['s2'][code])} | {show(code, R['tb'])} | {ok(GT['tb'][code])} | {show(code, R['s1'])} |")
L += ["", "T1 = ViFactCheck / ViWikiFC; G5 = SIB-200 / Belebele; G3 kèm CI bootstrap cụm 1.000 lần. T3a của TB = trung bình precision hai seed "
      f"(dạng gộp: {f2(R['tb']['T3a']['gop'], 3)}).", ""]
L += ["**Thống kê:**", ""]
for k, lab in (("s2", "2b-s2 − C-s1"), ("tb", "TB 2b − C-s1"), ("s1", "2b-s1 − C-s1")):
    t = STAT["T4_hieu"][k]
    L.append(f"- VI macro, {lab}: {f2(t['hieu'])} điểm, CI 95% bootstrap cụm 2.000 lần [{f2(t['ci95_bootstrap_cum_2000'][0])}; {f2(t['ci95_bootstrap_cum_2000'][1])}].")
for k in ("T2", "T3b"):
    m = STAT["mcnemar_s2_C"][k]
    L.append(f"- McNemar {k}, s2 so với C-s1: C đúng s2 sai {m['base_dung_m_sai']}, C sai s2 đúng {m['base_sai_m_dung']}, p = {f2(m['p'], 4)}.")
L += ["", "**Biên theo số câu:**", "", "| Luật | 2b-s2 | TB 2 seed |", "|---|---|---|"]
L.append(f"| T3b (sàn 94/120) | {f2(BI['s2']['T3b']['dung'], 1)} → biên {f2(BI['s2']['T3b']['bien_cau'], 1)} câu | {f2(BI['tb']['T3b']['dung'], 1)} → biên {f2(BI['tb']['T3b']['bien_cau'], 1)} câu |")
L.append(f"| T3a (< 0,70 khi) | mất {BI['s2']['T3a']['mat_doan_dung_de_truot']} lần đoán NEI đúng, hoặc thêm {BI['s2']['T3a']['them_doan_sai_de_truot']} lần đoán NEI sai | "
         f"mất {BI['tb']['T3a']['mat_doan_dung_de_truot']}, hoặc thêm {BI['tb']['T3a']['them_doan_sai_de_truot']} |")
gs = lambda k: "; ".join(f"{s} {v['cau_sai_them_de_truot']}" for s, v in BI[k]["G1"]["theo_nguon"].items())
L.append(f"| G1 (sàn 62,47) | biên {f2(BI['s2']['G1']['bien_diem'])} điểm ≈ số câu không dấu phải sai thêm: {gs('s2')} | biên {f2(BI['tb']['G1']['bien_diem'])} điểm: {gs('tb')} |")
L += ["", "**Chỉ báo mục 10:**", "", "(a) Recall \"Có\" của câu Có/Không:", "", "| Nguồn \\| chế độ | n | C-s1 | 2b-s1 | 2b-s2 | TB |", "|---|---|---|---|---|---|"]
for k, v in ind_a.items():
    L.append(f"| {k.replace('|', ' | ').replace('orig', 'có dấu').replace('nodiac', 'không dấu')} | {v['n']} | {f2(v['C'], 3)} | {f2(v['s1'], 3)} | {f2(v['s2'], 3)} | {f2(v['tb'], 3)} |"
             .replace(" | có dấu |", " có dấu |").replace(" | không dấu |", " không dấu |"))
L += ["", f"(b) Đoán NEI trên ViWikiFC không dấu (choice, gold NEI {ind_b['gold_NEI']}/{ind_b['C']['n']}): C-s1 {ind_b['C']['doan_NEI']}, "
      f"2b-s1 {ind_b['s1']['doan_NEI']}, 2b-s2 {ind_b['s2']['doan_NEI']} (/ {ind_b['s2']['n']}; ngưỡng câu hỏi mục 10: > 50% = > {f2(ind_b['s2']['n'] / 2, 1)}).", "",
      f"(c) Recall Bác bỏ ViWikiFC có dấu (n = {ind_c['n']}): C-s1 {f2(ind_c['C'], 3)}, 2b-s1 {f2(ind_c['s1'], 3)}, 2b-s2 {f2(ind_c['s2'], 3)}, TB {f2(ind_c['tb'], 3)}.", "",
      "(d) 2b-s2 so với 2b-s1:", "", "| Đo | 2b-s1 | 2b-s2 | s2 − s1 | \\|Δ\\| |", "|---|---|---|---|---|"]
for k, v in ind_d.items():
    L.append(f"| {k.replace('|orig', ' có dấu').replace('|nodiac', ' không dấu')} | {f2(v['s1'])} | {f2(v['s2'])} | {f2(v['s2_tru_s1'])} | {f2(v['abs'])} |")
L += ["", "**Nhiễu seed (dev, 2b-s1 so với 2b-s2):**", "", "| Nguồn \\| chế độ | n | số câu trả lời khác | 2b-s1 | 2b-s2 | s2 − s1 |", "|---|---|---|---|---|---|"]
for k, v in noise.items():
    L.append(f"| {k.replace('|', ' ')} | {v['n']} | {v['tra_loi_khac']} ({f2(100 * v['ti_le_khac'], 1)}%) | {f2(v['s1'], 1)} | {f2(v['s2'], 1)} | {f2(v['s2_tru_s1'], 1)} |")
L += ["", "**Chỉ báo khác:**", "", "| Bộ | C-s1 | 2b-s1 | 2b-s2 | TB |", "|---|---|---|---|---|"]
for ph in ("pb", "p1b"):
    o = other[ph]
    L.append(f"| {ph}, acc (mức đổi so với dev), n = {o['n']} | " + " | ".join(f"{f2(o[k]['acc'], 1)} ({f2(o[k]['muc_doi_so_voi_dev'], 1)})" for k in ("C", "s1", "s2", "tb")) + " |")
pk = other["proxy_kiem_chung"]
for v, lab in (("orig", "có dấu"), ("nodiac", "không dấu")):
    L.append(f"| proxy kiểm chứng {lab} (n = {next(x[v]['n'] for x in pk.values() if x)}) | " + " | ".join(f2(pk[k][v]["acc"], 1) if pk[k] else "—" for k in ("C", "s1", "s2")) + " | — |")
vk = other["val_kc_v2"]
L.append(f"| val_kc_v2 (n = {next(x['n'] for x in vk.values() if x)}) | " + " | ".join(f2(vk[k]["acc"], 1) if vk[k] else "—" for k in ("C", "s1", "s2")) + " | — |")
L += ["", "**Train:**", "", "| | 2b-s1 | 2b-s2 |", "|---|---|---|"]
rowv = lambda key, fmt: " | ".join(fmt(TI[k][key]) if TI[k] else "—" for k in ("s1", "s2"))
L.append("| val loss bắt đầu / epoch 1 / epoch 2 | " + rowv("val_loss_by_epoch", lambda x: " / ".join(f2(y, 3) for y in x)) + " |")
L.append("| Số bước | " + rowv("steps", str) + " |")
L.append("| Thời gian train (wall, giây) | " + rowv("wall_time_s", str) + " |")
L.append("| VRAM đỉnh (GB) | " + rowv("peak_vram_gb", f2) + " |")
L.append("| Nhiệt độ tối đa khi train (°C; số lần đo) | " + " | ".join(f"{TI[k]['nhiet_do_toi_da_train_c']} ({TI[k]['so_lan_do_nhiet']})" if TI[k] else "—" for k in ("s1", "s2")) + " |")
L.append("| GPU cộng dồn cả lượt (giây) | " + rowv("gpu_giay_cong_don", str) + " |")
a.out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(L))
