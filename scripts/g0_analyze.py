"""Tổng hợp G0 (BRIEF_buoc0_G0.md) theo đúng cách đọc đã chốt trước ở results/G0/CACH_DOC_chot_truoc.md. Chỉ đọc preds / json đã có; không chạy model.
Điểm = trung bình đều 6 nguồn (macro) trên dòng dev 1.3 theo biến thể; CI 95% theo cặp = bootstrap cụm (trường cluster) rút lại riêng từng nguồn,
1.000 lần, seed 20261012 cho từng phép so (như scripts/train_v1/trend_C.py). Thư viện chuẩn + numpy.
Dùng: python scripts/g0_analyze.py  → results/G0/g0_tong_hop.json"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import SEED, read_jsonl, sha256_file  # noqa: E402

DEV = ROOT / "data/bench_v1_3/dev_v1_3.jsonl"
assert sha256_file(DEV) == "c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e"
G = ROOT / "results/G0"
SOURCES = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
dev = read_jsonl(DEV)
D = {r["id"]: r for r in dev}
MODELS = {
    "Qwen3-1.7B gốc (M0)": "results/probe_v1/eval/dev/Qwen3-1.7B",
    "Qwen3.5-2B gốc": "results/bench_v1_3_trial/eval/Qwen3.5-2B",
    "M1 (dự án)": "results/probe_v1/eval/dev/Qwen3-1.7B-fullvocab-s1 (dự án)",
    "C (train thử)": "results/probe_v1/eval/dev/C-s1",
    "decider-2b": "results/probe_v1/eval/dev/decider-2b-text",
    "Quyet-Medium": "results/probe_v1/eval/dev/normal/quyet-medium-text",
    "Qwen3.5-2B-Base": "results/G0/eval/dev/Qwen3.5-2B-Base",
    "Qwen3.5-4B @851bf6e806ef": "results/G0/eval/dev/Qwen3.5-4B",
}
SETS = {"VI có dấu": ("vi", "orig"), "VI không dấu": ("vi", "nodiac"), "EN": ("en", "orig")}


def load(rel):
    f = ROOT / rel / "preds.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else None


def rows(lang, var, src=None):
    return [r for r in dev if r["lang"] == lang and r["variant"] == var and (src is None or r["source"] == src)]


def acc(P, rs):
    rs = [r for r in rs if r["id"] in P]
    return sum(P[r["id"]]["correct"] for r in rs) / len(rs) if rs else None


def macro(P, lang, var):
    v = [acc(P, rows(lang, var, s)) for s in SOURCES if rows(lang, var, s)]
    return 100 * sum(v) / len(v)


def diff_ci(P2, P1, lang, var):
    """macro(P2) − macro(P1) và CI 95% bootstrap cụm theo nguồn."""
    rng = np.random.default_rng(SEED)
    per, d = [], []
    for s in SOURCES:
        rs = [r for r in rows(lang, var, s) if r["id"] in P1 and r["id"] in P2]
        if not rs:
            continue
        cl = defaultdict(lambda: [0.0, 0])
        for r in rs:
            cl[r["cluster"]][0] += int(P2[r["id"]]["correct"]) - int(P1[r["id"]]["correct"]); cl[r["cluster"]][1] += 1
        S = np.array([v[0] for v in cl.values()]); N = np.array([v[1] for v in cl.values()])
        idx = rng.integers(0, len(S), size=(1000, len(S)))
        per.append(S[idx].sum(1) / N[idx].sum(1)); d.append(S.sum() / N.sum())
    v = np.sort(np.mean(per, axis=0))
    return {"chenh": round(100 * float(np.mean(d)), 2), "ci95": [round(100 * float(v[25]), 2), round(100 * float(v[974]), 2)]}   # như trend_C.py


res = {"cach_doc": "results/G0/CACH_DOC_chot_truoc.md", "g0b": {}, "g0c": {}, "latency": {}}
P = {k: load(v) for k, v in MODELS.items()}
for k, p in P.items():
    if p is None:
        res["g0b"][k] = "chua_co_preds"; continue
    res["g0b"][k] = {"nguon_preds": MODELS[k], **{s: round(macro(p, *lv), 2) for s, lv in SETS.items()},
                     "theo_nguon_VI_co_dau": {s: round(100 * acc(p, rows("vi", "orig", s)), 1) for s in SOURCES}}
B, M0 = P["Qwen3.5-2B-Base"], P["Qwen3-1.7B gốc (M0)"]
if B:
    res["g0b"]["2B-Base − M0 (VI có dấu)"] = diff_ci(B, M0, "vi", "orig")
    for k in ("Qwen3.5-2B gốc", "C (train thử)"):
        res["g0b"][f"2B-Base − {k} (VI có dấu)"] = diff_ci(B, P[k], "vi", "orig")
F = {"Qwen3-1.7B gốc": ("results/probe_v1/eval/dev/Qwen3-1.7B", "results/G0/eval/g0c/Qwen3-1.7B_khungEN", ".venv"),
     "Qwen3.5-2B gốc": ("results/G0/eval/g0c/Qwen3.5-2B_khungVI", "results/G0/eval/g0c/Qwen3.5-2B_khungEN", ".venv (FLA không chạy được)"),
     "Qwen3.5-2B-Base": ("results/G0/eval/dev/Qwen3.5-2B-Base", "results/G0/eval/g0c/Qwen3.5-2B-Base_khungEN", ".venv (FLA không chạy được)")}
for k, (vi, en) in ((k, v[:2]) for k, v in F.items()):
    pv, pe = load(vi), load(en)
    if not (pv and pe):
        res["g0c"][k] = "chua_co_preds"; continue
    d = diff_ci(pe, pv, "vi", "orig")
    res["g0c"][k] = {"khung_VI": round(macro(pv, "vi", "orig"), 2), "khung_EN": round(macro(pe, "vi", "orig"), 2), "EN_tru_VI": d,
                     "moi_truong": F[k][2], "dat_nguong_EN_hon_2_diem_CI_khong_chua_0": d["chenh"] >= 2.0 and d["ci95"][0] > 0,
                     "theo_nguon_EN_tru_VI": {s: round(100 * (acc(pe, rows("vi", "orig", s)) - acc(pv, rows("vi", "orig", s))), 1) for s in SOURCES}}
for f in sorted((G / "latency").glob("latency_*.json")):
    x = json.loads(f.read_text(encoding="utf-8"))
    e = {"fla": x.get("fla"), "do_dai_prompt": x["do_dai_prompt"], "vram_dinh_gb": round(x["vram_dinh_gb"], 2)}
    for m in ("eager", "graphs"):
        if m in x:
            e[m] = {**{k: round(v, 2) for k, v in x[m]["trung_binh_cac_lan"].items()}, "so_lan": x[m]["so_lan"],
                    "cac_lan_trung_vi_p90": [[round(p["trung_vi"], 2), round(p["p90"], 2)] for p in x[m]["cac_lan"]],
                    "p90_theo_nhom_lan1": {g: (round(v["p90"], 2) if v["p90"] is not None else None, v["n"]) for g, v in x[m]["cac_lan"][0]["p90_theo_nhom_do_dai"].items()},
                    **({"CANH_BAO": x[m]["CANH_BAO"]} if "CANH_BAO" in x[m] else {})}
    e["cong_dung"] = x.get("cong_dung"); e["graphs_loi"] = (x.get("graphs_loi") or "")[-600:] or None
    res["latency"][f.stem.replace("latency_", "")] = e
(G / "g0_tong_hop.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False)[:6000])
