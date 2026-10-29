"""Bảng cho bench 1.1 trên dev_v1_1 (mục D, E, F). Ra: results/dev_v1_1/tables_v1_1.md, results/dev_v1_1/metrics_v1_1.json

Preds (results/dev_v1_1/eval/, ghép bằng merge_preds.py: dòng Belebele/SIB/XCOPA trùng từng byte với dev_v1 dùng lại preds cũ, dòng
MASSIVE và kiểm chứng chấm lại bằng đúng đường chấm cũ, batch 1).
decider-2b và decider-ft: chấm cả ánh xạ "text" lẫn "keyed"; bảng chính dùng ánh xạ cho điểm cao hơn (trung bình năm lát chính).
Accuracy = trung bình đều theo nguồn; CI 95% bootstrap theo state (cluster) trong từng nguồn; McNemar chính xác theo cặp, gộp dòng.
Mức sàn (mục D, tính trên ĐÁP ÁN của dòng tiếng Việt có dấu): ngẫu nhiên = trung bình 1/k; "luôn chọn đáp án phổ biến nhất" theo chữ
của đáp án đúng, và theo vị trí đáp án đúng.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LETTERS, RESULTS, SEED, load_json, read_jsonl, sha256_file  # noqa: E402
from stats import B, mcnemar  # noqa: E402

R = RESULTS / "dev_v1_1"
DEV = DATA / "bench_v1_1" / "dev_v1_1.jsonl"
assert sha256_file(DEV) == load_json(DATA / "bench_v1_1" / "manifest_dev_v1_1.json")["sha256"]["dev_v1_1.jsonl"]
rows = {r["id"]: r for r in read_jsonl(DEV)}
load = lambda p: {x["id"]: x for x in read_jsonl(p)}
E = R / "eval"
SLICES = {"song song · EN": dict(block="parallel", lang="en"), "song song · VI có dấu": dict(block="parallel", lang="vi", variant="orig"),
          "song song · VI không dấu": dict(block="parallel", lang="vi", variant="nodiac"), "bản địa · có dấu": dict(block="native", variant="orig"),
          "bản địa · không dấu": dict(block="native", variant="nodiac")}


def sel(**kw):
    return sorted(i for i, r in rows.items() if all(r[k] == v for k, v in kw.items()))


def split(ids):
    g = defaultdict(list)
    for i in ids:
        g[rows[i]["source"]].append(i)
    return dict(sorted(g.items()))


def macro_boot(vals, ids_by_src, seed):
    rng = np.random.default_rng(seed)
    est = np.zeros((B, len(vals)))
    for s, src in enumerate(vals):
        v = vals[src]; g = defaultdict(list)
        for j, i in enumerate(ids_by_src[src]):
            g[rows[i]["cluster"]].append(j)
        cl = [np.array(x) for _, x in sorted(g.items())]
        sums = np.array([v[c].sum() for c in cl]); cnt = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        est[:, s] = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(np.mean([v.mean() for v in vals.values()])), [float(np.percentile(est.mean(1), 2.5)), float(np.percentile(est.mean(1), 97.5))]


def acc(P, ids, seed):
    bs = split(ids)
    a, ci = macro_boot({s: np.array([float(P[i]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed)
    return {"acc": a, "ci": ci, "n": len(ids), "per_source": {s: float(np.mean([P[i]["correct"] for i in v])) for s, v in bs.items()}}


def paired(PA, PB, ids, seed):
    bs = split(ids)
    d, ci = macro_boot({s: np.array([float(PA[i]["correct"]) - float(PB[i]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed)
    return {"diff": d, "ci": ci, **mcnemar([PA[i]["correct"] for i in ids], [PB[i]["correct"] for i in ids])}


def five(P):
    return float(np.mean([acc(P, sel(**kw), SEED)["acc"] for kw in SLICES.values()]))


# ---------------------------------------------------------------- model và chọn ánh xạ
cand = {"decider-2b": ("decider-2b-text", "decider-2b-keyed"), "decider-ft ep2": ("decider-ft-ep2-text", "decider-ft-ep2-keyed")}
P, mapping = {}, {}
for name, (t, k) in cand.items():
    if (E / t / "preds.jsonl").exists():
        pt, pk = load(E / t / "preds.jsonl"), load(E / k / "preds.jsonl")
        st, sk = five(pt), five(pk)
        P[name], mapping[name] = (pt, "text") if st >= sk else (pk, "keyed")
        mapping[name] = {"used": mapping[name], "mean_five_slices_text": st, "mean_five_slices_keyed": sk}
for name, d in (("adapter đối chứng", "Qwen3-1.7B-fullvocab-s1"), ("Qwen3-1.7B gốc", "Qwen3-1.7B"), ("strip (Qwen3)", "strip-s1")):
    if (E / d / "preds.jsonl").exists():
        P[name] = load(E / d / "preds.jsonl")
order = [m for m in ("decider-2b", "adapter đối chứng", "Qwen3-1.7B gốc", "strip (Qwen3)", "decider-ft ep2") if m in P]
for m in order:
    assert set(P[m]) == set(rows), m
DEC, CTL = "decider-2b", "adapter đối chứng"

M = {"dev_sha256": sha256_file(DEV), "mapping": mapping, "slices": {}, "paired": {}, "by_kind": {}, "floors": {}, "noul": {}}
for j, (name, kw) in enumerate(SLICES.items()):
    ids = sel(**kw)
    M["slices"][name] = {m: acc(P[m], ids, SEED + j) for m in order}
    M["paired"][name] = {f"{m} − {ref}": paired(P[m], P[ref], ids, SEED + 50 + j) for ref in (DEC, CTL) for m in order if m != ref and ref in P}
for kind in ("choice", "noul"):
    for var in ("orig", "nodiac"):
        ids = sel(block="native", kind=kind, variant=var)
        key = f"bản địa · {kind} · {'có dấu' if var == 'orig' else 'không dấu'}"
        M["by_kind"][key] = {m: acc(P[m], ids, SEED + 100)["acc"] for m in order} | {"n": len(ids)}

# ---------------------------------------------------------------- mục D: mức sàn (dòng VI có dấu)
for (src, kind) in sorted({(r["source"], r["kind"]) for r in rows.values() if r["lang"] == "vi" and r["variant"] == "orig"}):
    g = [r for r in rows.values() if r["source"] == src and r["kind"] == kind and r["lang"] == "vi" and r["variant"] == "orig"]
    ans_txt = Counter(r["options"][r["target"]] for r in g)
    pos = Counter(r["target"] for r in g)
    M["floors"][f"{src}|{kind}"] = {"n": len(g), "k": dict(Counter(len(r["options"]) for r in g)),
                                    "random": float(np.mean([1 / len(r["options"]) for r in g])),
                                    "majority_answer_text": ans_txt.most_common(1)[0][1] / len(g), "majority_answer_text_value": ans_txt.most_common(1)[0][0],
                                    "majority_position": pos.most_common(1)[0][1] / len(g), "majority_position_value": LETTERS[pos.most_common(1)[0][0]],
                                    "answer_text_dist": dict(ans_txt.most_common(5)) if len(ans_txt) <= 5 else None}


# ---------------------------------------------------------------- noul bản địa: tỉ lệ dự đoán "Có" và AUROC
def auroc(scores, labels):
    s = np.array(scores); y = np.array(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = s.argsort().argsort() + 1.0
    # xử lý hạng bằng nhau: trung bình hạng
    from collections import defaultdict as dd
    grp = dd(list)
    for i, v in enumerate(s):
        grp[v].append(i)
    for idx in grp.values():
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


for var in ("orig", "nodiac"):
    ids = sel(block="native", kind="noul", variant=var)
    gold_co = [int(rows[i]["target"] == 0) for i in ids]
    e = {"n": len(ids), "gold_co_share": float(np.mean(gold_co))}
    for m in order:
        e[m] = {"pred_co_share": float(np.mean([P[m][i]["pred"] == 0 for i in ids])), "auroc_p_co": auroc([P[m][i]["probs"][0] for i in ids], gold_co),
                "acc": float(np.mean([P[m][i]["correct"] for i in ids]))}
    M["noul"]["có dấu" if var == "orig" else "không dấu"] = e

# ---------------------------------------------------------------- mục F: đọc lại kết luận cũ
F = {}
if "strip (Qwen3)" in P:
    pv = lambda s: paired(P["strip (Qwen3)"], P[CTL], sel(**SLICES[s]), SEED + 700)
    c11 = {"nodiac_parallel_gain": pv("song song · VI không dấu"), "nodiac_native_gain": pv("bản địa · không dấu"),
           "vi_orig_parallel_change": pv("song song · VI có dấu"), "vi_orig_native_change": pv("bản địa · có dấu"), "en_change": pv("song song · EN")}
    ok = all(c11[k]["diff"] >= 0.05 and c11[k]["ci"][0] > 0 for k in ("nodiac_parallel_gain", "nodiac_native_gain")) and \
        all(c11[k]["diff"] >= -0.02 for k in ("vi_orig_parallel_change", "vi_orig_native_change", "en_change"))
    old = load_json(RESULTS / "ft_strip/metrics_strip_v0.json")["criteria"]
    F["strip_v0"] = {"v1_0": {k: old[k]["diff"] for k in c11} | {"verdict": "ĐẠT" if old["ALL_PASS"] else "KHÔNG ĐẠT"},
                     "v1_1": {k: c11[k]["diff"] for k in c11} | {"cis": {k: c11[k]["ci"] for k in c11}, "verdict": "ĐẠT" if ok else "KHÔNG ĐẠT"}}
if "decider-ft ep2" in P:
    PN, PD = load(E / "decider-ft-ep2-text/preds.jsonl"), load(E / "decider-2b-text/preds.jsonl")   # cùng ánh xạ "text" như báo cáo gốc
    g = lambda s: paired(PN, PD, sel(**SLICES[s]), SEED + 800)
    en, vio, nat, nod = g("song song · EN"), g("song song · VI có dấu"), g("bản địa · có dấu"), g("song song · VI không dấu")
    keep = en["diff"] >= -0.02 and vio["diff"] >= -0.02
    gain = (nat["diff"] >= 0.03 and nat["ci"][0] > 0) or nod["diff"] >= 0.10
    harm = (en["diff"] < -0.02 and en["ci"][1] < 0) or (vio["diff"] < -0.02 and vio["ci"][1] < 0)
    old = load_json(RESULTS / "ft_decider/metrics_ft_decider_v0.json")["reading"]
    F["ft_decider_v0"] = {"v1_0": {"en": old["en"], "vi_orig_parallel": old["vi_orig_parallel"], "native_orig": old["native_orig"],
                                   "nodiac_parallel": old["nodiac_parallel"], "verdict": old["verdict"]},
                          "v1_1": {"en": en["diff"], "vi_orig_parallel": vio["diff"], "native_orig": nat["diff"], "native_orig_ci": nat["ci"],
                                   "nodiac_parallel": nod["diff"], "vi_orig_parallel_ci": vio["ci"],
                                   "verdict": "ĐÁNG THEO" if keep and gain else "GÂY HẠI" if harm else "CÒN LẠI: báo số, không kết luận"}}
# restore: chỉ tính lại được trên nguồn không đổi chữ (Belebele, SIB-200, XCOPA) từ preds đã có
rs = load_json(RESULTS / "restore/metrics_restore.json")
PDr = load(RESULTS / "restore/eval/vit5/decider-2b-text/preds.jsonl"); PDs = load(RESULTS / "restore/eval/small/decider-2b-text/preds.jsonl")
PD0 = load(RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl")
ids3 = [i for i in sel(block="parallel", lang="vi", variant="orig") if rows[i]["source"] in ("belebele", "sib200", "xcopa")]
bs = split(ids3)
gap = lambda Pr, suf: macro_boot({s: np.array([float(PD0[i]["correct"]) - float(Pr[f"{i}:restore:{suf}"]["correct"]) for i in v]) for s, v in bs.items()}, bs, SEED + 900)
F["restore"] = {"v1_0_4_sources": {"vit5": rs["reading"]["song song (4 nguồn)|khôi phục vit5"]["gap_points"],
                                   "small": rs["reading"]["song song (4 nguồn)|khôi phục small"]["gap_points"]},
                "v1_1_3_sources_belebele_sib_xcopa": {"vit5": 100 * gap(PDr, "vit5")[0], "small": 100 * gap(PDs, "small")[0]},
                "note": "MASSIVE và kiểm chứng đổi chữ: bản khôi phục cũ không còn khớp; KHÔNG tính lại (không chạy vit5/small). "
                        "Con số 3 nguồn giống hệt nhau ở 1.0 và 1.1 vì ba nguồn này không đổi; không so được với ngưỡng định cho 4 nguồn."}
M["F"] = F
(R / "metrics_v1_1.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ---------------------------------------------------------------- markdown
f1 = lambda x: f"{100 * x:.1f}"; fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}; {100 * c[1]:.1f}]"; fcs = lambda c: f"[{100 * c[0]:+.1f}; {100 * c[1]:+.1f}]"
L = [f"dev_v1_1 SHA-256 `{M['dev_sha256'][:16]}…` (khớp manifest). Ánh xạ dùng cho decider: " +
     "; ".join(f"{k}: {v['used']} (TB năm lát text {100 * v['mean_five_slices_text']:.1f} / keyed {100 * v['mean_five_slices_keyed']:.1f})" for k, v in mapping.items()), "",
     "## 1. Năm lát chính — accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]", "",
     "| Lát | n | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
for name, e in M["slices"].items():
    L.append(f"| {name} | {e[order[0]]['n']} | " + " | ".join(f"{f1(e[m]['acc'])} {fc(e[m]['ci'])}" for m in order) + " |")
L += ["", "## 2. Chênh lệch theo cặp (điểm, [CI 95%], McNemar p)", "", "| Lát | So sánh | Chênh |", "|---|---|---|"]
for name, e in M["paired"].items():
    for k, v in e.items():
        L.append(f"| {name} | {k} | {fs(v['diff'])} {fcs(v['ci'])} (p={v['p']:.2g}) |")
L += ["", "## 3. Theo nguồn — accuracy (%)", "", "| Lát · nguồn | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
for name, e in M["slices"].items():
    for s in e[order[0]]["per_source"]:
        L.append(f"| {name} · {s} | " + " | ".join(f1(e[m]["per_source"][s]) for m in order) + " |")
L += ["", "## 4. Theo kiểu câu (khối bản địa; accuracy %, trung bình đều theo nguồn)", "", "| Lát | n | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
for k, e in M["by_kind"].items():
    L.append(f"| {k} | {e['n']} | " + " | ".join(f1(e[m]) for m in order) + " |")
L += ["", "## 5. Mức sàn (mục D) — dòng tiếng Việt có dấu, tính trên ĐÁP ÁN", "",
      "| Nguồn · kiểu câu | n | số lựa chọn | ngẫu nhiên % | luôn chọn đáp án (chữ) phổ biến nhất % | đáp án đó | luôn chọn vị trí phổ biến nhất % | vị trí |",
      "|---|---|---|---|---|---|---|---|"]
for k, e in M["floors"].items():
    L.append(f"| {k} | {e['n']} | {e['k']} | {f1(e['random'])} | {f1(e['majority_answer_text'])} | {e['majority_answer_text_value'][:60]} | {f1(e['majority_position'])} | {e['majority_position_value']} |")
L += ["", "## 6. Câu noul của khối bản địa — tỉ lệ dự đoán \"Có\" và AUROC của P(Có)", "",
      "| Biến thể | n | tỉ lệ đáp án \"Có\" | " + " | ".join(f"{m}: dự đoán Có % / AUROC / acc %" for m in order) + " |", "|---|---|---|" + "---|" * len(order)]
for var, e in M["noul"].items():
    L.append(f"| {var} | {e['n']} | {f1(e['gold_co_share'])} | " + " | ".join(
        f"{f1(e[m]['pred_co_share'])} / {e[m]['auroc_p_co']:.3f} / {f1(e[m]['acc'])}" for m in order) + " |")
L += ["", "## 7. Đọc lại kết luận cũ trên bản 1.1 (mục F)", ""]
if "strip_v0" in F:
    a, b = F["strip_v0"]["v1_0"], F["strip_v0"]["v1_1"]
    L += ["**strip_v0** (adapter bỏ dấu − adapter đối chứng; ngưỡng: không dấu tăng ≥ 5 ở song song và bản địa, CI không chứa 0; VI có dấu và EN giảm ≤ 2):", "",
          "| Chỉ số | bản 1.0 | bản 1.1 |", "|---|---|---|"]
    for k in ("nodiac_parallel_gain", "nodiac_native_gain", "vi_orig_parallel_change", "vi_orig_native_change", "en_change"):
        L.append(f"| {k} | {fs(a[k])} | {fs(b[k])} {fcs(b['cis'][k])} |")
    L.append(f"| Kết luận | {a['verdict']} | {b['verdict']} |")
if "ft_decider_v0" in F:
    a, b = F["ft_decider_v0"]["v1_0"], F["ft_decider_v0"]["v1_1"]
    L += ["", "**ft_decider_v0** (decider-ft ep2 − decider-2b, cùng ánh xạ \"text\" như báo cáo gốc):", "", "| Chỉ số | bản 1.0 | bản 1.1 |", "|---|---|---|"]
    for k in ("en", "vi_orig_parallel", "native_orig", "nodiac_parallel"):
        L.append(f"| {k} | {fs(a[k])} | {fs(b[k])}" + (f" {fcs(b[k + '_ci'])}" if k + "_ci" in b else "") + " |")
    L.append(f"| Kết luận | {a['verdict']} | {b['verdict']} |")
r_ = F["restore"]
L += ["", "**restore** (decider-2b: có dấu − khôi phục, khối song song):", "", "| Bộ | bản 1.0, 4 nguồn (lát định trước) | bản 1.1 | chỉ Belebele + SIB + XCOPA (như nhau ở 1.0 và 1.1) |", "|---|---|---|---|"]
for k in ("vit5", "small"):
    L.append(f"| {k} | {r_['v1_0_4_sources'][k]:.1f} | không tính lại | {r_['v1_1_3_sources_belebele_sib_xcopa'][k]:.1f} |")
L.append(f"\n{r_['note']}")
(R / "tables_v1_1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:12]))
