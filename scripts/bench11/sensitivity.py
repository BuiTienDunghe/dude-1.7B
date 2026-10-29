"""Hai phân tích độ nhạy trên dev, CHỈ dùng preds đã lưu (không chạy model). Ra: results/dev_v1_1/metrics_sensitivity.json, sens_tables.md

1. Câu chữ (ViFactCheck, ViWikiFC; có dấu và không dấu): preds chữ 1.0 ("hỗ trợ", results/dev_v1/eval/* và preds cũ của strip/decider-ft)
   và chữ 1.1 ("xác nhận", results/dev_v1_1/eval/*), ghép theo id (id giữ nguyên giữa hai bản). Accuracy trung bình đều theo nguồn,
   CI 95% bootstrap theo state (cluster) trong từng nguồn; chênh theo cặp trên cùng câu + McNemar. "Trung bình hai cách hỏi": trung bình
   0/1 của hai lượt trên từng câu, rồi trung bình đều theo nguồn. Khoảng cách "adapter đối chứng − decider-2b" ở khối bản địa theo ba cách.
2. Ngưỡng noul: trên các câu noul của val_v0 (174 câu), quét ngưỡng t của P(Yes/Có) (lựa chọn Yes/Có xác định theo CHỮ vì thứ tự đã xáo
   lúc dựng val), accuracy so với argmax của target; chọn t có accuracy cao nhất, hòa thì gần 0,5 nhất. Áp t lên câu noul bản địa
   của dev_v1.1 (lựa chọn cố định [Có, Không], P(Có) = probs[0]). Chỉ hai adapter của dự án được chỉnh; decider-2b giữ ngưỡng 0,5.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, read_jsonl  # noqa: E402
from stats import B, mcnemar  # noqa: E402

R = RESULTS / "dev_v1_1"
rows = {r["id"]: r for r in read_jsonl(DATA / "bench_v1_1/dev_v1_1.jsonl")}
load = lambda p: {x["id"]: x for x in read_jsonl(p)}
OLD = {"Qwen3-1.7B gốc": RESULTS / "dev_v1/eval/Qwen3-1.7B/preds.jsonl", "adapter đối chứng": RESULTS / "dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl",
       "strip (Qwen3)": RESULTS / "ft_strip/eval/normal/strip-s1/preds.jsonl", "decider-2b": RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl",
       "decider-ft ep2": RESULTS / "ft_decider/eval/dev_v1_ep2/decider-2b-text/preds.jsonl",
       "decider-2b (keyed)": RESULTS / "dev_v1/eval/decider-2b-keyed/preds.jsonl", "decider-ft ep2 (keyed)": RESULTS / "ft_decider/eval/dev_v1_ep2/decider-2b-keyed/preds.jsonl"}
NEW = {"Qwen3-1.7B gốc": R / "eval/Qwen3-1.7B/preds.jsonl", "adapter đối chứng": R / "eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl",
       "strip (Qwen3)": R / "eval/strip-s1/preds.jsonl", "decider-2b": R / "eval/decider-2b-text/preds.jsonl", "decider-ft ep2": R / "eval/decider-ft-ep2-text/preds.jsonl",
       "decider-2b (keyed)": R / "eval/decider-2b-keyed/preds.jsonl", "decider-ft ep2 (keyed)": R / "eval/decider-ft-ep2-keyed/preds.jsonl"}
MAIN = ["Qwen3-1.7B gốc", "adapter đối chứng", "strip (Qwen3)", "decider-2b", "decider-ft ep2"]
PO, PN = {m: load(p) for m, p in OLD.items()}, {m: load(p) for m, p in NEW.items()}
FC = [i for i, r in rows.items() if r["source"] in ("vifactcheck", "viwikifc")]
for m in OLD:
    assert all(i in PO[m] and i in PN[m] for i in FC), m


def sel(var, kind=None):
    return sorted(i for i in FC if rows[i]["variant"] == var and (kind is None or rows[i]["kind"] == kind))


def macro_boot(vals, ids, seed):
    """vals: {id: giá trị}; trung bình đều theo nguồn, bootstrap theo state trong từng nguồn."""
    by = defaultdict(list)
    for i in ids:
        by[rows[i]["source"]].append(i)
    rng = np.random.default_rng(seed)
    est = np.zeros((B, len(by)))
    point = []
    for s, (src, ii) in enumerate(sorted(by.items())):
        v = np.array([vals[i] for i in ii], float); point.append(v.mean())
        g = defaultdict(list)
        for j, i in enumerate(ii):
            g[rows[i]["cluster"]].append(j)
        cl = [np.array(x) for _, x in sorted(g.items())]
        sums = np.array([v[c].sum() for c in cl]); cnt = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        est[:, s] = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(np.mean(point)), [float(np.percentile(est.mean(1), 2.5)), float(np.percentile(est.mean(1), 97.5))]


# ---------------------------------------------------------------- 1. câu chữ
S1 = {}
for var in ("orig", "nodiac"):
    for kind in (None, "choice", "noul"):
        ids = sel(var, kind)
        key = f"{'có dấu' if var == 'orig' else 'không dấu'} · {kind or 'mọi kiểu'}"
        e = {"n": len(ids)}
        for j, m in enumerate(OLD):
            o = {i: float(PO[m][i]["correct"]) for i in ids}; n = {i: float(PN[m][i]["correct"]) for i in ids}
            a = {i: (o[i] + n[i]) / 2 for i in ids}; d = {i: n[i] - o[i] for i in ids}
            e[m] = {"v1_0": macro_boot(o, ids, SEED + j), "v1_1": macro_boot(n, ids, SEED + j), "avg": macro_boot(a, ids, SEED + j),
                    "diff_1_1_minus_1_0": macro_boot(d, ids, SEED + 100 + j), **{"mcnemar_" + k: v for k, v in mcnemar([n[i] for i in ids], [o[i] for i in ids]).items()},
                    "flip_correct_to_wrong": sum(o[i] == 1 and n[i] == 0 for i in ids), "flip_wrong_to_correct": sum(o[i] == 0 and n[i] == 1 for i in ids),
                    "pred_changed": sum(PO[m][i]["pred"] != PN[m][i]["pred"] for i in ids)}
        # khoảng cách adapter đối chứng − decider-2b theo ba cách
        gap = {}
        for how in ("v1_0", "v1_1", "avg"):
            def val(m, i):
                o, n = float(PO[m][i]["correct"]), float(PN[m][i]["correct"])
                return o if how == "v1_0" else n if how == "v1_1" else (o + n) / 2
            dd = {i: val("adapter đối chứng", i) - val("decider-2b", i) for i in ids}
            gap[how] = macro_boot(dd, ids, SEED + 200)
        e["gap_control_minus_decider"] = gap
        S1[key] = e

# ---------------------------------------------------------------- 2. ngưỡng noul
val = {r["id"]: r for r in read_jsonl(DATA / "val_v0.jsonl")}
VALP = {"adapter đối chứng": RESULTS / "ft_loss/eval/val/Qwen3-1.7B-fullvocab-s1/preds.jsonl", "strip (Qwen3)": RESULTS / "ft_strip/eval/val/strip-s1/preds.jsonl"}
YES = ("Yes", "Có")
nl = [r for r in val.values() if r["kind"] == "noul"]
T = np.round(np.arange(0.01, 1.0, 0.01), 2)
S2 = {"val_noul_n": len(nl), "val_noul_by_text_lang": {k: sum(r["text_lang"] == k for r in nl) for k in ("en", "vi", "multi")}, "models": {}}
devn = {var: [i for i in FC if rows[i]["kind"] == "noul" and rows[i]["variant"] == var] for var in ("orig", "nodiac")}
for i in devn["orig"]:
    assert rows[i]["options"] == ["Có", "Không"]
for m, p in VALP.items():
    P = load(p)
    py, gy = [], []
    checks = []
    for r in nl:
        iy = [o in YES for o in r["options"]].index(True)
        pr = P[r["id"]]["probs"]
        assert len(pr) == 2 and abs(sum(pr) - 1) < 1e-4
        py.append(pr[iy]); gy.append(int(np.argmax(r["target"]) == iy))
        if len(checks) < 4:
            checks.append({"id": r["id"][:70], "options": r["options"], "target": r["target"], "probs": [round(x, 3) for x in pr], "P_yes": round(pr[iy], 3)})
    py, gy = np.array(py), np.array(gy)
    curve = {float(t): float(np.mean((py >= t).astype(int) == gy)) for t in T}
    best = max(curve.values())
    cands = [t for t, a in curve.items() if a == best]
    t_star = min(cands, key=lambda t: (abs(t - 0.5), t))
    near = [t for t, a in curve.items() if a >= best - 0.005]
    e = {"direction_checks": checks, "val_acc_at_0.5": curve[0.5], "val_best_acc": best, "t_star": t_star, "t_star_logit_offset": float(np.log(t_star / (1 - t_star))),
         "thresholds_with_max_acc": cands, "thresholds_within_0.5pt_of_max": [min(near), max(near)], "val_yes_share_gold": float(gy.mean()),
         "val_pred_yes_share_at_0.5": float((py >= 0.5).mean()), "val_pred_yes_share_at_t": float((py >= t_star).mean()),
         "curve": {f"{t:.2f}": a for t, a in curve.items() if abs(round(t * 100) % 5) == 0}, "dev": {}}
    PD = PN[m]
    for var, ids in devn.items():
        before = {i: float(PD[i]["pred"] == rows[i]["target"]) for i in ids}
        after = {i: float((0 if PD[i]["probs"][0] >= t_star else 1) == rows[i]["target"]) for i in ids}
        e["dev"]["có dấu" if var == "orig" else "không dấu"] = {
            "n": len(ids), "acc_before": macro_boot(before, ids, SEED + 300), "acc_after": macro_boot(after, ids, SEED + 300),
            "diff_after_minus_before": macro_boot({i: after[i] - before[i] for i in ids}, ids, SEED + 301),
            "pred_co_before": float(np.mean([PD[i]["probs"][0] >= 0.5 for i in ids])), "pred_co_after": float(np.mean([PD[i]["probs"][0] >= t_star for i in ids]))}
    S2["models"][m] = e
for var, ids in devn.items():
    PD = PN["decider-2b"]
    S2.setdefault("decider-2b_at_0.5", {})["có dấu" if var == "orig" else "không dấu"] = {
        "acc": macro_boot({i: float(PD[i]["correct"]) for i in ids}, ids, SEED + 300), "pred_co": float(np.mean([PD[i]["pred"] == 0 for i in ids])),
        "gold_co": float(np.mean([rows[i]["target"] == 0 for i in ids])),
        "floor_majority": float(max(np.mean([rows[i]["target"] == 0 for i in ids]), np.mean([rows[i]["target"] == 1 for i in ids])))}
M = {"wording": S1, "noul_threshold": S2}
(R / "metrics_sensitivity.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ---------------------------------------------------------------- bảng
f1 = lambda x: f"{100 * x:.1f}"; fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}; {100 * c[1]:.1f}]"; fcs = lambda c: f"[{100 * c[0]:+.1f}; {100 * c[1]:+.1f}]"
L = ["## 1. Câu chữ 1.0 (\"hỗ trợ\") so với 1.1 (\"xác nhận\") — ViFactCheck + ViWikiFC (accuracy %, trung bình đều theo nguồn)", ""]
for key, e in S1.items():
    L += [f"### {key} (n = {e['n']})", "", "| Model | chữ 1.0 | chữ 1.1 | 1.1 − 1.0 [CI] (McNemar p) | đúng→sai / sai→đúng / đổi dự đoán | trung bình hai cách hỏi |", "|---|---|---|---|---|---|"]
    for m in OLD:
        x = e[m]
        L.append(f"| {m} | {f1(x['v1_0'][0])} {fc(x['v1_0'][1])} | {f1(x['v1_1'][0])} {fc(x['v1_1'][1])} | {fs(x['diff_1_1_minus_1_0'][0])} {fcs(x['diff_1_1_minus_1_0'][1])} (p={x['mcnemar_p']:.2g}) | "
                 f"{x['flip_correct_to_wrong']} / {x['flip_wrong_to_correct']} / {x['pred_changed']} | {f1(x['avg'][0])} {fc(x['avg'][1])} |")
    g = e["gap_control_minus_decider"]
    L += ["", f"Khoảng cách adapter đối chứng − decider-2b: chữ 1.0 {fs(g['v1_0'][0])} {fcs(g['v1_0'][1])}; chữ 1.1 {fs(g['v1_1'][0])} {fcs(g['v1_1'][1])}; "
              f"trung bình hai cách hỏi {fs(g['avg'][0])} {fcs(g['avg'][1])}", ""]
L += ["## 2. Ngưỡng câu noul (chọn trên val_v0, áp lên dev_v1.1)", "",
      f"val_v0: {S2['val_noul_n']} câu noul ({S2['val_noul_by_text_lang']}).", "",
      "| Model | t* (P(Có) ≥ t*) | độ lệch logit | acc val ở 0,5 → ở t* | ngưỡng cùng acc tối đa | dải trong 0,5 điểm của tối đa | val: tỉ lệ đáp án Yes / dự đoán Yes ở 0,5 → t* |",
      "|---|---|---|---|---|---|---|"]
for m, e in S2["models"].items():
    L.append(f"| {m} | {e['t_star']:.2f} | {e['t_star_logit_offset']:+.2f} | {f1(e['val_acc_at_0.5'])} → {f1(e['val_best_acc'])} | {e['thresholds_with_max_acc'][0]:.2f}–{e['thresholds_with_max_acc'][-1]:.2f} | "
             f"{e['thresholds_within_0.5pt_of_max'][0]:.2f}–{e['thresholds_within_0.5pt_of_max'][1]:.2f} | {f1(e['val_yes_share_gold'])} / {f1(e['val_pred_yes_share_at_0.5'])} → {f1(e['val_pred_yes_share_at_t'])} |")
L += ["", "Đường accuracy trên val theo ngưỡng (mỗi 0,05):", "", "| t | " + " | ".join(S2["models"]) + " |", "|---|" + "---|" * len(S2["models"])]
for t in next(iter(S2["models"].values()))["curve"]:
    L.append(f"| {t} | " + " | ".join(f1(e["curve"][t]) for e in S2["models"].values()) + " |")
L += ["", "Áp lên câu noul bản địa của dev_v1.1:", "", "| Biến thể | Model | acc trước (0,5) | acc sau (t*) | sau − trước [CI] | dự đoán \"Có\" trước → sau |", "|---|---|---|---|---|---|"]
for var in ("có dấu", "không dấu"):
    for m, e in S2["models"].items():
        d = e["dev"][var]
        L.append(f"| {var} | {m} | {f1(d['acc_before'][0])} {fc(d['acc_before'][1])} | {f1(d['acc_after'][0])} {fc(d['acc_after'][1])} | {fs(d['diff_after_minus_before'][0])} {fcs(d['diff_after_minus_before'][1])} | "
                 f"{f1(d['pred_co_before'])} → {f1(d['pred_co_after'])} |")
    dd = S2["decider-2b_at_0.5"][var]
    L.append(f"| {var} | decider-2b (0,5, không chỉnh) | {f1(dd['acc'][0])} {fc(dd['acc'][1])} | — | — | {f1(dd['pred_co'])} (đáp án Có: {f1(dd['gold_co'])}; sàn đa số {f1(dd['floor_majority'])}) |")
(R / "sens_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({m: {"t*": e["t_star"], "val": [round(e["val_acc_at_0.5"], 3), round(e["val_best_acc"], 3)],
                      "dev_orig": [round(e["dev"]["có dấu"]["acc_before"][0], 3), round(e["dev"]["có dấu"]["acc_after"][0], 3)]} for m, e in S2["models"].items()}, ensure_ascii=False))
