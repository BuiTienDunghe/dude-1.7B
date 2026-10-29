"""Tính chỉ số từ preds.jsonl của các model: accuracy, Brier (CI bootstrap theo state), mức tụt Anh→Việt (McNemar),
so sánh hai model (McNemar), mốc đoán đều / nhãn phổ biến nhất, phân bố chữ cái được chọn.

Dùng: report.py <run_dir>  → <run_dir>/metrics.json, <run_dir>/report.md
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common import DATA, LETTERS, SEED, read_jsonl

run = Path(sys.argv[1])
SOURCES = ["belebele", "sib200", "massive", "xcopa"]
LANGS = ["en", "vi"]
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B"]
B = 2000

rows = {r["id"]: r for r in read_jsonl(DATA / "dev_v0.jsonl")}
preds = {m: {p["id"]: p for p in read_jsonl(run / m / "preds.jsonl")} for m in MODELS}
for m in MODELS:
    assert set(preds[m]) == set(rows), m


def brier(p, t):
    v = np.asarray(p, dtype=np.float64).copy()
    v[t] -= 1.0
    return float((v ** 2).sum())


def ids_of(src, lang):
    return sorted(i for i, r in rows.items() if r["source"] == src and r["lang"] == lang)


def cluster_index(ids):
    """Gom các câu theo state (Belebele: theo đoạn văn/link; nguồn khác: mỗi câu một state)."""
    groups = defaultdict(list)
    for j, i in enumerate(ids):
        groups[rows[i]["cluster"]].append(j)
    return [np.array(g) for _, g in sorted(groups.items())]


def boot_stat(per_source_vals, rng):
    """per_source_vals: {src: (values array, clusters)}. Mỗi lần lặp lấy mẫu lại cụm trong từng nguồn;
    trả về mảng [B, n_src] trung bình theo nguồn."""
    out = np.zeros((B, len(per_source_vals)))
    for s, (vals, cl) in enumerate(per_source_vals.values()):
        sums = np.array([vals[c].sum() for c in cl]); cnts = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        out[:, s] = sums[pick].sum(1) / cnts[pick].sum(1)
    return out


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def mcnemar(a, b):
    """Kiểm định McNemar chính xác (binomial trên các cặp bất đồng). a, b: mảng bool đúng/sai."""
    n01 = int(((~a) & b).sum()); n10 = int((a & (~b)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)
    return {"a_only_correct": n10, "b_only_correct": n01, "p_exact": p}


metrics = {"bootstrap": {"B": B, "seed": SEED, "unit": "state (Belebele: passage/link; others: sentence)",
                         "macro": "per-replicate mean of per-source means"},
           "brier_definition": "multiclass sum_i (p_i - y_i)^2 over the k options (uniform guess = 1 - 1/k)",
           "models": {}, "baselines": {}, "en_vs_vi": {}, "model_vs_model": {}, "letter_bias": {}, "letter_mass": {}}

# ---------- accuracy / Brier theo nguồn × ngôn ngữ + trung bình đều theo nguồn
for m in MODELS:
    rng = np.random.default_rng(SEED)
    res = {}
    for lang in LANGS:
        acc_vals, bri_vals = {}, {}
        for src in SOURCES:
            ids = ids_of(src, lang)
            acc = np.array([preds[m][i]["correct"] for i in ids], dtype=float)
            bri = np.array([brier(preds[m][i]["probs"], rows[i]["target"]) for i in ids])
            cl = cluster_index(ids)
            acc_vals[src], bri_vals[src] = (acc, cl), (bri, cl)
        ba, bb = boot_stat(acc_vals, rng), boot_stat(bri_vals, rng)
        for s, src in enumerate(SOURCES):
            res[f"{src}/{lang}"] = {"n": len(acc_vals[src][0]), "n_states": len(acc_vals[src][1]),
                                    "acc": float(acc_vals[src][0].mean()), "acc_ci": ci(ba[:, s]),
                                    "brier": float(bri_vals[src][0].mean()), "brier_ci": ci(bb[:, s])}
        res[f"macro/{lang}"] = {"acc": float(np.mean([acc_vals[s][0].mean() for s in SOURCES])), "acc_ci": ci(ba.mean(1)),
                                "brier": float(np.mean([bri_vals[s][0].mean() for s in SOURCES])), "brier_ci": ci(bb.mean(1))}
    metrics["models"][m] = res

# ---------- mốc tham chiếu: đoán đều, luôn chọn nhãn phổ biến nhất (theo lớp gốc, và theo chữ cái)
for src in SOURCES:
    for lang in LANGS:
        ids = ids_of(src, lang)
        ks = np.array([len(rows[i]["options"]) for i in ids])
        tgt = [rows[i]["target"] for i in ids]
        uni_acc = float(np.mean(1 / ks)); uni_brier = float(np.mean(1 - 1 / ks))
        top_letter = Counter(tgt).most_common(1)[0][0]
        maj_letter = float(np.mean([t == top_letter for t in tgt]))
        gl = [rows[i]["gold_label"] for i in ids]
        if gl[0] is not None:
            top_cls = Counter(gl).most_common(1)[0][0]
            # chọn lựa chọn mang nhãn phổ biến nhất nếu có trong các lựa chọn của câu, ngược lại đoán đều
            hits = []
            for i in ids:
                r = rows[i]
                labels = [r["gold_label"]] + (r["flags"][0].split("=", 1)[1].split(",") if r["source"] == "massive" else [])
                present = top_cls in labels if r["source"] == "massive" else True
                hits.append((1.0 if r["gold_label"] == top_cls else 0.0) if present else 1 / len(r["options"]))
            maj_cls = {"label": top_cls, "acc": float(np.mean(hits))}
        else:
            maj_cls = None
        metrics["baselines"][f"{src}/{lang}"] = {"uniform_acc": uni_acc, "uniform_brier": uni_brier,
                                                 "majority_letter": {"letter": LETTERS[top_letter], "acc": maj_letter},
                                                 "majority_class": maj_cls}
metrics["baselines"]["macro"] = {"uniform_acc": float(np.mean([metrics["baselines"][f"{s}/en"]["uniform_acc"] for s in SOURCES]))}

# ---------- mức tụt Anh → Việt theo pair_id, McNemar; CI bootstrap của hiệu theo cụm
for m in MODELS:
    rng = np.random.default_rng(SEED + 1)
    res, diff_vals = {}, {}
    for src in SOURCES:
        en_ids = ids_of(src, "en"); vi_ids = [i[:-2] + "vi" for i in en_ids]
        a = np.array([preds[m][i]["correct"] for i in en_ids]); b = np.array([preds[m][i]["correct"] for i in vi_ids])
        d = a.astype(float) - b.astype(float)
        diff_vals[src] = (d, cluster_index(en_ids))
        res[src] = {"acc_en": float(a.mean()), "acc_vi": float(b.mean()), "drop": float(d.mean()), **mcnemar(a, b)}
    bd = boot_stat(diff_vals, rng)
    for s, src in enumerate(SOURCES):
        res[src]["drop_ci"] = ci(bd[:, s])
    allen = [i for s in SOURCES for i in ids_of(s, "en")]
    a = np.array([preds[m][i]["correct"] for i in allen]); b = np.array([preds[m][i[:-2] + "vi"]["correct"] for i in allen])
    res["macro"] = {"drop": float(np.mean([res[s]["drop"] for s in SOURCES])), "drop_ci": ci(bd.mean(1)),
                    "pooled_mcnemar": mcnemar(a, b)}
    metrics["en_vs_vi"][m] = res

# ---------- so sánh hai model trên cùng câu (McNemar) + CI của hiệu
rng = np.random.default_rng(SEED + 2)
mm = {}
for lang in LANGS + ["all"]:
    dv = {}
    for src in SOURCES:
        ids = ids_of(src, "en") + ids_of(src, "vi") if lang == "all" else ids_of(src, lang)
        a = np.array([preds["Qwen3-1.7B"][i]["correct"] for i in ids]); b = np.array([preds["Qwen3-0.6B"][i]["correct"] for i in ids])
        dv[src] = (a.astype(float) - b.astype(float), cluster_index(ids))
        mm[f"{src}/{lang}"] = {"acc_1.7B": float(a.mean()), "acc_0.6B": float(b.mean()), "diff": float((a.astype(float) - b).mean()),
                               **mcnemar(a, b)}
    bd = boot_stat(dv, rng)
    for s, src in enumerate(SOURCES):
        mm[f"{src}/{lang}"]["diff_ci"] = ci(bd[:, s])
    ids = [i for s in SOURCES for i in (ids_of(s, "en") + ids_of(s, "vi") if lang == "all" else ids_of(s, lang))]
    a = np.array([preds["Qwen3-1.7B"][i]["correct"] for i in ids]); b = np.array([preds["Qwen3-0.6B"][i]["correct"] for i in ids])
    mm[f"macro/{lang}"] = {"diff": float(np.mean([mm[f"{s}/{lang}"]["diff"] for s in SOURCES])), "diff_ci": ci(bd.mean(1)),
                           "pooled_mcnemar": mcnemar(a, b)}
metrics["model_vs_model"] = {"a": "Qwen3-1.7B", "b": "Qwen3-0.6B", **mm}

# ---------- phân bố chữ cái được chọn vs chữ cái đúng; khối lượng xác suất trên các chữ cái
for m in MODELS:
    lb, lm = {}, {}
    for src in SOURCES:
        for lang in LANGS:
            ids = ids_of(src, lang); k = len(rows[ids[0]]["options"])
            pc = Counter(preds[m][i]["pred"] for i in ids); tc = Counter(rows[i]["target"] for i in ids)
            lb[f"{src}/{lang}"] = {"pred": {LETTERS[j]: pc.get(j, 0) for j in range(k)},
                                  "gold": {LETTERS[j]: tc.get(j, 0) for j in range(k)},
                                  "max_pred_share": max(pc.values()) / len(ids),
                                  "mean_prob_by_letter": {LETTERS[j]: float(np.mean([preds[m][i]["probs"][j] for i in ids])) for j in range(k)}}
            mass = np.array([preds[m][i]["letter_mass"] for i in ids])
            lm[f"{src}/{lang}"] = {"mean": float(mass.mean()), "min": float(mass.min()),
                                   "share_below_0.5": float((mass < 0.5).mean()),
                                   "top_tokens": Counter(preds[m][i]["top_token"] for i in ids).most_common(3)}
    metrics["letter_bias"][m], metrics["letter_mass"][m] = lb, lm

(run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------- bảng markdown
f = lambda x: f"{100 * x:.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
L = ["# Kết quả model gốc trên dev_v0", "",
     "Accuracy (%) [CI 95% bootstrap theo state] · Brier (tổng đa lớp, đoán đều = 1−1/k)", "",
     "| Nguồn | k | Đoán đều acc | Nhãn phổ biến acc | " + " | ".join(f"{m} {lang} acc" for m in MODELS for lang in LANGS) + " | "
     + " | ".join(f"{m} {lang} Brier" for m in MODELS for lang in LANGS) + " |",
     "|" + "---|" * (4 + 8)]
for src in SOURCES:
    k = len(rows[ids_of(src, 'en')[0]]["options"])
    bl = metrics["baselines"][f"{src}/vi"]
    maj = bl["majority_class"]["acc"] if bl["majority_class"] else bl["majority_letter"]["acc"]
    cells = [f"{f(metrics['models'][m][f'{src}/{lang}']['acc'])} {fc(metrics['models'][m][f'{src}/{lang}']['acc_ci'])}" for m in MODELS for lang in LANGS]
    cells += [f"{metrics['models'][m][f'{src}/{lang}']['brier']:.3f}" for m in MODELS for lang in LANGS]
    L.append(f"| {src} | {k} | {f(bl['uniform_acc'])} | {f(maj)} | " + " | ".join(cells) + " |")
cells = [f"**{f(metrics['models'][m][f'macro/{lang}']['acc'])}** {fc(metrics['models'][m][f'macro/{lang}']['acc_ci'])}" for m in MODELS for lang in LANGS]
cells += [f"{metrics['models'][m][f'macro/{lang}']['brier']:.3f}" for m in MODELS for lang in LANGS]
L.append(f"| **trung bình đều** | | {f(metrics['baselines']['macro']['uniform_acc'])} | | " + " | ".join(cells) + " |")

L += ["", "## Mức tụt Anh → Việt (điểm %, cùng pair_id; McNemar chính xác)", "",
      "| Nguồn | " + " | ".join(f"{m} tụt [CI] | {m} p" for m in MODELS) + " |", "|---|" + "---|---|" * 2]
for src in SOURCES + ["macro"]:
    cells = []
    for m in MODELS:
        e = metrics["en_vs_vi"][m][src]
        p = e["p_exact"] if src != "macro" else e["pooled_mcnemar"]["p_exact"]
        cells += [f"{f(e['drop'])} {fc(e['drop_ci'])}", f"{p:.3g}"]
    L.append(f"| {src} | " + " | ".join(cells) + " |")

L += ["", "## 1.7B − 0.6B (điểm %, cùng câu; McNemar chính xác)", "",
      "| Nguồn | en hiệu [CI] | en p | vi hiệu [CI] | vi p |", "|---|---|---|---|---|"]
for src in SOURCES + ["macro"]:
    cells = []
    for lang in LANGS:
        e = metrics["model_vs_model"][f"{src}/{lang}"]
        p = e["p_exact"] if src != "macro" else e["pooled_mcnemar"]["p_exact"]
        cells += [f"{f(e['diff'])} {fc(e['diff_ci'])}", f"{p:.3g}"]
    L.append(f"| {src} | " + " | ".join(cells) + " |")

L += ["", "## Phân bố chữ cái được chọn (số câu) so với chữ cái đúng", ""]
for m in MODELS:
    L.append(f"**{m}**"); L.append("")
    L.append("| Nguồn/ngôn ngữ | dự đoán | đúng | tỉ lệ chữ được chọn nhiều nhất | khối xác suất trên các chữ (TB / min) |")
    L.append("|---|---|---|---|---|")
    for key, v in metrics["letter_bias"][m].items():
        lm = metrics["letter_mass"][m][key]
        L.append(f"| {key} | {' '.join(f'{a}:{b}' for a, b in v['pred'].items())} | {' '.join(f'{a}:{b}' for a, b in v['gold'].items())}"
                 f" | {f(v['max_pred_share'])} | {lm['mean']:.3f} / {lm['min']:.3f} |")
    L.append("")
(run / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
