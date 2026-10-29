"""Bảng kết quả dev_v1 theo lát. Ra: results/dev_v1/tables_dev_v1.md, results/dev_v1/metrics_dev_v1.json

Model: Qwen3-1.7B gốc, Qwen3-1.7B full_vocab (dự án), decider-2b (choice: cách ánh xạ có accuracy cao hơn trên toàn bộ câu
choice; noul/score: kiểu gốc). Accuracy trong mỗi lát là trên mọi câu của lát (micro), CI 95% bootstrap theo state (cụm =
đoạn văn/bài báo/văn bản; bản có dấu và không dấu cùng một cụm). Mốc: đoán đều = TB 1/k; nhãn phổ biến = luôn chọn vị trí
đáp án đúng phổ biến nhất của lát (với choice/score/noul là chỉ số lựa chọn).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, read_jsonl  # noqa: E402
from stats import mcnemar  # noqa: E402

R = RESULTS / "dev_v1"
rows = {r["id"]: r for r in read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")}
P = {"Qwen3-1.7B gốc": {p["id"]: p for p in read_jsonl(R / "eval/Qwen3-1.7B/preds.jsonl")},
     "dự án (full_vocab)": {p["id"]: p for p in read_jsonl(R / "eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl")}}
dec = {m: {p["id"]: p for p in read_jsonl(R / f"eval/decider-2b-{m}/preds.jsonl")} for m in ("text", "keyed")}
ch = [i for i, r in rows.items() if r["kind"] not in ("noul", "score")]   # dòng dev_v0 mang kind cũ, đều là choice
acc_map = {m: np.mean([dec[m][i]["correct"] for i in ch]) for m in dec}
best = max(acc_map, key=acc_map.get)
P["decider-2b"] = dec[best]
MODELS = list(P)
for m in MODELS:
    assert set(P[m]) == set(rows), m
B = 2000


def cl_boot(vals, ids, seed):
    """CI 95% bootstrap theo cụm của trung bình vals (micro)."""
    g = defaultdict(list)
    for v, i in zip(vals, ids):
        g[rows[i]["cluster"]].append(v)
    sums = np.array([sum(x) for x in g.values()]); cnts = np.array([len(x) for x in g.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(B, len(sums)))
    est = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def brier(p, t):
    v = np.array(p, dtype=float); v[t] -= 1
    return float((v ** 2).sum())


def stats(m, ids, seed=SEED):
    c = [float(P[m][i]["correct"]) for i in ids]
    out = {"n": len(ids), "n_states": len({rows[i]["cluster"] for i in ids}), "acc": float(np.mean(c)), "acc_ci": cl_boot(c, ids, seed),
           "brier": float(np.mean([brier(P[m][i]["probs"], rows[i]["target"]) for i in ids]))}
    return out


def baselines(ids):
    ks = [len(rows[i]["options"]) for i in ids]
    tc = Counter(rows[i]["target"] for i in ids)
    return {"uniform": float(np.mean([1 / k for k in ks])), "majority_position": tc.most_common(1)[0][1] / len(ids)}


def sel(**kw):
    return sorted(i for i, r in rows.items() if all((r[k] in v if isinstance(v, (list, tuple, set)) else r[k] == v) for k, v in kw.items()))


def bin_of(r):
    n = r["n_state_tokens"]
    return "≤256" if n <= 256 else ("257–1024" if n <= 1024 else ">1024")


SLICES = {
    "song song · en · có dấu": sel(block="parallel", lang="en"),
    "song song · vi · có dấu": sel(block="parallel", lang="vi", variant="orig"),
    "song song · vi · không dấu": sel(block="parallel", lang="vi", variant="nodiac"),
    "bản địa · có dấu": sel(block="native", variant="orig"),
    "bản địa · không dấu": sel(block="native", variant="nodiac"),
}
for s in ("belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc", "vihsd"):
    for lang, var in (("en", "orig"), ("vi", "orig"), ("vi", "nodiac")):
        ids = sel(source=s, lang=lang, variant=var)
        if ids:
            SLICES[f"{s} · {lang} · {'có dấu' if var == 'orig' else 'không dấu'}"] = ids
for k in ("choice", "noul", "score"):
    for var in ("orig", "nodiac"):
        SLICES[f"bản địa · {k} · {'có dấu' if var == 'orig' else 'không dấu'}"] = sel(block="native", kind=k, variant=var)
for b in ("≤256", "257–1024", ">1024"):
    SLICES[f"độ dài state {b} (vi, có dấu)"] = sorted(i for i, r in rows.items() if r["lang"] == "vi" and r["variant"] == "orig" and bin_of(r) == b)
    SLICES[f"độ dài state {b} (vi, không dấu)"] = sorted(i for i, r in rows.items() if r["lang"] == "vi" and r["variant"] == "nodiac" and bin_of(r) == b)

M = {"decider_choice_mapping": best, "decider_choice_acc_by_mapping": {k: float(v) for k, v in acc_map.items()}, "slices": {}}
for j, (name, ids) in enumerate(SLICES.items()):
    if not ids:
        continue
    e = {"baselines": baselines(ids)}
    for m in MODELS:
        e[m] = stats(m, ids, SEED + j)
    a, b = np.array([P["decider-2b"][i]["correct"] for i in ids]), np.array([P["dự án (full_vocab)"][i]["correct"] for i in ids])
    e["decider_minus_project"] = {"diff": float(a.mean() - b.mean()), **mcnemar(a, b),
                                  "ci": cl_boot(list(a.astype(float) - b.astype(float)), ids, SEED + 900 + j)}
    M["slices"][name] = e


def paired_drop(m, ids_a, map_b, seed):
    """Mức tụt a → b trên cùng pair (b = map_b(id_a)); McNemar."""
    a = np.array([P[m][i]["correct"] for i in ids_a]); b = np.array([P[m][map_b(i)]["correct"] for i in ids_a])
    return {"n": len(ids_a), "acc_a": float(a.mean()), "acc_b": float(b.mean()), "drop": float(a.mean() - b.mean()), **mcnemar(a, b),
            "ci": cl_boot(list(a.astype(float) - b.astype(float)), ids_a, seed)}


M["diacritics_drop"] = {}
M["en_vi_drop"] = {}
for s in ("belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc", "vihsd", "ALL_PARALLEL", "ALL_NATIVE"):
    if s == "ALL_PARALLEL":
        ids = sel(block="parallel", lang="vi", variant="orig")
    elif s == "ALL_NATIVE":
        ids = sel(block="native", variant="orig")
    else:
        ids = sel(source=s, lang="vi", variant="orig")
    M["diacritics_drop"][s] = {m: paired_drop(m, ids, lambda i: i + ":nodiac", SEED + 300) for m in MODELS}
    if s in ("belebele", "sib200", "massive", "xcopa", "ALL_PARALLEL"):
        en = sel(source=s, lang="en") if s != "ALL_PARALLEL" else sel(block="parallel", lang="en")
        M["en_vi_drop"][s] = {m: paired_drop(m, en, lambda i: i[:-2] + "vi", SEED + 400) for m in MODELS}

# score: sai số tuyệt đối của mức kỳ vọng; noul: accuracy ở ngưỡng 0,5 và Brier nhị phân (P(có) − y)^2
M["score_noul"] = {}
for var in ("orig", "nodiac"):
    for k in ("score", "noul"):
        ids = sel(block="native", kind=k, variant=var)
        if not ids:
            continue
        for s in sorted({rows[i]["source"] for i in ids}):
            sid = [i for i in ids if rows[i]["source"] == s]
            e = {}
            for m in MODELS:
                if k == "score":
                    err = [abs(sum(j * p for j, p in enumerate(P[m][i]["probs"])) - rows[i]["target"]) for i in sid]
                    e[m] = {"mae_expected_level": float(np.mean(err)), "acc_argmax": float(np.mean([P[m][i]["correct"] for i in sid]))}
                else:
                    y = [1.0 if rows[i]["target"] == 0 else 0.0 for i in sid]          # lựa chọn 0 = "Có"
                    pyes = [P[m][i]["probs"][0] for i in sid]
                    e[m] = {"acc_at_0.5": float(np.mean([(p >= 0.5) == (t == 1.0) for p, t in zip(pyes, y)])),
                            "brier_binary": float(np.mean([(p - t) ** 2 for p, t in zip(pyes, y)])), "share_true": float(np.mean(y))}
            if k == "score":
                gold = [rows[i]["target"] for i in sid]
                e["baseline_mae_predict_mean"] = float(np.mean([abs(np.mean(gold) - g) for g in gold]))
            else:
                e["baseline_brier_predict_rate"] = float(np.mean(y) * (1 - np.mean(y)))
            M["score_noul"][f"{s} · {k} · {var}"] = e
(R / "metrics_dev_v1.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ======================================================================= markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
L = [f"decider-2b, câu choice: cách ánh xạ **{best}** (accuracy trên mọi câu choice: " +
     ", ".join(f"{k} {f(v)}" for k, v in acc_map.items()) + "). noul/score dùng kiểu câu hỏi gốc.", "",
     "## Accuracy theo lát (%), CI 95% bootstrap theo state", "",
     "| Lát | n (state) | đoán đều | nhãn phổ biến | " + " | ".join(MODELS) + " | decider − dự án [CI] (p) |",
     "|---|---|---|---|" + "---|" * len(MODELS) + "---|"]
for name, e in M["slices"].items():
    m0 = e[MODELS[0]]
    d = e["decider_minus_project"]
    L.append(f"| {name} | {m0['n']} ({m0['n_states']}) | {f(e['baselines']['uniform'])} | {f(e['baselines']['majority_position'])} | "
             + " | ".join(f"{f(e[m]['acc'])} {fc(e[m]['acc_ci'])}" for m in MODELS)
             + f" | {fs(d['diff'])} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] ({d['p']:.2g}) |")
L += ["", "## Brier đa lớp theo lát (tổng (p−y)²; đoán đều = 1 − 1/k)", "", "| Lát | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for name, e in M["slices"].items():
    L.append(f"| {name} | " + " | ".join(f"{e[m]['brier']:.3f}" for m in MODELS) + " |")
for title, key in (("Mức tụt khi bỏ dấu (có dấu − không dấu, cùng pair_id; McNemar)", "diacritics_drop"),
                   ("Mức tụt Anh → Việt trên khối song song (en − vi có dấu, cùng pair_id; McNemar)", "en_vi_drop")):
    L += ["", f"## {title}", "", "| Nguồn | " + " | ".join(f"{m}: tụt [CI] (p)" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for s, v in M[key].items():
        L.append(f"| {s} (n={v[MODELS[0]]['n']}) | " + " | ".join(
            f"{f(v[m]['acc_a'])}→{f(v[m]['acc_b'])}: {fs(v[m]['drop'])} [{100 * v[m]['ci'][0]:+.1f}, {100 * v[m]['ci'][1]:+.1f}] ({v[m]['p']:.2g})" for m in MODELS) + " |")
L += ["", "## score: sai số tuyệt đối của mức kỳ vọng (MAE) · noul: accuracy ở ngưỡng 0,5 và Brier nhị phân", "",
      "| Lát | " + " | ".join(MODELS) + " | mốc |", "|---|" + "---|" * (len(MODELS) + 1)]
for name, e in M["score_noul"].items():
    if " · score · " in name:
        cells = [f"MAE {e[m]['mae_expected_level']:.3f} / acc {f(e[m]['acc_argmax'])}" for m in MODELS]
        base = f"MAE luôn đoán mức TB {e['baseline_mae_predict_mean']:.3f}"
    else:
        cells = [f"acc {f(e[m]['acc_at_0.5'])} / Brier {e[m]['brier_binary']:.3f}" for m in MODELS]
        base = f"Brier luôn đoán tỉ lệ {e['baseline_brier_predict_rate']:.3f} (tỉ lệ đúng {f(e[MODELS[0]]['share_true'])})"
    L.append(f"| {name} | " + " | ".join(cells) + f" | {base} |")
(R / "tables_dev_v1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
