"""So adapter 1.7B seed 1 loss mới (full_vocab) với adapter cũ (letters_k, bước 3) trên dev_v0 và val nội bộ.

Ra: results/ft_loss/metrics_loss_fix.json, results/ft_loss/tables_loss_fix.md
"""
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.special import log_softmax

from common import DATA, LETTERS, RESULTS, SEED, read_jsonl
from stats import boot, brier, ci, cluster_index, mcnemar, temper

R, FT = RESULTS / "ft_loss", RESULTS / "ft"
SOURCES = ["belebele", "sib200", "massive", "xcopa"]
LANGS = ["vi", "en"]
CONDS = {  # tên: (normal, reverse, val, temperature)
    "1.7B gốc": (RESULTS / "base_rerun/Qwen3-1.7B", FT / "eval/reverse/Qwen3-1.7B", FT / "eval/val/Qwen3-1.7B", FT / "temp/Qwen3-1.7B.json"),
    "1.7B cũ (letters_k)": (FT / "eval/normal/Qwen3-1.7B-ft-s1", FT / "eval/reverse/Qwen3-1.7B-ft-s1", FT / "eval/val/Qwen3-1.7B-ft-s1",
                            FT / "temp/Qwen3-1.7B-ft-s1.json"),
    "1.7B mới (full_vocab)": (R / "eval/normal/Qwen3-1.7B-fullvocab-s1", R / "eval/reverse/Qwen3-1.7B-fullvocab-s1",
                              R / "eval/val/Qwen3-1.7B-fullvocab-s1", R / "temp/Qwen3-1.7B-fullvocab-s1.json"),
}
OLD, NEW = "1.7B cũ (letters_k)", "1.7B mới (full_vocab)"
rows = {r["id"]: r for r in read_jsonl(DATA / "dev_v0.jsonl")}
val = {r["id"]: r for r in read_jsonl(DATA / "val_v0.jsonl")}
ids_by = {(s, l): sorted(i for i, r in rows.items() if r["source"] == s and r["lang"] == l) for s in SOURCES for l in LANGS}
P = {n: {k: {p["id"]: p for p in read_jsonl(d / "preds.jsonl")} for k, d in zip(("normal", "reverse", "val"), c[:3])} for n, c in CONDS.items()}
T = {n: json.load(open(c[3]))["T"] for n, c in CONDS.items()}
M = {"conditions": {n: [str(x) for x in c] for n, c in CONDS.items()}, "T": T}


def correct(Pn, ids):
    return np.array([Pn[i]["correct"] for i in ids], dtype=float)


def acc_brier(Pn, lang, t=None, seed=SEED):
    per_a, per_b, out = [], [], {}
    for s in SOURCES:
        ids = ids_by[(s, lang)]; cl = cluster_index(ids, rows)
        a = correct(Pn, ids)
        b = np.array([brier(Pn[i]["probs"] if t is None else temper(Pn[i]["probs"], t), rows[i]["target"]) for i in ids])
        per_a.append((a, cl)); per_b.append((b, cl)); out[s] = {"acc": a.mean(), "brier": b.mean()}
    ba, bb = boot(per_a, seed), boot(per_b, seed + 1)
    for j, s in enumerate(SOURCES):
        out[s]["acc_ci"], out[s]["brier_ci"] = ci(ba[:, j]), ci(bb[:, j])
    out["macro"] = {"acc": float(np.mean([out[s]["acc"] for s in SOURCES])), "acc_ci": ci(ba.mean(1)),
                    "brier": float(np.mean([out[s]["brier"] for s in SOURCES])), "brier_ci": ci(bb.mean(1))}
    return out


def paired(PA, PB, lang, seed):
    per, out = [], {}
    for s in SOURCES:
        ids = ids_by[(s, lang)]
        a, b = correct(PA, ids), correct(PB, ids)
        per.append((a - b, cluster_index(ids, rows)))
        out[s] = {"diff": float((a - b).mean()), **mcnemar(a.astype(bool), b.astype(bool))}
    bd = boot(per, seed)
    for j, s in enumerate(SOURCES):
        out[s]["ci"] = ci(bd[:, j])
    allids = [i for s in SOURCES for i in ids_by[(s, lang)]]
    out["macro"] = {"diff": float(np.mean([out[s]["diff"] for s in SOURCES])), "ci": ci(bd.mean(1)),
                    **mcnemar(correct(PA, allids).astype(bool), correct(PB, allids).astype(bool))}
    return out


# ---- letter_mass
lm = {}
for n in CONDS:
    d = {}
    for order in ("normal", "reverse"):
        m = np.array([P[n][order][i]["letter_mass"] for i in rows])
        d[order] = {"n_below_0.5": int((m < 0.5).sum()), "min": float(m.min()), "mean": float(m.mean()),
                    "median": float(np.median(m)),
                    **{f"mean_{l}": float(np.mean([P[n][order][i]["letter_mass"] for i in rows if rows[i]["lang"] == l])) for l in LANGS}}
    vm = np.array([p["letter_mass"] for p in P[n]["val"].values()])
    d["val"] = {"n_below_0.5": int((vm < 0.5).sum()), "mean": float(vm.mean()), "min": float(vm.min())}
    d["top_tokens_normal"] = Counter(P[n]["normal"][i]["top_token"] for i in rows).most_common(5)
    lm[n] = d
M["letter_mass"] = lm

# ---- accuracy / Brier, McNemar mới vs cũ
M["acc_brier"] = {n: {l: acc_brier(P[n]["normal"], l) for l in LANGS} for n in CONDS}
M["brier_T"] = {n: {l: acc_brier(P[n]["normal"], l, t=T[n])["macro"] for l in LANGS} for n in CONDS}
M["new_vs_old"] = {l: paired(P[NEW]["normal"], P[OLD]["normal"], l, SEED + 60) for l in LANGS}

# ---- đảo thứ tự: nhất quán, acc hai lượt, chênh theo vị trí
order_stats = {}
for n in CONDS:
    N, Rv = P[n]["normal"], P[n]["reverse"]; d = {}
    for l in LANGS:
        for s in SOURCES:
            ids = ids_by[(s, l)]; k = N[ids[0]]["k"]
            same = np.mean([N[i]["pred"] == k - 1 - Rv[i]["pred"] for i in ids])
            both = {}
            for j in range(k):
                v = [N[i]["correct"] for i in ids if rows[i]["target"] == j] + [Rv[i]["correct"] for i in ids if k - 1 - rows[i]["target"] == j]
                both[LETTERS[j]] = float(np.mean(v))
            d[f"{s}/{l}"] = {"acc_normal": correct(N, ids).mean(), "acc_reversed": correct(Rv, ids).mean(),
                             "acc_two_orders": (correct(N, ids).mean() + correct(Rv, ids).mean()) / 2, "consistency": float(same),
                             "acc_by_gold_pos_both": both, "pos_spread": max(both.values()) - min(both.values())}
        d[f"macro/{l}"] = {key: float(np.mean([d[f"{s}/{l}"][key] for s in SOURCES]))
                           for key in ["acc_normal", "acc_reversed", "acc_two_orders", "consistency", "pos_spread"]}
    order_stats[n] = d
M["order"] = order_stats

# ---- val nội bộ theo loại câu hỏi: loss (hai định nghĩa) và accuracy theo target_argmax
val_kind = {}
for n in CONDS:
    d = defaultdict(lambda: {"n": 0, "ce_letters_k": 0.0, "ce_full_vocab": 0.0, "correct": 0})
    for i, p in P[n]["val"].items():
        r = val[i]; lg = np.array(p["logits"], dtype=np.float64); t = np.array(r["target"])
        ce_k = float(-(t * log_softmax(lg)).sum())
        ce_full = ce_k - float(np.log(max(p["letter_mass"], 1e-300)))
        for key in (r["kind"], "all"):
            d[key]["n"] += 1; d[key]["ce_letters_k"] += ce_k; d[key]["ce_full_vocab"] += ce_full
            d[key]["correct"] += int(p["pred"] == r["target_argmax"])
    val_kind[n] = {k: {"n": v["n"], "ce_letters_k": v["ce_letters_k"] / v["n"], "ce_full_vocab": v["ce_full_vocab"] / v["n"],
                       "acc_argmax": v["correct"] / v["n"]} for k, v in d.items()}
M["val_by_kind"] = val_kind

# ---- probe, greedy, train
M["probe"] = {n: json.load(open(R / f"probe_{tag}.json", encoding="utf-8")) for n, tag in ((OLD, "old"), (NEW, "new"))}
M["greedy"] = {n: json.load(open(R / f"greedy_{tag}.json", encoding="utf-8"))["summary"] for n, tag in ((OLD, "old"), (NEW, "new"))}
M["train"] = {OLD: json.load(open(FT / "train/Qwen3-1.7B-s1/summary.json")), NEW: json.load(open(R / "train/Qwen3-1.7B-s1/summary.json"))}
M["train_monitor"] = [json.loads(l) for l in open(R / "train/Qwen3-1.7B-s1/train_log.jsonl") if '"letter_mass"' in l or '"val"' in l]

# ---- tiêu chí
new_lm = lm[NEW]["normal"]
crit = {
    "letter_mass": {"n_below_0.5": new_lm["n_below_0.5"], "mean": new_lm["mean"],
                    "pass": new_lm["n_below_0.5"] <= 20 and new_lm["mean"] >= 0.95},
    "vi_accuracy": {"new": M["acc_brier"][NEW]["vi"]["macro"]["acc"], "old": M["acc_brier"][OLD]["vi"]["macro"]["acc"],
                    "diff": M["new_vs_old"]["vi"]["macro"]["diff"], "pass": M["new_vs_old"]["vi"]["macro"]["diff"] >= -0.02},
    "greedy_valid": {"valid": M["greedy"][NEW]["valid_strict"], "n": M["greedy"][NEW]["n"],
                     "pass": M["greedy"][NEW]["valid_strict"] >= 99},
}
crit["all_pass"] = all(v["pass"] for v in crit.values())
M["criteria"] = crit
R.mkdir(exist_ok=True)

# ======================================================================= markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
fcs = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
names = list(CONDS)
L = ["### letter_mass trên dev_v0 (2.000 câu)", "", "| | số câu < 0,5 (thường / đảo) | min | trung bình | TB vi | TB en | val nội bộ: số < 0,5 / TB |",
     "|---|---|---|---|---|---|---|"]
for n in names:
    a, b, v = lm[n]["normal"], lm[n]["reverse"], lm[n]["val"]
    L.append(f"| {n} | {a['n_below_0.5']} / {b['n_below_0.5']} | {a['min']:.3f} | {a['mean']:.3f} | {a['mean_vi']:.3f} | {a['mean_en']:.3f} | {v['n_below_0.5']} / {v['mean']:.3f} |")
L.append("")
for l in LANGS:
    L += [f"### Accuracy (%) và Brier — {'tiếng Việt' if l == 'vi' else 'tiếng Anh'} [CI 95%]", "",
          "| Nguồn | " + " | ".join(f"{n} acc" for n in names) + " | " + " | ".join(f"{n} Brier" for n in names) + " |",
          "|---|" + "---|" * (2 * len(names))]
    for s in SOURCES + ["macro"]:
        lab = "**trung bình đều**" if s == "macro" else s
        L.append(f"| {lab} | " + " | ".join(f"{f(M['acc_brier'][n][l][s]['acc'])} {fc(M['acc_brier'][n][l][s]['acc_ci'])}" for n in names)
                 + " | " + " | ".join(f"{M['acc_brier'][n][l][s]['brier']:.3f} {[round(x, 3) for x in M['acc_brier'][n][l][s]['brier_ci']]}" for n in names) + " |")
    L.append("")
L += ["### Mới − cũ trên cùng câu (điểm %, McNemar chính xác)", "",
      "| Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|" + "---|" * (len(SOURCES) + 2)]
for l in LANGS:
    e = M["new_vs_old"][l]
    L.append(f"| {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={e[s]['p']:.2g})" for s in SOURCES) + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {e['macro']['p']:.2g} |")
L.append("")
L += ["### Đảo thứ tự lựa chọn (trung bình đều theo nguồn)", "",
      "| | Ngôn ngữ | acc thường | acc đảo | acc TB hai lượt | nhất quán | chênh acc theo vị trí đáp án (max−min) |", "|---|---|---|---|---|---|---|"]
for n in names:
    for l in LANGS:
        e = order_stats[n][f"macro/{l}"]
        L.append(f"| {n} | {l} | {f(e['acc_normal'])} | {f(e['acc_reversed'])} | {f(e['acc_two_orders'])} | {f(e['consistency'])} | {f(e['pos_spread'])} |")
L.append("")
L += ["### Temperature (fit trên val nội bộ) và Brier trung bình đều trước / sau", "",
      "| | T | Brier vi T=1 → T | Brier en T=1 → T |", "|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {T[n]:.3f} | {M['acc_brier'][n]['vi']['macro']['brier']:.3f} → {M['brier_T'][n]['vi']['brier']:.3f} | "
             f"{M['acc_brier'][n]['en']['macro']['brier']:.3f} → {M['brier_T'][n]['en']['brier']:.3f} |")
L.append("")
L += ["### Val nội bộ theo loại câu hỏi: loss (CE trên k chữ cái / CE toàn từ vựng) và accuracy theo target_argmax", "",
      "| | " + " | ".join(f"{k} (n)" for k in ["choice", "noul", "score", "all"]) + " |", "|---|---|---|---|---|"]
for n in names:
    cells = []
    for k in ["choice", "noul", "score", "all"]:
        e = val_kind[n][k]
        cells.append(f"{e['ce_letters_k']:.3f} / {e['ce_full_vocab']:.3f} / acc {f(e['acc_argmax'])} (n={e['n']})")
    L.append(f"| {n} | " + " | ".join(cells) + " |")
L.append("")
L += ["### probe_full_softmax.py trên 5 câu cũ", "",
      "| | Câu | k | khối trên k chữ | biến thể ' A' | biến thể 'Ａ' | entropy | logit max | logit std | top-3 |", "|---|---|---|---|---|---|---|---|---|---|"]
for n in (OLD, NEW):
    for x in M["probe"][n]:
        L.append(f"| {n} | {x['id'][:40]} | {x['k']} | {x['mass_k']:.3f} | {x['mass_space_variants_k']:.4f} | {x['mass_fullwidth_variants_k']:.4f} | "
                 f"{x['entropy']:.2f} | {x['logit_max']:.1f} | {x['logit_std']:.2f} | {x['top10'][:3]} |")
L.append("")
L += ["### Sinh greedy 1 token, 100 câu dev_v0 (25 mỗi nguồn)", "", "| | chữ cái hợp lệ (chặt) | biến thể | trùng đáp án đường đọc logit | token không hợp lệ |", "|---|---|---|---|---|"]
for n in (OLD, NEW):
    g = M["greedy"][n]
    L.append(f"| {n} | {g['valid_strict']}/{g['n']} | {g['variant']} | {g['agrees_with_logit_pred']}/{g['n']} | {g['invalid_tokens'][:8]} |")
L.append("")
L += ["### Train", "", "| | thời gian (phút) | VRAM đỉnh (GB) | val loss của lần chạy: đầu → epoch 1 → epoch 2 |", "|---|---|---|---|"]
for n in (OLD, NEW):
    t = M["train"][n]
    L.append(f"| {n} | {t['wall_time_s'] / 60:.1f} | {t['peak_vram_gb']:.1f} | {' → '.join(f'{x:.3f}' for x in t['val_loss_by_epoch'])} |")
L.append("")
old_vl = M["train"][OLD]["val_loss_by_epoch"]   # adapter cũ chỉ ghi CE trên k chữ cái
new_val = [x for x in M["train_monitor"] if x["event"] == "val"]
M["val_loss_split"] = [{"point": lab, "new_letters_k": x["val_loss_letters_k"],
                        "new_neg_log_letter_mass": x["val_loss_full_vocab"] - x["val_loss_letters_k"],
                        "new_full_vocab": x["val_loss_full_vocab"], "old_letters_k": o}
                       for lab, x, o in zip(["bước 0", "cuối epoch 1", "cuối epoch 2"], new_val, old_vl)]
L += ["### Val loss của lần train mới, tách hai phần (val nội bộ 1.000 câu, đo trong lúc train)", "",
      "(b) = full_vocab − letters_k = trung bình −log letter_mass trên val, vì Σ target = 1.", "",
      "| Điểm đo | (a) letters_k, mới | (b) −log letter_mass, mới | full_vocab = (a) + (b), mới | letters_k, adapter cũ |", "|---|---|---|---|---|"]
for x in M["val_loss_split"]:
    L.append(f"| {x['point']} | {x['new_letters_k']:.3f} | {x['new_neg_log_letter_mass']:.3f} | {x['new_full_vocab']:.3f} | {x['old_letters_k']:.3f} |")
L.append("")
L += ["letter_mass trên 200 câu val cố định trong lúc train (loss mới):", ""]
L.append(", ".join(f"bước {x['step']}: {x['mean']:.3f} ({x['n_below_0.5']} < 0,5)" for x in M["train_monitor"] if x["event"] == "letter_mass"))
L.append("")
L += ["### Tiêu chí đạt", "",
      f"- letter_mass: {crit['letter_mass']['n_below_0.5']} câu < 0,5 (ngưỡng ≤ 20), trung bình {crit['letter_mass']['mean']:.3f} (ngưỡng ≥ 0,95) → **{'ĐẠT' if crit['letter_mass']['pass'] else 'KHÔNG ĐẠT'}**",
      f"- Accuracy tiếng Việt: mới {f(crit['vi_accuracy']['new'])} so với cũ {f(crit['vi_accuracy']['old'])}, chênh {fs(crit['vi_accuracy']['diff'])} (ngưỡng ≥ −2,0) → **{'ĐẠT' if crit['vi_accuracy']['pass'] else 'KHÔNG ĐẠT'}**",
      f"- Sinh greedy: {crit['greedy_valid']['valid']}/{crit['greedy_valid']['n']} hợp lệ (ngưỡng ≥ 99) → **{'ĐẠT' if crit['greedy_valid']['pass'] else 'KHÔNG ĐẠT'}**",
      f"- Cả ba: **{'ĐẠT' if crit['all_pass'] else 'KHÔNG ĐẠT'}**", ""]
(R / "tables_loss_fix.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(R / "metrics_loss_fix.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("\n".join(L))
