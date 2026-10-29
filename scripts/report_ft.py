"""Báo cáo so sánh model gốc và sau fine-tune trên dev_v0.

Đọc:
  results/base_rerun/<model>/preds.jsonl                 gốc, thứ tự thường (khớp từng byte với run1 của bước dev_v0)
  results/ft/eval/normal/<model>-ft-s<seed>/preds.jsonl  sau fine-tune, thứ tự thường
  results/ft/eval/reverse/<name>/preds.jsonl             thứ tự lựa chọn đảo ngược (gốc và sau fine-tune)
  results/ft/temp/<name>.json                            temperature fit trên val nội bộ
  results/ft/train/<model>-s<seed>/summary.json          thời gian train, VRAM đỉnh, val loss
Ra: results/ft/metrics_ft.json, results/ft/report_ft.md
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, LETTERS, RESULTS, SEED, read_jsonl
from stats import boot, brier, ci, cluster_index, mcnemar, temper

FT = RESULTS / "ft"
SOURCES = ["belebele", "sib200", "massive", "xcopa"]
LANGS = ["vi", "en"]
MODELS = os.environ.get("REPORT_MODELS", "Qwen3-0.6B,Qwen3-1.7B").split(",")   # chỉ để chạy thử từng phần
SEEDS = sorted({int(p.name.rsplit("-s", 1)[1]) for p in (FT / "train").glob("*-s*") if (p / "summary.json").exists()})
BOTH = len(MODELS) == 2

rows = {r["id"]: r for r in read_jsonl(DATA / "dev_v0.jsonl")}
ids_by = {(s, l): sorted(i for i, r in rows.items() if r["source"] == s and r["lang"] == l) for s in SOURCES for l in LANGS}

conds = []   # (model, label, name)
for m in MODELS:
    conds.append((m, "gốc", m))
    for s in SEEDS:
        conds.append((m, f"ft s{s}", f"{m}-ft-s{s}"))


def load(name, order):
    if order == "normal":
        path = (RESULTS / "base_rerun" / name if "-ft-" not in name else FT / "eval" / "normal" / name) / "preds.jsonl"
    else:
        path = FT / "eval" / "reverse" / name / "preds.jsonl"
    P = {p["id"]: p for p in read_jsonl(path)}
    assert set(P) == set(rows), path
    return P


normal = {c[2]: load(c[2], "normal") for c in conds}
reverse = {c[2]: load(c[2], "reverse") for c in conds}
temps = {c[2]: json.load(open(FT / "temp" / f"{c[2]}.json")) for c in conds}
M = {"seeds": SEEDS, "conditions": [c[2] for c in conds]}


def correct(P, ids):
    return np.array([P[i]["correct"] for i in ids], dtype=float)


def acc_brier(P, lang, T=None, seed=SEED):
    per_a, per_b, out = [], [], {}
    for s in SOURCES:
        ids = ids_by[(s, lang)]
        cl = cluster_index(ids, rows)
        a = correct(P, ids)
        b = np.array([brier(P[i]["probs"] if T is None else temper(P[i]["probs"], T), rows[i]["target"]) for i in ids])
        per_a.append((a, cl)); per_b.append((b, cl))
        out[s] = {"acc": a.mean(), "brier": b.mean()}
    ba, bb = boot(per_a, seed), boot(per_b, seed + 1)
    for j, s in enumerate(SOURCES):
        out[s]["acc_ci"], out[s]["brier_ci"] = ci(ba[:, j]), ci(bb[:, j])
    out["macro"] = {"acc": float(np.mean([out[s]["acc"] for s in SOURCES])), "acc_ci": ci(ba.mean(1)),
                    "brier": float(np.mean([out[s]["brier"] for s in SOURCES])), "brier_ci": ci(bb.mean(1))}
    return out


def paired_diff(PA, PB, lang_ids_fn, seed):
    """Hiệu accuracy A−B trên cùng câu: theo nguồn + trung bình đều, CI bootstrap theo state, McNemar chính xác."""
    per, out = [], {}
    for s in SOURCES:
        ids = lang_ids_fn(s)
        a, b = correct(PA, ids), correct(PB, ids)
        per.append((a - b, cluster_index(ids, rows)))
        out[s] = {"diff": float((a - b).mean()), **mcnemar(a.astype(bool), b.astype(bool))}
    bd = boot(per, seed)
    for j, s in enumerate(SOURCES):
        out[s]["ci"] = ci(bd[:, j])
    allids = [i for s in SOURCES for i in lang_ids_fn(s)]
    out["macro"] = {"diff": float(np.mean([out[s]["diff"] for s in SOURCES])), "ci": ci(bd.mean(1)),
                    **mcnemar(correct(PA, allids).astype(bool), correct(PB, allids).astype(bool))}
    return out


# ---- A. accuracy / Brier (thứ tự thường), trước & sau temperature
M["acc_brier"] = {c[2]: {l: acc_brier(normal[c[2]], l) for l in LANGS} for c in conds}
M["brier_T"] = {c[2]: {"T": temps[c[2]]["T"], **{l: acc_brier(normal[c[2]], l, T=temps[c[2]]["T"])["macro"] for l in LANGS}} for c in conds}

# ---- B. mỗi model sau fine-tune so với chính nó trước fine-tune
M["ft_vs_base"] = {f"{m}-ft-s{s}": {l: paired_diff(normal[f"{m}-ft-s{s}"], normal[m], lambda src, l=l: ids_by[(src, l)], SEED + 10)
                                    for l in LANGS} for m in MODELS for s in SEEDS}

# ---- C. 0.6B vs 1.7B sau fine-tune (cùng seed), và chênh lệch giữa hai seed của cùng model
if BOTH:
    M["small_vs_large_ft"] = {f"s{s}": {l: paired_diff(normal[f"Qwen3-1.7B-ft-s{s}"], normal[f"Qwen3-0.6B-ft-s{s}"],
                                                       lambda src, l=l: ids_by[(src, l)], SEED + 20) for l in LANGS} for s in SEEDS}
    M["small_vs_large_base"] = {l: paired_diff(normal["Qwen3-1.7B"], normal["Qwen3-0.6B"], lambda src, l=l: ids_by[(src, l)], SEED + 21)
                                for l in LANGS}
if len(SEEDS) >= 2:
    a, b = SEEDS[0], SEEDS[1]
    M["seed_gap"] = {m: {l: paired_diff(normal[f"{m}-ft-s{a}"], normal[f"{m}-ft-s{b}"], lambda src, l=l: ids_by[(src, l)], SEED + 30)
                         for l in LANGS} for m in MODELS}
    # 1.7B − 0.6B theo mọi cặp seed (cùng seed và chéo), và trung bình các cặp
    if BOTH:
        M["small_vs_large_ft_pairs"] = {f"1.7B s{x} − 0.6B s{y}": {l: paired_diff(normal[f"Qwen3-1.7B-ft-s{x}"], normal[f"Qwen3-0.6B-ft-s{y}"],
                                                                                  lambda src, l=l: ids_by[(src, l)], SEED + 50) for l in LANGS}
                                        for x in SEEDS for y in SEEDS}
        M["small_vs_large_ft_pairs_mean"] = {l: {s: float(np.mean([v[l][s]["diff"] for v in M["small_vs_large_ft_pairs"].values()]))
                                                 for s in SOURCES + ["macro"]} for l in LANGS}
    # accuracy theo seed: s1, s2, trung bình, s1 − s2
    M["per_seed"] = {m: {l: {s: {**{f"s{x}": M["acc_brier"][f"{m}-ft-s{x}"][l][s]["acc"] for x in SEEDS},
                                 "mean": float(np.mean([M["acc_brier"][f"{m}-ft-s{x}"][l][s]["acc"] for x in SEEDS])),
                                 "s1_minus_s2": M["acc_brier"][f"{m}-ft-s{a}"][l][s]["acc"] - M["acc_brier"][f"{m}-ft-s{b}"][l][s]["acc"]}
                             for s in SOURCES + ["macro"]} for l in LANGS} for m in MODELS}

# ---- D. mức tụt Anh → Việt (ghép theo pair_id) trước và sau
M["en_vi_drop"] = {c[2]: paired_diff(normal[c[2]], {i[:-2] + "en": normal[c[2]][i] for i in normal[c[2]] if i.endswith(":vi")},
                                     lambda src: ids_by[(src, "en")], SEED + 40) for c in conds}
# (PA = câu tiếng Anh, PB = câu tiếng Việt cùng pair_id, đánh chỉ số theo id tiếng Anh)

# ---- E/G/H. phân bố chữ cái, đảo thứ tự lựa chọn, khối xác suất trên chữ cái
pos = {}
for c in conds:
    name = c[2]; N, R = normal[name], reverse[name]
    d = {}
    for l in LANGS:
        for s in SOURCES:
            ids = ids_by[(s, l)]; k = N[ids[0]]["k"]
            cn, cr = correct(N, ids), correct(R, ids)
            same = np.array([N[i]["pred"] == k - 1 - R[i]["pred"] for i in ids], dtype=float)
            by_pos = {LETTERS[j]: float(np.mean([N[i]["correct"] for i in ids if rows[i]["target"] == j]))
                      for j in range(k) if any(rows[i]["target"] == j for i in ids)}
            # gộp hai lượt: vị trí của đáp án đúng ở lượt đảo là k-1-target
            both = {}
            for j in range(k):
                v = [N[i]["correct"] for i in ids if rows[i]["target"] == j] + [R[i]["correct"] for i in ids if k - 1 - rows[i]["target"] == j]
                both[LETTERS[j]] = float(np.mean(v))
            d[f"{s}/{l}"] = {
                "acc_normal": cn.mean(), "acc_reversed": cr.mean(), "acc_mean_two_orders": (cn.mean() + cr.mean()) / 2,
                "consistency": same.mean(), "acc_by_gold_pos_normal": by_pos, "acc_by_gold_pos_both_orders": both,
                "pos_spread_both": max(both.values()) - min(both.values()),
                "pred_hist_normal": {LETTERS[j]: Counter(N[i]["pred"] for i in ids).get(j, 0) for j in range(k)},
                "max_pred_share_normal": max(Counter(N[i]["pred"] for i in ids).values()) / len(ids),
                "letter_mass_mean": float(np.mean([N[i]["letter_mass"] for i in ids])),
                "letter_mass_lt_0.5": int(sum(N[i]["letter_mass"] < 0.5 for i in ids)),
                "letter_mass_lt_0.5_reversed": int(sum(R[i]["letter_mass"] < 0.5 for i in ids)),
            }
        ids = [i for s in SOURCES for i in ids_by[(s, l)]]
        d[f"macro/{l}"] = {key: float(np.mean([d[f"{s}/{l}"][key] for s in SOURCES]))
                           for key in ["acc_normal", "acc_reversed", "acc_mean_two_orders", "consistency", "pos_spread_both"]}
    d["letter_mass_lt_0.5_total"] = int(sum(normal[name][i]["letter_mass"] < 0.5 for i in rows))
    d["letter_mass_lt_0.5_total_reversed"] = int(sum(reverse[name][i]["letter_mass"] < 0.5 for i in rows))
    d["letter_mass_min"] = float(min(normal[name][i]["letter_mass"] for i in rows))
    pos[name] = d
M["order_and_position"] = pos

# ---- I. tốc độ, thời gian train, VRAM
M["timing_eval"] = {m: json.load(open(RESULTS / "base_rerun" / m / "timing.json")) for m in MODELS}
M["timing_eval_ft"] = {f"{m}-ft-s{s}": json.load(open(FT / "eval" / "normal" / f"{m}-ft-s{s}" / "timing.json")) for m in MODELS for s in SEEDS}
M["train"] = {f"{m}-s{s}": json.load(open(FT / "train" / f"{m}-s{s}" / "summary.json")) for m in MODELS for s in SEEDS}

FT.mkdir(exist_ok=True)
(FT / "metrics_ft.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ======================================================================= markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
fcs = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
fp = lambda p: f"{p:.2g}"
names = [c[2] for c in conds]
short = {c[2]: f"{c[0].split('-')[1]} {c[1]}" for c in conds}
L = []
for l in LANGS:
    L += [f"### Accuracy (%) — {'tiếng Việt' if l == 'vi' else 'tiếng Anh'} [CI 95%]", "",
          "| Nguồn | " + " | ".join(short[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for s in SOURCES + ["macro"]:
        lab = "**trung bình đều**" if s == "macro" else s
        L.append(f"| {lab} | " + " | ".join(f"{f(M['acc_brier'][n][l][s]['acc'])} {fc(M['acc_brier'][n][l][s]['acc_ci'])}" for n in names) + " |")
    L.append("")
    L += [f"### Brier — {'tiếng Việt' if l == 'vi' else 'tiếng Anh'} [CI 95%] (đoán đều: belebele 0.750, sib200 0.857, massive 0.900, xcopa 0.500)", "",
          "| Nguồn | " + " | ".join(short[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for s in SOURCES + ["macro"]:
        lab = "**trung bình đều**" if s == "macro" else s
        L.append(f"| {lab} | " + " | ".join(f"{M['acc_brier'][n][l][s]['brier']:.3f} {[round(x, 3) for x in M['acc_brier'][n][l][s]['brier_ci']]}" for n in names) + " |")
    L.append("")

L += ["### Sau fine-tune so với chính model đó trước fine-tune (điểm %, cùng câu, McNemar chính xác)", "",
      "| Model | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
for key, v in M["ft_vs_base"].items():
    for l in LANGS:
        e = v[l]
        L.append(f"| {key} | {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={fp(e[s]['p'])})" for s in SOURCES)
                 + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {fp(e['macro']['p'])} |")
L.append("")

L += ["### 1.7B − 0.6B trên cùng câu (điểm %, McNemar chính xác)", "",
      "| So sánh | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
cmp_rows = ([("gốc", M["small_vs_large_base"])] + [(f"sau ft, seed {s}", M["small_vs_large_ft"][f"s{s}"]) for s in SEEDS]) if BOTH else []
for lab, v in cmp_rows:
    for l in LANGS:
        e = v[l]
        L.append(f"| {lab} | {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={fp(e[s]['p'])})" for s in SOURCES)
                 + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {fp(e['macro']['p'])} |")
L.append("")
if "per_seed" in M:
    L += ["### Accuracy sau fine-tune theo seed (%): " + ", ".join(f"s{x}" for x in SEEDS) + ", trung bình, s1 − s2", "",
          "| Model | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều |", "|---|---|" + "---|" * (len(SOURCES) + 1)]
    for m in MODELS:
        for l in LANGS:
            cells = []
            for s in SOURCES + ["macro"]:
                e = M["per_seed"][m][l][s]
                cells.append(" / ".join(f(e[f"s{x}"]) for x in SEEDS) + f" / TB {f(e['mean'])} / Δ {fs(e['s1_minus_s2'])}")
            L.append(f"| {m} | {l} | " + " | ".join(cells) + " |")
    L.append("")
if "small_vs_large_ft_pairs" in M:
    L += ["### 1.7B − 0.6B sau fine-tune theo từng cặp seed (điểm %, McNemar chính xác, cùng câu)", "",
          "| Cặp seed | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
    for lab, v in M["small_vs_large_ft_pairs"].items():
        for l in LANGS:
            e = v[l]
            L.append(f"| {lab} | {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={fp(e[s]['p'])})" for s in SOURCES)
                     + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {fp(e['macro']['p'])} |")
    for l in LANGS:
        e = M["small_vs_large_ft_pairs_mean"][l]
        L.append(f"| **trung bình {len(M['small_vs_large_ft_pairs'])} cặp** | {l} | " + " | ".join(fs(e[s]) for s in SOURCES) + f" | {fs(e['macro'])} | |")
    L.append("")
if "seed_gap" in M:
    L += [f"### Chênh lệch giữa hai seed (seed {SEEDS[0]} − seed {SEEDS[1]}, cùng model, cùng câu)", "",
          "| Model | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
    for m, v in M["seed_gap"].items():
        for l in LANGS:
            e = v[l]
            L.append(f"| {m} | {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={fp(e[s]['p'])})" for s in SOURCES)
                     + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {fp(e['macro']['p'])} |")
    L.append("")

L += ["### Mức tụt Anh → Việt (điểm %, ghép pair_id; dương = tiếng Việt kém hơn)", "",
      "| Điều kiện | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|" + "---|" * (len(SOURCES) + 2)]
for n in names:
    e = M["en_vi_drop"][n]
    L.append(f"| {short[n]} | " + " | ".join(f"{fs(e[s]['diff'])} (p={fp(e[s]['p'])})" for s in SOURCES)
             + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {fp(e['macro']['p'])} |")
L.append("")

L += ["### Thứ tự lựa chọn đảo ngược: độ nhất quán và phụ thuộc vị trí (trung bình đều theo nguồn)", "",
      "| Điều kiện | Ngôn ngữ | acc thường | acc đảo | acc TB hai lượt | nhất quán | chênh acc giữa các vị trí đáp án (max−min, gộp hai lượt) |",
      "|---|---|---|---|---|---|---|"]
for n in names:
    for l in LANGS:
        e = pos[n][f"macro/{l}"]
        L.append(f"| {short[n]} | {l} | {f(e['acc_normal'])} | {f(e['acc_reversed'])} | {f(e['acc_mean_two_orders'])} | {f(e['consistency'])} | {f(e['pos_spread_both'])} |")
L.append("")
L += ["#### Theo nguồn (tiếng Việt): acc TB hai lượt / nhất quán / acc theo vị trí đáp án đúng (gộp hai lượt)", "",
      "| Điều kiện | " + " | ".join(SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
for n in names:
    cells = []
    for s in SOURCES:
        e = pos[n][f"{s}/vi"]
        cells.append(f"{f(e['acc_mean_two_orders'])} / {f(e['consistency'])} / " + " ".join(f"{k}:{f(v)}" for k, v in e["acc_by_gold_pos_both_orders"].items()))
    L.append(f"| {short[n]} | " + " | ".join(cells) + " |")
L.append("")

L += ["### Phân bố chữ cái được chọn (thứ tự thường): tỉ lệ của chữ được chọn nhiều nhất (%)", "",
      "| Điều kiện | " + " | ".join(f"{s}/{l}" for l in LANGS for s in SOURCES) + " |", "|---|" + "---|" * (2 * len(SOURCES))]
for n in names:
    L.append(f"| {short[n]} | " + " | ".join(f(pos[n][f'{s}/{l}']['max_pred_share_normal']) for l in LANGS for s in SOURCES) + " |")
L.append("")
L += ["Số câu theo chữ được chọn, tiếng Việt (đúng: xem dev_v0 manifest):", ""]
for n in names:
    L.append(f"- {short[n]}: " + "; ".join(f"{s} " + " ".join(f"{k}{v}" for k, v in pos[n][f'{s}/vi']['pred_hist_normal'].items()) for s in SOURCES))
L.append("")

L += ["### Khối xác suất trên các chữ cái (letter_mass)", "",
      "| Điều kiện | số câu < 0.5 (thường) | số câu < 0.5 (đảo) | min | TB vi | TB en |", "|---|---|---|---|---|---|"]
for n in names:
    d = pos[n]
    tb = lambda l: np.mean([d[f"{s}/{l}"]["letter_mass_mean"] for s in SOURCES])
    L.append(f"| {short[n]} | {d['letter_mass_lt_0.5_total']} | {d['letter_mass_lt_0.5_total_reversed']} | {d['letter_mass_min']:.3f} | {tb('vi'):.3f} | {tb('en'):.3f} |")
L.append("")

L += ["### Brier trước / sau temperature (T fit trên val nội bộ; trung bình đều theo nguồn)", "",
      "| Điều kiện | T | Brier vi T=1 | Brier vi T | Brier en T=1 | Brier en T |", "|---|---|---|---|---|---|"]
for n in names:
    t = M["brier_T"][n]
    L.append(f"| {short[n]} | {t['T']:.3f} | {M['acc_brier'][n]['vi']['macro']['brier']:.3f} | {t['vi']['brier']:.3f} | "
             f"{M['acc_brier'][n]['en']['macro']['brier']:.3f} | {t['en']['brier']:.3f} |")
L.append("")

L += ["### Tốc độ, thời gian train, VRAM", "",
      "| Model | độ trễ trung vị batch 1 (ms) | câu/giây batch 16 | thời gian train (phút) | VRAM đỉnh khi train (GB, allocated / reserved) | val loss đầu → epoch 1 → epoch 2 |",
      "|---|---|---|---|---|---|"]
for m in MODELS:
    t = M["timing_eval"][m]
    L.append(f"| {m} gốc | {t['batch1_median_latency_ms']:.1f} | {t['batched_rows_per_s']:.0f} | | | |")
    for s in SEEDS:
        te = M["timing_eval_ft"][f"{m}-ft-s{s}"]; tr = M["train"][f"{m}-s{s}"]
        vl = tr["val_loss_by_epoch"]
        L.append(f"| {m} ft s{s} (LoRA đã merge) | {te['batch1_median_latency_ms']:.1f} | {te['batched_rows_per_s']:.0f} | {tr['wall_time_s'] / 60:.1f} | "
                 f"{tr['peak_vram_gb']:.1f} / {tr['peak_vram_reserved_gb']:.1f} | {' → '.join(f'{x:.3f}' for x in vl)} |")
L.append("")
(FT / "report_ft.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
