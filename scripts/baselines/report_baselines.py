"""Bảng so model ngoài với các mốc của dự án trên dev_v0. Ra: results/baselines/tables_baselines.md, metrics_baselines.json

Mốc dự án: Qwen3-1.7B gốc, Qwen3-1.7B full_vocab (adapter loss mới), Qwen3-0.6B ft s1/s2.
Model ngoài: Qwen3.5-2B gốc (đường chấm chữ cái của dự án), decider-2b (hai cách ánh xạ lựa chọn: text, keyed).
Với decider-2b, "tốt nhất" = cách ánh xạ có accuracy trung bình đều cao hơn, chọn riêng cho từng ngôn ngữ (có lợi cho model ngoài).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, read_jsonl
from stats import boot, brier, ci, cluster_index, mcnemar

B = RESULTS / "baselines"
SOURCES = ["belebele", "sib200", "massive", "xcopa"]
LANGS = ["vi", "en"]
CONDS = {  # tên: (normal, reverse, timing, params)
    "Qwen3-1.7B gốc": (RESULTS / "base_rerun/Qwen3-1.7B", RESULTS / "ft/eval/reverse/Qwen3-1.7B", RESULTS / "base_rerun/Qwen3-1.7B/timing.json"),
    "Qwen3-1.7B full_vocab (dự án)": (RESULTS / "ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1", RESULTS / "ft_loss/eval/reverse/Qwen3-1.7B-fullvocab-s1",
                                      RESULTS / "ft_loss/eval/normal/Qwen3-1.7B-fullvocab-s1/timing.json"),
    "Qwen3-0.6B ft s1": (RESULTS / "ft/eval/normal/Qwen3-0.6B-ft-s1", RESULTS / "ft/eval/reverse/Qwen3-0.6B-ft-s1", RESULTS / "ft/eval/normal/Qwen3-0.6B-ft-s1/timing.json"),
    "Qwen3-0.6B ft s2": (RESULTS / "ft/eval/normal/Qwen3-0.6B-ft-s2", RESULTS / "ft/eval/reverse/Qwen3-0.6B-ft-s2", RESULTS / "ft/eval/normal/Qwen3-0.6B-ft-s2/timing.json"),
    "Qwen3.5-2B gốc": (B / "eval/normal/Qwen3.5-2B", B / "eval/reverse/Qwen3.5-2B", B / "eval/normal/Qwen3.5-2B/timing.json"),
    "decider-2b (text)": (B / "eval/normal/decider-2b-text", B / "eval/reverse/decider-2b-text", B / "eval/normal/decider-2b-text/timing.json"),
    "decider-2b (keyed)": (B / "eval/normal/decider-2b-keyed", B / "eval/reverse/decider-2b-keyed", B / "eval/normal/decider-2b-keyed/timing.json"),
}
PARAMS = {"Qwen3-1.7B gốc": "2,03 B (safetensors)", "Qwen3-1.7B full_vocab (dự án)": "2,03 B + LoRA 17,4 M (đã merge)",
          "Qwen3-0.6B ft s1": "0,75 B + LoRA 10,1 M (đã merge)", "Qwen3-0.6B ft s2": "0,75 B + LoRA 10,1 M (đã merge)",
          "Qwen3.5-2B gốc": "2,27 B trong repo (gồm bộ mã hóa ảnh); chỉ nạp phần văn bản",
          "decider-2b (text)": "1,88 B", "decider-2b (keyed)": "1,88 B"}
REF = "Qwen3-1.7B full_vocab (dự án)"
rows = {r["id"]: r for r in read_jsonl(DATA / "dev_v0.jsonl")}
ids_by = {(s, l): sorted(i for i, r in rows.items() if r["source"] == s and r["lang"] == l) for s in SOURCES for l in LANGS}
P, R, TIM = {}, {}, {}
for n, (pn, pr, pt) in CONDS.items():
    P[n] = {p["id"]: p for p in read_jsonl(pn / "preds.jsonl")}
    R[n] = {p["id"]: p for p in read_jsonl(pr / "preds.jsonl")}
    TIM[n] = json.load(open(pt))
    assert set(P[n]) == set(rows) and set(R[n]) == set(rows), n
    excl = [i for i, p in P[n].items() if p.get("excluded")]
    assert not excl, (n, len(excl))


def corr(Pn, ids):
    return np.array([Pn[i]["correct"] for i in ids], dtype=float)


def acc_brier(Pn, lang, sources=SOURCES, seed=SEED):
    per_a, per_b, out = [], [], {}
    for s in sources:
        ids = ids_by[(s, lang)]; cl = cluster_index(ids, rows)
        a = corr(Pn, ids); b = np.array([brier(Pn[i]["probs"], rows[i]["target"]) for i in ids])
        per_a.append((a, cl)); per_b.append((b, cl)); out[s] = {"acc": a.mean(), "brier": b.mean()}
    ba, bb = boot(per_a, seed), boot(per_b, seed + 1)
    for j, s in enumerate(sources):
        out[s]["acc_ci"], out[s]["brier_ci"] = ci(ba[:, j]), ci(bb[:, j])
    out["macro"] = {"acc": float(np.mean([out[s]["acc"] for s in sources])), "acc_ci": ci(ba.mean(1)),
                    "brier": float(np.mean([out[s]["brier"] for s in sources])), "brier_ci": ci(bb.mean(1))}
    return out


def paired(PA, PB, idfn, seed, sources=SOURCES):
    """Hiệu accuracy A − B trên cùng câu: theo nguồn (McNemar chính xác) và trung bình đều (CI bootstrap theo state)."""
    per, out = [], {}
    for s in sources:
        ids = idfn(s); a, b = corr(PA, ids), corr(PB, ids)
        per.append((a - b, cluster_index(ids, rows)))
        out[s] = {"diff": float((a - b).mean()), **mcnemar(a.astype(bool), b.astype(bool))}
    bd = boot(per, seed)
    for j, s in enumerate(sources):
        out[s]["ci"] = ci(bd[:, j])
    allids = [i for s in sources for i in idfn(s)]
    out["macro"] = {"diff": float(np.mean([out[s]["diff"] for s in sources])), "ci": ci(bd.mean(1)),
                    **mcnemar(corr(PA, allids).astype(bool), corr(PB, allids).astype(bool))}
    return out


M = {"ref": REF}
NO_MASSIVE = [s for s in SOURCES if s != "massive"]
M["acc_brier"] = {n: {l: acc_brier(P[n], l) for l in LANGS} for n in CONDS}
M["acc_no_massive"] = {n: {l: acc_brier(P[n], l, NO_MASSIVE)["macro"] for l in LANGS} for n in CONDS}
M["en_vi_drop"] = {n: paired(P[n], {i[:-2] + "en": P[n][i] for i in P[n] if i.endswith(":vi")}, lambda s: ids_by[(s, "en")], SEED + 40)
                   for n in CONDS}
M["vs_ref"] = {n: {l: paired(P[n], P[REF], lambda s, l=l: ids_by[(s, l)], SEED + 70) for l in LANGS} for n in CONDS if n != REF}
M["vs_ref_no_massive"] = {n: {l: paired(P[n], P[REF], lambda s, l=l: ids_by[(s, l)], SEED + 71, NO_MASSIVE)["macro"] for l in LANGS}
                          for n in CONDS if n != REF}
order = {}
for n in CONDS:
    d = {}
    for l in LANGS:
        cons, a2 = [], []
        for s in SOURCES:
            ids = ids_by[(s, l)]
            cons.append(np.mean([P[n][i]["pred"] == P[n][i]["k"] - 1 - R[n][i]["pred"] for i in ids]))
            a2.append((corr(P[n], ids).mean() + corr(R[n], ids).mean()) / 2)
        d[l] = {"consistency": float(np.mean(cons)), "acc_two_orders": float(np.mean(a2)),
                "acc_reversed": float(np.mean([corr(R[n], ids_by[(s, l)]).mean() for s in SOURCES]))}
    order[n] = d
M["order"] = order
best = {l: max(["decider-2b (text)", "decider-2b (keyed)"], key=lambda n: M["acc_brier"][n][l]["macro"]["acc"]) for l in LANGS}
M["decider_best_mapping"] = best
M["trunc"] = {n: {"over_default_cap": TIM[n].get("n_over_default_cap_1536"), "excluded": TIM[n].get("n_excluded_over_32768", 0)} for n in CONDS}
M["latency_ms"] = {n: TIM[n]["batch1_median_latency_ms"] for n in CONDS}
# câu hiển nhiên
obv = {}
orows = {r["id"]: r for r in read_jsonl(B / "obvious_v0.jsonl")}
for n, sub in (("Qwen3.5-2B gốc", "Qwen3.5-2B"), ("decider-2b (text)", "decider-2b-text"), ("decider-2b (keyed)", "decider-2b-keyed")):
    pn = {p["id"]: p for p in read_jsonl(B / "obvious/normal" / sub / "preds.jsonl")}
    pr = {p["id"]: p for p in read_jsonl(B / "obvious/reverse" / sub / "preds.jsonl")}
    res = {}
    for l in LANGS:
        ids = [i for i in orows if orows[i]["lang"] == l]
        both = sum(pn[i]["correct"] and pr[i]["correct"] for i in ids)
        sums_ok = all(abs(sum(pn[i]["probs"]) - 1) < 1e-4 and len(pn[i]["probs"]) == len(orows[i]["options"]) for i in ids)
        res[l] = {"both_orders_correct": both, "n": len(ids), "probs_sum_and_len_ok": sums_ok,
                  "failed": [i for i in ids if not (pn[i]["correct"] and pr[i]["correct"])]}
    obv[n] = res
M["obvious"] = obv
(B / "metrics_baselines.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ======================================================================= markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
fcs = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
names = list(CONDS)
L = ["## Kiểm tra adapter: 10 câu hiển nhiên × 2 thứ tự", "", "| Model | en: đúng ở cả hai thứ tự | vi: đúng ở cả hai thứ tự | xác suất cộng = 1, đủ phần tử | câu sai |", "|---|---|---|---|---|"]
for n, r in obv.items():
    L.append(f"| {n} | {r['en']['both_orders_correct']}/{r['en']['n']} | {r['vi']['both_orders_correct']}/{r['vi']['n']} | "
             f"{r['en']['probs_sum_and_len_ok'] and r['vi']['probs_sum_and_len_ok']} | {r['en']['failed'] + r['vi']['failed']} |")
L.append("")
for l in LANGS:
    L += [f"## Accuracy (%) — {'tiếng Việt' if l == 'vi' else 'tiếng Anh'} [CI 95% bootstrap theo state]", "",
          "| Model | " + " | ".join(SOURCES) + " | trung bình đều | không MASSIVE | Brier TB đều |", "|---|" + "---|" * (len(SOURCES) + 3)]
    for n in names:
        e = M["acc_brier"][n][l]
        L.append(f"| {n} | " + " | ".join(f"{f(e[s]['acc'])} {fc(e[s]['acc_ci'])}" for s in SOURCES)
                 + f" | **{f(e['macro']['acc'])}** {fc(e['macro']['acc_ci'])} | {f(M['acc_no_massive'][n][l]['acc'])} | {e['macro']['brier']:.3f} {[round(x, 3) for x in e['macro']['brier_ci']]} |")
    L.append("")
for l in LANGS:
    L += [f"## Brier — {'tiếng Việt' if l == 'vi' else 'tiếng Anh'} (đoán đều: belebele 0,750, sib200 0,857, massive 0,900, xcopa 0,500)", "",
          "| Model | " + " | ".join(SOURCES) + " |", "|---|" + "---|" * len(SOURCES)]
    for n in names:
        e = M["acc_brier"][n][l]
        L.append(f"| {n} | " + " | ".join(f"{e[s]['brier']:.3f}" for s in SOURCES) + " |")
    L.append("")
L += [f"## So với {REF} trên cùng câu (điểm %, dương = model kia hơn; McNemar chính xác)", "",
      "| Model | Ngôn ngữ | " + " | ".join(SOURCES) + " | trung bình đều [CI] | p (gộp) |", "|---|---|" + "---|" * (len(SOURCES) + 2)]
for n, v in M["vs_ref"].items():
    for l in LANGS:
        e = v[l]
        L.append(f"| {n} | {l} | " + " | ".join(f"{fs(e[s]['diff'])} (p={e[s]['p']:.2g})" for s in SOURCES)
                 + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} | {e['macro']['p']:.2g} |")
L.append("")
L += [f"Không tính MASSIVE (3 nguồn), model kia − {REF}, trung bình đều [CI] (p gộp):", ""]
for n, v in M["vs_ref_no_massive"].items():
    L.append(f"- {n}: vi {fs(v['vi']['diff'])} {fcs(v['vi']['ci'])} (p={v['vi']['p']:.2g}); en {fs(v['en']['diff'])} {fcs(v['en']['ci'])} (p={v['en']['p']:.2g})")
L.append("")
L += ["## Mức tụt Anh → Việt (điểm %, ghép pair_id; dương = tiếng Việt kém hơn)", "",
      "| Model | " + " | ".join(SOURCES) + " | trung bình đều [CI] |", "|---|" + "---|" * (len(SOURCES) + 1)]
for n in names:
    e = M["en_vi_drop"][n]
    L.append(f"| {n} | " + " | ".join(f"{fs(e[s]['diff'])}" for s in SOURCES) + f" | {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} |")
L.append("")
L += ["## Đảo thứ tự lựa chọn, tham số, độ trễ, cắt cụt", "",
      "| Model | tham số | nhất quán vi / en | acc TB hai lượt vi / en | độ trễ trung vị batch 1 (ms) | câu bị cắt / bị loại |", "|---|---|---|---|---|---|"]
for n in names:
    o = M["order"][n]; t = M["trunc"][n]
    tr = "0 / 0" if t["over_default_cap"] is None else f"{t['over_default_cap']} vượt 1.536 (mặc định) / {t['excluded']} bị loại (đặt 32.768)"
    L.append(f"| {n} | {PARAMS[n]} | {f(o['vi']['consistency'])} / {f(o['en']['consistency'])} | {f(o['vi']['acc_two_orders'])} / {f(o['en']['acc_two_orders'])} | "
             f"{M['latency_ms'][n]:.1f} | {tr} |")
L.append("")
# bảng cùng cỡ
same = ["decider-2b (text)", "decider-2b (keyed)", "Qwen3.5-2B gốc", "Qwen3-1.7B gốc", REF]
L += ["## Bảng cùng cỡ", "", f"decider-2b: cách ánh xạ tốt hơn là **{best['vi']}** ở tiếng Việt và **{best['en']}** ở tiếng Anh.", "",
      "| Model | vi TB đều [CI] | en TB đều [CI] | tụt Anh→Việt | nhất quán vi | độ trễ batch 1 (ms) | vi: dự án − model này [CI] (p) | en: dự án − model này [CI] (p) |",
      "|---|---|---|---|---|---|---|---|"]
for n in same:
    e = M["acc_brier"][n]
    if n == REF:
        dv = de = "—"
    else:
        v, w = M["vs_ref"][n]["vi"]["macro"], M["vs_ref"][n]["en"]["macro"]
        dv = f"{fs(-v['diff'])} {fcs([-v['ci'][1], -v['ci'][0]])} (p={v['p']:.2g})"
        de = f"{fs(-w['diff'])} {fcs([-w['ci'][1], -w['ci'][0]])} (p={w['p']:.2g})"
    L.append(f"| {n} | {f(e['vi']['macro']['acc'])} {fc(e['vi']['macro']['acc_ci'])} | {f(e['en']['macro']['acc'])} {fc(e['en']['macro']['acc_ci'])} | "
             f"{fs(M['en_vi_drop'][n]['macro']['diff'])} | {f(M['order'][n]['vi']['consistency'])} | {M['latency_ms'][n]:.1f} | {dv} | {de} |")
q35 = {l: paired(P["Qwen3.5-2B gốc"], P["Qwen3-1.7B gốc"], lambda s, l=l: ids_by[(s, l)], SEED + 80) for l in LANGS}
M["qwen35_vs_qwen3_base"] = q35
L += ["", "Qwen3.5-2B gốc − Qwen3-1.7B gốc (cùng đường chấm, cùng câu):", ""]
for l in LANGS:
    e = q35[l]
    L.append(f"- {l}: {fs(e['macro']['diff'])} {fcs(e['macro']['ci'])} (p={e['macro']['p']:.2g}); theo nguồn: " +
             ", ".join(f"{s} {fs(e[s]['diff'])} (p={e[s]['p']:.2g})" for s in SOURCES))
L.append("")
(B / "tables_baselines.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(B / "metrics_baselines.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("\n".join(L))
