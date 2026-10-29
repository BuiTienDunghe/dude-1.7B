"""Báo cáo train thử có bỏ dấu ngẫu nhiên. Ra: results/ft_strip/tables_strip_v0.md, results/ft_strip/metrics_strip_v0.json

Model:
  strip    : Qwen3-1.7B + adapter mới (full_vocab + bỏ dấu ngẫu nhiên p 0,30), results/ft_strip/eval/
  control  : adapter đối chứng results/ft_loss (preds dev_v1 đã lưu ở results/dev_v1/eval, đã qua G1 trùng từng byte)
  decider  : Mapika/decider-2b, preds dev_v1 đã lưu (choice: cách ánh xạ "text", như REPORT_dev_v1); noul/score kiểu gốc
  base     : Qwen3-1.7B gốc, preds dev_v1 đã lưu
Accuracy trong mỗi lát = trung bình đều theo nguồn (macro). CI 95% bootstrap theo state, lấy mẫu lại trong từng nguồn.
Chênh lệch theo cặp: cùng câu, CI bootstrap theo state; McNemar chính xác gộp mọi câu của lát.
ECE: top-label, 15 bin đều, tính trên mọi câu của lát (micro). Brier: tổng đa lớp, trung bình đều theo nguồn.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, load_json, read_jsonl, sha256_file  # noqa: E402
from stats import B, mcnemar, temper  # noqa: E402

R = RESULTS / "ft_strip"
DEV = DATA / "bench_v1" / "dev_v1.jsonl"
assert sha256_file(DEV) == load_json(DATA / "bench_v1" / "manifest_dev_v1.json")["sha256"]["dev_v1.jsonl"], "dev_v1 khác manifest"
rows = {r["id"]: r for r in read_jsonl(DEV)}
diag_rows = {r["id"]: r for r in read_jsonl(R / "diag" / "dev_v1_vi_state_only_nodiac.jsonl")}


def load(p):
    return {x["id"]: x for x in read_jsonl(p)}


P = {"strip (mới)": load(R / "eval/normal/strip-s1/preds.jsonl"),
     "control (đối chứng)": load(RESULTS / "dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl"),
     "decider-2b": load(RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl"),
     "Qwen3-1.7B gốc": load(RESULTS / "dev_v1/eval/Qwen3-1.7B/preds.jsonl")}
REV = {"strip (mới)": load(R / "eval/reverse/strip-s1/preds.jsonl"), "control (đối chứng)": load(R / "eval/reverse/control-s1/preds.jsonl")}
DIAG = {"strip (mới)": load(R / "eval/diag/strip-s1/preds.jsonl"), "control (đối chứng)": load(R / "eval/diag/control-s1/preds.jsonl"),
        "decider-2b": load(R / "eval/diag/decider-2b-text/preds.jsonl")}
T = {"strip (mới)": load_json(R / "temp/strip-s1.json")["T"], "control (đối chứng)": load_json(RESULTS / "ft_loss/temp/Qwen3-1.7B-fullvocab-s1.json")["T"]}
MODELS = list(P)
NEW, CTRL = "strip (mới)", "control (đối chứng)"
for m in MODELS:
    assert set(P[m]) == set(rows), m
for m in REV:
    assert set(REV[m]) == set(rows), m
SOURCES_PAR = ["belebele", "sib200", "massive", "xcopa"]
SOURCES_NAT = ["vifactcheck", "viwikifc", "vihsd"]


def sel(**kw):
    return sorted(i for i, r in rows.items() if all(r[k] == v for k, v in kw.items()))


SLICES = {
    "song song · EN": ("parallel", dict(block="parallel", lang="en")),
    "song song · VI có dấu": ("parallel", dict(block="parallel", lang="vi", variant="orig")),
    "song song · VI không dấu": ("parallel", dict(block="parallel", lang="vi", variant="nodiac")),
    "bản địa · có dấu": ("native", dict(block="native", variant="orig")),
    "bản địa · không dấu": ("native", dict(block="native", variant="nodiac")),
}


def by_source(ids, rws=rows):
    g = defaultdict(list)
    for i in ids:
        g[rws[i]["source"]].append(i)
    return dict(sorted(g.items()))


def clusters(ids, rws=rows):
    g = defaultdict(list)
    for j, i in enumerate(ids):
        g[rws[i]["cluster"]].append(j)
    return [np.array(v) for _, v in sorted(g.items())]


def macro_boot(vals_by_src, cl_by_src, seed):
    """vals_by_src: {src: np.array}; trả về (trung bình đều, CI) với bootstrap theo cụm trong từng nguồn."""
    rng = np.random.default_rng(seed)
    est = np.zeros((B, len(vals_by_src)))
    for s, (src, v) in enumerate(vals_by_src.items()):
        cl = cl_by_src[src]
        sums = np.array([v[c].sum() for c in cl]); cnt = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        est[:, s] = sums[pick].sum(1) / cnt[pick].sum(1)
    point = float(np.mean([v.mean() for v in vals_by_src.values()]))
    return point, [float(np.percentile(est.mean(1), 2.5)), float(np.percentile(est.mean(1), 97.5))]


def brier(p, t):
    v = np.array(p, dtype=float); v[t] -= 1
    return float((v ** 2).sum())


def ece(probs, targets, bins=15):
    conf = np.array([max(p) for p in probs]); acc = np.array([int(np.argmax(p) == t) for p, t in zip(probs, targets)])
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(e)


def slice_metrics(Pm, ids, seed, rws=rows, T_=None):
    bs = by_source(ids, rws)
    acc = {s: np.array([float(Pm[i]["correct"]) for i in v]) for s, v in bs.items()}
    cl = {s: clusters(v, rws) for s, v in bs.items()}
    a, ci = macro_boot(acc, cl, seed)
    pr = {i: (Pm[i]["probs"] if T_ is None else list(temper(Pm[i]["probs"], T_))) for i in ids}
    probs = [pr[i] for i in ids]
    tg = [rws[i]["target"] for i in ids]
    br = float(np.mean([np.mean([brier(pr[i], rws[i]["target"]) for i in v]) for v in bs.values()]))
    lm = [Pm[i].get("letter_mass") for i in ids]
    out = {"n": len(ids), "acc": a, "acc_ci": ci, "brier": br, "ece": ece(probs, tg),
           "per_source": {s: float(v.mean()) for s, v in acc.items()}}
    if lm[0] is not None:
        lm = np.array(lm); out["letter_mass_mean"] = float(lm.mean()); out["letter_mass_lt_0.5"] = int((lm < 0.5).sum())
    return out


def paired(PA, PB, ids, seed, rws=rows):
    bs = by_source(ids, rws)
    d = {s: np.array([float(PA[i]["correct"]) - float(PB[i]["correct"]) for i in v]) for s, v in bs.items()}
    cl = {s: clusters(v, rws) for s, v in bs.items()}
    diff, ci = macro_boot(d, cl, seed)
    a = np.array([PA[i]["correct"] for i in ids], bool); b = np.array([PB[i]["correct"] for i in ids], bool)
    per = {s: {"diff": float(v.mean()), **mcnemar(np.array([PA[i]["correct"] for i in bs[s]], bool), np.array([PB[i]["correct"] for i in bs[s]], bool))}
           for s, v in d.items()}
    return {"diff": diff, "ci": ci, **mcnemar(a, b), "per_source": per}


M = {"dev_v1_sha256": sha256_file(DEV), "T": T, "slices": {}, "paired": {}, "consistency": {}, "by_kind": {}, "diag": {}}
for j, (name, (blk, kw)) in enumerate(SLICES.items()):
    ids = sel(**kw)
    M["slices"][name] = {m: slice_metrics(P[m], ids, SEED + j) for m in MODELS}
    for m in (NEW, CTRL):
        M["slices"][name][m + " · sau T"] = slice_metrics(P[m], ids, SEED + j, T_=T[m])
    M["paired"][name] = {f"{NEW} − {o}": paired(P[NEW], P[o], ids, SEED + 50 + j) for o in MODELS if o != NEW}
    M["consistency"][name] = {m: float(np.mean([P[m][i]["pred"] == P[m][i]["k"] - 1 - REV[m][i]["pred"] for i in ids])) for m in REV}
# theo kiểu câu hỏi (khối bản địa) và theo biến thể
for kind in ("choice", "noul", "score"):
    for var in ("orig", "nodiac"):
        ids = sel(block="native", kind=kind, variant=var)
        if not ids:
            continue
        key = f"bản địa · {kind} · {'có dấu' if var == 'orig' else 'không dấu'}"
        bs_k = by_source(ids)
        M["by_kind"][key] = {"n": len(ids), **{m: float(np.mean([np.mean([P[m][i]["correct"] for i in v]) for v in bs_k.values()])) for m in MODELS},
                             "paired_new_minus_control": paired(P[NEW], P[CTRL], ids, SEED + 200),
                             "paired_new_minus_decider": paired(P[NEW], P["decider-2b"], ids, SEED + 201)}
# chẩn đoán phụ: có dấu / chỉ state không dấu / không dấu toàn bộ
for blk, srcs in (("song song", SOURCES_PAR), ("bản địa", SOURCES_NAT)):
    orig = sel(block="parallel" if blk == "song song" else "native", lang="vi", variant="orig")
    so = [i + ":stateonly" for i in orig]
    full = [i + ":nodiac" for i in orig]
    e = {}
    for m in DIAG:
        e[m] = {"có dấu": slice_metrics(P[m], orig, SEED + 300)["acc"],
                "chỉ state không dấu": slice_metrics(DIAG[m], so, SEED + 301, rws=diag_rows)["acc"],
                "không dấu toàn bộ": slice_metrics(P[m], full, SEED + 302)["acc"]}
    M["diag"][blk] = e
# ngưỡng đã định trước
pv = M["paired"]
f_new = lambda s: pv[s][f"{NEW} − {CTRL}"]
crit = {
    "nodiac_parallel_gain": {"diff": f_new("song song · VI không dấu")["diff"], "ci": f_new("song song · VI không dấu")["ci"]},
    "nodiac_native_gain": {"diff": f_new("bản địa · không dấu")["diff"], "ci": f_new("bản địa · không dấu")["ci"]},
    "vi_orig_parallel_change": {"diff": f_new("song song · VI có dấu")["diff"], "ci": f_new("song song · VI có dấu")["ci"]},
    "vi_orig_native_change": {"diff": f_new("bản địa · có dấu")["diff"], "ci": f_new("bản địa · có dấu")["ci"]},
    "en_change": {"diff": f_new("song song · EN")["diff"], "ci": f_new("song song · EN")["ci"]},
}
for k in ("nodiac_parallel_gain", "nodiac_native_gain"):
    crit[k]["pass"] = crit[k]["diff"] >= 0.05 and crit[k]["ci"][0] > 0
for k in ("vi_orig_parallel_change", "vi_orig_native_change", "en_change"):
    crit[k]["pass"] = crit[k]["diff"] >= -0.02
    crit[k]["noise_lt_1.5"] = abs(crit[k]["diff"]) < 0.015
crit["ALL_PASS"] = all(v["pass"] for v in crit.values() if isinstance(v, dict))
M["criteria"] = crit
M["train"] = {"strip": load_json(R / "train/Qwen3-1.7B-s1/summary.json"), "control": load_json(RESULTS / "ft_loss/train/Qwen3-1.7B-s1/summary.json")}
M["train_log_val"] = [json.loads(l) for l in open(R / "train/Qwen3-1.7B-s1/train_log.jsonl") if '"event": "val"' in l or '"strip_counts"' in l]
(R / "metrics_strip_v0.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ======================================================================= markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}; {100 * c[1]:.1f}]"
fcs = lambda c: f"[{100 * c[0]:+.1f}; {100 * c[1]:+.1f}]"
L = [f"dev_v1 SHA-256 khớp manifest: `{M['dev_v1_sha256'][:16]}…`. Nhiệt độ (fit trên val_v0): strip {T[NEW]:.3f}, control {T[CTRL]:.3f}.", "",
     "## 1. Năm lát chính — accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]", "",
     "| Lát | n | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
for name, e in M["slices"].items():
    L.append(f"| {name} | {e[NEW]['n']} | " + " | ".join(f"{f(e[m]['acc'])} {fc(e[m]['acc_ci'])}" for m in MODELS) + " |")
L += ["", "## 2. Chênh lệch theo cặp (cùng câu): strip − model khác (điểm %, [CI 95%], McNemar p gộp)", "",
      "| Lát | " + " | ".join(f"strip − {o}" for o in MODELS if o != NEW) + " |", "|---|" + "---|" * (len(MODELS) - 1)]
for name, e in M["paired"].items():
    L.append(f"| {name} | " + " | ".join(f"{fs(v['diff'])} {fcs(v['ci'])} (p={v['p']:.2g})" for v in e.values()) + " |")
L += ["", "## 3. Brier, ECE, letter_mass, nhất quán khi đảo thứ tự", "",
      "| Lát | Model | Brier T=1 | Brier sau T | ECE T=1 | ECE sau T | letter_mass TB / số < 0,5 | nhất quán |", "|---|---|---|---|---|---|---|---|"]
for name, e in M["slices"].items():
    for m in MODELS:
        aT = e.get(m + " · sau T")
        lm = f"{e[m]['letter_mass_mean']:.3f} / {e[m]['letter_mass_lt_0.5']}" if "letter_mass_mean" in e[m] else "—"
        L.append(f"| {name} | {m} | {e[m]['brier']:.3f} | {aT['brier']:.3f} | {e[m]['ece']:.3f} | {aT['ece']:.3f} | {lm} | "
                 f"{f(M['consistency'][name][m]) if m in M['consistency'][name] else '—'} |" if aT else
                 f"| {name} | {m} | {e[m]['brier']:.3f} | — | {e[m]['ece']:.3f} | — | {lm} | — |")
L += ["", "## 4. Theo nguồn — accuracy (%) và strip − control (McNemar p)", "",
      "| Lát · nguồn | " + " | ".join(MODELS) + " | strip − control (p) |", "|---|" + "---|" * (len(MODELS) + 1)]
for name, e in M["slices"].items():
    pc = M["paired"][name][f"{NEW} − {CTRL}"]["per_source"]
    for s in e[NEW]["per_source"]:
        L.append(f"| {name} · {s} | " + " | ".join(f(e[m]["per_source"][s]) for m in MODELS) + f" | {fs(pc[s]['diff'])} (p={pc[s]['p']:.2g}) |")
L += ["", "## 5. Theo kiểu câu hỏi (khối bản địa) — accuracy (%, trung bình đều theo nguồn), strip − control và strip − decider; McNemar p gộp theo dòng", "",
      "| Lát | n | " + " | ".join(MODELS) + " | strip − control [CI] (p) | strip − decider [CI] (p) |", "|---|---|" + "---|" * (len(MODELS) + 2)]
for k, e in M["by_kind"].items():
    a, b = e["paired_new_minus_control"], e["paired_new_minus_decider"]
    L.append(f"| {k} | {e['n']} | " + " | ".join(f(e[m]) for m in MODELS) + f" | {fs(a['diff'])} {fcs(a['ci'])} ({a['p']:.2g}) | {fs(b['diff'])} {fcs(b['ci'])} ({b['p']:.2g}) |")
L += ["", "## 6. Chẩn đoán phụ — accuracy tiếng Việt (%) theo phần bị bỏ dấu (không phải số chính thức)", "",
      "| Khối | Model | có dấu | chỉ state không dấu | không dấu toàn bộ |", "|---|---|---|---|---|"]
for blk, e in M["diag"].items():
    for m, v in e.items():
        L.append(f"| {blk} | {m} | {f(v['có dấu'])} | {f(v['chỉ state không dấu'])} | {f(v['không dấu toàn bộ'])} |")
L += ["", "## 7. Ngưỡng đã định trước", ""]
c = M["criteria"]
L.append(f"- Không dấu, song song: strip − control {fs(c['nodiac_parallel_gain']['diff'])} {fcs(c['nodiac_parallel_gain']['ci'])} (cần ≥ +5,0 và CI không chứa 0) → **{'ĐẠT' if c['nodiac_parallel_gain']['pass'] else 'KHÔNG ĐẠT'}**")
L.append(f"- Không dấu, bản địa: {fs(c['nodiac_native_gain']['diff'])} {fcs(c['nodiac_native_gain']['ci'])} → **{'ĐẠT' if c['nodiac_native_gain']['pass'] else 'KHÔNG ĐẠT'}**")
for k, lab in (("vi_orig_parallel_change", "VI có dấu, song song"), ("vi_orig_native_change", "VI có dấu, bản địa"), ("en_change", "EN")):
    L.append(f"- {lab}: {fs(c[k]['diff'])} {fcs(c[k]['ci'])} (cần giảm không quá 2,0) → **{'ĐẠT' if c[k]['pass'] else 'KHÔNG ĐẠT'}**"
             + (" (dưới 1,5 điểm: coi là nhiễu)" if c[k]["noise_lt_1.5"] else ""))
L.append(f"- Tổng: **{'ĐẠT' if c['ALL_PASS'] else 'KHÔNG ĐẠT'}**")
(R / "tables_strip_v0.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
