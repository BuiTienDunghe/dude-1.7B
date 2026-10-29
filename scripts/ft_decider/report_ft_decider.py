"""Bảng của lần train thử từ decider-2b (ft_decider_v0). Ra: results/ft_decider/tables_ft_decider_v0.md, metrics_ft_decider_v0.json

Model (mọi preds từ đường chấm của decider, batch 1, choice ánh xạ "text", noul/score kiểu gốc, T phát hành):
  mới      : decider-2b + LoRA (epoch 2, bản chính thức)       results/ft_decider/eval/dev_v1_ep2/decider-2b-text
  mới ep1  : checkpoint cuối epoch 1 (chỉ để xem có quên dần)   results/ft_decider/eval/dev_v1_ep1/decider-2b-text
  decider  : Mapika/decider-2b, preds đã lưu (G1 trùng từng byte) results/dev_v1/eval/decider-2b-text
  strip    : Qwen3-1.7B + adapter bỏ dấu (results/ft_strip), preds đã lưu
Accuracy = trung bình đều theo nguồn; CI 95% bootstrap theo state (lấy mẫu trong từng nguồn); McNemar chính xác gộp mọi câu.
Brier đa lớp (trung bình đều theo nguồn), ECE top-label 15 bin (micro) — trên xác suất đường chấm trả ra (đã chia T phát hành).
Cách đọc định trước (so với decider-2b):
  Đáng theo: EN và VI có dấu song song giảm ≤ 2 điểm, VÀ (bản địa có dấu tăng ≥ 3 điểm với CI không chứa 0, HOẶC không dấu song song tăng ≥ 10).
  Gây hại : EN hoặc VI có dấu (song song) giảm > 2 điểm với CI không chứa 0.   Còn lại: báo số, không kết luận.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, load_json, read_jsonl, sha256_file  # noqa: E402
from stats import B, mcnemar  # noqa: E402

R = RESULTS / "ft_decider"
DEV = DATA / "bench_v1" / "dev_v1.jsonl"
assert sha256_file(DEV) == load_json(DATA / "bench_v1" / "manifest_dev_v1.json")["sha256"]["dev_v1.jsonl"], "dev_v1 khác manifest"
rows = {r["id"]: r for r in read_jsonl(DEV)}
load = lambda p: {x["id"]: x for x in read_jsonl(p)}
NEW, EP1, DEC, STR = "decider-ft (mới, ep2)", "decider-ft ep1", "decider-2b", "strip (Qwen3-1.7B)"
P = {NEW: load(R / "eval/dev_v1_ep2/decider-2b-text/preds.jsonl"), DEC: load(RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl"),
     STR: load(RESULTS / "ft_strip/eval/normal/strip-s1/preds.jsonl")}
if (R / "eval/dev_v1_ep1/decider-2b-text/preds.jsonl").exists():
    P[EP1] = load(R / "eval/dev_v1_ep1/decider-2b-text/preds.jsonl")
MODELS = [NEW, DEC, STR] + ([EP1] if EP1 in P else [])
for m in MODELS:
    assert set(P[m]) == set(rows), m

SLICES = {
    "song song · EN": dict(block="parallel", lang="en"),
    "song song · VI có dấu": dict(block="parallel", lang="vi", variant="orig"),
    "song song · VI không dấu": dict(block="parallel", lang="vi", variant="nodiac"),
    "bản địa · có dấu": dict(block="native", variant="orig"),
    "bản địa · không dấu": dict(block="native", variant="nodiac"),
}


def sel(**kw):
    return sorted(i for i, r in rows.items() if all(r[k] == v for k, v in kw.items()))


def split(ids, rws=rows):
    g = defaultdict(list)
    for i in ids:
        g[rws[i]["source"]].append(i)
    return dict(sorted(g.items()))


def macro_boot(vals, ids_by_src, seed, rws=rows):
    rng = np.random.default_rng(seed)
    est = np.zeros((B, len(vals)))
    for s, src in enumerate(vals):
        v = vals[src]; g = defaultdict(list)
        for j, i in enumerate(ids_by_src[src]):
            g[rws[i]["cluster"]].append(j)
        cl = [np.array(x) for _, x in sorted(g.items())]
        sums = np.array([v[c].sum() for c in cl]); cnt = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        est[:, s] = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(np.mean([v.mean() for v in vals.values()])), [float(np.percentile(est.mean(1), 2.5)), float(np.percentile(est.mean(1), 97.5))]


def brier(p, t):
    v = np.array(p, dtype=float); v[t] -= 1
    return float((v ** 2).sum())


def ece(probs, targets, bins=15):
    conf = np.array([max(p) for p in probs]); acc = np.array([int(np.argmax(p) == t) for p, t in zip(probs, targets)])
    e = 0.0
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return float(e)


def metrics(pr, ids, seed, rws=rows, key=lambda i: i):
    bs = split(ids, rws)
    a, ci = macro_boot({s: np.array([float(pr[key(i)]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed, rws)
    return {"n": len(ids), "acc": a, "acc_ci": ci,
            "brier": float(np.mean([np.mean([brier(pr[key(i)]["probs"], rws[i]["target"]) for i in v]) for v in bs.values()])),
            "ece": ece([pr[key(i)]["probs"] for i in ids], [rws[i]["target"] for i in ids]),
            "per_source": {s: float(np.mean([pr[key(i)]["correct"] for i in v])) for s, v in bs.items()}}


def paired(pa, pb, ids, seed, rws=rows, ka=lambda i: i, kb=lambda i: i):
    bs = split(ids, rws)
    d, ci = macro_boot({s: np.array([float(pa[ka(i)]["correct"]) - float(pb[kb(i)]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed, rws)
    out = {"diff": d, "ci": ci, **mcnemar([pa[ka(i)]["correct"] for i in ids], [pb[kb(i)]["correct"] for i in ids])}
    out["per_source"] = {s: {"diff": float(np.mean([float(pa[ka(i)]["correct"]) - float(pb[kb(i)]["correct"]) for i in v])),
                             **mcnemar([pa[ka(i)]["correct"] for i in v], [pb[kb(i)]["correct"] for i in v])} for s, v in bs.items()}
    return out


M = {"dev_v1_sha256": sha256_file(DEV), "slices": {}, "paired": {}, "by_kind": {}, "restored": {}}
for j, (name, kw) in enumerate(SLICES.items()):
    ids = sel(**kw)
    M["slices"][name] = {m: metrics(P[m], ids, SEED + j) for m in MODELS}
    M["paired"][name] = {f"{NEW} − {o}": paired(P[NEW], P[o], ids, SEED + 50 + j) for o in MODELS if o != NEW}
for kind in ("choice", "noul", "score"):
    for var in ("orig", "nodiac"):
        ids = sel(block="native", kind=kind, variant=var)
        if ids:
            key = f"bản địa · {kind} · {'có dấu' if var == 'orig' else 'không dấu'}"
            M["by_kind"][key] = {"n": len(ids), **{m: metrics(P[m], ids, SEED + 200)["acc"] for m in MODELS},
                                 "new_minus_decider": paired(P[NEW], P[DEC], ids, SEED + 201),
                                 "new_minus_strip": paired(P[NEW], P[STR], ids, SEED + 202)}

# bản đã khôi phục dấu bằng vit5 (results/restore): mới + vit5 so với decider + vit5, và với chính nó ở có dấu / không dấu
RV = R / "eval/restored_vit5/decider-2b-text/preds.jsonl"
if RV.exists():
    PN = load(RV); PD = load(RESULTS / "restore/eval/vit5/decider-2b-text/preds.jsonl")
    for j, (blk, b) in enumerate((("song song", "parallel"), ("bản địa", "native"))):
        ids = sel(block=b, lang="vi", variant="orig")
        rk = lambda i: f"{i}:restore:vit5"
        M["restored"][blk] = {
            "new_restored": metrics(PN, ids, SEED + 400 + j, key=rk), "decider_restored": metrics(PD, ids, SEED + 400 + j, key=rk),
            "new_restored_minus_decider_restored": paired(PN, PD, ids, SEED + 410 + j, ka=rk, kb=rk),
            "new_restored_minus_new_nodiac": paired(PN, P[NEW], ids, SEED + 420 + j, ka=rk, kb=lambda i: i + ":nodiac"),
            "new_restored_minus_new_orig": paired(PN, P[NEW], ids, SEED + 430 + j, ka=rk),
            "new_nodiac_minus_decider_restored": paired(P[NEW], PD, ids, SEED + 440 + j, ka=lambda i: i + ":nodiac", kb=rk)}

# cách đọc định trước
pv = M["paired"]; g = lambda s: pv[s][f"{NEW} − {DEC}"]
en, vio, nat, nod = g("song song · EN"), g("song song · VI có dấu"), g("bản địa · có dấu"), g("song song · VI không dấu")
keep_ok = en["diff"] >= -0.02 and vio["diff"] >= -0.02
gain = (nat["diff"] >= 0.03 and nat["ci"][0] > 0) or nod["diff"] >= 0.10
harm = (en["diff"] < -0.02 and en["ci"][1] < 0) or (vio["diff"] < -0.02 and vio["ci"][1] < 0)
M["reading"] = {"en": en["diff"], "vi_orig_parallel": vio["diff"], "native_orig": nat["diff"], "native_orig_ci": nat["ci"], "nodiac_parallel": nod["diff"],
                "keep_ok": keep_ok, "gain": gain, "harm": harm,
                "verdict": "ĐÁNG THEO" if keep_ok and gain else "GÂY HẠI" if harm else "CÒN LẠI: báo số, không kết luận"}
tr = R / "train/decider-2b-s1"
if (tr / "summary.json").exists():
    M["train"] = load_json(tr / "summary.json")
    M["train_log"] = [json.loads(l) for l in open(tr / "train_log.jsonl") if '"event": "val"' in l or '"event": "mass"' in l or "strip_counts" in l]
(R / "metrics_ft_decider_v0.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ------------------------------------------------------------------ markdown
f = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:.1f}; {100 * c[1]:.1f}]"
fcs = lambda c: f"[{100 * c[0]:+.1f}; {100 * c[1]:+.1f}]"
L = [f"dev_v1 SHA-256 khớp manifest: `{M['dev_v1_sha256'][:16]}…`. Mọi preds từ đường chấm của decider (batch 1, T phát hành).", "",
     "## 1. Năm lát chính — accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]", "",
     "| Lát | n | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
for name, e in M["slices"].items():
    L.append(f"| {name} | {e[NEW]['n']} | " + " | ".join(f"{f(e[m]['acc'])} {fc(e[m]['acc_ci'])}" for m in MODELS) + " |")
L += ["", "## 2. Chênh lệch theo cặp (cùng câu): mới − model khác (điểm %, [CI 95%], McNemar p gộp)", "",
      "| Lát | " + " | ".join(f"mới − {o}" for o in MODELS if o != NEW) + " |", "|---|" + "---|" * (len(MODELS) - 1)]
for name, e in M["paired"].items():
    L.append(f"| {name} | " + " | ".join(f"{fs(v['diff'])} {fcs(v['ci'])} (p={v['p']:.2g})" for v in e.values()) + " |")
L += ["", "## 3. Brier và ECE (xác suất đường chấm)", "", "| Lát | " + " | ".join(f"{m} Brier / ECE" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for name, e in M["slices"].items():
    L.append(f"| {name} | " + " | ".join(f"{e[m]['brier']:.3f} / {e[m]['ece']:.3f}" for m in MODELS) + " |")
L += ["", "## 4. Theo nguồn — accuracy (%) và mới − decider-2b (McNemar p)", "",
      "| Lát · nguồn | " + " | ".join(MODELS) + " | mới − decider (p) | mới − strip (p) |", "|---|" + "---|" * (len(MODELS) + 2)]
for name, e in M["slices"].items():
    pd_, ps_ = M["paired"][name][f"{NEW} − {DEC}"]["per_source"], M["paired"][name][f"{NEW} − {STR}"]["per_source"]
    for s in e[NEW]["per_source"]:
        L.append(f"| {name} · {s} | " + " | ".join(f(e[m]["per_source"][s]) for m in MODELS) +
                 f" | {fs(pd_[s]['diff'])} (p={pd_[s]['p']:.2g}) | {fs(ps_[s]['diff'])} (p={ps_[s]['p']:.2g}) |")
L += ["", "## 5. Theo kiểu câu hỏi (khối bản địa; accuracy %, trung bình đều theo nguồn)", "",
      "| Lát | n | " + " | ".join(MODELS) + " | mới − decider [CI] (p) | mới − strip [CI] (p) |", "|---|---|" + "---|" * (len(MODELS) + 2)]
for k, e in M["by_kind"].items():
    a, b = e["new_minus_decider"], e["new_minus_strip"]
    L.append(f"| {k} | {e['n']} | " + " | ".join(f(e[m]) for m in MODELS) + f" | {fs(a['diff'])} {fcs(a['ci'])} ({a['p']:.2g}) | {fs(b['diff'])} {fcs(b['ci'])} ({b['p']:.2g}) |")
if M["restored"]:
    L += ["", "## 6. Bản khôi phục dấu bằng vit5 (results/restore) — câu tiếng Việt", "",
          "| Khối | mới + vit5 | decider + vit5 | (mới + vit5) − (decider + vit5) | mới + vit5 − mới không dấu | mới + vit5 − mới có dấu | mới không dấu − (decider + vit5) |",
          "|---|---|---|---|---|---|---|"]
    for blk, e in M["restored"].items():
        cells = [e[k] for k in ("new_restored_minus_decider_restored", "new_restored_minus_new_nodiac", "new_restored_minus_new_orig", "new_nodiac_minus_decider_restored")]
        L.append(f"| {blk} | {f(e['new_restored']['acc'])} | {f(e['decider_restored']['acc'])} | " +
                 " | ".join(f"{fs(c['diff'])} {fcs(c['ci'])} (p={c['p']:.2g})" for c in cells) + " |")
r = M["reading"]
L += ["", "## 7. Cách đọc định trước (so với decider-2b)", "",
      f"- EN: {fs(r['en'])}; VI có dấu song song: {fs(r['vi_orig_parallel'])} → giữ (giảm ≤ 2): **{'có' if r['keep_ok'] else 'không'}**",
      f"- Bản địa có dấu: {fs(r['native_orig'])} {fcs(r['native_orig_ci'])} (cần ≥ +3, CI không chứa 0); không dấu song song: {fs(r['nodiac_parallel'])} (cần ≥ +10) → tăng: **{'có' if r['gain'] else 'không'}**",
      f"- Gây hại (EN hoặc VI có dấu giảm > 2, CI không chứa 0): **{'có' if r['harm'] else 'không'}**",
      f"- Kết luận theo quy tắc: **{r['verdict']}**"]
if "train_log" in M:
    L += ["", "## 8. Train — val_v0 mỗi epoch và khối lượng nhãn hợp lệ (monitor 200 dòng)", "",
          "| Sự kiện | bước | " + " | ".join(("val_loss", "val_ce_eval_dist", "val_mass_mean", "val_acc_argmax", "val_nodiac_loss", "val_nodiac_ce_eval_dist", "val_nodiac_acc_argmax", "mass_mean")) + " |",
          "|---|---|" + "---|" * 8]
    for e in M["train_log"]:
        if e.get("event") in ("val", "mass"):
            L.append(f"| {e['event']} | {e['step']} | " + " | ".join(f"{e[k]:.4f}" if k in e else "—" for k in
                     ("val_loss", "val_ce_eval_dist", "val_mass_mean", "val_acc_argmax", "val_nodiac_loss", "val_nodiac_ce_eval_dist", "val_nodiac_acc_argmax", "mean")) + " |")
(R / "tables_ft_decider_v0.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[-6:]))
