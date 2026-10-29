"""Bảng phần A (khôi phục dấu trước khi chấm). Ra: results/restore/tables_restore.md, results/restore/metrics_restore.json

Model: control = adapter đối chứng (results/ft_loss), strip = adapter bỏ dấu (results/ft_strip), decider = Mapika/decider-2b (choice: "text").
Biến thể của mỗi câu tiếng Việt dev_v1 (cùng id gốc): có dấu, không dấu (preds đã lưu), khôi phục <bộ> (đầu ra thô — KẾT QUẢ CHÍNH),
đã sửa <bộ> (chiếu về đầu vào không dấu — KIỂM ĐỘ NHẠY).
Accuracy trong khối = trung bình đều theo nguồn; CI 95% bootstrap theo state (cluster), lấy mẫu trong từng nguồn.
Chênh lệch theo cặp: cùng câu; McNemar chính xác gộp mọi câu của khối.
Cách đọc đã định trước (áp riêng từng bộ, trên đầu ra thô, khối song song đủ 4 nguồn): khoảng cách = decider có dấu − decider khôi phục;
≤ 5 điểm: lợi thế không dấu của model dự án không còn khi so theo pipeline; > 10: khả năng chịu không dấu vẫn có giá trị; ở giữa: chỉ báo số.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, SEED, load_json, read_jsonl, sha256_file  # noqa: E402
from stats import B, mcnemar  # noqa: E402

R = RESULTS / "restore"
DEV = DATA / "bench_v1" / "dev_v1.jsonl"
assert sha256_file(DEV) == load_json(DATA / "bench_v1" / "manifest_dev_v1.json")["sha256"]["dev_v1.jsonl"], "dev_v1 khác manifest"
rows = {r["id"]: r for r in read_jsonl(DEV)}
VI = sorted(i for i, r in rows.items() if r["lang"] == "vi" and r["variant"] == "orig")
RESTORERS = ["vit5", "small"]
load = lambda p: {x["id"]: x for x in read_jsonl(p)}

SRC = {"control": {"base": RESULTS / "dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl", "dir": "control-s1"},
       "strip": {"base": RESULTS / "ft_strip/eval/normal/strip-s1/preds.jsonl", "dir": "strip-s1"},
       "decider-2b": {"base": RESULTS / "dev_v1/eval/decider-2b-text/preds.jsonl", "dir": "decider-2b-text"}}
MODELS = list(SRC)
VARS = ["có dấu", "không dấu"] + [f"khôi phục {n}" for n in RESTORERS] + [f"đã sửa {n}" for n in RESTORERS]
P = defaultdict(dict)  # P[model][variant][orig_id] = pred
for m, s in SRC.items():
    base = load(s["base"])
    P[m]["có dấu"] = {i: base[i] for i in VI}
    P[m]["không dấu"] = {i: base[i + ":nodiac"] for i in VI}
    for n in RESTORERS:
        for tag, suf, sub in ((f"khôi phục {n}", n, n), (f"đã sửa {n}", f"{n}_fixed", f"{n}_fixed")):
            f = R / "eval" / sub / s["dir"] / "preds.jsonl"
            if f.exists():
                pr = load(f)
                P[m][tag] = {i: pr[f"{i}:restore:{suf}"] for i in VI}
VARS = [v for v in VARS if all(v in P[m] for m in MODELS)]

BLOCKS = {"song song (4 nguồn)": ["belebele", "sib200", "massive", "xcopa"],
          "song song · chỉ MASSIVE + XCOPA": ["massive", "xcopa"],
          "bản địa (3 nguồn)": ["vifactcheck", "viwikifc", "vihsd"]}
ALL_SRC = BLOCKS["song song (4 nguồn)"] + BLOCKS["bản địa (3 nguồn)"]


def ids_of(srcs):
    return [i for i in VI if rows[i]["source"] in srcs]


def macro_boot(vals_by_src, ids_by_src, seed):
    rng = np.random.default_rng(seed)
    est = np.zeros((B, len(vals_by_src)))
    for s, src in enumerate(vals_by_src):
        v = vals_by_src[src]
        g = defaultdict(list)
        for j, i in enumerate(ids_by_src[src]):
            g[rows[i]["cluster"]].append(j)
        cl = [np.array(x) for _, x in sorted(g.items())]
        sums = np.array([v[c].sum() for c in cl]); cnt = np.array([len(c) for c in cl])
        pick = rng.integers(0, len(cl), size=(B, len(cl)))
        est[:, s] = sums[pick].sum(1) / cnt[pick].sum(1)
    point = float(np.mean([v.mean() for v in vals_by_src.values()]))
    return point, [float(np.percentile(est.mean(1), 2.5)), float(np.percentile(est.mean(1), 97.5))]


def split(ids):
    g = defaultdict(list)
    for i in ids:
        g[rows[i]["source"]].append(i)
    return dict(sorted(g.items()))


def acc(pred, ids, seed):
    bs = split(ids)
    return macro_boot({s: np.array([float(pred[i]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed)


def paired(pa, pb, ids, seed):
    bs = split(ids)
    d, ci = macro_boot({s: np.array([float(pa[i]["correct"]) - float(pb[i]["correct"]) for i in v]) for s, v in bs.items()}, bs, seed)
    mc = mcnemar([pa[i]["correct"] for i in ids], [pb[i]["correct"] for i in ids])
    return {"diff": d, "ci": ci, "p": mc["p"], "a_only": mc["a_only"], "b_only": mc["b_only"]}


M = {"dev_v1_sha256": sha256_file(DEV), "variants": VARS, "acc": {}, "paired": {}, "per_source": {}, "reading": {}, "pipeline": {}}
for bj, (blk, srcs) in enumerate(BLOCKS.items()):
    ids = ids_of(srcs)
    for m in MODELS:
        for vj, v in enumerate(VARS):
            a, ci = acc(P[m][v], ids, SEED + 10 * bj + vj)
            M["acc"][f"{blk}|{m}|{v}"] = {"acc": a, "ci": ci, "n": len(ids)}
        for vj, v in enumerate(VARS[2:]):
            M["paired"][f"{blk}|{m}|{v} − không dấu"] = paired(P[m][v], P[m]["không dấu"], ids, SEED + 100 + 10 * bj + vj)
            M["paired"][f"{blk}|{m}|{v} − có dấu"] = paired(P[m][v], P[m]["có dấu"], ids, SEED + 200 + 10 * bj + vj)
    # so theo pipeline: model dự án (không dấu / khôi phục) − decider (khôi phục), cùng câu
    for vj, v in enumerate(VARS[2:]):
        for m in ("control", "strip"):
            M["pipeline"][f"{blk}|{m} không dấu − decider-2b {v}"] = paired(P[m]["không dấu"], P["decider-2b"][v], ids, SEED + 300 + 10 * bj + vj)
            M["pipeline"][f"{blk}|{m} {v} − decider-2b {v}"] = paired(P[m][v], P["decider-2b"][v], ids, SEED + 400 + 10 * bj + vj)
M["paired_source"] = {}
for m in MODELS:
    for sj, s in enumerate(ALL_SRC):
        ids = ids_of([s])
        M["per_source"][f"{m}|{s}"] = {v: float(np.mean([P[m][v][i]["correct"] for i in ids])) for v in VARS} | {"n": len(ids)}
        for vj, v in enumerate(VARS[2:]):
            M["paired_source"][f"{m}|{s}|{v} − không dấu"] = paired(P[m][v], P[m]["không dấu"], ids, SEED + 500 + 10 * sj + vj)
            M["paired_source"][f"{m}|{s}|{v} − có dấu"] = paired(P[m][v], P[m]["có dấu"], ids, SEED + 600 + 10 * sj + vj)


def zone(gap):
    return "≤ 5: lợi thế không dấu không còn khi so theo pipeline" if gap <= 0.05 else \
        "> 10: khả năng chịu không dấu của model vẫn có giá trị" if gap > 0.10 else "ở giữa (5–10): chỉ báo số, không kết luận"


for blk in BLOCKS:
    for v in VARS[2:]:
        pr = M["paired"][f"{blk}|decider-2b|{v} − có dấu"]
        gap = -pr["diff"]
        M["reading"][f"{blk}|{v}"] = {"gap_points": 100 * gap, "gap_ci": [-100 * pr["ci"][1], -100 * pr["ci"][0]], "zone": zone(gap)}

# lỗi ở lựa chọn sau khôi phục (đầu ra thô): lựa chọn bị đổi chữ-số, hai lựa chọn trùng nhau, lựa chọn đúng mất từ
import re as _re  # noqa: E402
import unicodedata as _ud  # noqa: E402
from common import strip_diacritics  # noqa: E402

akey = lambda s: "".join(c for c in strip_diacritics(s) if c.isalnum())
nwords = lambda s: sum(any(c.isalnum() for c in w) for w in s.split())
norm = lambda s: _re.sub(r"\s+", " ", _ud.normalize("NFC", s)).strip().lower()
OPT = {}
for n in RESTORERS:
    rr = {r["id"].split(":restore:")[0]: r for r in read_jsonl(R / "rows" / f"dev_v1_vi_restored_{n}.jsonl")}
    flags = {}
    for i in VI:
        nd, rs = rows[i + ":nodiac"], rr[i]
        g = int(nd["target"]) if isinstance(nd["target"], (int, str)) and str(nd["target"]).isdigit() else None
        changed = any(akey(a) != akey(b) for a, b in zip(nd["options"], rs["options"]))
        dup = len({norm(o) for o in rs["options"]}) < len(rs["options"]) and len({norm(o) for o in nd["options"]}) == len(nd["options"])
        lost = g is not None and nwords(rs["options"][g]) < nwords(nd["options"][g])
        flags[i] = {"changed": changed, "dup": dup, "gold_lost_words": lost}
    OPT[n] = flags
M["options_damage"] = {}
for n in RESTORERS:
    for s in ALL_SRC:
        ids = ids_of([s])
        e = {"rows": len(ids), "options_changed": sum(OPT[n][i]["changed"] for i in ids),
             "two_options_identical": sum(OPT[n][i]["dup"] for i in ids), "gold_option_lost_words": sum(OPT[n][i]["gold_lost_words"] for i in ids)}
        for m in MODELS:
            for grp, sel in (("đổi", True), ("giữ", False)):
                g_ids = [i for i in ids if OPT[n][i]["changed"] == sel]
                e[f"{m}|{grp}"] = {"n": len(g_ids), **({v: float(np.mean([P[m][v][i]["correct"] for i in g_ids]))
                                                       for v in ("có dấu", "không dấu", f"khôi phục {n}", f"đã sửa {n}") if v in P[m]} if g_ids else {})}
        M["options_damage"][f"{n}|{s}"] = e

# chất lượng khôi phục và độ trễ (từ check_restore.py / make_fixed.py)
Q = {n: load_json(R / "rows" / f"quality_{n}.json") for n in RESTORERS + [f"{n}_fixed" for n in RESTORERS]
     if (R / "rows" / f"quality_{n}.json").exists()}
FX = {n: load_json(R / "rows" / f"fixed_stats_{n}.json") for n in RESTORERS if (R / "rows" / f"fixed_stats_{n}.json").exists()}
M["quality"] = {n: {k: q[k] for k in ("invariant_violations", "violation_kinds", "violations_by_source", "truncated_outputs",
                                         "forced_splits", "quality", "latency", "dropped_chars_top")} for n, q in Q.items()}
M["fixed_stats"] = FX
(R / "metrics_restore.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ bảng
f1 = lambda x: f"{100 * x:.1f}"
fs = lambda x: f"{100 * x:+.1f}"
fc = lambda c: f"[{100 * c[0]:+.1f}; {100 * c[1]:+.1f}]"
fp = lambda p: f"{p:.2g}"
L = [f"dev_v1 SHA-256 khớp manifest: `{M['dev_v1_sha256'][:16]}…`. 1.763 câu tiếng Việt (1.000 song song, 763 bản địa).",
     "Kết quả chính: **khôi phục <bộ>** = đầu ra thô của bộ khôi phục. **đã sửa <bộ>** = kiểm độ nhạy (chiếu về đầu vào không dấu).", ""]
L += ["## 1. Accuracy (%, trung bình đều theo nguồn) [CI 95% bootstrap theo state]", "",
      "| Khối | Model | " + " | ".join(VARS) + " |", "|---|---|" + "---|" * len(VARS)]
for blk in BLOCKS:
    for m in MODELS:
        L.append(f"| {blk} | {m} | " + " | ".join(
            f"{f1(M['acc'][f'{blk}|{m}|{v}']['acc'])} [{f1(M['acc'][f'{blk}|{m}|{v}']['ci'][0])}; {f1(M['acc'][f'{blk}|{m}|{v}']['ci'][1])}]"
            for v in VARS) + " |")
L += ["", "## 2. Chênh lệch theo cặp (điểm %, [CI 95%], McNemar p)", "",
      "| Khối | Model | Biến thể | − không dấu | − có dấu |", "|---|---|---|---|---|"]
for blk in BLOCKS:
    for m in MODELS:
        for v in VARS[2:]:
            a = M["paired"][f"{blk}|{m}|{v} − không dấu"]; b = M["paired"][f"{blk}|{m}|{v} − có dấu"]
            L.append(f"| {blk} | {m} | {v} | {fs(a['diff'])} {fc(a['ci'])} (p={fp(a['p'])}) | {fs(b['diff'])} {fc(b['ci'])} (p={fp(b['p'])}) |")
L += ["", "## 3. Cách đọc đã định trước — decider-2b: khoảng cách có dấu − khôi phục (điểm)", "",
      "Áp cho khối song song đủ 4 nguồn và đầu ra thô. Các dòng khác (đã sửa; chỉ MASSIVE + XCOPA; bản địa) chỉ để đối chiếu.", "",
      "| Khối | Biến thể | Khoảng cách [CI 95%] | Vùng |", "|---|---|---|---|"]
for k, e in M["reading"].items():
    blk, v = k.split("|")
    L.append(f"| {blk} | {v} | {e['gap_points']:.1f} [{e['gap_ci'][0]:.1f}; {e['gap_ci'][1]:.1f}] | {e['zone']} |")
L += ["", "## 4. So theo pipeline: model dự án − decider-2b sau khôi phục (cùng câu; điểm, [CI], p)", "",
      "| Khối | So sánh | Chênh |", "|---|---|---|"]
for k, e in M["pipeline"].items():
    blk, c = k.split("|")
    L.append(f"| {blk} | {c} | {fs(e['diff'])} {fc(e['ci'])} (p={fp(e['p'])}) |")
L += ["", "## 5. Theo nguồn — accuracy (%)", "", "| Nguồn | n | Model | " + " | ".join(VARS) + " |", "|---|---|---|" + "---|" * len(VARS)]
for s in ALL_SRC:
    for m in MODELS:
        e = M["per_source"][f"{m}|{s}"]
        L.append(f"| {s} | {e['n']} | {m} | " + " | ".join(f1(e[v]) for v in VARS) + " |")
L += ["", "## 5b. Theo nguồn — chênh lệch theo cặp (điểm, [CI 95% bootstrap theo state], McNemar p)", "",
      "| Nguồn | Model | Biến thể | − không dấu | − có dấu |", "|---|---|---|---|---|"]
for s in ALL_SRC:
    for m in MODELS:
        for v in VARS[2:]:
            a_ = M["paired_source"][f"{m}|{s}|{v} − không dấu"]; b_ = M["paired_source"][f"{m}|{s}|{v} − có dấu"]
            L.append(f"| {s} | {m} | {v} | {fs(a_['diff'])} {fc(a_['ci'])} (p={fp(a_['p'])}) | {fs(b_['diff'])} {fc(b_['ci'])} (p={fp(b_['p'])}) |")
L += ["", "## 6. Chất lượng khôi phục so với bản có dấu gốc (state; NFC, phân biệt hoa/thường, bỏ qua khoảng trắng)", "",
      "| Bộ | Nguồn | dòng | âm tiết đúng % | âm tiết có dấu đúng % | state khớp hoàn toàn % | câu hỏi khớp % | mọi lựa chọn khớp % | dòng vi phạm bất biến |",
      "|---|---|---|---|---|---|---|---|---|"]
for n, q in Q.items():
    for s in ALL_SRC + ["block:parallel", "block:native", "all"]:
        e = q["quality"].get(s)
        if e:
            L.append(f"| {n} | {s} | {e['rows']} | {f1(e['syllable_acc'])} | {f1(e['diacritic_syllable_acc'])} | {f1(e['state_exact'])} | "
                     f"{f1(e['question_exact'])} | {f1(e['options_all_exact'])} | {e['rows_with_invariant_violation']} |")
L += ["", "## 7. Bất biến (bỏ dấu bản khôi phục = bản không dấu, chỉ cho lệch khoảng trắng)", "",
      "| Bộ | dòng vi phạm | state | câu hỏi | lựa chọn | chạm max_length | chỉ mất/thêm ký tự ngoài chữ-số | đổi chữ-số | tách câu cưỡng bức | ký tự bị rơi nhiều nhất |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for n, q in Q.items():
    v, k = q["invariant_violations"], q.get("violation_kinds", {})
    top = " ".join(f"`{c}`×{c_n}" for c, c_n in q.get("dropped_chars_top", [])[:8])
    L.append(f"| {n} | {v['row']} | {v['state']} | {v['question']} | {v['options']} | {k.get('truncated_output', 0)} | "
             f"{k.get('only_non_alnum_chars_lost_or_added', 0)} | {k.get('letters_or_digits_changed', 0)} | {q['forced_splits']} | {top} |")
L += ["", "Dòng vi phạm theo nguồn: " + "; ".join(f"{n}: " + ", ".join(f"{s} {c}" for s, c in sorted(q["violations_by_source"].items()))
                                                 for n, q in Q.items() if not n.endswith("_fixed"))]
L += ["", "## 7b. Lỗi ở lựa chọn sau khôi phục (đầu ra thô) và accuracy theo nhóm dòng", "",
      "\"đổi\" = có ít nhất một lựa chọn mà chuỗi chữ-số (sau khi bỏ dấu) khác bản không dấu; \"giữ\" = mọi lựa chọn giữ nguyên chữ-số. "
      "\"trùng\" = hai lựa chọn trở thành giống hệt nhau (NFC, không phân biệt hoa/thường) dù bản không dấu khác nhau; "
      "\"đúng mất từ\" = lựa chọn đúng còn ít từ hơn bản không dấu (vd. \"suc khoe\" → \"sức\").", "",
      "| Bộ | Nguồn | dòng | lựa chọn bị đổi | trùng | đúng mất từ |", "|---|---|---|---|---|---|"]
for k, e in M["options_damage"].items():
    n, s = k.split("|")
    L.append(f"| {n} | {s} | {e['rows']} | {e['options_changed']} | {e['two_options_identical']} | {e['gold_option_lost_words']} |")
L += ["", "Accuracy (%) theo nhóm dòng, cùng các câu đó ở mọi biến thể:", "",
      "| Bộ | Nguồn | Model | nhóm | n | có dấu | không dấu | khôi phục (thô) | đã sửa |", "|---|---|---|---|---|---|---|---|---|"]
for k, e in M["options_damage"].items():
    n, s = k.split("|")
    for m in MODELS:
        for grp in ("đổi", "giữ"):
            x = e[f"{m}|{grp}"]
            if x["n"]:
                L.append(f"| {n} | {s} | {m} | {grp} | {x['n']} | {f1(x['có dấu'])} | {f1(x['không dấu'])} | {f1(x[f'khôi phục {n}'])} | "
                         f"{f1(x[f'đã sửa {n}']) if f'đã sửa {n}' in x else '—'} |")
if FX:
    L += ["", "## 8. Bản đã sửa — số từ (có chữ-số) phải giữ nguyên không dấu, theo nguồn", "",
          "| Bộ | Nguồn | dòng bị sửa | từ | từ giữ không dấu | tỉ lệ % |", "|---|---|---|---|---|---|"]
    for n, fx in FX.items():
        for s in ALL_SRC:
            e = fx["by_source"].get(s)
            if e:
                L.append(f"| {n} | {s} | {e['rows_changed_by_fix']}/{e['rows']} | {e['all_words']} | {e['all_kept_nodiac']} | {100 * e['kept_share']:.2f} |")
L += ["", "## 9. Độ trễ bộ khôi phục, batch 1 (state + câu hỏi + lựa chọn của một câu; fp32, greedy, theo thẻ model)", "",
      "| Bộ | state (token Qwen) | dòng | p50 ms | p90 ms | TB ms | số lần gọi TB |", "|---|---|---|---|---|---|---|"]
for n, q in Q.items():
    if n.endswith("_fixed"):
        continue
    for b in ("≤128", "129–512", "513–1024", ">1024", "all"):
        e = q["latency"].get(b)
        if e:
            L.append(f"| {n} | {b} | {e['rows']} | {e['total_p50_ms']:.0f} | {e['total_p90_ms']:.0f} | {e['total_mean_ms']:.0f} | {e['calls_mean']:.1f} |")
lat = load_json(RESULTS / "ft" / "latency_ft.json")
dm = {t: np.median([r["p50_ms"] for r in lat["results"] if "1.7B" in r["model"] and r["target"] == t]) for t in (128, 512, 2048)}
tc = load_json(RESULTS / "dev_v1/eval/Qwen3-1.7B-fullvocab-s1/timing.json"); td = load_json(RESULTS / "dev_v1/eval/decider-2b-text/timing.json")
L += ["", f"Model quyết định (đo trước, batch 1, bf16): Qwen3-1.7B + adapter p50 {dm[128]:.0f} / {dm[512]:.0f} / {dm[2048]:.0f} ms ở prompt "
      f"128 / 512 / 2.048 token (`results/ft/latency_ft.json`); trung vị trên dev_v1: adapter đối chứng "
      f"{tc['batch1_median_latency_ms']:.0f} ms, decider-2b {td['batch1_median_latency_ms']:.0f} ms."]
(R / "tables_restore.md").write_text("\n".join(L) + "\n", encoding="utf-8")
for k, e in M["reading"].items():
    print(k, f"{e['gap_points']:.1f}", e["zone"])
