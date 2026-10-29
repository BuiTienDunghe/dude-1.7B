"""Bước E: bảng so model ngoài với mốc dự án trên DEV 1.2 (data/bench_v1_2/dev_v1_2.jsonl). Chạy bằng .venv chính (cần scipy).
Ra: results/baselines_v1_2/metrics_baselines_v1_2.json, tables_baselines_v1_2.md, ung_vien_soat_nhan.jsonl.

- Biến thể: VI có dấu (6 nguồn), VI không dấu (6 nguồn), EN (4 nguồn song song). Accuracy và Brier theo nguồn + trung bình đều, CI 95% bootstrap theo
  state (cụm = trường cluster). Cột "1.2 chỉ dòng source": cặp có vi_provenance = source và gold_provenance = source.
- Câu bị cắt hoặc bị từ chối (excluded) không tính điểm; đếm riêng. So ghép cặp chỉ trên câu cả hai model đều tính.
- So với Qwen3-1.7B full_vocab (dự án): hiệu theo nguồn (McNemar chính xác) và trung bình đều (CI bootstrap), có bản bỏ MASSIVE.
- Tụt Anh → Việt: ghép theo pair_id trên 4 nguồn song song.
- Nhất quán khi đảo thứ tự: pred_normal == k-1-pred_reverse, trên câu KHÔNG phải noul (model ngoài chấm noul bằng đường yes/no gốc, đầu vào không phụ
  thuộc thứ tự; với model dự án noul vẫn đảo chữ cái nhưng để so cùng tập câu, mọi model đều tính trên câu không phải noul).
- Model "đáng tin" = qua đủ kiểm adapter (README, câu hiển nhiên hai thứ tự, xác suất đủ phần tử và tổng = 1). Model chưa qua: KHÔNG vào bảng chính.
- Ứng viên soát nhãn: câu dev mà >= 2/3 số model đáng tin (7 preds của scores_v1_2.json + model ngoài đáng tin, mỗi model ngoài 1 phiếu theo cách ánh xạ
  điểm cao hơn) cùng chọn một đáp án khác đáp án chuẩn. Chỉ lập danh sách."""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import SEED  # noqa: E402
from stats import boot, brier, ci, cluster_index, mcnemar  # noqa: E402

DEV = ROOT / "data/bench_v1_2/dev_v1_2.jsonl"
DEV_SHA = "ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b"
B = ROOT / "results/baselines_v1_2"
D12 = ROOT / "results/dev_v1_2/eval"
assert hashlib.sha256(DEV.read_bytes()).hexdigest() == DEV_SHA
rows = {json.loads(l)["id"]: json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip()}
PAIR_SRC = {r["pair_id"]: r["vi_provenance"] == "source" and r["gold_provenance"] == "source" for r in rows.values() if r["lang"] == "vi" and r["variant"] == "orig"}
VARS = {"vi": ("vi", "orig"), "vi_nodiac": ("vi", "nodiac"), "en": ("en", "orig")}
SRC_VI = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
SRC_EN = ["belebele", "sib200", "massive", "xcopa"]
SRCS = {"vi": SRC_VI, "vi_nodiac": SRC_VI, "en": SRC_EN}
ids_by = {(v, s): sorted(i for i, r in rows.items() if (r["lang"], r["variant"]) == VARS[v] and r["source"] == s) for v in VARS for s in SRC_VI}
read = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
jl = lambda p: json.load(open(p, encoding="utf-8")) if Path(p).exists() else {}

# ---------------------------------------------------------------- model
# tên: (preds normal, preds reverse | None, timing, nhóm, tham số)
M = {
    "Qwen3-1.7B gốc": (D12 / "Qwen3-1.7B", B / "eval/reverse/Qwen3-1.7B", B / "eval/reverse/Qwen3-1.7B/timing.json", "mốc dự án", "2,03 B"),
    "Qwen3-1.7B full_vocab (dự án)": (D12 / "Qwen3-1.7B-fullvocab-s1", B / "eval/reverse/Qwen3-1.7B-fullvocab-s1",
                                      B / "eval/reverse/Qwen3-1.7B-fullvocab-s1/timing.json", "mốc dự án", "2,03 B + LoRA 17,4 M (đã merge)"),
    "Qwen3.5-2B gốc": (B / "eval/normal/Qwen3.5-2B", B / "eval/reverse/Qwen3.5-2B", B / "eval/normal/Qwen3.5-2B/timing.json", "cùng cỡ",
                       "2,27 B trong repo (gồm bộ mã hóa ảnh); chỉ nạp phần văn bản"),
    "decider-2b (text)": (D12 / "decider-2b-text", B / "eval/reverse/decider/decider-2b-text", B / "eval/reverse/decider/decider-2b-text/timing.json",
                          "cùng cỡ", "1,88 B"),
    "decider-2b (keyed)": (D12 / "decider-2b-keyed", B / "eval/reverse/decider/decider-2b-keyed", B / "eval/reverse/decider/decider-2b-keyed/timing.json",
                           "cùng cỡ", "1,88 B"),
}
EXT = {"quyet-tiny": ("Quyet-1.0-Tiny", "nhỏ hơn", "183 M (mmBERT-small)"), "quyet-small": ("Quyet-1.0-Small", "nhỏ hơn", "328 M (SEA-LION-ModernBERT-300M)"),
       "laya-multilingual": ("laya-multilingual", "nhỏ hơn", "322 M (mmBERT-base)"), "laya-vi": ("laya-vi", "nhỏ hơn", "322 M (mmBERT-base)"),
       "quyet-medium": ("Quyet-1.0-Medium", "lớn hơn", "4,66 B (Qwen3.5-4B + LoRA đã merge)")}
TRUST = {}
for key, (nm, grp, par) in EXT.items():
    for mp in ("text", "keyed"):
        d = B / f"eval/normal/{key}-{mp}"
        if (d / "preds.jsonl").exists():
            M[f"{nm} ({mp})"] = (d, B / f"eval/reverse/{key}-{mp}", d / "timing.json", grp, par)
            TRUST[f"{nm} ({mp})"] = bool(jl(d / "timing.json").get("tin_cay"))
# lượt chạy lại khi GPU rảnh (eval/rerun): dùng độ trễ của lượt này; dự đoán phải trùng lượt đầu (kiểm tái lập)
RERUN = {"Qwen3.5-2B gốc": ("normal", B / "eval/rerun/normal/Qwen3.5-2B"), "Qwen3-1.7B full_vocab (dự án)": ("reverse", B / "eval/rerun/reverse/Qwen3-1.7B-fullvocab-s1"),
         "decider-2b (text)": ("reverse", B / "eval/rerun/reverse/decider/decider-2b-text"), "decider-2b (keyed)": ("reverse", B / "eval/rerun/reverse/decider/decider-2b-keyed")}
TAI_LAP = {}
for n, (order, d) in RERUN.items():
    if (d / "preds.jsonl").exists():
        first = read((M[n][0] if order == "normal" else M[n][1]) / "preds.jsonl"); again = read(d / "preds.jsonl")
        TAI_LAP[n] = {"thu_tu": order, "dong": len(again), "cung_du_doan": sum(first[i]["pred"] == again[i]["pred"] for i in again if i in first) / len(again),
                      "xac_suat_lech_max": max(max(abs(x - y) for x, y in zip(first[i]["probs"], again[i]["probs"])) for i in again if i in first)}
        pn, pr, pt, g, par = M[n]
        M[n] = (pn, pr, d / "timing.json", g, par)
for n in M:
    TRUST.setdefault(n, True)    # mốc dự án, Qwen3.5-2B (eval_base, đã kiểm câu hiển nhiên), decider-2b (adapter đã kiểm ở bước baseline)
REF = "Qwen3-1.7B full_vocab (dự án)"
P, R, TIM = {}, {}, {}
for n, (pn, pr, pt, _, _) in M.items():
    P[n] = read(pn / "preds.jsonl")
    assert set(P[n]) == set(rows), (n, len(set(rows) - set(P[n])), len(set(P[n]) - set(rows)))
    R[n] = read(pr / "preds.jsonl") if pr and (pr / "preds.jsonl").exists() else {}
    TIM[n] = jl(pt)
ok = lambda n, i: not P[n][i].get("excluded")
corr = lambda Pn, ids: np.array([Pn[i]["correct"] for i in ids], dtype=float)


def acc_brier(n, v, sources, filt=None, seed=SEED):
    per_a, per_b, out = [], [], {}
    for s in sources:
        ids = [i for i in ids_by[(v, s)] if ok(n, i) and (filt is None or filt(i))]
        if not ids:
            out[s] = {"acc": float("nan"), "n": 0}; per_a.append(None); continue
        cl = cluster_index(ids, rows)
        a = corr(P[n], ids); b = np.array([brier(P[n][i]["probs"], rows[i]["target"]) for i in ids])
        per_a.append((a, cl)); per_b.append((b, cl)); out[s] = {"acc": float(a.mean()), "brier": float(b.mean()), "n": len(ids)}
    good = [x for x in per_a if x is not None]
    if len(good) == len(sources):
        ba, bb = boot(per_a, seed), boot(per_b, seed + 1)
        for j, s in enumerate(sources):
            out[s]["acc_ci"], out[s]["brier_ci"] = ci(ba[:, j]), ci(bb[:, j])
        out["macro"] = {"acc": float(np.mean([out[s]["acc"] for s in sources])), "acc_ci": ci(ba.mean(1)),
                        "brier": float(np.mean([out[s]["brier"] for s in sources])), "brier_ci": ci(bb.mean(1))}
    else:
        out["macro"] = {"acc": float("nan")}
    out["excluded"] = sum(1 for s in sources for i in ids_by[(v, s)] if not ok(n, i))
    return out


def paired(nA, nB, v, sources, seed, mapA=None):
    """Hiệu A − B trên cùng câu (bỏ câu bị loại ở một trong hai)."""
    per, out = [], {}
    for s in sources:
        ids = [i for i in ids_by[(v, s)] if ok(nB, i) and ok(nA, (mapA or (lambda x: x))(i))]
        a = corr(P[nA], [(mapA or (lambda x: x))(i) for i in ids]); b = corr(P[nB], ids)
        per.append((a - b, cluster_index(ids, rows)))
        out[s] = {"diff": float((a - b).mean()), **mcnemar(a.astype(bool), b.astype(bool)), "n": len(ids)}
    bd = boot(per, seed)
    for j, s in enumerate(sources):
        out[s]["ci"] = ci(bd[:, j])
    allA = np.concatenate([x[0] for x in per]); out["macro"] = {"diff": float(np.mean([out[s]["diff"] for s in sources])), "ci": ci(bd.mean(1))}
    a_all = np.concatenate([corr(P[nA], [(mapA or (lambda x: x))(i) for i in ids_by[(v, s)] if ok(nB, i) and ok(nA, (mapA or (lambda x: x))(i))]) for s in sources])
    b_all = np.concatenate([corr(P[nB], [i for i in ids_by[(v, s)] if ok(nB, i) and ok(nA, (mapA or (lambda x: x))(i))]) for s in sources])
    out["macro"].update(mcnemar(a_all.astype(bool), b_all.astype(bool)))
    del allA
    return out


out = {"dev_sha": DEV_SHA, "ref": REF, "trust": TRUST, "tai_lap": TAI_LAP, "acc": {}, "acc_source_only": {}, "acc_no_massive": {}, "vs_ref": {}, "vs_ref_no_massive": {},
       "en_vi_drop": {}, "order": {}, "latency_ms": {}, "excluded": {}, "params": {n: M[n][4] for n in M}, "group": {n: M[n][3] for n in M}}
for n in M:
    out["acc"][n] = {v: acc_brier(n, v, SRCS[v]) for v in VARS}
    out["acc_source_only"][n] = {v: acc_brier(n, v, SRCS[v], filt=lambda i: PAIR_SRC.get(rows[i]["pair_id"], False))["macro"] for v in VARS}
    out["acc_no_massive"][n] = {v: acc_brier(n, v, [s for s in SRCS[v] if s != "massive"])["macro"] for v in VARS}
    out["excluded"][n] = {v: out["acc"][n][v]["excluded"] for v in VARS}
    out["latency_ms"][n] = TIM[n].get("batch1_median_latency_ms")
    if n != REF:
        out["vs_ref"][n] = {v: paired(n, REF, v, SRCS[v], SEED + 70) for v in VARS}
        out["vs_ref_no_massive"][n] = {v: paired(n, REF, v, [s for s in SRCS[v] if s != "massive"], SEED + 71)["macro"] for v in VARS}
    # Anh -> Việt: hiệu EN − VI có dấu trên cùng pair_id (4 nguồn song song)
    per, d = [], {}
    for s in SRC_EN:
        ids = [i for i in ids_by[("en", s)] if ok(n, i) and ok(n, rows[i]["pair_id"] + ":vi")]
        a = corr(P[n], ids); b = corr(P[n], [rows[i]["pair_id"] + ":vi" for i in ids])
        per.append((a - b, cluster_index(ids, rows))); d[s] = float((a - b).mean())
    bd = boot(per, SEED + 40)
    out["en_vi_drop"][n] = {"theo_nguon": d, "macro": float(np.mean(list(d.values()))), "ci": ci(bd.mean(1))}
    if R[n]:
        o = {}
        for v in VARS:
            cons, a2 = [], []
            for s in SRCS[v]:
                ids = [i for i in ids_by[(v, s)] if rows[i]["kind"] != "noul" and i in R[n] and ok(n, i) and not R[n][i].get("excluded")]
                if ids:
                    cons.append(np.mean([P[n][i]["pred"] == P[n][i]["k"] - 1 - R[n][i]["pred"] for i in ids]))
                    a2.append((corr(P[n], ids).mean() + corr(R[n], ids).mean()) / 2)
            o[v] = {"consistency": float(np.mean(cons)) if cons else None, "acc_two_orders": float(np.mean(a2)) if a2 else None}
        out["order"][n] = o

# câu hỏi 3 của brief: Qwen3.5-2B gốc − Qwen3-1.7B gốc (ghép cặp)
out["qwen35_vs_qwen3"] = {v: paired("Qwen3.5-2B gốc", "Qwen3-1.7B gốc", v, SRCS[v], SEED + 90) for v in VARS}
out["qwen35_vs_qwen3_no_massive"] = {v: paired("Qwen3.5-2B gốc", "Qwen3-1.7B gốc", v, [s for s in SRCS[v] if s != "massive"], SEED + 91)["macro"] for v in VARS}

# ---------------------------------------------------------------- ứng viên soát nhãn
P7 = {m: read(D12 / m / "preds.jsonl") for m in ("decider-2b-text", "decider-2b-keyed", "decider-ft-ep2-text", "decider-ft-ep2-keyed",
                                                   "Qwen3-1.7B-fullvocab-s1", "Qwen3-1.7B", "strip-s1")}
voters = dict(P7)
if TRUST.get("Qwen3.5-2B gốc"):
    voters["Qwen3.5-2B"] = P["Qwen3.5-2B gốc"]
for key, (nm, _, _) in EXT.items():
    cands = [f"{nm} ({mp})" for mp in ("text", "keyed") if f"{nm} ({mp})" in M and TRUST.get(f"{nm} ({mp})")]
    if cands:
        best = max(cands, key=lambda c: out["acc"][c]["vi"]["macro"]["acc"])
        voters[best] = P[best]
cand = []
for i, r in rows.items():
    votes = {m: p[i]["pred"] for m, p in voters.items() if i in p and not p[i].get("excluded") and p[i].get("pred") is not None}
    if not votes:
        continue
    cnt = Counter(votes.values())
    maj, nmaj = cnt.most_common(1)[0]
    if maj != r["target"] and nmaj * 3 >= 2 * len(votes):
        cand.append({"id": i, "pair_id": r["pair_id"], "nguon": r["source"], "lang": r["lang"], "variant": r["variant"], "dap_an_chuan": r["target"],
                     "dap_an_da_so": maj, "ty_le_dong_thuan": round(nmaj / len(votes), 3), "so_model": len(votes),
                     "model_chon_da_so": sorted(m for m, v in votes.items() if v == maj), "vi_provenance": rows[r["pair_id"] + ":vi"]["vi_provenance"]
                     if r["pair_id"] + ":vi" in rows else None})
cand.sort(key=lambda x: (-x["ty_le_dong_thuan"], x["nguon"], x["id"]))
with open(B / "ung_vien_soat_nhan.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for c in cand:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
out["ung_vien"] = {"so_dong": len(cand), "so_cap": len({c["pair_id"] for c in cand}), "theo_nguon_bien_the": dict(Counter(f"{c['nguon']}|{c['lang']}|{c['variant']}" for c in cand)),
                   "nguoi_bo_phieu": sorted(voters)}
clean = lambda x: None if isinstance(x, float) and math.isnan(x) else x
(B / "metrics_baselines_v1_2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=clean), encoding="utf-8")
print(json.dumps({"models": list(M), "trust": TRUST, "ung_vien": out["ung_vien"]}, ensure_ascii=False))


# ---------------------------------------------------------------- bảng markdown
NAMEV = {"vi": "VI có dấu", "vi_nodiac": "VI không dấu", "en": "EN"}
NM = {"belebele": "Belebele", "sib200": "SIB-200", "massive": "MASSIVE", "xcopa": "XCOPA", "vifactcheck": "ViFactCheck", "viwikifc": "ViWikiFC"}
f1 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}".replace(".", ",")
fs = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}".replace(".", ",")
fci = lambda c: f"[{f1(c[0])}; {f1(c[1])}]" if c else ""
fsci = lambda c: f"[{fs(c[0])}; {fs(c[1])}]" if c else ""
f3 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}".replace(".", ",")
fp = lambda p: "—" if p is None else (f"{p:.0e}" if p < 1e-3 else f"{p:.3f}".replace(".", ","))
trusted = [n for n in M if TRUST[n]]
L = ["# Bảng số bước E (DEV 1.2)", "", f"Dev: `data/bench_v1_2/dev_v1_2.jsonl` (sha `{DEV_SHA}`). Accuracy (%), trung bình đều theo nguồn, [CI 95% bootstrap theo state].", ""]
for v in VARS:
    srcs = SRCS[v]
    L += [f"## {NAMEV[v]}", "", "| Model | Trung bình đều [CI] | Chỉ dòng source | Bỏ MASSIVE | " + " | ".join(NM[s] for s in srcs) + " | Brier | Câu bị loại |",
          "|---" * (6 + len(srcs)) + "|"]
    for n in trusted:
        a = out["acc"][n][v]
        L.append(f"| {n} | {f1(a['macro']['acc'])} {fci(a['macro'].get('acc_ci'))} | {f1(out['acc_source_only'][n][v]['acc'])} | {f1(out['acc_no_massive'][n][v]['acc'])} | "
                 + " | ".join(f1(a[s]["acc"]) for s in srcs) + f" | {f3(a['macro'].get('brier'))} | {a['excluded']} |")
    L.append("")
L += [f"## So với {REF} (A − dự án, điểm %; [CI]; p McNemar gộp)", "", "| Model | VI có dấu | VI không dấu | EN | VI bỏ MASSIVE | EN bỏ MASSIVE |", "|---|---|---|---|---|---|"]
for n in trusted:
    if n == REF:
        continue
    d = out["vs_ref"][n]; dn = out["vs_ref_no_massive"][n]
    L.append(f"| {n} | " + " | ".join(f"{fs(d[v]['macro']['diff'])} {fsci(d[v]['macro']['ci'])} (p={fp(d[v]['macro']['p'])})" for v in VARS)
             + f" | {fs(dn['vi']['diff'])} {fsci(dn['vi']['ci'])} | {fs(dn['en']['diff'])} {fsci(dn['en']['ci'])} |")
L += ["", "### Theo nguồn, VI có dấu (A − dự án; p McNemar chính xác)", "", "| Model | " + " | ".join(NM[s] for s in SRC_VI) + " |", "|---" * (1 + len(SRC_VI)) + "|"]
for n in trusted:
    if n != REF:
        L.append(f"| {n} | " + " | ".join(f"{fs(out['vs_ref'][n]['vi'][s]['diff'])} (p={fp(out['vs_ref'][n]['vi'][s]['p'])})" for s in SRC_VI) + " |")
for v in ("en", "vi_nodiac"):
    L += ["", f"### Theo nguồn, {NAMEV[v]} (A − dự án; p McNemar chính xác)", "", "| Model | " + " | ".join(NM[s] for s in SRCS[v]) + " |",
          "|---" * (1 + len(SRCS[v])) + "|"]
    for n in trusted:
        if n != REF:
            L.append(f"| {n} | " + " | ".join(f"{fs(out['vs_ref'][n][v][s]['diff'])} (p={fp(out['vs_ref'][n][v][s]['p'])})" for s in SRCS[v]) + " |")
for v in VARS:
    L += ["", f"### Brier theo nguồn, {NAMEV[v]} (thấp hơn là tốt hơn) [CI]", "", "| Model | " + " | ".join(NM[s] for s in SRCS[v]) + " | Trung bình đều |",
          "|---" * (2 + len(SRCS[v])) + "|"]
    for n in trusted:
        a = out["acc"][n][v]
        L.append(f"| {n} | " + " | ".join(f"{f3(a[s].get('brier'))}" for s in SRCS[v]) + f" | {f3(a['macro'].get('brier'))} "
                 + (f"[{f3(a['macro']['brier_ci'][0])}; {f3(a['macro']['brier_ci'][1])}]" if a['macro'].get('brier_ci') else "") + " |")
L += ["", "## Tụt Anh → Việt, nhất quán khi đảo thứ tự, độ trễ", "",
      "Nhất quán khi đảo tính trên câu KHÔNG phải Có/Không (noul) cho MỌI model, kể cả mốc dự án, để cùng một tập câu. Độ trễ: trung vị batch 1 trên toàn bộ "
      "3.851 dòng dev, riêng decider-2b đo ở lượt đảo trên 3.445 dòng không phải noul (khác tập dòng, cùng cách đo).", "",
      "| Model | Tham số | EN − VI có dấu [CI] | Nhất quán khi đảo VI / VI không dấu / EN | Accuracy trung bình hai thứ tự VI | Độ trễ trung vị batch 1 (ms) |",
      "|---|---|---|---|---|---|"]
for n in trusted:
    o = out["order"].get(n, {})
    L.append(f"| {n} | {out['params'][n]} | {fs(out['en_vi_drop'][n]['macro'])} {fsci(out['en_vi_drop'][n]['ci'])} | "
             + " / ".join(f1((o.get(v) or {}).get("consistency")) for v in VARS) + f" | {f1((o.get('vi') or {}).get('acc_two_orders'))} | "
             + (f"{out['latency_ms'][n]:.1f}".replace(".", ",") if out["latency_ms"][n] else "—") + " |")
L += ["", f"Tái lập (lượt chạy lại khi GPU rảnh so với lượt đầu): " + "; ".join(f"{n} {v['thu_tu']}: cùng dự đoán {f1(v['cung_du_doan'])}%, xác suất lệch tối đa {v['xac_suat_lech_max']:.2e}"
                                                                         for n, v in TAI_LAP.items()), ""]
# trạng thái kiểm adapter của model ngoài (model chưa qua: KHÔNG ghi điểm)
L += ["## Kiểm adapter model ngoài", "", "| Model | README | Câu hiển nhiên sai, text / keyed (trên 40 câu × 2 thứ tự) | Khớp predict() công khai (4 chữ số) | "
      "Câu vượt giới hạn mặc định | Câu bị cắt / từ chối ở giới hạn đã nâng | Vào bảng chính |", "|---|---|---|---|---|---|---|"]
for key, (nm, _, _) in EXT.items():
    ck = jl(B / f"checks/{key}.json")
    over = cut = 0
    for mp in ("text", "keyed"):
        d = B / f"eval/normal/{key}-{mp}/preds.jsonl"
        if d.exists():
            pp = read(d)
            over = max(over, sum(1 for x in pp.values() if x.get("note") == "vuot_mac_dinh"))
            cut = max(cut, sum(1 for x in pp.values() if x.get("excluded")))
    w = ck.get("wiring", {})
    wmax = max((v["lech_lon_nhat"] for v in w.values()), default=float("nan"))
    L.append(f"| {nm} | {'đạt' if ck.get('readme', {}).get('pass') else 'KHÔNG đạt'} | "
             + " / ".join(str(len(ck.get("mappings", {}).get(mp, {}).get("sai", []))) for mp in ("text", "keyed"))
             + f" | {'có' if w and all(v['khop_4_chu_so'] for v in w.values()) else 'không'} (lệch tối đa {wmax:.1e}) | {over} | {cut} | "
             + f"{'có' if any(TRUST.get(f'{nm} ({mp})') for mp in ('text', 'keyed')) else 'không (chưa qua kiểm)'} |")
    out.setdefault("kiem_adapter", {})[key] = {"readme": ck.get("readme"), "sai_hien_nhien": {mp: ck.get("mappings", {}).get(mp, {}).get("sai") for mp in ("text", "keyed")},
                                               "wiring": w, "readme_fp32": ck.get("readme_fp32"), "vuot_mac_dinh": over, "bi_loai": cut,
                                               "medium_keyed_bang_text": ck.get("medium_keyed_bang_text")}
SIDE_NOTE = {"quyet-tiny": "keyed sai câu Có/Không số 7 (VI, EN; hai thứ tự); text sai thêm câu chọn đáp án 2 (VI), 5 (EN), 6 (VI, EN)",
             "quyet-small": "keyed sai câu Có/Không số 7 (VI, EN; hai thứ tự); text sai thêm câu chọn đáp án 1 (VI, EN), 2 (EN), 5 (EN)",
             "laya-vi": "sai câu Có/Không số 4 (VI) và số 6 (EN), hai thứ tự; keyed sai thêm câu chọn đáp án 5 (VI, một thứ tự)",
             "laya-multilingual": "sai mọi câu Có/Không có đáp án 'Có' (0, 2, 4, 6, 8; VI và EN) và câu 7 EN; câu chọn đáp án 5 (VI); ví dụ README "
                                  "trượt cả ở fp32 (chọn 'sales' thay vì 'billing'); gói laya tự ghi revision 'đã duyệt' e4e9ddf2…, khác pin 1720e3e3… của dự án"}
L += ["", "## Bảng phụ: CHƯA QUA KIỂM CÂU HIỂN NHIÊN – chỉ tham khảo", "",
      "Bốn model dưới đây trượt câu hiển nhiên; adapter khớp predict() công khai đến 4 chữ số nên số phản ánh model, nhưng KHÔNG thuộc bảng chính. "
      "Dùng cách ánh xạ cho VI có dấu cao hơn (quy tắc brief).", "",
      "| Model | Ánh xạ dùng | VI có dấu [CI] | VI không dấu | EN [CI] | Brier VI / EN | Nhất quán khi đảo VI / VI không dấu / EN | Độ trễ (ms) | Câu hiển nhiên sai |",
      "|---|---|---|---|---|---|---|---|---|"]
for key, (nm, _, _) in EXT.items():
    if any(TRUST.get(f"{nm} ({mp})") for mp in ("text", "keyed")):
        continue
    cands = [f"{nm} ({mp})" for mp in ("text", "keyed") if f"{nm} ({mp})" in M]
    if not cands:
        continue
    n = max(cands, key=lambda c: out["acc"][c]["vi"]["macro"]["acc"])
    a = out["acc"][n]; o = out["order"].get(n, {})
    L.append(f"| {nm} | {n.split('(')[-1].rstrip(')')} | {f1(a['vi']['macro']['acc'])} {fci(a['vi']['macro'].get('acc_ci'))} | {f1(a['vi_nodiac']['macro']['acc'])} | "
             f"{f1(a['en']['macro']['acc'])} {fci(a['en']['macro'].get('acc_ci'))} | {f3(a['vi']['macro'].get('brier'))} / {f3(a['en']['macro'].get('brier'))} | "
             + " / ".join(f1((o.get(v) or {}).get("consistency")) for v in VARS) + f" | {out['latency_ms'][n]:.1f} | {SIDE_NOTE[key]} |".replace(".", ",", 1))
L += ["", "Quyet-1.0-Medium: hai cách ánh xạ cho cùng prompt (gói ẩn khóa khi có mô tả; đã kiểm xác suất trùng trên câu hiển nhiên), nên chỉ chấm và ghi một lần.", ""]
(B / "tables_baselines_v1_2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(B / "metrics_baselines_v1_2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=clean), encoding="utf-8")
