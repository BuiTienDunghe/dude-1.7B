"""Bench 1.3 DỰNG THỬ, bước 4: preds dev 1.3 = preds dev 1.2 cho dòng có nội dung không đổi (bỏ riêng trường mo_ho khi so) + preds chấm lại cho dòng
mới / đổi (results/bench_v1_3_trial/eval_rescore); dòng bị loại bỏ khỏi điểm. Bảng so 1.2 với 1.3: điểm chính và lát bỏ câu mơ hồ, theo 9 model.
Accuracy: trung bình đều theo nguồn (6 nguồn cho VI, 4 cho EN) và trung bình trên mọi câu. Python Windows. Ra: eval/<model>/preds.jsonl,
scores_v1_3_trial.json, và mục 7 trong REPORT_bench_v1_3_trial.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/bench_v1_3_trial"
R = T / "eval_rescore"
read = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
FLAG = "mo_ho"
core = lambda r: {k: v for k, v in r.items() if k != FLAG}
d12, d13 = read(ROOT / "data/bench_v1_2/dev_v1_2.jsonl"), read(T / "dev_v1_3_trial.jsonl")
MODELS = {  # tên: (preds dev 1.2, preds chấm lại 1.3)
    "decider-2b-text": (ROOT / "results/dev_v1_2/eval/decider-2b-text", R / "decider/decider-2b-text"),
    "decider-2b-keyed": (ROOT / "results/dev_v1_2/eval/decider-2b-keyed", R / "decider/decider-2b-keyed"),
    "decider-ft-ep2-text": (ROOT / "results/dev_v1_2/eval/decider-ft-ep2-text", R / "ftd/decider-2b-text"),
    "decider-ft-ep2-keyed": (ROOT / "results/dev_v1_2/eval/decider-ft-ep2-keyed", R / "ftd/decider-2b-keyed"),
    "Qwen3-1.7B-fullvocab-s1 (dự án)": (ROOT / "results/dev_v1_2/eval/Qwen3-1.7B-fullvocab-s1", R / "control"),
    "Qwen3-1.7B": (ROOT / "results/dev_v1_2/eval/Qwen3-1.7B", R / "base"),
    "strip-s1": (ROOT / "results/dev_v1_2/eval/strip-s1", R / "strip"),
    "Qwen3.5-2B": (ROOT / "results/baselines_v1_2/eval/normal/Qwen3.5-2B", R / "Qwen3.5-2B"),
    "Quyet-1.0-Medium": (ROOT / "results/baselines_v1_2/eval/normal/quyet-medium-text", R / "ext/normal/quyet-medium-text"),
}
SRC_VI = ["belebele", "sib200", "massive", "xcopa", "vifactcheck", "viwikifc"]
VARS = {"vi": ("vi", "orig", SRC_VI), "vi_nodiac": ("vi", "nodiac", SRC_VI), "en": ("en", "orig", SRC_VI[:4])}
same = {i for i, r in d13.items() if i in d12 and core(r) == core(d12[i])}
need = set(d13) - same


def score(rows, preds, v, keep=lambda r: True):
    lang, var, srcs = VARS[v]
    per, n_all, c_all = {}, 0, 0
    for s in srcs:
        ids = [i for i, r in rows.items() if r["source"] == s and r["lang"] == lang and r["variant"] == var and keep(r)]
        if not ids:          # lát con (ví dụ chỉ câu mơ hồ) có thể không có dòng ở một số nguồn
            continue
        c = sum(preds[i]["pred"] == rows[i]["target"] for i in ids)
        per[s] = c / len(ids); n_all += len(ids); c_all += c
    return {"macro": 100 * sum(per.values()) / len(per), "tren_moi_cau": 100 * c_all / n_all, "theo_nguon": {s: 100 * x for s, x in per.items()}, "n": n_all}


out = {"dong_dung_preds_1_2": len(same), "dong_cham_lai": sorted(need), "scores": {}}
for m, (p12, p13) in MODELS.items():
    P12, PR = read(p12 / "preds.jsonl"), read(p13 / "preds.jsonl")
    assert set(d12) <= set(P12), m
    assert need <= set(PR), (m, sorted(need - set(PR)))
    P13 = {i: (P12[i] if i in same else PR[i]) for i in d13}
    (T / f"eval/{m}").mkdir(parents=True, exist_ok=True)
    with open(T / f"eval/{m}/preds.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in d13.items():
            x = {k: v for k, v in P13[i].items() if k not in ("rescored", "nguon_preds")}
            x.update(target=r["target"], correct=x["pred"] == r["target"], nguon_preds="dev_1_2" if i in same else "cham_lai_1_3")
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    out["scores"][m] = {v: {"1.2": score(d12, P12, v), "1.3": score(d13, P13, v), "1.3_bo_mo_ho": score(d13, P13, v, lambda r: not r.get(FLAG)),
                            "1.3_chi_mo_ho": score(d13, P13, v, lambda r: bool(r.get(FLAG))) if v != "en" or any(r.get(FLAG) and r["lang"] == "en" for r in d13.values()) else None}
                        for v in VARS}
    out["scores"][m]["cap_mo_ho_dung"] = {v: f"{sum(P13[i]['pred'] == d13[i]['target'] for i, r in d13.items() if r.get(FLAG) and (r['lang'], r['variant']) == VARS[v][:2])}/"
                                             f"{sum(1 for r in d13.values() if r.get(FLAG) and (r['lang'], r['variant']) == VARS[v][:2])}" for v in VARS}
    out["scores"][m]["airline_vi"] = {"1.2": P12["belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1:vi"]["pred"] == d12["belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1:vi"]["target"],
                                      "1.3": P13["belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1:vi"]["pred"] == d13["belebele:https://en.wikivoyage.org/wiki/Airline_alliances#1:vi"]["target"]}
(T / "scores_v1_3_trial.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

f1 = lambda x: f"{x:.1f}".replace(".", ",")
d = lambda a, b: f"{b - a:+.1f}".replace(".", ",")
L = ["", "## 7. Điểm dev 1.2 so với 1.3 (chỉ dev; batch 1)", "",
     f"Preds dev 1.3: {len(same)} dòng nội dung không đổi dùng lại preds dev 1.2; {len(need)} dòng chấm lại (VI và VI không dấu của Airline_alliances#1) bằng cả 9 model; "
     "4 dòng của 2 cặp bị loại không còn trong điểm. Accuracy (%) trung bình đều theo nguồn (6 nguồn VI, 4 nguồn EN); cột cuối là trung bình trên mọi câu của 1.3.", ""]
for v, nm in (("vi", "VI có dấu"), ("vi_nodiac", "VI không dấu"), ("en", "EN")):
    L += [f"### {nm}", "", "| Model | 1.2 | 1.3 chính | 1.3 − 1.2 | 1.3 bỏ câu mơ hồ | 1.3 trên mọi câu |", "|---|---|---|---|---|---|"]
    for m, sc in out["scores"].items():
        a, b, c = sc[v]["1.2"]["macro"], sc[v]["1.3"]["macro"], sc[v]["1.3_bo_mo_ho"]["macro"]
        L.append(f"| {m} | {f1(a)} | {f1(b)} | {d(a, b)} | {f1(c)} | {f1(sc[v]['1.3']['tren_moi_cau'])} |")
    L.append("")
L += ["### Câu mơ hồ (26 cặp): số dòng model chọn đúng đáp án chuẩn", "", "| Model | VI có dấu | VI không dấu | EN |", "|---|---|---|---|"]
for m, sc in out["scores"].items():
    L.append(f"| {m} | " + " | ".join(sc["cap_mo_ho_dung"][v] for v in VARS) + " |")
L += ["", "### Theo nguồn, VI có dấu: 1.2 → 1.3 (chỉ ViWikiFC và Belebele có dòng đổi)", "", "| Model | " + " | ".join(SRC_VI) + " |", "|---" * 7 + "|"]
for m, sc in out["scores"].items():
    L.append(f"| {m} | " + " | ".join(f"{f1(sc['vi']['1.2']['theo_nguon'][s])} → {f1(sc['vi']['1.3']['theo_nguon'][s])}" for s in SRC_VI) + " |")
L += ["", "Lát MLQA: tạm dừng (mục 1). Chưa đóng băng."]
rp = T / "REPORT_bench_v1_3_trial.md"
txt = rp.read_text(encoding="utf-8")
cut = txt.find("\n## 7. Điểm dev 1.2 so với 1.3")
rp.write_text((txt if cut < 0 else txt[:cut]).rstrip("\n") + "\n" + "\n".join(L) + "\n", encoding="utf-8")
print(json.dumps({m: {v: [round(sc[v]["1.2"]["macro"], 1), round(sc[v]["1.3"]["macro"], 1), round(sc[v]["1.3_bo_mo_ho"]["macro"], 1)] for v in VARS}
                  for m, sc in out["scores"].items()}, ensure_ascii=False))
