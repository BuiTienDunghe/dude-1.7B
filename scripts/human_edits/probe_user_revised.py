"""Thử lại 6 cặp dev người dùng tự dịch (Buffalo-Pittsburgh_Highway #1/#2, Harper Clean Air Act #1/#2, Canyoning #2, sib200:810) — CHỈ DEV,
chưa phải bước D.
  extract <thư mục dựng>: lấy 18 dòng (EN, VI có dấu, VI không dấu) từ bản dựng tạm theo lớp ghi đè hiện tại -> user_revised_probe/probe_dev.jsonl;
                          kiểm NFC và bản không dấu = strip_diacritics(bản có dấu). Chạy bằng venv WSL.
  table: bảng model × cặp × biến thể, dự đoán + đúng/sai + xác suất đáp án đúng ở 1.1, bản thử 1.2 trước đó, bản mới -> REPORT_probe.md, probe_table.json.
Không ghi data/, không chạy test, không train."""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "results/bench_v1_2_trial/user_revised_probe"
B = "belebele:https://en.wikivoyage.org/wiki/Buffalo-Pittsburgh_Highway"
H = "belebele:https://en.wikinews.org/wiki/Canadian_Prime_Minister_Harper_agrees_to_send_%27Clean_Air_Act%27_to_committee"
C = "belebele:https://en.wikivoyage.org/wiki/Canyoning#2"
PAIRS = [B + "#1", B + "#2", H + "#1", H + "#2", C, "sib200:810"]
SHORT = {B + "#1": "Buffalo#1", B + "#2": "Buffalo#2", H + "#1": "Harper#1", H + "#2": "Harper#2", C: "Canyoning#2", "sib200:810": "sib200:810"}
MODELS = {"decider-2b-text": "decider/decider-2b-text", "decider-2b-keyed": "decider/decider-2b-keyed", "decider-ft-ep2-text": "ftd/decider-2b-text",
          "decider-ft-ep2-keyed": "ftd/decider-2b-keyed", "Qwen3-1.7B-fullvocab-s1": "control", "Qwen3-1.7B": "base", "strip-s1": "strip"}
VARS = [("en", ":en"), ("vi", ":vi"), ("vi không dấu", ":vi:nodiac")]
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
LET = "ABCDEFGHIJ"

if sys.argv[1] == "extract":
    sys.path.insert(0, str(ROOT / "scripts"))
    from common import strip_diacritics  # noqa: E402
    new = {r["id"]: r for r in read(Path(sys.argv[2]) / "dev_v1_2_trial.jsonl") if r["pair_id"] in PAIRS}
    trial = {r["id"]: r for r in read(ROOT / "results/bench_v1_2_trial/dev_v1_2_trial.jsonl") if r["pair_id"] in PAIRS}
    assert len(new) == 3 * len(PAIRS) and set(new) == set(trial), (len(new), sorted(set(new) ^ set(trial)))
    chk = {}
    for i, r in sorted(new.items()):
        t = trial[i]
        chk[i] = {"truong_khac_ban_thu": [k for k in ("state", "question", "options", "target") if r[k] != t[k]], "vi_revised_fields": r.get("vi_revised_fields"),
                  "gold_provenance": r["gold_provenance"]}
        if r["lang"] == "vi":
            chk[i]["nfc"] = all(unicodedata.is_normalized("NFC", s) for s in [r["state"], r["question"], *r["options"]])
            if r["variant"] == "orig":
                nd = new[i + ":nodiac"]
                chk[i]["khong_dau_sinh_lai_dung"] = (nd["state"] == strip_diacritics(r["state"]) and nd["question"] == strip_diacritics(r["question"])
                                                     and nd["options"] == [strip_diacritics(o) for o in r["options"]] and nd["target"] == r["target"])
    P.mkdir(parents=True, exist_ok=True)
    with open(P / "probe_dev.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(new):
            f.write(json.dumps(new[i], ensure_ascii=False) + "\n")
    (P / "probe_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({SHORT[new[i]["pair_id"]] + i[len(new[i]["pair_id"]):]: v for i, v in chk.items()}, ensure_ascii=False))
    sys.exit(0)

# ---------------------------------------------------------------- table
gold = {"1.1": {r["id"]: r["target"] for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl") if r["pair_id"] in PAIRS},
        "thu": {r["id"]: r["target"] for r in read(ROOT / "results/bench_v1_2_trial/dev_v1_2_trial.jsonl") if r["pair_id"] in PAIRS},
        "moi": {r["id"]: r["target"] for r in read(P / "probe_dev.jsonl")}}
src = {"1.1": lambda m: ROOT / f"results/dev_v1_1/eval/{m}/preds.jsonl", "thu": lambda m: ROOT / f"results/bench_v1_2_trial/eval/{m}/preds.jsonl",
       "moi": lambda m: P / f"eval/{MODELS[m]}/preds.jsonl"}
tab = []
for m in MODELS:
    pr = {v: {x["id"]: x for x in read(src[v](m)) if x["pair_id"] in PAIRS} for v in src}
    for pid in PAIRS:
        for vn, suf in VARS:
            i = pid + suf
            cell = {"model": m, "cap": SHORT[pid], "bien_the": vn}
            for v in src:
                x = pr[v].get(i)
                g = gold[v].get(i)
                cell[v] = None if x is None else {"du_doan": x["pred"], "dap_an": g, "dung": x["pred"] == g,
                                                  "p_dap_an": round(x["probs"][g], 3) if x.get("probs") else None}
            tab.append(cell)
(P / "probe_table.json").write_text(json.dumps(tab, ensure_ascii=False, indent=1), encoding="utf-8")


def fmt(c):
    if c is None:
        return "—"
    return f"{'✓' if c['dung'] else '✗'}{LET[c['du_doan']]} {c['p_dap_an']:.2f}" if c["p_dap_an"] is not None else f"{'✓' if c['dung'] else '✗'}{LET[c['du_doan']]}"


L = [f"# Thử lại {len(PAIRS)} cặp dev người dùng tự dịch", "",
     "Ô = 1.1 → bản thử 1.2 trước đó → bản mới (lớp ghi đè hiện tại). Mỗi mốc: ✓/✗, chữ cái dự đoán, xác suất model gán cho đáp án đúng.", ""]
for pid in PAIRS:
    gs = {v: [gold[v].get(pid + s) for _, s in VARS] for v in gold}
    L += [f"## {SHORT[pid]} (đáp án: " + ", ".join(f"{v} {LET[gs[v][1]]}" for v in gold) + ")", "",
          "| Model | EN | VI có dấu | VI không dấu |", "|---|---|---|---|"]
    for m in MODELS:
        cs = [next(c for c in tab if c["model"] == m and c["cap"] == SHORT[pid] and c["bien_the"] == vn) for vn, _ in VARS]
        L.append(f"| {m} | " + " | ".join(" → ".join(fmt(c[v]) for v in ("1.1", "thu", "moi")) for c in cs) + " |")
    L.append("")
summ = {}
for vn, _ in VARS:
    for v in ("1.1", "thu", "moi"):
        cs = [c[v] for c in tab if c["bien_the"] == vn and c[v] is not None]
        summ[f"{vn}|{v}"] = f"{sum(c['dung'] for c in cs)}/{len(cs)}"
L += [f"## Tổng số ô đúng (7 model × {len(PAIRS)} cặp)", "", "| Biến thể | 1.1 | thử 1.2 | mới |", "|---|---|---|---|"]
L += [f"| {vn} | {summ[vn + '|1.1']} | {summ[vn + '|thu']} | {summ[vn + '|moi']} |" for vn, _ in VARS]
(P / "REPORT_probe.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
