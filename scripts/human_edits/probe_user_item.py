"""Chấm thử MỘT câu dev với bản dịch người dùng gửi (file JSON {pair_id, state, question, options, target}; CHƯA vào lớp ghi đè). Câu cùng
đoạn văn không chấm, không đổi. CHỈ DEV; không test, không train, không ghi data/.
Dùng: probe_user_item.py build|table <đường dẫn JSON>, ví dụ results/bench_v1_2_trial/user_revised_probe/grandcanyon1_user.json.
Tên = tên file bỏ đuôi "_user"; mọi file ra ở user_revised_probe/<tên>/ (<tên>_dev.jsonl, REPORT_<tên>.md, <tên>_table.json, eval/).
  build (venv WSL): EN = bản 1.1 nguyên (nếu nguồn có EN); VI = dòng VI bản thử 1.2, các trường có trong JSON thay bằng bản người dùng
        (thiếu trường nào thì giữ bản thử); không dấu = common.strip_diacritics; n_tokens / n_state_tokens tính lại như build_revised.
  table: model × biến thể, dự đoán, đúng/sai, P(đáp án đúng) ở 1.1 / bản thử 1.2 / bản người dùng.
Thay cho probe_goma1.py (giữ lại dạng lệnh cũ; báo cáo Goma chạy lại ra y hệt)."""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
P = ROOT / "results/bench_v1_2_trial/user_revised_probe"
J = Path(sys.argv[2])
J = J if J.is_absolute() else ROOT / J
NAME = J.stem[:-len("_user")] if J.stem.endswith("_user") else J.stem
assert re.fullmatch(r"[A-Za-z0-9_-]+", NAME), NAME
G = P / NAME
U = json.load(open(J, encoding="utf-8"))
PID = U["pair_id"]
SHORT = PID.split("/wiki/")[-1] if "/wiki/" in PID else PID
MODELS = {"decider-2b-text": "decider/decider-2b-text", "decider-2b-keyed": "decider/decider-2b-keyed", "decider-ft-ep2-text": "ftd/decider-2b-text",
          "decider-ft-ep2-keyed": "ftd/decider-2b-keyed", "Qwen3-1.7B-fullvocab-s1": "control", "Qwen3-1.7B": "base", "strip-s1": "strip"}
LET = "ABCDEFGHIJ"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
V11 = {r["id"]: r for r in read(ROOT / "data/bench_v1_1/dev_v1_1.jsonl") if r["pair_id"] == PID}
VTR = {r["id"]: r for r in read(ROOT / "results/bench_v1_2_trial/dev_v1_2_trial.jsonl") if r["pair_id"] == PID}
assert V11 and set(V11) == set(VTR), f"{PID}: không có trong dev 1.1 hoặc bản thử (chỉ câu dev, không bị loại)"
VARS = [(vn, s) for vn, s in (("EN", ":en"), ("VI có dấu", ":vi"), ("VI không dấu", ":vi:nodiac")) if PID + s in V11]

if sys.argv[1] == "build":
    from common import CONFIG, load_json, model_revisions, render_prompt, strip_diacritics  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
    frames = load_json(CONFIG / "prompt_frames.json")
    ntok = lambda r: len(tok.encode(render_prompt(r, tok, frames), add_special_tokens=False))
    nst = lambda r: len(tok.encode(r["state"], add_special_tokens=False))
    vi = dict(VTR[PID + ":vi"])
    new = {k: U[k] for k in ("state", "question", "options") if k in U}
    assert new, "JSON không có trường nào để thay"
    if "options" in new:
        assert len(new["options"]) == len(vi["options"])
        new["options"] = list(new["options"])
    assert U.get("target", vi["target"]) == vi["target"] == V11[PID + ":vi"]["target"], "đáp án khác bản thử / 1.1"
    for s in [new.get("state", ""), new.get("question", ""), *new.get("options", [])]:
        assert unicodedata.is_normalized("NFC", s)
    assert all(str(x).strip() for x in [new.get("state", "x"), new.get("question", "x"), *new.get("options", [])])
    vi.update(new)
    old = V11[PID + ":vi"]
    rev = [f for f, a, b in [("state", vi["state"], old["state"]), ("question", vi["question"], old["question"])] if a != unicodedata.normalize("NFC", b)]
    rev += [f"option:{j}" for j, (a, b) in enumerate(zip(vi["options"], old["options"])) if a != unicodedata.normalize("NFC", b)]
    vi.update(vi_provenance="revised" if rev else "source", vi_revised_fields=rev, n_tokens=ntok(vi), n_state_tokens=nst(vi))
    nd = {**vi, "id": vi["id"] + ":nodiac", "variant": "nodiac", "state": strip_diacritics(vi["state"]), "question": strip_diacritics(vi["question"]),
          "options": [strip_diacritics(o) for o in vi["options"]]}
    for a, b in ((vi["state"], nd["state"]), (vi["question"], nd["question"])) + tuple(zip(vi["options"], nd["options"])):
        assert re.findall(r"\d+", a) == re.findall(r"\d+", b)
    nd.update(n_tokens=ntok(nd), n_state_tokens=nst(nd))
    out = ([V11[PID + ":en"]] if PID + ":en" in V11 else []) + [vi, nd]
    G.mkdir(parents=True, exist_ok=True)
    with open(G / f"{NAME}_dev.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"ten": NAME, "rows": len(out), "vi_revised_fields": rev, "target": vi["target"], "n_tokens": [r["n_tokens"] for r in out],
                      "en_bang_1_1": (out[0] == V11[PID + ":en"]) if PID + ":en" in V11 else None}, ensure_ascii=False))
    sys.exit(0)

# ---------------------------------------------------------------- table
gold = {"1.1": {i: r["target"] for i, r in V11.items()}, "thu": {i: r["target"] for i, r in VTR.items()},
        "nguoi_dung": {r["id"]: r["target"] for r in read(G / f"{NAME}_dev.jsonl")}}
src = {"1.1": lambda m: ROOT / f"results/dev_v1_1/eval/{m}/preds.jsonl", "thu": lambda m: ROOT / f"results/bench_v1_2_trial/eval/{m}/preds.jsonl",
       "nguoi_dung": lambda m: G / f"eval/{MODELS[m]}/preds.jsonl"}
tab = []
for m in MODELS:
    pr = {v: {x["id"]: x for x in read(src[v](m)) if x["pair_id"] == PID} for v in src}
    for vn, suf in VARS:
        i = PID + suf
        cell = {"model": m, "bien_the": vn}
        for v in src:
            x, g = pr[v].get(i), gold[v][i]
            cell[v] = None if x is None else {"du_doan": x["pred"], "dap_an": g, "dung": x["pred"] == g, "p_dap_an": round(x["probs"][g], 3)}
        tab.append(cell)
(G / f"{NAME}_table.json").write_text(json.dumps(tab, ensure_ascii=False, indent=1), encoding="utf-8")
fmt = lambda c: "—" if c is None else f"{'✓' if c['dung'] else '✗'}{LET[c['du_doan']]} {c['p_dap_an']:.2f}"
L = [f"# {SHORT}: chấm thử bản dịch của người dùng (DEV, một câu; đáp án {LET[gold['nguoi_dung'][PID + ':vi']]})", "",
     "Ô = 1.1 → bản thử 1.2 → bản người dùng. Mỗi mốc: ✓/✗, chữ cái dự đoán, P(đáp án đúng). "
     + ("EN giữ nguyên bản 1.1." if VARS[0][0] == "EN" else "Nguồn tiếng Việt gốc, không có EN."), "",
     "| Model | " + " | ".join(vn for vn, _ in VARS) + " |", "|---|" + "---|" * len(VARS)]
for m in MODELS:
    cs = [next(c for c in tab if c["model"] == m and c["bien_the"] == vn) for vn, _ in VARS]
    L.append(f"| {m} | " + " | ".join(" → ".join(fmt(c[v]) for v in src) for c in cs) + " |")
L += ["", "| Biến thể | 1.1 | thử 1.2 | người dùng |", "|---|---|---|---|"]
for vn, _ in VARS:
    L.append(f"| {vn} | " + " | ".join(f"{sum(c[v]['dung'] for c in tab if c['bien_the'] == vn and c[v])}/{len(MODELS)}" for v in src) + " |")
(G / f"REPORT_{NAME}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
