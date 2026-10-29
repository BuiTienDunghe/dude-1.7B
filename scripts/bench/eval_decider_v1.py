"""Chấm Mapika/decider-2b trên dev_v1 (choice, noul, score). Dùng: eval_decider_v1.py <rows.jsonl> <out_dir> [--checks] [--adapter DIR]

- --adapter DIR: chấm model dẫn xuất (LoRA của scripts/ft_decider, gộp vào trọng số bf16); không có cờ thì chạy y như cũ.

- Nạp và kiểm gói `decider/` như scripts/baselines/eval_decider.py (sha256 từng file .py, eager, temperature phát hành).
- choice: decide() một câu một hàng, hai cách ánh xạ "text" và "keyed" (đã kiểm ở bước baseline). T choice = 1,164.
- noul : kiểu câu hỏi gốc của decider, {"type": "noul", "instructions": <câu hỏi>} qua đường system_one
         (_system_one_items + _system_one_probs, chính là phần thân của Decider.system_one, nhưng giữ xác suất không làm tròn).
         Lựa chọn của nó là no/yes; T noul = 1,624. Ghi probs theo thứ tự lựa chọn của dev_v1 [Có, Không] = [P(yes), P(no)].
- score: {"type": "score", "instructions": <câu hỏi>, "criteria": [3 mức]}; isolated levels như cấu hình phát hành
         (mỗi mức một hàng no/yes, chuẩn hóa P(yes)); T score = 1,124. probs theo đúng thứ tự mức.
- Cắt cụt: đặt giới hạn 32.768 token cho state; đếm câu có "Context:\n"+state > 1.536 (mức cắt mặc định của decide()).
- --checks: so xác suất noul/score của đường riêng với Decider.system_one() (làm tròn 4 chữ số) trên vài câu, rồi thôi.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, read_jsonl, write_jsonl  # noqa: E402

REPO, REV = "Mapika/decider-2b", "ff99089b6810691bd79134ec6febb5d9afa6e619"
CAP_DEFAULT, CAP = 1536, 32768
rows_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
folder = snapshot_download(REPO, revision=REV)
reviewed = RESULTS / "baselines" / "meta" / "Mapika__decider-2b"
for p in (Path(folder) / "decider").glob("*.py"):
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256((reviewed / f"decider__{p.name}").read_bytes()).hexdigest(), p.name
sys.path.insert(0, folder)
from decider import temperature as TT  # noqa: E402
from decider.infer import Decider  # noqa: E402
from decider.systemone import combine_isolated  # noqa: E402

torch.manual_seed(0)
d = Decider(folder, device="cuda", dtype=torch.bfloat16, use_graphs=False)
assert abs(d.T - 1.145) < 1e-9 and d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124}
assert d.neutralize_none is False and d.chat is None and d.isolated_levels is True
ADAPTER = sys.argv[sys.argv.index("--adapter") + 1] if "--adapter" in sys.argv else None
if ADAPTER:   # model dẫn xuất (scripts/ft_decider): gộp LoRA vào trọng số bf16; mọi phần khác của đường chấm giữ nguyên
    from peft import PeftModel
    d.m.lm = PeftModel.from_pretrained(d.m.lm, ADAPTER).merge_and_unload().eval()
tok = d.m.tok


def typed_probs(state, spec):
    """Xác suất không làm tròn cho một câu noul/score, theo đúng đường của Decider.system_one."""
    rqs, index, items = d._system_one_items(state, {"q": spec}, True, CAP, "state_first", None)
    flat = d._system_one_probs(items, "state_first", 65536, TT.for_items(d.T, d.T_by_type, items))
    # Mỗi hàng là một vectơ rộng MAX_OPTIONS (255), ô ngoài số lựa chọn có xác suất 0. Đọc giống hệt systemone.assemble /
    # format_answer của gói: noul cắt 2 ô đầu [no, yes] rồi chuẩn hóa; score isolated lấy P(yes) = ô 1 của từng hàng mức.
    assert all(sum(x[2:]) < 1e-6 for x in flat), "xác suất rơi ngoài 2 lựa chọn no/yes"
    if spec["type"] == "noul":
        assert len(flat) == 1
        p_no, p_yes = flat[0][:2]; s = p_no + p_yes
        return [p_yes / s, p_no / s]                           # thứ tự lựa chọn của dev_v1: [Có, Không]
    assert len(flat) == len(spec["criteria"])                  # isolated: một hàng no/yes cho mỗi mức
    probs, _ = combine_isolated([float(x[1]) for x in flat])
    return list(probs)


def spec_for(r):
    if r["kind"] == "noul":
        return {"type": "noul", "instructions": r["question"]}
    return {"type": "score", "instructions": r["question"], "criteria": list(r["options"])}


rows = read_jsonl(rows_path)
if "--checks" in sys.argv:
    res = []
    for kind in ("noul", "score"):
        for r in [x for x in rows if x["kind"] == kind][:4]:
            mine = typed_probs(r["state"], spec_for(r))
            ans = d.system_one(r["state"], {"q": spec_for(r)}, max_state_tokens=CAP)["answers"]["q"]
            ref = [ans["noul"], 1 - ans["noul"]] if kind == "noul" else [ans["probabilities"][str(i)] for i in range(len(r["options"]))]
            res.append({"id": r["id"][:60], "kind": kind, "max_abs_diff_vs_system_one_rounded": max(abs(a - b) for a, b in zip(mine, ref))})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checks_decider_v1.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)); sys.exit(0)

for mapping in ("text", "keyed"):
    preds, lat, n_over = [], [], 0
    for r in rows[:10]:   # warmup
        d.decide(r["state"], [{"question": r["question"], "options": list(r["options"])}], max_ctx_tokens=CAP)
    for r in rows:
        n_ctx = len(tok.encode("Context:\n" + r["state"], add_special_tokens=False))
        assert n_ctx <= CAP, r["id"]
        n_over += n_ctx > CAP_DEFAULT
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if r["kind"] not in ("noul", "score"):   # dòng dev_v0 có kind reading_mc/topic/intent/causal: đều là choice
            opts = list(r["options"]) if mapping == "text" else [f"opt{i + 1}: {o}" for i, o in enumerate(r["options"])]
            p = d.decide(r["state"], [{"question": r["question"], "options": opts}], max_ctx_tokens=CAP)[0]["probs_list"]
        else:
            p = typed_probs(r["state"], spec_for(r))
        torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
        assert len(p) == len(r["options"]) and abs(sum(p) - 1) < 1e-4, (r["id"], p)
        pred = max(range(len(p)), key=p.__getitem__)
        preds.append({"id": r["id"], "pair_id": r["pair_id"], "source": r["source"], "lang": r["lang"], "kind": r["kind"],
                      "variant": r["variant"], "k": len(p), "target": r["target"], "probs": p, "pred": pred, "correct": pred == r["target"],
                      "letter_mass": None, "n_ctx_tokens": n_ctx, "over_default_cap_1536": n_ctx > CAP_DEFAULT,
                      "mapping": mapping if r["kind"] not in ("noul", "score") else "native_" + r["kind"]})
    name = f"decider-2b-{mapping}"
    write_jsonl(out_dir / name / "preds.jsonl", preds)
    (out_dir / name / "timing.json").write_text(json.dumps({
        "repo": REPO, "revision": REV, "T_by_type": d.T_by_type, "mapping_choice": mapping, "rows": len(rows),
        "batch1_median_latency_ms": 1000 * statistics.median(lat), "n_over_default_cap_1536": int(n_over), "cap": CAP,
        "latency_includes": "tokenize + collate + forward", "flash_linear_attention": False,
        **({"adapter": ADAPTER} if ADAPTER else {})}, indent=1), encoding="utf-8")
    print(mapping, "acc", sum(x["correct"] for x in preds) / len(preds), "over1536", n_over)
