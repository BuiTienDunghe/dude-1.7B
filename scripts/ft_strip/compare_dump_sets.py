"""G2 bổ sung: so TẬP id dòng của từng batch hiệu dụng (không tính thứ tự trong batch) giữa strip_p 0,3 và tham chiếu,
và đo bao nhiêu bước có cách chia micro-batch (4 × 8, theo thứ tự đã sắp độ dài) khác đi. Chạy lại đường dữ liệu (không model)."""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, apply_strip, load_json, model_revisions, read_jsonl, render_prompt, strip_decision  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=model_revisions()["Qwen/Qwen3-1.7B"])
frames = load_json(CONFIG / "prompt_frames.json")
train = read_jsonl(DATA / "train_v0.jsonl")
SEED, EB, N = 1, 32, 200
spe = len(train) // EB
enc = lambda r: tok.encode(render_prompt(r, tok, frames), add_special_tokens=False)


def shuffled(row, rng):
    perm = rng.permutation(len(row["options"]))
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}


res = {"steps": 0, "same_id_set": 0, "same_order": 0, "same_microbatch_partition": 0}
step = 0
for epoch in range(2):
    rng = np.random.default_rng(SEED * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(spe):
        idx = order[b * EB:(b + 1) * EB]
        base = [shuffled(train[i], rng) for i in idx]
        strip = [apply_strip(r, strip_decision(SEED, epoch, r["id"], 0.3, 0.5)) if r["text_lang"] == "vi" else r for r in base]
        sa = [r["id"] for _, r in sorted(((len(enc(r)), k), r) for k, r in enumerate(base))]
        sb = [r["id"] for _, r in sorted(((len(enc(r)), k), r) for k, r in enumerate(strip))]
        res["steps"] += 1
        res["same_id_set"] += sorted(sa) == sorted(sb)
        res["same_order"] += sa == sb
        res["same_microbatch_partition"] += [sorted(sa[i:i + 8]) for i in range(0, EB, 8)] == [sorted(sb[i:i + 8]) for i in range(0, EB, 8)]
        step += 1
        if step >= N:
            break
    if step >= N:
        break
(RESULTS / "ft_strip" / "gates" / "g2_sets.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res))
