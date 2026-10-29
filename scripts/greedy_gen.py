"""Sinh greedy đúng 1 token cho 100 câu dev_v0 cố định (25 câu mỗi nguồn, chọn theo seed), qua model.generate.

Dùng: greedy_gen.py <model_id> <adapter_dir> <preds.jsonl của đường đọc logit (thứ tự thường)> <out.json>
Đếm:
  hợp lệ (chặt)  — token sinh ra là đúng một trong k token chữ cái mà đường chấm đọc logit cho câu đó;
  biến thể       — " A" (có dấu cách) hoặc "Ａ" (full-width) của một trong k chữ cái; báo riêng, không cộng vào hợp lệ;
  trùng đáp án   — chữ cái sinh ra trùng với lựa chọn có xác suất cao nhất của đường đọc logit.
"""
import json
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, LETTERS, SEED, load_json, model_revisions, read_jsonl, render_prompt

mid, adapter, preds_path, out_path = sys.argv[1:5]
rev = model_revisions()[mid]
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
tok = AutoTokenizer.from_pretrained(mid, revision=rev)
model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval().to(torch.bfloat16)
frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(DATA / "dev_v0.jsonl")
preds = {p["id"]: p for p in read_jsonl(preds_path)}

rng = np.random.default_rng(SEED)
pick = []
for src in ["belebele", "sib200", "massive", "xcopa"]:
    ids = sorted(r["id"] for r in rows if r["source"] == src)
    pick += [ids[i] for i in sorted(rng.choice(len(ids), 25, replace=False))]
byid = {r["id"]: r for r in rows}

letter_ids = [tok.convert_tokens_to_ids(L) for L in LETTERS]
def single(s):
    t = tok.encode(s, add_special_tokens=False)
    return t[0] if len(t) == 1 else None
variant = {}
for i, L in enumerate(LETTERS):
    for s in (" " + L, chr(ord("Ａ") + i)):
        t = single(s)
        if t is not None:
            variant[t] = i

res = []
with torch.inference_mode():
    for rid in pick:
        r = byid[rid]
        k = len(r["options"])
        ids = tok.encode(render_prompt(r, tok, frames), add_special_tokens=False)
        x = torch.tensor([ids], device="cuda")
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=1, do_sample=False, temperature=None,
                           top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        t = int(g[0, -1])
        valid = t in letter_ids[:k]
        var = (not valid) and t in variant and variant[t] < k
        letter = letter_ids.index(t) if valid else (variant[t] if var else None)
        res.append({"id": rid, "k": k, "token": tok.convert_ids_to_tokens(t), "valid_strict": valid, "variant": var,
                    "agrees_with_logit_pred": valid and letter == preds[rid]["pred"]})
summary = {"n": len(res), "valid_strict": sum(x["valid_strict"] for x in res), "variant": sum(x["variant"] for x in res),
           "agrees_with_logit_pred": sum(x["agrees_with_logit_pred"] for x in res),
           "invalid_tokens": [x["token"] for x in res if not x["valid_strict"]][:20],
           "adapter": adapter, "preds": preds_path}
json.dump({"summary": summary, "rows": res}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
