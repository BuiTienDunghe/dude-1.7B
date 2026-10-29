"""Kiểm tra nạp Qwen/Qwen3.5-2B (revision ghim) bằng AutoModelForCausalLM như eval_base.py, và kiểm tra chữ cái A..J
là một token ở ranh giới trả lời với chat template của nó. In thông tin, không chấm.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ — dùng common.py của dự án
from common import CONFIG, DATA, LETTERS, load_json, read_jsonl, render_prompt

MID, REV = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
tok = AutoTokenizer.from_pretrained(MID, revision=REV)
frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(DATA / "dev_v0.jsonl")
r = next(x for x in rows if x["lang"] == "vi" and x["source"] == "massive")
p = render_prompt(r, tok, frames)
print("prompt tail:", repr(p[-120:]))
ids = tok.encode(p, add_special_tokens=False)
bad = []
for L in LETTERS:
    li = tok.encode(L, add_special_tokens=False)
    full = tok.encode(p + L, add_special_tokens=False)
    if len(li) != 1 or full[:-1] != ids or full[-1] != li[0]:
        bad.append((L, li, full[-3:]))
print("letters single-token at boundary:", "OK" if not bad else bad)
model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
print("class:", type(model).__name__, "| config:", type(model.config).__name__,
      "| max_pos:", getattr(model.config, "max_position_embeddings", "MISSING"), "| dtype:", next(model.parameters()).dtype)
with torch.inference_mode():
    lg = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1].float()
k = len(r["options"])
lid = [tok.convert_tokens_to_ids(L) for L in LETTERS[:k]]
pk = torch.softmax(lg[lid], -1)
print("k", k, "target", r["target"], "pred", int(pk.argmax()), "letter_mass", float(torch.softmax(lg, -1)[lid].sum()),
      "top token", tok.convert_ids_to_tokens(int(lg.argmax())))
