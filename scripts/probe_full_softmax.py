"""Xem phân phối trên toàn bộ từ vựng tại vị trí trả lời: top-8 token, entropy, khối trên A..J, cho model gốc và model đã fine-tune.

Dùng: probe_full_softmax.py <model_id> [adapter_dir] [out.json]   (5 câu cố định: câu tiếng Việt đầu tiên của mỗi nguồn
      + câu MASSIVE tiếng Anh đầu tiên; in tối đa 5 câu)
Ngoài khối trên k chữ cái, đo thêm xác suất rơi vào biến thể của chữ cái: có dấu cách phía trước (" A") và full-width ("Ａ"),
chỉ tính những biến thể là đúng một token.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, LETTERS, load_json, model_revisions, read_jsonl, render_prompt

mid = sys.argv[1]
adapter = sys.argv[2] if len(sys.argv) > 2 else None
rev = model_revisions()[mid]
tok = AutoTokenizer.from_pretrained(mid, revision=rev)
model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval().to(torch.bfloat16)
frames = load_json(CONFIG / "prompt_frames.json")
rows = read_jsonl(DATA / "dev_v0.jsonl")
pick = [next(r for r in rows if r["source"] == s and r["lang"] == "vi") for s in ["belebele", "sib200", "massive", "xcopa"]]
pick.append(next(r for r in rows if r["source"] == "massive" and r["lang"] == "en"))
letter_ids = [tok.convert_tokens_to_ids(L) for L in LETTERS]


def single(s):
    t = tok.encode(s, add_special_tokens=False)
    return t[0] if len(t) == 1 else None


space_ids = [single(" " + L) for L in LETTERS]
fw_ids = [single(chr(ord("Ａ") + i)) for i in range(len(LETTERS))]
out = []
with torch.inference_mode():
    for r in pick:
        ids = tok.encode(render_prompt(r, tok, frames), add_special_tokens=False)
        lg = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg, -1)
        ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
        top = torch.topk(p, 10)
        k = len(r["options"])
        rec = {"id": r["id"][:60], "k": k, "mass_k": float(p[letter_ids[:k]].sum()), "mass_AJ": float(p[letter_ids].sum()),
               "mass_space_variants_k": float(sum(p[i] for i in space_ids[:k] if i is not None)),
               "mass_fullwidth_variants_k": float(sum(p[i] for i in fw_ids[:k] if i is not None)),
               "n_single_token_variants": [sum(i is not None for i in space_ids[:k]), sum(i is not None for i in fw_ids[:k])],
               "entropy": ent, "logit_max": float(lg.max()), "logit_std": float(lg.std()),
               "top10": [(tok.convert_ids_to_tokens(int(i)), round(float(v), 4)) for v, i in zip(top.values, top.indices)]}
        out.append(rec)
        print(f"{rec['id']} k={k} mass_k={rec['mass_k']:.3f} mass_AJ={rec['mass_AJ']:.3f} space={rec['mass_space_variants_k']:.4f} "
              f"fullwidth={rec['mass_fullwidth_variants_k']:.4f} entropy={ent:.2f} logit_max={rec['logit_max']:.1f} logit_std={rec['logit_std']:.2f}")
        print("   top:", rec["top10"][:8])
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
