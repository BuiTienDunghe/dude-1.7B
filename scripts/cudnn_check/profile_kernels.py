"""Bước 0 (bổ sung): tên kernel CUDA thực sự được gọi trong attention, qua torch.profiler key_averages(), ở chế độ mặc định:
  (i) đường chấm: batch 1, không mask, inference;  (ii) đường train: micro-batch 8 câu right-padding, attention_mask,
  bf16 autocast, train + gradient checkpointing, forward + backward. In mọi kernel có tên liên quan tới attention.
Ra: results/cudnn_check/kernels.json
"""
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, render_prompt  # noqa: E402

MID = "Qwen/Qwen3-1.7B"; rev = model_revisions()[MID]
tok = AutoTokenizer.from_pretrained(MID, revision=rev)
model = AutoModelForCausalLM.from_pretrained(MID, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
frames = load_json(CONFIG / "prompt_frames.json")
KEYS = ("cudnn", "fmha", "flash", "attention", "attn", "efficient", "sdp", "cutlass", "softmax")


def kernels(fn):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        if any(k in e.key.lower() for k in KEYS):
            out[e.key[:160]] = e.count
    return dict(sorted(out.items(), key=lambda x: -x[1]))


rows = read_jsonl(DATA / "dev_v0.jsonl")
ids = torch.tensor([tok.encode(render_prompt(rows[0], tok, frames), add_special_tokens=False)], device="cuda")
model.eval()
res = {}
with torch.inference_mode():
    res["eval_batch1_nomask"] = kernels(lambda: model(input_ids=ids, logits_to_keep=1))

train = read_jsonl(DATA / "train_v0.jsonl")
srt = sorted(train, key=lambda r: (r["n_state_tokens"], r["id"]))
pick = [srt[int(i * (len(srt) - 1) / 7)] for i in range(8)]
encs = [tok.encode(render_prompt(r, tok, frames), add_special_tokens=False) for r in pick]
L = max(len(e) for e in encs)
x = torch.full((8, L), tok.pad_token_id, dtype=torch.long); msk = torch.zeros((8, L), dtype=torch.long)
for i, e in enumerate(encs):
    x[i, :len(e)] = torch.tensor(e); msk[i, :len(e)] = 1
x, msk = x.cuda(), msk.cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()


def train_step():
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=x, attention_mask=msk).last_hidden_state
    h.float().mean().backward()


res["train_padded_fwd_bwd"] = kernels(train_step)
res["train_batch_lengths"] = [len(e) for e in encs]
(RESULTS / "cudnn_check" / "kernels.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
