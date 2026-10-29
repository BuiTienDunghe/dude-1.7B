"""Bước 0: trạng thái backend SDPA mặc định trong .venv chính, attn_implementation transformers chọn cho Qwen3-1.7B,
và kernel attention thực sự được gọi (torch profiler) ở hai tình huống:
  (i)  giống đường chấm: batch 1, không mask;
  (ii) giống đường train: micro-batch 8 câu right-padding có attention_mask, bf16 autocast, chế độ train + gradient checkpointing.
Ra: results/cudnn_check/backend_state.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, RESULTS, load_json, model_revisions, read_jsonl, render_prompt  # noqa: E402

OUT = RESULTS / "cudnn_check"; OUT.mkdir(parents=True, exist_ok=True)
MID = "Qwen/Qwen3-1.7B"; rev = model_revisions()[MID]
b = torch.backends.cuda
state = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
         "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
         "flags_default": {"cudnn_sdp_enabled": b.cudnn_sdp_enabled(), "flash_sdp_enabled": b.flash_sdp_enabled(),
                           "mem_efficient_sdp_enabled": b.mem_efficient_sdp_enabled(), "math_sdp_enabled": b.math_sdp_enabled()}}
tok = AutoTok = transformers.AutoTokenizer.from_pretrained(MID, revision=rev)
model = transformers.AutoModelForCausalLM.from_pretrained(MID, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
state["attn_implementation_requested_sdpa"] = model.config._attn_implementation
m2 = transformers.AutoConfig.from_pretrained(MID, revision=rev)
state["attn_implementation_if_not_specified"] = getattr(transformers.AutoModelForCausalLM.from_config(m2, dtype=torch.bfloat16).config, "_attn_implementation", None)
frames = load_json(CONFIG / "prompt_frames.json")


def kernels(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    names = Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    attn = {n: c for n, c in names.items() if any(s in n.lower() for s in ("cudnn", "fmha", "flash", "attention", "efficient", "sdpa", "softmax", "bmm"))}
    return dict(sorted(attn.items(), key=lambda x: -x[1])[:25])


rows = read_jsonl(DATA / "dev_v0.jsonl")
ids = tok.encode(render_prompt(rows[0], tok, frames), add_special_tokens=False)
model.eval()
with torch.inference_mode():
    model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)   # warmup
    state["kernels_eval_batch1_nomask"] = kernels(lambda: model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1))

train = read_jsonl(DATA / "train_v0.jsonl")
srt = sorted(train, key=lambda r: r["n_state_tokens"])
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


def train_fwd():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=x, attention_mask=msk).last_hidden_state
    h.float().sum().backward()


train_fwd()
state["train_batch_lengths"] = [len(e) for e in encs]
state["kernels_train_padded_fwd_bwd"] = kernels(train_fwd)
(OUT / "backend_state.json").write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
print(json.dumps(state, indent=1, default=str))
