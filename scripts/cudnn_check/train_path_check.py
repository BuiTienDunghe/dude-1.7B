"""Bước 2: đường train. Micro-batch 8 câu train_v0 có độ dài khác nhau rõ (right padding thật sự xảy ra), dựng đúng như
train_lora.py (render_prompt, right padding, hidden tại len-1 rồi lm_head, bf16 autocast, gradient checkpointing, chế độ train).
Model: Qwen3-1.7B gốc + adapter full_vocab (LoRA chưa merge, như lúc train).

Ba chế độ SDPA: (a) mặc định, (b) tắt cuDNN SDPA, (c) chỉ math. Dropout của LoRA được giữ y hệt giữa các chế độ bằng cách
đặt cùng seed trước mỗi forward. Báo:
- giữa từng cặp chế độ: max |Δlogit| ở vị trí trả lời (k chữ cái; toàn từ vựng), |Δloss| (letters_k, full_vocab; trung bình
  batch và từng câu), |Δ chuẩn gradient LoRA|;
- lỗi mask trực tiếp: logit vị trí trả lời của từng câu trong batch có padding so với chạy riêng câu đó ở batch 1, cùng chế độ.
Ra: results/cudnn_check/train_path.json
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, LETTERS, RESULTS, load_json, model_revisions, read_jsonl, render_prompt  # noqa: E402

OUT = RESULTS / "cudnn_check"; OUT.mkdir(parents=True, exist_ok=True)
MID = "Qwen/Qwen3-1.7B"; rev = model_revisions()[MID]
ADAPTER = RESULTS / "ft_loss/train/Qwen3-1.7B-s1/adapter"
SEED = 1
torch.backends.cuda.matmul.allow_tf32 = False
frames = load_json(CONFIG / "prompt_frames.json")
tok = AutoTokenizer.from_pretrained(MID, revision=rev)
letter_ids = torch.tensor([tok.convert_tokens_to_ids(L) for L in LETTERS], device="cuda")
base = AutoModelForCausalLM.from_pretrained(MID, revision=rev, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.enable_input_require_grads()
model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=True)
model.train()
inner = model.base_model.model
backbone, lm_head = inner.model, inner.lm_head

train = read_jsonl(DATA / "train_v0.jsonl")
srt = sorted(train, key=lambda r: (r["n_state_tokens"], r["id"]))
rows = [srt[int(i * (len(srt) - 1) / 7)] for i in range(8)]       # 8 câu trải từ ngắn nhất tới dài nhất
encs = [tok.encode(render_prompt(r, tok, frames), add_special_tokens=False) for r in rows]


def answer_logits(encs):
    """Chép đúng answer_logits của train_lora.py: right padding, hidden tại len-1, lm_head, bf16 autocast."""
    L = max(len(e) for e in encs)
    ids = torch.full((len(encs), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        ids[i, :len(e)] = torch.tensor(e); mask[i, :len(e)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        last = torch.tensor([len(e) - 1 for e in encs], device="cuda")
        logits = lm_head(h[torch.arange(len(encs), device="cuda"), last]).float()
    return logits


def losses(rs, logits):
    lk, lf = [], []
    for i, r in enumerate(rs):
        k = len(r["options"]); t = torch.tensor(r["target"], device="cuda", dtype=torch.float32)
        lk.append(-(t * torch.log_softmax(logits[i, letter_ids[:k]], -1)).sum())
        lf.append(-(t * torch.log_softmax(logits[i], -1)[letter_ids[:k]]).sum())
    return torch.stack(lk), torch.stack(lf)


@contextmanager
def mode(name):
    b = torch.backends.cuda
    old = (b.cudnn_sdp_enabled(), b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled())
    if name == "no_cudnn":
        b.enable_cudnn_sdp(False)
    elif name == "math_only":
        b.enable_cudnn_sdp(False); b.enable_flash_sdp(False); b.enable_mem_efficient_sdp(False); b.enable_math_sdp(True)
    try:
        yield
    finally:
        b.enable_cudnn_sdp(old[0]); b.enable_flash_sdp(old[1]); b.enable_mem_efficient_sdp(old[2]); b.enable_math_sdp(old[3])


res = {}
params = [p for p in model.parameters() if p.requires_grad]
for name in ("default", "no_cudnn", "math_only"):
    with mode(name):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(SEED)
        lg = answer_logits(encs)
        lk, lf = losses(rows, lg)
        lf.mean().backward()                   # backward của loss full_vocab, như lúc train
        gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)).item()
        single = []
        for i, e in enumerate(encs):           # từng câu chạy riêng, batch 1, không padding
            torch.manual_seed(SEED)
            with torch.no_grad():
                single.append(answer_logits([e])[0])
        res[name] = {"logits": lg.detach(), "lk": lk.detach(), "lf": lf.detach(), "gnorm": gnorm, "single": torch.stack(single)}

k_of = [len(r["options"]) for r in rows]


def maxdiff(A, Bm, letters_only):
    if letters_only:
        return max(float((A[i, letter_ids[:k]] - Bm[i, letter_ids[:k]]).abs().max()) for i, k in enumerate(k_of))
    return float((A - Bm).abs().max())


out = {"rows": [{"id": r["id"][:80], "n_tokens": len(e), "k": len(r["options"])} for r, e in zip(rows, encs)],
       "padded_length": max(len(e) for e in encs), "dropout_note": "train mode, LoRA dropout 0.05 with identical seed per forward",
       "pairs": {}, "padding_vs_single": {}, "loss_values": {}}
for a, b2 in (("default", "no_cudnn"), ("default", "math_only"), ("no_cudnn", "math_only")):
    A, Bm = res[a], res[b2]
    out["pairs"][f"{a} vs {b2}"] = {
        "max_abs_dlogit_k_letters": maxdiff(A["logits"], Bm["logits"], True),
        "max_abs_dlogit_full_vocab": maxdiff(A["logits"], Bm["logits"], False),
        "abs_dloss_letters_k_mean": float((A["lk"].mean() - Bm["lk"].mean()).abs()),
        "max_abs_dloss_letters_k_row": float((A["lk"] - Bm["lk"]).abs().max()),
        "abs_dloss_full_vocab_mean": float((A["lf"].mean() - Bm["lf"].mean()).abs()),
        "max_abs_dloss_full_vocab_row": float((A["lf"] - Bm["lf"]).abs().max()),
        "abs_dgrad_norm": abs(A["gnorm"] - Bm["gnorm"]), "rel_dgrad_norm": abs(A["gnorm"] - Bm["gnorm"]) / Bm["gnorm"],
    }
for name, R in res.items():
    out["padding_vs_single"][name] = {
        "max_abs_dlogit_k_letters": maxdiff(R["logits"], R["single"], True),
        "max_abs_dlogit_full_vocab": maxdiff(R["logits"], R["single"], False),
        "per_row_max_abs_dlogit_k": [float((R["logits"][i, letter_ids[:k]] - R["single"][i, letter_ids[:k]]).abs().max()) for i, k in enumerate(k_of)],
        "argmax_k_agree": sum(int(R["logits"][i, letter_ids[:k]].argmax() == R["single"][i, letter_ids[:k]].argmax()) for i, k in enumerate(k_of)),
    }
    out["loss_values"][name] = {"letters_k_mean": float(R["lk"].mean()), "full_vocab_mean": float(R["lf"].mean()), "grad_norm": R["gnorm"]}
(OUT / "train_path.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
