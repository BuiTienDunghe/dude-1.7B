"""Fine-tune LoRA cho decision model: loss chỉ tại token kế tiếp sau prompt, trên các chữ cái của k lựa chọn.

Dùng: train_lora.py <model_id> <seed> <out_dir> [--lr 2e-4] [--loss letters_k|full_vocab]
       [--monitor_frac 0.1 --gate_step_frac 0.2 --gate_letter_mass 0.5] [--check_identity N]
Prompt dựng bằng common.render_prompt — đúng module chấm dùng, không có bản thứ hai.
Mỗi lần nạp một câu train, thứ tự lựa chọn (và target) được xáo lại.

Loss (--loss):
  letters_k  (mặc định, loss của bước 3): CE giữa softmax trên riêng k logit chữ cái và target.
  full_vocab : CE trên toàn từ vựng, target mềm đặt lên k token chữ cái: −Σ_i target_i · log p_vocab(chữ cái i)
               = letters_k − log letter_mass, với letter_mass = Σ_{i<k} p_vocab(chữ cái i).
Theo dõi (--monitor_frac > 0): cứ mỗi phần số bước đó, đo letter_mass trung bình trên 200 câu đầu (cố định) của val_v0
  ở chế độ eval/no_grad (không đổi trọng số, không tiêu RNG); nếu --gate_letter_mass > 0 và ở mốc --gate_step_frac
  giá trị dưới ngưỡng thì lưu adapter tạm, ghi log và dừng (exit 3).
--check_identity N: chỉ kiểm đẳng thức full_vocab = letters_k − log letter_mass trên N câu val bằng model chưa train rồi thoát.
Ra: <out_dir>/adapter/ (checkpoint cuối epoch 2), train_log.jsonl, summary.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONFIG, DATA, LETTERS, load_json, model_revisions, read_jsonl, render_prompt, sha256_file

ap = argparse.ArgumentParser()
ap.add_argument("model_id")
ap.add_argument("seed", type=int)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--micro_batch", type=int, default=8)
ap.add_argument("--effective_batch", type=int, default=32)
ap.add_argument("--warmup_frac", type=float, default=0.03)
ap.add_argument("--loss", choices=["letters_k", "full_vocab"], default="letters_k")
ap.add_argument("--monitor_frac", type=float, default=0.0)
ap.add_argument("--monitor_rows", type=int, default=200)
ap.add_argument("--gate_step_frac", type=float, default=0.2)
ap.add_argument("--gate_letter_mass", type=float, default=0.0)
ap.add_argument("--check_identity", type=int, default=0)
ap.add_argument("--check_adapter", type=Path)
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
accum = args.effective_batch // args.micro_batch
assert accum * args.micro_batch == args.effective_batch

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
rev = model_revisions()[args.model_id]
frames = load_json(CONFIG / "prompt_frames.json")
tok = AutoTokenizer.from_pretrained(args.model_id, revision=rev)
letter_ids = torch.tensor([tok.convert_tokens_to_ids(L) for L in LETTERS], device="cuda")
assert all(len(tok.encode(L, add_special_tokens=False)) == 1 for L in LETTERS)

model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=rev, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
torch.manual_seed(args.seed)
if args.check_identity and args.check_adapter:   # chỉ dùng để kiểm đẳng thức trên một adapter đã train
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(args.check_adapter))
else:
    model = get_peft_model(model, lcfg)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
inner = model.base_model.model           # Qwen3ForCausalLM
backbone, lm_head = inner.model, inner.lm_head

train = read_jsonl(DATA / "train_v0.jsonl")
val = read_jsonl(DATA / "val_v0.jsonl")


def encode(row):
    ids = tok.encode(render_prompt(row, tok, frames), add_special_tokens=False)
    return ids


def shuffled(row, rng):
    perm = rng.permutation(len(row["options"]))
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}


def answer_logits(rows, encs=None):
    """Logit toàn từ vựng tại vị trí cuối của prompt. Right padding, lấy hidden tại len-1. Trả về [B, V] float32."""
    encs = encs or [encode(r) for r in rows]
    L = max(len(e) for e in encs)
    ids = torch.full((len(encs), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        ids[i, :len(e)] = torch.tensor(e); mask[i, :len(e)] = 1
    ids, mask = ids.cuda(), mask.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        last = torch.tensor([len(e) - 1 for e in encs], device="cuda")
        logits = lm_head(h[torch.arange(len(encs), device="cuda"), last]).float()   # [B, V]
    return logits


def row_losses(rows, logits, mode):
    """Loss từng câu. letters_k: đúng phép tính của bước 3. full_vocab: CE trên toàn từ vựng với target trên k chữ cái."""
    losses = []
    for i, r in enumerate(rows):
        k = len(r["options"])
        tgt = torch.tensor(r["target"], device="cuda", dtype=torch.float32)
        if mode == "letters_k":
            logp = torch.log_softmax(logits[i, letter_ids[:k]], -1)
        else:
            logp = torch.log_softmax(logits[i], -1)[letter_ids[:k]]
        losses.append(-(tgt * logp).sum())
    return torch.stack(losses)


def row_log_letter_mass(rows, logits):
    return torch.stack([torch.logsumexp(torch.log_softmax(logits[i], -1)[letter_ids[:len(r["options"])]], -1)
                        for i, r in enumerate(rows)])


def batch_loss(rows, encs=None):
    return row_losses(rows, answer_logits(rows, encs), args.loss)


@torch.no_grad()
def val_eval():
    """Val loss theo cả hai định nghĩa (loss của lần chạy là `val_loss`) và letter_mass trung bình."""
    model.eval()
    tot = {"letters_k": 0.0, "full_vocab": 0.0}; mass = 0.0
    vs = sorted(val, key=lambda r: len(encode(r)))   # tổng không phụ thuộc thứ tự; sắp để bớt padding
    for s in range(0, len(vs), 16):
        rs = vs[s:s + 16]
        lg = answer_logits(rs)
        for mode in tot:
            tot[mode] += row_losses(rs, lg, mode).sum().item()
        mass += row_log_letter_mass(rs, lg).exp().sum().item()
    model.train()
    return {"val_loss": tot[args.loss] / len(val), "val_loss_letters_k": tot["letters_k"] / len(val),
            "val_loss_full_vocab": tot["full_vocab"] / len(val), "val_letter_mass_mean": mass / len(val)}


monitor_rows = val[:args.monitor_rows]   # 200 câu đầu của val_v0 (thứ tự đã lưu), cố định


@torch.no_grad()
def monitor_letter_mass():
    model.eval()
    vals = []
    for s in range(0, len(monitor_rows), 16):
        rs = monitor_rows[s:s + 16]
        vals += row_log_letter_mass(rs, answer_logits(rs)).exp().tolist()
    model.train()
    return float(np.mean(vals)), int(sum(v < 0.5 for v in vals))


if args.check_identity:
    # Kiểm bằng số: full_vocab = letters_k − log letter_mass, trên N câu val, model gốc (LoRA khởi tạo = 0).
    model.eval()
    with torch.no_grad():
        rs = val[:args.check_identity]
        lg = answer_logits(rs)
        a, b, lm = row_losses(rs, lg, "letters_k"), row_losses(rs, lg, "full_vocab"), row_log_letter_mass(rs, lg)
    res = [{"id": r["id"][:60], "k": len(r["options"]), "letters_k": float(x), "full_vocab": float(y), "log_letter_mass": float(z),
            "letters_k_minus_log_mass": float(x - z), "abs_diff": float(abs(y - (x - z)))} for r, x, y, z in zip(rs, a, b, lm)]
    (args.out_dir / "identity_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for x in res[:5]:
        print(json.dumps(x))
    print("max_abs_diff", max(x["abs_diff"] for x in res))
    raise SystemExit(0)


steps_per_epoch = len(train) // args.effective_batch
total_steps = steps_per_epoch * args.epochs
warmup = max(1, int(args.warmup_frac * total_steps))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

log = open(args.out_dir / "train_log.jsonl", "w", encoding="utf-8")
def emit(rec):
    log.write(json.dumps(rec) + "\n"); log.flush()

monitor_every = max(1, round(args.monitor_frac * total_steps)) if args.monitor_frac > 0 else 0
gate_step = round(args.gate_step_frac * total_steps)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
ve = val_eval()
v0 = ve["val_loss"]
emit({"event": "val", "step": 0, "epoch": 0, **ve, "loss_mode": args.loss})
if monitor_every:
    lm_mean, lm_low = monitor_letter_mass()
    emit({"event": "letter_mass", "step": 0, "mean": lm_mean, "n_below_0.5": lm_low, "n": len(monitor_rows)})
model.train()
step, run_loss = 0, []
for epoch in range(args.epochs):
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(steps_per_epoch):
        idx = order[b * args.effective_batch:(b + 1) * args.effective_batch]
        rows = [shuffled(train[i], rng) for i in idx]
        # Sắp 32 câu của batch hiệu dụng theo độ dài trước khi chia micro-batch: gradient vẫn là trung bình trên
        # đúng 32 câu đó (toán học không đổi), chỉ giảm phần padding.
        encs = [encode(r) for r in rows]
        srt = sorted(range(len(rows)), key=lambda i: len(encs[i]))
        rows, encs = [rows[i] for i in srt], [encs[i] for i in srt]
        for m in range(accum):
            sl = slice(m * args.micro_batch, (m + 1) * args.micro_batch)
            loss = batch_loss(rows[sl], encs[sl]).mean() / accum
            loss.backward()
            run_loss.append(loss.item() * accum)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            emit({"event": "train", "step": step, "epoch": epoch + (b + 1) / steps_per_epoch,
                  "loss": float(np.mean(run_loss)), "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                  "elapsed_s": time.time() - t0})
            run_loss = []
        if monitor_every and step % monitor_every == 0:
            lm_mean, lm_low = monitor_letter_mass()
            emit({"event": "letter_mass", "step": step, "mean": lm_mean, "n_below_0.5": lm_low, "n": len(monitor_rows),
                  "elapsed_s": time.time() - t0})
            if args.gate_letter_mass > 0 and step == gate_step and lm_mean < args.gate_letter_mass:
                model.save_pretrained(args.out_dir / f"adapter_gate_step{step}")
                emit({"event": "gate_stop", "step": step, "letter_mass_mean": lm_mean, "threshold": args.gate_letter_mass})
                print(json.dumps({"gate_stop": True, "step": step, "letter_mass_mean": lm_mean}))
                raise SystemExit(3)
    emit({"event": "val", "step": step, "epoch": epoch + 1, **val_eval(), "elapsed_s": time.time() - t0})

model.save_pretrained(args.out_dir / "adapter")
summary = {
    "model": args.model_id, "revision": rev, "seed": args.seed, "lr": args.lr, "epochs": args.epochs, "loss": args.loss,
    "monitor_every_steps": monitor_every, "gate": {"step": gate_step, "letter_mass_threshold": args.gate_letter_mass},
    "effective_batch": args.effective_batch, "micro_batch": args.micro_batch, "grad_accum": accum,
    "steps": total_steps, "warmup_steps": warmup, "schedule": "linear warmup + cosine decay to 0", "grad_clip": 1.0,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(lcfg.target_modules)},
    "trainable_params": n_trainable, "precision": "bf16 weights + bf16 autocast, LoRA params fp32 (peft default)",
    "gradient_checkpointing": True, "train_rows": len(train), "val_rows": len(val),
    "train_sha256": sha256_file(DATA / "train_v0.jsonl"), "val_sha256": sha256_file(DATA / "val_v0.jsonl"),
    "val_loss_start": v0, "val_loss_by_epoch": [json.loads(l)["val_loss"] for l in open(args.out_dir / "train_log.jsonl") if '"val"' in l],
    "wall_time_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
}
(args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["model", "seed", "val_loss_by_epoch", "wall_time_s", "peak_vram_gb"]}))
