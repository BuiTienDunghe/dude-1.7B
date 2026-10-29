"""Fine-tune LoRA cho decision model: loss chỉ tại token kế tiếp sau prompt, trên các chữ cái của k lựa chọn.

Dùng: train_lora.py <model_id> <seed> <out_dir> [--lr 2e-4] [--loss full_vocab|letters_k] [--token_budget N] [--vram_probe T]
       [--monitor_frac 0.1 --gate_step_frac 0.2 --gate_letter_mass 0.5] [--check_identity N]
       [--strip_p 0.3 --strip_state_only_frac 0.5] [--val_nodiac] [--dump_batches N]
Prompt dựng bằng common.render_prompt — đúng module chấm dùng, không có bản thứ hai.
Mỗi lần nạp một câu train, thứ tự lựa chọn (và target) được xáo lại.

Sửa 28/10 (bước 0, audit 27/10): mặc định --loss là full_vocab; chỉ câu choice được xáo thứ tự lựa chọn (noul/score giữ nguyên);
  --token_budget N > 0 chia micro-batch theo ngân sách token sau pad (để câu dài tới 2.048 token không tràn 16 GB); --vram_probe T đo VRAM
  một lượt forward+backward trên nhóm câu dài T token rồi thoát (không train).
Loss (--loss):
  letters_k  (loss của bước 3, đã biết là hỏng): CE giữa softmax trên riêng k logit chữ cái và target.
  full_vocab (mặc định từ 28/10): CE trên toàn từ vựng, target mềm đặt lên k token chữ cái: −Σ_i target_i · log p_vocab(chữ cái i)
               = letters_k − log letter_mass, với letter_mass = Σ_{i<k} p_vocab(chữ cái i).
Theo dõi (--monitor_frac > 0): cứ mỗi phần số bước đó, đo letter_mass trung bình trên 200 câu đầu (cố định) của val_v0
  ở chế độ eval/no_grad (không đổi trọng số, không tiêu RNG); nếu --gate_letter_mass > 0 và ở mốc --gate_step_frac
  giá trị dưới ngưỡng thì lưu adapter tạm, ghi log và dừng (exit 3).
--check_identity N: chỉ kiểm đẳng thức full_vocab = letters_k − log letter_mass trên N câu val bằng model chưa train rồi thoát.
Bỏ dấu khi nạp dữ liệu (--strip_p > 0): mỗi epoch, mỗi dòng train có text_lang == "vi" bị bỏ dấu với xác suất strip_p;
  trong số đó, phần --strip_state_only_frac chỉ bỏ ở state, phần còn lại bỏ ở state, câu hỏi và lựa chọn.
  Quyết định lấy từ common.strip_decision(seed, epoch, id) — tách khỏi RNG xáo batch, nên thứ tự batch và hoán vị lựa
  chọn giống hệt khi không bỏ dấu. Hàm bỏ dấu là common.strip_diacritics (chính hàm đã tạo bản không dấu của bench_v1).
--val_nodiac: mỗi lần đo val, ghi thêm loss trên val_v0 với mọi dòng tiếng Việt bỏ dấu toàn bộ (chỉ ghi, không dùng để chọn).
--dump_batches N: chỉ chạy đường dữ liệu N bước đầu (không nạp model), ghi hash id dòng và hash token từng bước rồi thoát.
--train_file / --val_file: file dữ liệu train / val (mặc định data/train_v0.jsonl, data/val_v0.jsonl = hành vi cũ). Không truyền thì
  summary.json giữ đúng các trường cũ.
--save_fracs "0.25,0.5,0.75": lưu thêm <out_dir>/adapter_step<N>/ ở các mốc đó của tổng số bước (mặc định trống = tắt; không đổi RNG
  hay thứ tự batch). Mốc 100% là adapter/ cuối train.
Ra: <out_dir>/adapter/ (checkpoint cuối epoch 2), train_log.jsonl, summary.json
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (CONFIG, DATA, LETTERS, apply_strip, load_json, model_revisions, read_jsonl, render_prompt, sha256_file, strip_decision,
                    strip_diacritics)

ap = argparse.ArgumentParser()
ap.add_argument("model_id")
ap.add_argument("seed", type=int)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--micro_batch", type=int, default=8)
ap.add_argument("--effective_batch", type=int, default=32)
ap.add_argument("--warmup_frac", type=float, default=0.03)
ap.add_argument("--loss", choices=["letters_k", "full_vocab"], default="full_vocab")   # đổi mặc định 28/10 (audit 27/10); letters_k đã biết là hỏng
ap.add_argument("--token_budget", type=int, default=0)   # > 0: chia micro-batch theo ngân sách token sau pad thay cho --micro_batch cố định
ap.add_argument("--vram_probe", type=int, default=0)     # > 0: chỉ đo VRAM 1 lượt forward+backward trên nhóm câu dài N token rồi thoát
ap.add_argument("--monitor_frac", type=float, default=0.0)
ap.add_argument("--monitor_rows", type=int, default=200)
ap.add_argument("--gate_step_frac", type=float, default=0.2)
ap.add_argument("--gate_letter_mass", type=float, default=0.0)
ap.add_argument("--check_identity", type=int, default=0)
ap.add_argument("--check_adapter", type=Path)
ap.add_argument("--strip_p", type=float, default=0.0)
ap.add_argument("--strip_state_only_frac", type=float, default=0.5)
ap.add_argument("--val_nodiac", action="store_true")
ap.add_argument("--dump_batches", type=int, default=0)
ap.add_argument("--train_file", type=Path, default=DATA / "train_v0.jsonl")   # mặc định như cũ; ví dụ results/train_v1/components/C.jsonl
ap.add_argument("--val_file", type=Path, default=DATA / "val_v0.jsonl")
ap.add_argument("--save_fracs", default="")   # ví dụ "0.25,0.5,0.75": lưu thêm adapter_step<N> ở các mốc đó; trống = tắt (mặc định)
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

train = read_jsonl(args.train_file)
val = read_jsonl(args.val_file)
steps_per_epoch = len(train) // args.effective_batch
total_steps = steps_per_epoch * args.epochs
# Mốc lưu adapter giữa chừng (--save_fracs); mốc 100% chính là <out_dir>/adapter/ cuối train nên không lưu lặp
save_steps = sorted({min(total_steps, max(1, round(float(f) * total_steps))) for f in args.save_fracs.split(",") if f.strip()} - {total_steps})


def encode(row):
    ids = tok.encode(render_prompt(row, tok, frames), add_special_tokens=False)
    return ids


def shuffled(row, rng):
    """Xáo thứ tự lựa chọn (và target) cho câu kind == "choice". Câu "noul" (Có/Không) và "score" (thang có thứ tự) GIỮ NGUYÊN thứ tự
    (quyết định 12/10, thực thi sau audit 27/10). Hoán vị vẫn được rút cho MỌI câu để RNG tiêu y như trước: thứ tự batch và hoán vị
    của câu choice giống hệt các lần train cũ cùng seed."""
    perm = rng.permutation(len(row["options"]))
    if row["kind"] != "choice":
        return row
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}


def pad64(n):
    return ((n + 63) // 64) * 64


def token_groups(encs):
    """Chia một batch hiệu dụng (đã sắp theo độ dài tăng dần) thành các nhóm micro-batch cho --token_budget > 0: tham lam, mỗi nhóm có
    (số câu × độ dài dài nhất làm tròn lên bội 64) <= ngân sách (như ft_decider). Một câu dài hơn ngân sách đứng riêng một nhóm."""
    out, cur = [], []
    for i, e in enumerate(encs):
        if cur and (len(cur) + 1) * pad64(max(len(e), max(len(encs[j]) for j in cur))) > args.token_budget:
            out.append(cur); cur = []
        cur.append(i)
    return out + [cur]


def maybe_strip(row, epoch):
    """Bỏ dấu theo quyết định hash; chỉ dòng tiếng Việt (trường text_lang của dữ liệu). Trả về (row, mode)."""
    if args.strip_p <= 0 or row["text_lang"] != "vi":
        return row, None
    mode = strip_decision(args.seed, epoch, row["id"], args.strip_p, args.strip_state_only_frac)
    return apply_strip(row, mode), mode


def epoch_batches(epoch):
    """Các batch hiệu dụng của một epoch, đúng như vòng train: (rows, encs, modes) đã sắp theo độ dài."""
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(steps_per_epoch):
        idx = order[b * args.effective_batch:(b + 1) * args.effective_batch]
        rows, modes, before = [], [], []
        for i in idx:
            r0 = shuffled(train[i], rng)                           # shuffled tiêu RNG như cũ; bỏ dấu không tiêu RNG
            r, m = maybe_strip(r0, epoch)
            rows.append(r); modes.append(m); before.append(r0)
        # Sắp 32 câu của batch hiệu dụng theo độ dài trước khi chia micro-batch: gradient vẫn là trung bình trên
        # đúng 32 câu đó (toán học không đổi), chỉ giảm phần padding. Khóa sắp là độ dài token của câu TRƯỚC khi bỏ dấu
        # (quyết định của người dùng ở cổng G2): thứ tự trong batch và cách chia micro-batch giống hệt lần train không bỏ dấu.
        encs = [encode(r) for r in rows]
        key = [len(encode(b0)) if m else len(e) for b0, m, e in zip(before, modes, encs)]
        srt = sorted(range(len(rows)), key=lambda i: key[i])
        yield [rows[i] for i in srt], [encs[i] for i in srt], [modes[i] for i in srt]


if args.dump_batches:
    # Đường dữ liệu N bước đầu, không nạp model: hash id dòng và hash token từng bước.
    out, step = [], 0
    for epoch in range(args.epochs):
        for rows, encs, modes in epoch_batches(epoch):
            out.append({"step": step + 1, "epoch": epoch,
                        "ids_sha": hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest(),
                        "tokens_sha": hashlib.sha256(json.dumps(encs).encode()).hexdigest(),
                        "n_stripped": sum(m is not None for m in modes), "modes": [m for m in modes if m]})
            step += 1
            if step >= args.dump_batches:
                break
        if step >= args.dump_batches:
            break
    (args.out_dir / "batch_dump.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    # Kiểm loại câu (file riêng, để batch_dump.jsonl giữ đúng định dạng cũ): câu noul/score phải giữ nguyên thứ tự lựa chọn
    # (so theo bản bỏ dấu để không lẫn với bỏ dấu); đếm câu choice bị đổi thứ tự; số micro-batch theo --token_budget.
    TR = {r["id"]: r for r in train}
    kc = {"noul_score": 0, "noul_score_giu_thu_tu": 0, "choice": 0, "choice_doi_thu_tu": 0, "so_micro_batch": 0, "max_padded_tokens_micro": 0}
    step = 0
    for epoch in range(args.epochs):
        for rows, encs, modes in epoch_batches(epoch):
            for r in rows:
                same = ([strip_diacritics(o) for o in r["options"]] == [strip_diacritics(o) for o in TR[r["id"]]["options"]]
                        and r["target"] == TR[r["id"]]["target"])
                if r["kind"] == "choice":
                    kc["choice"] += 1; kc["choice_doi_thu_tu"] += not same
                else:
                    kc["noul_score"] += 1; kc["noul_score_giu_thu_tu"] += same
            gs = token_groups(encs) if args.token_budget > 0 else [list(range(m * args.micro_batch, (m + 1) * args.micro_batch)) for m in range(accum)]
            kc["so_micro_batch"] += len(gs)
            kc["max_padded_tokens_micro"] = max(kc["max_padded_tokens_micro"], max(len(g) * pad64(max(len(encs[i]) for i in g)) for g in gs))
            step += 1
            if step >= args.dump_batches:
                break
        if step >= args.dump_batches:
            break
    (args.out_dir / "batch_dump_kiem_loai_cau.json").write_text(json.dumps(kc, indent=1), encoding="utf-8")
    print(json.dumps({"dumped_steps": len(out), "strip_p": args.strip_p, "stripped_rows": sum(x["n_stripped"] for x in out),
                      "total_steps": total_steps, "save_steps": save_steps, "kiem_loai_cau": kc}))
    raise SystemExit(0)

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
val_nodiac = [apply_strip(r, "all") if r["text_lang"] == "vi" else r for r in val]


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
def val_eval(rows=None, prefix="val"):
    """Val loss theo cả hai định nghĩa (loss của lần chạy là `<prefix>_loss`) và letter_mass trung bình."""
    rows = val if rows is None else rows
    model.eval()
    tot = {"letters_k": 0.0, "full_vocab": 0.0}; mass = 0.0
    vs = sorted(rows, key=lambda r: len(encode(r)))   # tổng không phụ thuộc thứ tự; sắp để bớt padding
    for s in range(0, len(vs), 16):
        rs = vs[s:s + 16]
        lg = answer_logits(rs)
        for mode in tot:
            tot[mode] += row_losses(rs, lg, mode).sum().item()
        mass += row_log_letter_mass(rs, lg).exp().sum().item()
    model.train()
    return {f"{prefix}_loss": tot[args.loss] / len(rows), f"{prefix}_loss_letters_k": tot["letters_k"] / len(rows),
            f"{prefix}_loss_full_vocab": tot["full_vocab"] / len(rows), f"{prefix}_letter_mass_mean": mass / len(rows)}


def val_all():
    out = val_eval()
    if args.val_nodiac:
        out.update(val_eval(val_nodiac, "val_nodiac"))
    return out


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


if args.vram_probe:
    # Đo VRAM, KHÔNG train: một lượt forward + backward trên nhóm câu dài nhất có thể (câu giả dài --vram_probe token, lấy state thật dài
    # nhất của file train rồi lặp và cắt cho đủ độ dài); không optimizer step, không lưu adapter. Số câu trong nhóm = số câu tối đa
    # mà --token_budget cho phép ở độ dài đó (hoặc --micro_batch nếu không đặt ngân sách).
    base = max(train, key=lambda r: len(r["state"]))
    body = base["state"]
    while len(encode({**base, "state": body})) < args.vram_probe:
        body = body + " " + base["state"]
    lo, hi = 0, len(body)
    while lo < hi:                                     # cắt state dài nhất mà prompt vẫn <= vram_probe token
        mid = (lo + hi + 1) // 2
        if len(encode({**base, "state": body[:mid]})) <= args.vram_probe:
            lo = mid
        else:
            hi = mid - 1
    row = {**base, "state": body[:lo]}
    T = len(encode(row))
    n = max(1, args.token_budget // pad64(T)) if args.token_budget > 0 else args.micro_batch
    rows = [row] * n
    model.train()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t_ = time.time()
    rl = batch_loss(rows, [encode(r) for r in rows])
    (rl.sum() / len(rows)).backward()
    torch.cuda.synchronize()
    res = {"vram_probe_tokens": T, "rows_in_group": n, "token_budget": args.token_budget, "micro_batch": args.micro_batch,
           "padded_tokens": n * pad64(T), "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30, "seconds": time.time() - t_,
           "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30, "loss": args.loss,
           "ghi_chu": "chưa gồm trạng thái AdamW của tham số LoRA (khoảng 2 × 4 byte × số tham số train được)", "trainable_params": n_trainable}
    model.zero_grad(set_to_none=True)
    (args.out_dir / f"vram_probe_{T}_{args.token_budget}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    raise SystemExit(0)

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
ve = val_all()
v0 = ve["val_loss"]
emit({"event": "val", "step": 0, "epoch": 0, **ve, "loss_mode": args.loss})
if monitor_every:
    lm_mean, lm_low = monitor_letter_mass()
    emit({"event": "letter_mass", "step": 0, "mean": lm_mean, "n_below_0.5": lm_low, "n": len(monitor_rows)})
model.train()
step, run_loss = 0, []
strip_counts = []
for epoch in range(args.epochs):
    cnt = {"vi_rows_seen": 0, "state_only": 0, "all": 0}
    for b, (rows, encs, modes) in enumerate(epoch_batches(epoch)):
        cnt["vi_rows_seen"] += sum(r["text_lang"] == "vi" for r in rows)
        for m in modes:
            if m:
                cnt[m] += 1
        if args.token_budget <= 0:                     # đường cũ, giữ nguyên từng phép tính
            for m in range(accum):
                sl = slice(m * args.micro_batch, (m + 1) * args.micro_batch)
                loss = batch_loss(rows[sl], encs[sl]).mean() / accum
                loss.backward()
                run_loss.append(loss.item() * accum)
        else:                                          # theo ngân sách token: gradient vẫn = Σ loss các câu / số câu của batch hiệu dụng
            for g in token_groups(encs):
                rl = batch_loss([rows[i] for i in g], [encs[i] for i in g])
                (rl.sum() / len(rows)).backward()
                run_loss.append(rl.mean().item())
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            emit({"event": "train", "step": step, "epoch": epoch + (b + 1) / steps_per_epoch,
                  "loss": float(np.mean(run_loss)), "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                  "elapsed_s": time.time() - t0})
            run_loss = []
        if step in save_steps:
            model.save_pretrained(args.out_dir / f"adapter_step{step}")
            emit({"event": "save_adapter", "step": step, "frac": step / total_steps, "elapsed_s": time.time() - t0})
        if monitor_every and step % monitor_every == 0:
            lm_mean, lm_low = monitor_letter_mass()
            emit({"event": "letter_mass", "step": step, "mean": lm_mean, "n_below_0.5": lm_low, "n": len(monitor_rows),
                  "elapsed_s": time.time() - t0})
            if args.gate_letter_mass > 0 and step == gate_step and lm_mean < args.gate_letter_mass:
                model.save_pretrained(args.out_dir / f"adapter_gate_step{step}")
                emit({"event": "gate_stop", "step": step, "letter_mass_mean": lm_mean, "threshold": args.gate_letter_mass})
                print(json.dumps({"gate_stop": True, "step": step, "letter_mass_mean": lm_mean}))
                raise SystemExit(3)
    strip_counts.append({"epoch": epoch, **cnt})
    emit({"event": "strip_counts", **strip_counts[-1]})
    emit({"event": "val", "step": step, "epoch": epoch + 1, **val_all(), "elapsed_s": time.time() - t0})

model.save_pretrained(args.out_dir / "adapter")
summary = {
    "model": args.model_id, "revision": rev, "seed": args.seed, "lr": args.lr, "epochs": args.epochs, "loss": args.loss,
    "strip_p": args.strip_p, "strip_state_only_frac": args.strip_state_only_frac, "strip_counts": strip_counts,
    **({"save_steps": save_steps} if save_steps else {}),
    "monitor_every_steps": monitor_every, "gate": {"step": gate_step, "letter_mass_threshold": args.gate_letter_mass},
    "effective_batch": args.effective_batch, "micro_batch": args.micro_batch, "grad_accum": accum,
    **({"token_budget": args.token_budget} if args.token_budget > 0 else {}),
    "steps": total_steps, "warmup_steps": warmup, "schedule": "linear warmup + cosine decay to 0", "grad_clip": 1.0,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(lcfg.target_modules)},
    "trainable_params": n_trainable, "precision": "bf16 weights + bf16 autocast, LoRA params fp32 (peft default)",
    "gradient_checkpointing": True, "train_rows": len(train), "val_rows": len(val),
    "train_sha256": sha256_file(args.train_file), "val_sha256": sha256_file(args.val_file),
    **({"train_file": str(args.train_file)} if args.train_file != DATA / "train_v0.jsonl" else {}),
    **({"val_file": str(args.val_file)} if args.val_file != DATA / "val_v0.jsonl" else {}),
    "val_loss_start": v0, "val_loss_by_epoch": [json.loads(l)["val_loss"] for l in open(args.out_dir / "train_log.jsonl") if '"event": "val"' in l],
    "wall_time_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
}
(args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["model", "seed", "val_loss_by_epoch", "wall_time_s", "peak_vram_gb"]}))
