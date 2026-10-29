"""Fine-tune LoRA từ Mapika/decider-2b trên train_v0, ở định dạng gốc của decider (scripts/ft_decider/dcommon.py).

Dùng: train_decider.py <seed> <out_dir> [--lr 2e-4] [--strip_p 0.3 --strip_state_only_frac 0.5] [--val_nodiac]
       [--monitor_frac 0.1 --gate_step_frac 0.2 --gate_mass 0.5] [--g3 K] [--token_budget N]
Đường dữ liệu giống hệt train_lora.py của strip_v0: cùng hoán vị batch (rng seed*1000+epoch), cùng xáo lựa chọn từng dòng (shuffled
tiêu RNG như cũ), cùng quyết định bỏ dấu common.strip_decision(seed, epoch, id) chỉ cho dòng text_lang == "vi", và 32 dòng của batch
sắp theo độ dài TRƯỚC khi bỏ dấu (ở đây: tổng token decider của các hàng prompt của dòng) rồi chia 4 micro-batch × 8 dòng.
Loss: dcommon.row_loss (CE trên toàn từ vựng tại slot, chia T phát hành theo kiểu câu; score: CE với phân phối isolated + hạng mass).
Gradient của một bước = Σ loss 32 dòng / 32. --token_budget N > 0: một micro-batch được chạy forward/backward thành vài nhóm dòng
  sao cho số token đã pad của nhóm ≤ N (gradient không đổi về toán học); 0 = cả micro-batch một lần.
Theo dõi: mỗi --monitor_frac số bước, mass trung bình (khối lượng xác suất trên token nhãn hợp lệ, toàn từ vựng, sau chia T)
  trên 200 dòng đầu của val_v0; ở mốc --gate_step_frac nếu dưới --gate_mass thì lưu adapter tạm và dừng (exit 3).
Checkpoint: adapter_epoch1/ (cuối epoch 1, chỉ để xem), adapter/ (cuối epoch 2, bản chính thức).
--g3 K: không train thật. Lập kế hoạch token của toàn bộ 2 epoch (không cần model), đo VRAM đỉnh trên micro-batch nặng nhất
  (forward + backward), chạy K bước thật đầu tiên để đo thời gian, ước lượng tổng thời gian; ghi g3.json rồi thoát (không lưu adapter).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, apply_strip, read_jsonl, sha256_file, strip_decision  # noqa: E402
import dcommon as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--micro_batch", type=int, default=8)
ap.add_argument("--effective_batch", type=int, default=32)
ap.add_argument("--warmup_frac", type=float, default=0.03)
ap.add_argument("--strip_p", type=float, default=0.0)
ap.add_argument("--strip_state_only_frac", type=float, default=0.5)
ap.add_argument("--val_nodiac", action="store_true")
ap.add_argument("--monitor_frac", type=float, default=0.1)
ap.add_argument("--monitor_rows", type=int, default=200)
ap.add_argument("--gate_step_frac", type=float, default=0.2)
ap.add_argument("--gate_mass", type=float, default=0.0)
ap.add_argument("--token_budget", type=int, default=0)
ap.add_argument("--g3", type=int, default=0)
ap.add_argument("--loss_variant", choices=["A", "C", "N"], default=None)
args = ap.parse_args()
if args.loss_variant is None:            # chỉ G3 (đo VRAM/thời gian, không phụ thuộc loss) được bỏ trống; train thật phải chọn
    assert args.g3, "--loss_variant bắt buộc khi train"
    args.loss_variant = "A"
args.out_dir.mkdir(parents=True, exist_ok=True)
accum = args.effective_batch // args.micro_batch
assert accum * args.micro_batch == args.effective_batch

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
folder = D.load_folder()
from decider import temperature as TT  # noqa: E402,F401
from decider.infer import Decider  # noqa: E402

# Đối tượng Decider chỉ để dựng prompt và lấy T (đúng cấu hình phát hành); trọng số của nó nạp lên CPU rồi bỏ.
d = Decider(folder, device="cpu", dtype=torch.bfloat16, use_graphs=False)
assert d.T_by_type == {"choice": 1.164, "noul": 1.624, "score": 1.124} and d.neutralize_none is False and d.chat is None and d.isolated_levels
d.m.lm = None
tok = d.m.tok
pad_id = tok.pad_token_id

train = read_jsonl(DATA / "train_v0.jsonl")
val = read_jsonl(DATA / "val_v0.jsonl")
steps_per_epoch = len(train) // args.effective_batch
total_steps = steps_per_epoch * args.epochs
_cache = {}


def prep(row):
    """-> (câu đã ánh xạ, items, T từng slot). Cache theo nội dung (dòng bỏ dấu khác nội dung nên không lẫn)."""
    key = json.dumps([row["id"], row["state"], row["question"], row["options"], row["target"]], ensure_ascii=False)
    v = _cache.get(key)
    if v is None:
        m = D.to_decider(row)
        items = D.items_for(d, m)
        v = (m, items, D.slot_temps(d, items))
        if len(_cache) < 200_000:
            _cache[key] = v
    return v


def n_tok(row):
    return sum(len(it["ids"]) for it in prep(row)[1])


def shuffled(row, rng):
    perm = rng.permutation(len(row["options"]))
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}


def maybe_strip(row, epoch):
    if args.strip_p <= 0 or row["text_lang"] != "vi":
        return row, None
    mode = strip_decision(args.seed, epoch, row["id"], args.strip_p, args.strip_state_only_frac)
    return apply_strip(row, mode), mode


def epoch_batches(epoch):
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(steps_per_epoch):
        idx = order[b * args.effective_batch:(b + 1) * args.effective_batch]
        rows, modes, before = [], [], []
        for i in idx:
            r0 = shuffled(train[i], rng)
            r, m = maybe_strip(r0, epoch)
            rows.append(r); modes.append(m); before.append(r0)
        key = [n_tok(b0) for b0 in before]                      # độ dài TRƯỚC khi bỏ dấu
        srt = sorted(range(len(rows)), key=lambda i: key[i])
        yield [rows[i] for i in srt], [modes[i] for i in srt], [idx[i] for i in srt]


def padded_tokens(rows):
    items = [it for r in rows for it in prep(r)[1]]
    L = ((max(len(it["ids"]) for it in items) + 63) // 64) * 64
    return len(items) * L, len(items)


def groups(rows):
    if args.token_budget <= 0:
        return [rows]
    out, cur = [], []
    for r in rows:
        if cur and padded_tokens(cur + [r])[0] > args.token_budget:
            out.append(cur); cur = []
        cur.append(r)
    return out + [cur]


# ------------------------------------------------------------------ G3, phần không cần model: kế hoạch token 2 epoch
plan = []
if args.g3:
    t_plan = time.time()
    for epoch in range(args.epochs):
        for b, (rows, modes, idx) in enumerate(epoch_batches(epoch)):
            mbs = [padded_tokens(rows[m * args.micro_batch:(m + 1) * args.micro_batch]) for m in range(accum)]
            plan.append({"step": len(plan) + 1, "epoch": epoch, "padded_tokens": sum(x[0] for x in mbs),
                         "max_micro_padded_tokens": max(x[0] for x in mbs), "items": sum(x[1] for x in mbs), "batch": b})
    print("plan done", round(time.time() - t_plan), "s", flush=True)
    mt = np.array([x["max_micro_padded_tokens"] for x in plan])
    plan_stats = {"steps": len(plan), "total_padded_tokens": int(sum(x["padded_tokens"] for x in plan)),
                  "max_micro_padded_tokens": int(mt.max()), "p50_max_micro": float(np.median(mt)), "p99_max_micro": float(np.percentile(mt, 99)),
                  "steps_with_micro_over_16k": int((mt > 16384).sum()), "steps_with_micro_over_32k": int((mt > 32768).sum())}
    (args.out_dir / "plan_stats.json").write_text(json.dumps(plan_stats, indent=1), encoding="utf-8")
    print(json.dumps(plan_stats), flush=True)

# ------------------------------------------------------------------ model train: dcommon.build_lora_model (DecisionModel + LoRA)
M = D.build_lora_model(folder, args.seed)
model, letters, lora_modules, n_trainable, TARGETS = M["model"], M["letters"], M["lora_modules"], M["n_trainable"], D.TARGETS


def forward_rows(rows):
    """-> list[(câu đã ánh xạ, p giới hạn từng slot, log p toàn từ vựng tại nhãn từng slot)]."""
    preps = [prep(r) for r in rows]
    items = [it for _, its, _ in preps for it in its]
    temps = [t for _, _, ts in preps for t in ts]
    ps, lfs = D.readout(D.forward_full(M, items, pad_id), temps)
    out, c = [], 0
    for m, its, _ in preps:
        n = sum(len(it["slots"]) for it in its)
        out.append((m, ps[c:c + n], lfs[c:c + n])); c += n
    return out


def losses(rows):
    return torch.stack([D.row_loss(m, p, lf, args.loss_variant) for m, p, lf in forward_rows(rows)])


val_nodiac = [apply_strip(r, "all") if r["text_lang"] == "vi" else r for r in val]


@torch.no_grad()
def val_eval(rows=None, prefix="val"):
    rows = val if rows is None else rows
    model.eval()
    tot = ce = mass = 0.0; correct = 0
    vs = sorted(rows, key=n_tok)
    for s in range(0, len(vs), 8):
        for m, p, lf in [x for g in groups(vs[s:s + 8]) for x in forward_rows(g)]:
            tot += D.row_loss(m, p, lf, args.loss_variant).item()
            q = D.row_dist(m, p); t = torch.tensor(m["target"], device=q.device)
            ce += -(t * torch.log(q.clamp_min(1e-30))).sum().item()
            mass += D.row_log_mass(m, lf, D.mass_key(args.loss_variant)).exp().item()
            correct += int(torch.argmax(q) == torch.argmax(t))
    model.train()
    n = len(rows)
    return {f"{prefix}_loss": tot / n, f"{prefix}_ce_eval_dist": ce / n, f"{prefix}_mass_mean": mass / n, f"{prefix}_acc_argmax": correct / n}


def val_all():
    out = val_eval()
    if args.val_nodiac:
        out.update(val_eval(val_nodiac, "val_nodiac"))
    return out


monitor_rows = val[:args.monitor_rows]


@torch.no_grad()
def monitor_mass():
    model.eval()
    vals = []
    for s in range(0, len(monitor_rows), 8):
        vals += [D.row_log_mass(m, lf, D.mass_key(args.loss_variant)).exp().item()
                 for g in groups(monitor_rows[s:s + 8]) for m, _, lf in forward_rows(g)]
    model.train()
    return float(np.mean(vals)), int(sum(v < 0.5 for v in vals))


group_stats = {"micro_batches": 0, "micro_batches_split": 0, "max_groups": 1, "steps_with_split": 0}


def step_backward(rows):
    """Một micro-batch: mỗi nhóm backward (Σ loss từng dòng của nhóm) / effective_batch — gradient của bước đúng bằng
    Σ loss 32 dòng / 32 (không lấy trung bình theo nhóm hay theo micro-batch). -> (tổng loss, số nhóm)."""
    tot = 0.0
    gs = groups(rows)
    for g in gs:
        l = losses(g)
        (l.sum() / args.effective_batch).backward()
        tot += l.sum().item()
    group_stats["micro_batches"] += 1
    group_stats["micro_batches_split"] += len(gs) > 1
    group_stats["max_groups"] = max(group_stats["max_groups"], len(gs))
    return tot, len(gs)


warmup = max(1, int(args.warmup_frac * total_steps))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

# ------------------------------------------------------------------ G3: VRAM đỉnh trên micro-batch nặng nhất + thời gian K bước thật
if args.g3:
    model.train()
    worst = max(plan, key=lambda x: x["max_micro_padded_tokens"])
    ep, bi = worst["epoch"], worst["batch"]
    rows_w = next(r for b, (r, _, _) in enumerate(epoch_batches(ep)) if b == bi)
    mbs = [rows_w[m * args.micro_batch:(m + 1) * args.micro_batch] for m in range(accum)]
    heavy = max(mbs, key=lambda r: padded_tokens(r)[0])
    print("worst micro", padded_tokens(heavy), [prep(r)[0]["kind"] for r in heavy], "groups", [padded_tokens(g)[0] for g in groups(heavy)], flush=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time(); step_backward(heavy); torch.cuda.synchronize()
    peak_heavy = torch.cuda.max_memory_allocated() / 2**30; reserved_heavy = torch.cuda.max_memory_reserved() / 2**30
    t_heavy = time.time() - t
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    times, toks = [], []
    for b, (rows, modes, idx) in enumerate(epoch_batches(0)):
        if b >= args.g3:
            break
        torch.cuda.synchronize(); t = time.time()
        for m in range(accum):
            step_backward(rows[m * args.micro_batch:(m + 1) * args.micro_batch])
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t); toks.append(plan[b]["padded_tokens"])
    s_per_tok = float(np.sum(times[1:]) / np.sum(toks[1:]))           # bỏ bước đầu (khởi động)
    total_tok = sum(x["padded_tokens"] for x in plan)
    model.eval(); torch.cuda.synchronize(); t = time.time(); monitor_mass(); t_monitor = time.time() - t
    val_tok = sum(padded_tokens([r])[0] for r in val)
    est_train_h = s_per_tok * total_tok / 3600
    n_val_passes = 3 * (2 if args.val_nodiac else 1)                  # bước 0 + cuối 2 epoch; val gốc (+ val bỏ dấu)
    est_val_h = n_val_passes * val_tok * s_per_tok / 3 / 3600          # giả định: forward không grad ≈ 1/3 chi phí train mỗi token
    est_monitor_h = 11 * t_monitor / 3600
    res = {"worst_micro_batch": {"epoch": ep, "batch": bi, "rows": len(heavy), "padded_tokens": padded_tokens(heavy)[0],
                                 "items": padded_tokens(heavy)[1], "peak_alloc_gb": peak_heavy, "peak_reserved_gb": reserved_heavy,
                                 "seconds": t_heavy, "token_budget": args.token_budget},
           "steps_measured": len(times), "step_seconds": times, "step_padded_tokens": toks, "seconds_per_padded_token": s_per_tok,
           "peak_alloc_gb_steps": torch.cuda.max_memory_allocated() / 2**30,
           "plan_total_padded_tokens": total_tok, "plan_steps": len(plan),
           "plan_mean_items_per_step": float(np.mean([x["items"] for x in plan])),
           "monitor_seconds": t_monitor, "estimate_hours": {"train": est_train_h, "val": est_val_h, "monitor": est_monitor_h,
                                                            "total": est_train_h + est_val_h + est_monitor_h},
           "lora_modules_count": len(lora_modules), "lora_target_names": TARGETS, "trainable_params": n_trainable}
    (args.out_dir / "g3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (args.out_dir / "lora_modules.json").write_text(json.dumps(lora_modules, indent=0), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("worst_micro_batch", "seconds_per_padded_token", "estimate_hours", "peak_alloc_gb_steps")}))
    raise SystemExit(0)

# ------------------------------------------------------------------ train
log = open(args.out_dir / "train_log.jsonl", "w", encoding="utf-8")


def emit(rec):
    log.write(json.dumps(rec) + "\n"); log.flush()


monitor_every = max(1, round(args.monitor_frac * total_steps)) if args.monitor_frac > 0 else 0
gate_step = round(args.gate_step_frac * total_steps)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
emit({"event": "val", "step": 0, "epoch": 0, **val_all(), "elapsed_s": time.time() - t0})
if monitor_every:
    mm, ml = monitor_mass()
    emit({"event": "mass", "step": 0, "mean": mm, "n_below_0.5": ml, "n": len(monitor_rows)})
model.train()
step, run_loss, strip_counts = 0, [], []
for epoch in range(args.epochs):
    cnt = {"vi_rows_seen": 0, "state_only": 0, "all": 0}
    for b, (rows, modes, idx) in enumerate(epoch_batches(epoch)):
        cnt["vi_rows_seen"] += sum(r["text_lang"] == "vi" for r in rows)
        for m in modes:
            if m:
                cnt[m] += 1
        tot, ng = 0.0, []
        for m in range(accum):
            t_, n_ = step_backward(rows[m * args.micro_batch:(m + 1) * args.micro_batch])
            tot += t_; ng.append(n_)
        group_stats["steps_with_split"] += max(ng) > 1
        run_loss.append(tot / args.effective_batch)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            emit({"event": "train", "step": step, "epoch": epoch + (b + 1) / steps_per_epoch, "loss": float(np.mean(run_loss)),
                  "lr": sched.get_last_lr()[0], "grad_norm": float(gn), "elapsed_s": time.time() - t0})
            run_loss = []
        if monitor_every and step % monitor_every == 0:
            mm, ml = monitor_mass()
            emit({"event": "mass", "step": step, "mean": mm, "n_below_0.5": ml, "n": len(monitor_rows), "elapsed_s": time.time() - t0})
            if args.gate_mass > 0 and step == gate_step and mm < args.gate_mass:
                model.save_pretrained(args.out_dir / f"adapter_gate_step{step}")
                emit({"event": "gate_stop", "step": step, "mass_mean": mm, "threshold": args.gate_mass})
                print(json.dumps({"gate_stop": True, "step": step, "mass_mean": mm}))
                raise SystemExit(3)
    strip_counts.append({"epoch": epoch, **cnt})
    emit({"event": "strip_counts", **strip_counts[-1]})
    ve = val_all()
    emit({"event": "val", "step": step, "epoch": epoch + 1, **ve, "group_stats": dict(group_stats), "elapsed_s": time.time() - t0})
    model.save_pretrained(args.out_dir / ("adapter_epoch1" if epoch == 0 else "adapter"))
    print(json.dumps({"epoch_done": epoch + 1, "step": step, **{k: round(v, 4) for k, v in ve.items()}}), flush=True)

summary = {
    "model": D.REPO, "revision": D.REV, "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
    "loss_variant": args.loss_variant, "loss": D.LOSS_VARIANTS[args.loss_variant],
    "strip_p": args.strip_p, "strip_state_only_frac": args.strip_state_only_frac, "strip_counts": strip_counts,
    "monitor_every_steps": monitor_every, "gate": {"step": gate_step, "mass_threshold": args.gate_mass},
    "effective_batch": args.effective_batch, "micro_batch": args.micro_batch, "grad_accum": accum, "token_budget": args.token_budget,
    "group_stats": group_stats,
    "dropout_rng_note": "LoRA dropout dùng RNG toàn cục của torch; số lần gọi và hình dạng tensor phụ thuộc cách chia nhóm, nên mặt nạ "
                        "dropout phụ thuộc cách chia (cách chia cố định theo dữ liệu và ngân sách token, nên chạy lại vẫn tất định)",
    "steps": total_steps, "warmup_steps": warmup, "schedule": "linear warmup + cosine decay to 0", "grad_clip": 1.0,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": TARGETS, "modules": len(lora_modules)},
    "trainable_params": n_trainable, "precision": "bf16 weights + bf16 autocast, LoRA params fp32 (peft default)",
    "gradient_checkpointing": True, "train_rows": len(train), "val_rows": len(val),
    "train_sha256": sha256_file(DATA / "train_v0.jsonl"), "val_sha256": sha256_file(DATA / "val_v0.jsonl"),
    "val_by_epoch": [json.loads(l) for l in open(args.out_dir / "train_log.jsonl") if '"event": "val"' in l],
    "wall_time_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
}
(args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(args.out_dir / "lora_modules.json").write_text(json.dumps(lora_modules, indent=0), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("seed", "wall_time_s", "peak_vram_gb")}))
