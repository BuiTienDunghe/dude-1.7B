"""Khôi phục dấu cho bản KHÔNG DẤU TOÀN BỘ của dev_v1 (state, câu hỏi, lựa chọn) bằng một bộ khôi phục seq2seq.

Cách gọi theo thẻ model: AutoModelForSeq2SeqLM, fp32, generate greedy với max_length=256. Batch 1 (mỗi đoạn một lần gọi).
Giới hạn 256 token: tách văn bản theo xuống dòng, rồi theo ranh giới câu (sau . ! ? … và khoảng trắng); gom các câu liền nhau
thành đoạn ≤ CHUNK token của bộ khôi phục. Câu dài hơn CHUNK thì tách tại khoảng trắng (đếm và báo). Không cắt bỏ phần nào:
các đoạn đã khôi phục được ghép lại bằng đúng khoảng trắng gốc giữa chúng.

Cách dùng: restore_dev.py <vit5|small> <out_dir> [--pilot N]
Ra: <out_dir>/dev_v1_vi_restored_<name>.jsonl  (id = id bản có dấu + ":restore:<name>", variant = "restored_<name>", n_tokens = null)
    <out_dir>/log_<name>.jsonl  (thời gian từng dòng, số đoạn, số lần tách cưỡng bức, số đầu ra chạm max_length)
    <out_dir>/meta_<name>.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_json, read_jsonl, sha256_file, write_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("out_dir")
ap.add_argument("--pilot", type=int, default=0, help="chỉ chạy N dòng (nửa đầu danh sách + nửa dài nhất) để soi thử")
ap.add_argument("--chunk", type=int, default=160,
                help="số token tối đa của một đoạn đầu vào. 160 vì trên state dev_v1, số token bản có dấu / bản không dấu có p99 = 1,5 "
                     "(scripts/restore/tok_ratio.py), nên đầu ra ~240 < max_length 256. Chọn theo thống kê tokenizer, không theo điểm.")
args = ap.parse_args()

cfg = load_json(CONFIG / "restorers.json")[args.name]
DEV = DATA / "bench_v1" / "dev_v1.jsonl"
MAX_LEN = 256
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if args.name == "vit5":
    # transformers 5.19 không dựng được T5Tokenizer từ repo này (TypeError ở 'vocab'); nạp tokenizer.json của chính repo.
    # Đã kiểm (scripts/restore/tok_check_vit5.py): trên 7.254 chuỗi tiếng Việt của dev_v1, sau khi bỏ khoảng trắng đầu/cuối,
    # id trùng 100% với sentencepiece spiece.model + </s>. Mọi chuỗi đưa vào model ở đây đều đã bỏ khoảng trắng đầu/cuối.
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(cfg["repo"], "tokenizer.json", revision=cfg["revision"]),
                                  eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
else:
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
model = AutoModelForSeq2SeqLM.from_pretrained(cfg["repo"], revision=cfg["revision"], use_safetensors=True).cuda().eval()
EOS = model.generation_config.eos_token_id
EOS = set(EOS if isinstance(EOS, list) else [EOS])


def ntok(s):
    return len(tok(s, add_special_tokens=True)["input_ids"])


stats = {"calls": 0, "truncated": 0, "forced_splits": 0}
chunk_log = []  # mọi lần gọi: đầu vào, đầu ra, có chạm max_length không — để dựng lại bản ghép khác mà không chạy lại GPU
ctx = {}


@torch.no_grad()
def gen(text):
    enc = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_length=MAX_LEN, num_beams=1, do_sample=False)
    stats["calls"] += 1
    seq = out[0].tolist()
    trunc = len(seq) >= MAX_LEN and seq[-1] not in EOS
    stats["truncated"] += trunc
    txt = tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if ctx:
        chunk_log.append({**ctx, "k": ctx.setdefault("_k", 0), "in": text, "out": txt, "truncated": bool(trunc)})
        ctx["_k"] += 1
    return txt


SPLIT = re.compile(r"(\n+|(?<=[.!?…])[ \t]+)")


def pieces(core):
    """Một câu quá dài: tách tại khoảng trắng thành các mẩu ≤ CHUNK token. -> list[(mẩu, khoảng trắng sau)]."""
    parts = re.split(r"(\s+)", core)
    words = [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    out, cur = [], []
    for w in words:
        trial = "".join(a + b for a, b in cur) + w[0]
        if cur and ntok(trial) > args.chunk:
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    if len(out) > 1:
        stats["forced_splits"] += 1
    return [("".join(a + b for a, b in grp[:-1]) + grp[-1][0], grp[-1][1]) for grp in out]


def restore_text(text):
    """-> (văn bản đã khôi phục, số đoạn). Giữ nguyên mọi khoảng trắng ngoài các đoạn."""
    if not text.strip():
        return text, 0
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    body = text.strip()
    parts = SPLIT.split(body)  # [câu, sep, câu, sep, ..., câu]
    units = [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    # gom các câu liền nhau thành đoạn ≤ CHUNK token; không gom qua xuống dòng
    chunks, cur = [], []
    for u, sep in units:
        if ntok(u) > args.chunk:
            if cur:
                chunks.append(cur)
                cur = []
            ps = pieces(u)
            ps[-1] = (ps[-1][0], ps[-1][1] + sep)
            chunks.extend([[x] for x in ps])
            continue
        trial = "".join(a + b for a, b in cur) + u
        if cur and ntok(trial) > args.chunk:
            chunks.append(cur)
            cur = []
        cur.append((u, sep))
        if "\n" in sep:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    out = []
    for ch in chunks:
        src = "".join(a + b for a, b in ch[:-1]) + ch[-1][0]
        core = src.strip()  # khoảng trắng đầu/cuối đoạn giữ nguyên bên ngoài, không đưa vào model
        if not core:
            out.append(src + ch[-1][1])
            continue
        lw, rw = src[:len(src) - len(src.lstrip())], src[len(src.rstrip()):]
        out.append(lw + gen(core) + rw + ch[-1][1])
    return lead + "".join(out) + trail, len(chunks)


rows = read_jsonl(DEV)
src_rows = [r for r in rows if r["lang"] == "vi" and r["variant"] == "nodiac"]
if args.pilot:
    h = args.pilot // 2
    longest = sorted(src_rows, key=lambda r: -len(r["state"]))[:args.pilot - h]
    ids = {r["id"] for r in src_rows[:h]} | {r["id"] for r in longest}
    src_rows = [r for r in src_rows if r["id"] in ids]

# làm nóng (không tính giờ)
for r in src_rows[:5]:
    gen(r["question"].strip())
stats.update(calls=0, truncated=0, forced_splits=0)

out_rows, log = [], []
t_all = time.time()
for k, r in enumerate(src_rows):
    assert r["id"].endswith(":nodiac")
    oid = r["id"][: -len(":nodiac")]
    rec = {"id": oid, "source": r["source"], "block": r["block"], "n_state_tokens": r.get("n_state_tokens")}
    c0 = dict(stats)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ctx.clear(); ctx.update(id=oid, field="state")
    st, n_ch = restore_text(r["state"])
    torch.cuda.synchronize(); t1 = time.perf_counter()
    ctx.clear(); ctx.update(id=oid, field="question")
    q, _ = restore_text(r["question"])
    torch.cuda.synchronize(); t2 = time.perf_counter()
    opts = []
    for j, o in enumerate(r["options"]):
        ctx.clear(); ctx.update(id=oid, field=f"option{j}")
        opts.append(restore_text(o)[0])
    ctx.clear()
    torch.cuda.synchronize(); t3 = time.perf_counter()
    rec.update(state_ms=1000 * (t1 - t0), question_ms=1000 * (t2 - t1), options_ms=1000 * (t3 - t2), total_ms=1000 * (t3 - t0),
               state_chunks=n_ch, calls=stats["calls"] - c0["calls"], truncated=stats["truncated"] - c0["truncated"],
               forced_splits=stats["forced_splits"] - c0["forced_splits"])
    log.append(rec)
    out_rows.append({**r, "id": f"{oid}:restore:{args.name}", "variant": f"restored_{args.name}", "n_tokens": None,
                     "state": st, "question": q, "options": opts})
    if (k + 1) % 200 == 0:
        print(f"{k + 1}/{len(src_rows)} {time.time() - t_all:.0f}s", flush=True)

p = out_dir / f"dev_v1_vi_restored_{args.name}.jsonl"
write_jsonl(p, out_rows)
write_jsonl(out_dir / f"log_{args.name}.jsonl", log)
write_jsonl(out_dir / f"chunks_{args.name}.jsonl", [{k: v for k, v in c.items() if k != "_k"} for c in chunk_log])
meta = {"restorer": cfg, "dev_v1_sha256": sha256_file(DEV), "rows": len(out_rows), "chunk_tokens": args.chunk, "max_length": MAX_LEN,
        "decoding": "greedy, num_beams=1, fp32, batch 1", "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        **{f"total_{k}": v for k, v in stats.items()}, "wall_s": time.time() - t_all, "out_sha256": sha256_file(p)}
(out_dir / f"meta_{args.name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("rows", "total_calls", "total_truncated", "total_forced_splits", "wall_s")}))
