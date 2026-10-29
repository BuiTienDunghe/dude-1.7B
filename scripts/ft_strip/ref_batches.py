"""Cổng G2 (tham chiếu): đường dữ liệu của lần train đối chứng (results/ft_loss), chạy bằng CHÍNH các dòng code cũ.
Các đoạn code dưới đây được chép nguyên văn từ scripts/ref/train_lora_ft_loss.py (bản đã tạo adapter đối chứng; sha256 ghi
trong output) và script assert rằng từng đoạn có mặt nguyên văn trong file đó. Ghi hash id dòng và hash token cho N bước đầu.
Dùng: ref_batches.py <N> <out.jsonl>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, load_json, model_revisions, read_jsonl, render_prompt, sha256_file  # noqa: E402

REF = Path(__file__).resolve().parents[1] / "ref" / "train_lora_ft_loss.py"
SRC = REF.read_text(encoding="utf-8")
SNIPPETS = [
    '''def encode(row):
    ids = tok.encode(render_prompt(row, tok, frames), add_special_tokens=False)
    return ids''',
    '''def shuffled(row, rng):
    perm = rng.permutation(len(row["options"]))
    return {**row, "options": [row["options"][p] for p in perm], "target": [row["target"][p] for p in perm]}''',
    '''    rng = np.random.default_rng(args.seed * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(steps_per_epoch):
        idx = order[b * args.effective_batch:(b + 1) * args.effective_batch]
        rows = [shuffled(train[i], rng) for i in idx]''',
    '''        encs = [encode(r) for r in rows]
        srt = sorted(range(len(rows)), key=lambda i: len(encs[i]))
        rows, encs = [rows[i] for i in srt], [encs[i] for i in srt]''',
]
for s in SNIPPETS:
    assert s in SRC, "đoạn code tham chiếu không có nguyên văn trong file cũ"

N, out_path = int(sys.argv[1]), Path(sys.argv[2])


class A:   # tham số của lần train đối chứng
    seed, effective_batch, epochs = 1, 32, 2


args = A()
MID = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(MID, revision=model_revisions()[MID])
frames = load_json(CONFIG / "prompt_frames.json")
train = read_jsonl(DATA / "train_v0.jsonl")
steps_per_epoch = len(train) // args.effective_batch
exec(SNIPPETS[0]); exec(SNIPPETS[1])   # định nghĩa encode, shuffled đúng như file cũ

out, step = [], 0
for epoch in range(args.epochs):
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    order = rng.permutation(len(train))
    for b in range(steps_per_epoch):
        idx = order[b * args.effective_batch:(b + 1) * args.effective_batch]
        rows = [shuffled(train[i], rng) for i in idx]
        encs = [encode(r) for r in rows]
        srt = sorted(range(len(rows)), key=lambda i: len(encs[i]))
        rows, encs = [rows[i] for i in srt], [encs[i] for i in srt]
        out.append({"step": step + 1, "epoch": epoch, "ids_sha": hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest(),
                    "tokens_sha": hashlib.sha256(json.dumps(encs).encode()).hexdigest()})
        step += 1
        if step >= N:
            break
    if step >= N:
        break
out_path.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
print(json.dumps({"ref_file_sha256": sha256_file(REF), "dumped_steps": len(out)}))
