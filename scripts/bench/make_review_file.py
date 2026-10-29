"""Ghi 20 dòng dev_v1 mỗi nguồn bản địa (kèm bản không dấu) ra file để người làm dự án đọc duyệt.
Không in ra màn hình (ViHSD có thể chứa ngôn từ thù ghét). State cắt còn 1.200 ký tự trong file.
Ra: results/dev_v1/review_native_dev20.md
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, LETTERS, RESULTS, SEED, read_jsonl  # noqa: E402

rows = read_jsonl(DATA / "bench_v1" / "dev_v1.jsonl")
byid = {r["id"]: r for r in rows}
L = ["# 20 dòng dev_v1 mỗi nguồn bản địa (để duyệt)", "",
     "Mỗi mục: bản có dấu rồi bản không dấu (cùng pair_id). State cắt còn 1.200 ký tự. Đáp án đúng đánh dấu ✔.",
     "Nguồn ViHSD có thể chứa ngôn từ xúc phạm hoặc thù ghét.", ""]
rng = np.random.default_rng(SEED)
for src in ("vifactcheck", "viwikifc", "vihsd"):
    orig = sorted(r["id"] for r in rows if r["source"] == src and r["variant"] == "orig")
    pick = sorted(rng.choice(len(orig), 20, replace=False))
    L += [f"## {src}", ""]
    for j in pick:
        r = byid[orig[j]]; n = byid[r["id"] + ":nodiac"]
        L.append(f"### {r['pair_id']} — {r['kind']} — nhãn gốc: {r['gold_label']}")
        for tag, x in (("có dấu", r), ("không dấu", n)):
            st = x["state"] if len(x["state"]) <= 1200 else x["state"][:1200] + " …"
            L.append(f"**{tag}** — Văn bản: {st}")
            L.append(f"Câu hỏi: {x['question']}")
            L.append("Lựa chọn: " + " | ".join(f"{LETTERS[i]}. {o}{' ✔' if i == x['target'] else ''}" for i, o in enumerate(x["options"])))
            L.append("")
out = RESULTS / "dev_v1" / "review_native_dev20.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(out, "written")
