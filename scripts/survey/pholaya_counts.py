"""Khảo sát train_v1 (nhóm 1 + phần PhoLaya của nhóm 2): đếm dòng, tác vụ, kiểu câu hỏi, trung vị độ dài state (token Qwen3,
cùng tokenizer với bộ đếm của train_v0) cho split train của từng nguồn trong LakoreAI/PhoLaya-base (revision đã ghim của train_v0).
Chỉ tải split train, vào results/train_v1_survey/raw/pholaya/. Bỏ massive (bench_v1). viglue xử lý riêng (file 586 MB).
Đếm số dòng đã dùng trong train_v0 + val_v0 cho vinli / vsfc / visfd. Không in nội dung dữ liệu.
Ra: results/train_v1_survey/pholaya_counts.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, model_revisions, read_jsonl  # noqa: E402

PL = ("LakoreAI/PhoLaya-base", "034d4273fb2600582cb789c45defe86b1c07e12d")
OUT = RESULTS / "train_v1_survey"
RAW = OUT / "raw" / "pholaya"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
used = Counter()
for f in ("train_v0.jsonl", "val_v0.jsonl"):
    for r in read_jsonl(DATA / f):
        if r["dataset"].startswith("pholaya/"):
            used[r["source"]] += 1
res = {"repo": PL[0], "revision": PL[1], "sources": {}}
for ds in sys.argv[1:] or ["vinli", "vsfc", "visfd", "uts", "vianli", "victsd", "vsmec"]:
    path = Path(hf_hub_download(PL[0], f"sources/{ds}/train.jsonl", repo_type="dataset", revision=PL[1], local_dir=RAW))
    n_rows, per_q, kinds, nq_per_row, lens, keys = 0, Counter(), defaultdict(Counter), Counter(), [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            n_rows += 1
            keys[tuple(sorted(x))] += 1
            state = json.loads(x["state"]) if isinstance(x["state"], str) else x["state"]
            qs = json.loads(x["questions"]) if isinstance(x["questions"], str) else x["questions"]
            nq_per_row[len(qs)] += 1
            for qid, q in qs.items():
                task = "aspect" if qid.startswith("aspect_") else qid
                per_q[task] += 1
                kinds[task][q.get("type")] += 1
            if n_rows % 7 == 0 or n_rows < 200:  # mẫu đều để tính trung vị độ dài (đủ cho trung vị)
                text = "\n".join(str(v) for v in state.values())
                lens.append(len(tok.encode(text, add_special_tokens=False)))
    rec = {"file_bytes": path.stat().st_size, "rows": n_rows, "row_keys": [list(k) for k, _ in keys.most_common(2)],
           "questions_per_row": dict(nq_per_row), "tasks": {t: {"questions": c, "kinds": dict(kinds[t]),
                                                                "used_in_train_v0_val_v0": used.get(f"pholaya/{ds}/{t}", 0)}
                                                            for t, c in per_q.items()},
           "state_tokens_median": float(np.median(lens)), "state_tokens_p90": float(np.percentile(lens, 90)),
           "state_tokens_sampled": len(lens)}
    res["sources"][ds] = rec
    print(ds, n_rows, {t: (v["questions"], v["used_in_train_v0_val_v0"]) for t, v in rec["tasks"].items()},
          "median", rec["state_tokens_median"], flush=True)
(OUT / "pholaya_counts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
