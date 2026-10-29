"""Giai đoạn 0 train_v1, soi schema (không ghi dữ liệu): 2 dòng mẫu (cắt ngắn) mỗi nguồn tasksource phần B, cùng phân bố kind / câu hỏi;
revision hiện tại + danh sách file + features của các nguồn bản địa phần A. Ra: results/train_v1/inspect_sources.json."""
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, load_json  # noqa: E402

man = load_json(DATA / "manifest_train_v0.json")
TS = (man["sources"]["tasksource"]["repo"], man["sources"]["tasksource"]["revision"])
B = ["race/high", "race/middle", "quail", "cosmos_qa", "dream", "banking77", "clinc_oos/plus", "multidogo/airline/intent", "multidogo/fastfood/intent",
     "multidogo/finance/intent", "multidogo/insurance/intent", "multidogo/media/intent", "multidogo/software/intent", "vitaminc", "nli_fever", "health_fact",
     "super_glue/boolq", "super_glue/boolq_passage", "boolq-natural-perturbations", "sharc"]
cut = lambda s: (s or "")[:160]
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
src = ts["source"]
out = {"tasksource": {"repo": TS[0], "revision": TS[1], "columns": ts.column_names, "sources": {}}}
for s in B:
    idx = [i for i, x in enumerate(src) if x == s]
    rows = [ts[i] for i in idx]
    out["tasksource"]["sources"][s] = {
        "rows": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)), "license_use": dict(Counter(r["license_use"] for r in rows)),
        "license": dict(Counter(str(r.get("license")) for r in rows)), "variants": dict(Counter(r["variant"] for r in rows).most_common(5)),
        "questions_top": dict(Counter(cut(r["question"]) for r in rows).most_common(4)),
        "n_options": dict(Counter(len(r["options"] or []) for r in rows)),
        "examples": [{k: (cut(v) if isinstance(v, str) else (v[:6] if isinstance(v, list) else v)) for k, v in rows[j].items()} for j in (0, len(rows) // 2) if rows]}
api = HfApi()
NAT = {"VSMRC/mrc": None, "mhardalov/exams": None, "phamluan/vi-tourism-admin-intent": None, "kornwtp/VNTC-10Topics": None, "LakoreAI/PhoLaya-base": None,
       "uitnlp/ViANLI": None}
out["native"] = {}
for repo in NAT:
    info = api.dataset_info(repo, files_metadata=False)
    out["native"][repo] = {"sha": info.sha, "gated": info.gated, "license_tags": [t for t in (info.tags or []) if t.startswith("license:")],
                           "files": [s.rfilename for s in info.siblings][:60]}
(RESULTS / "train_v1").mkdir(parents=True, exist_ok=True)
(RESULTS / "train_v1" / "inspect_sources.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("ok")
