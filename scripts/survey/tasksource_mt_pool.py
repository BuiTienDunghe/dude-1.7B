"""Khảo sát train_v1, nhóm 3 (chỉ ước lượng, KHÔNG dịch gì): trong tasksource-jev-typed-decisions (revision của train_v0),
những tác vụ thuộc dạng bài còn thiếu ở tiếng Việt (đọc hiểu, chủ đề, ý định, kiểm chứng, yes/no) còn bao nhiêu dòng chưa dùng,
và mỗi dòng dài bao nhiêu token (Qwen3, state + câu hỏi + lựa chọn) — để ước lượng số dòng dịch được mỗi giờ.
Cũng đếm các dòng tiếng Việt sẵn có trong tasksource (nguồn multilingual/*vi) chưa dùng.
Ra: results/train_v1_survey/tasksource_pool.json. Không in nội dung dữ liệu.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS, load_json, model_revisions, read_jsonl  # noqa: E402

man = load_json(DATA / "manifest_train_v0.json")
TS = (man["sources"]["tasksource"]["repo"], man["sources"]["tasksource"]["revision"])
excluded = set(man["excluded_tasksource_sources"])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=model_revisions()["Qwen/Qwen3-0.6B"])
used = Counter(r["source"] for f in ("train_v0.jsonl", "val_v0.jsonl") for r in read_jsonl(DATA / f) if r["dataset"] == "tasksource")
ts = load_dataset(TS[0], "default", revision=TS[1], split="train")
src_col = ts["source"]
pool = Counter(src_col)
kind_col = ts["kind"]

# dạng bài còn thiếu -> mẫu tên nguồn (khớp theo tên nguồn của tasksource; danh sách cuối cùng do người duyệt)
FORMS = {
    "đọc hiểu": r"^(race|race-c|quail|cosmos_qa|mctest|dream|quartz|sciq|qasc|openbookqa|multirc|boolq|drop|squad|cloth|reclor|logiqa)(/|$)",
    "chủ đề": r"^(ag_news|yahoo_answers_topics|dbpedia|20_newsgroups|trec|open_question_type|stackoverflow-questions|tweet_topic)(/|$)",
    "ý định": r"(banking77|clinc|snips|hwu|atis|intent)",
    "kiểm chứng": r"^(vitaminc|nli_fever|fever|health_fact|liar|climate_fever|scifact|covid|pubhealth|averitec)(/|$)",
    "yes/no": r"(boolq|sharc|strategyqa|spartqa-yn|yes|bool|^yn)",
}
res = {"repo": TS[0], "revision": TS[1], "rows_total": len(ts), "forms": {}, "vi_sources": {}}
by_src_idx = {}
for i, s in enumerate(src_col):
    by_src_idx.setdefault(s, []).append(i)
rng = np.random.default_rng(0)


def stats(idx):
    pick = rng.choice(idx, size=min(300, len(idx)), replace=False)
    lens = []
    for j in pick:
        x = ts[int(j)]
        text = "\n".join([x["state"] or "", x["question"] or ""] + list(x["options"] or []))
        lens.append(len(tok.encode(text, add_special_tokens=False)))
    return {"tokens_median": float(np.median(lens)), "tokens_mean": float(np.mean(lens)), "tokens_p90": float(np.percentile(lens, 90)),
            "sampled": len(lens)}


for form, pat in FORMS.items():
    out = {}
    for s in sorted(pool):
        if re.search(pat, s, flags=re.I) and not s.startswith("multilingual/"):
            kinds = Counter(kind_col[i] for i in by_src_idx[s])
            out[s] = {"rows": pool[s], "used_v0": used.get(s, 0), "excluded": s in excluded, "kinds": dict(kinds), **stats(by_src_idx[s])}
    res["forms"][form] = out
    print(form, len(out), sum(v["rows"] - v["used_v0"] for v in out.values() if not v["excluded"]), flush=True)
for s in sorted(pool):
    if s.startswith("multilingual/") and re.search(r"(?:[-/_])vi$", s):
        res["vi_sources"][s] = {"rows": pool[s], "used_v0": used.get(s, 0), "excluded": s in excluded,
                                "kinds": dict(Counter(kind_col[i] for i in by_src_idx[s])), **stats(by_src_idx[s])}
print("vi sources", {k: (v["rows"], v["used_v0"], v["excluded"]) for k, v in res["vi_sources"].items()})
(RESULTS / "train_v1_survey" / "tasksource_pool.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
